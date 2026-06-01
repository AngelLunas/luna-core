"""``stash_records`` — catalog system tool.

Writes opaque records to a named ``ScratchpadStore`` collection. The
catalog scope means it's toggleable per agent via AgentOperation
(once the DB-seeding work lands). The input_schema is generic across
every flow: ``collection``, ``records``, optional ``record_ids``. The
per-node ``stash`` config in the inspector parameterizes the *prompt*
(suggests the collection name, drives the field-schema editor chips)
but does not constrain this tool's catalog shape — the same tool
serves every flow.

Per-call state (flow_run_id, redis client) arrives via the
``call_context`` argument that the AgentRunner threads in. The handler
constructs a ``ScratchpadStore`` per call — the store is cheap (a
thin wrapper over the shared Redis client) and per-call construction
keeps the handler stateless and easy to reason about.
"""
from __future__ import annotations

from typing import Any

from luna_core.mcp.system_tools.registry import SystemToolRegistry
from luna_core.services.scratchpad import ScratchpadStore

TOOL_NAME = "stash_records"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "collection": {
            "type": "string",
            "description": (
                "Name of the scratchpad collection to write into. "
                "Must be lowercase ASCII with underscores; the flow author "
                "tells you which name to use in the prompt."
            ),
        },
        "records": {
            "type": "array",
            "description": (
                "Records to stage. Each record is a free-form object — "
                "the prompt tells you which fields to include. Pass an "
                "empty array to stash nothing."
            ),
            "items": {"type": "object"},
        },
        "record_ids": {
            "type": "array",
            "description": (
                "Optional. When provided, must have the SAME LENGTH as "
                "records. Each entry is either a stable id string (use "
                "the remote source's id when one exists, for idempotency) "
                "or null to let the runtime hash the payload. If you "
                "don't have ids for any record, omit this field OR pass "
                "an empty array — both mean 'hash every record'. Do NOT "
                "pass a partial list."
            ),
            "items": {"type": ["string", "null"]},
        },
    },
    "required": ["collection", "records"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "Stage records into a named scratchpad collection for later "
    "consumption by a downstream node (typically another iterative "
    "agent that processes records one at a time). Records are opaque "
    "to the runtime — the prompt tells you the shape. Returns "
    "{stashed: N, collection, ids} where N is the count of newly "
    "stored records and `ids` lists their ids. The call is "
    "idempotent: if a record's id (explicit or hashed) is already in "
    "the collection, that record is reported under `already_stashed` "
    "instead of being re-inserted — there is NO need to retry on "
    "an already_stashed entry. Not terminal: you may call this "
    "multiple times per turn, but a redundant call simply produces "
    "more already_stashed entries and zero new stashes."
)


async def handler(
    args: dict[str, Any], *, call_context: dict[str, Any]
) -> dict[str, Any]:
    redis = call_context.get("redis")
    flow_run_id = call_context.get("flow_run_id")
    if redis is None or flow_run_id is None:
        # Programming error in the dispatcher, not an agent error —
        # raise so the bug is fixed at the source instead of being
        # quietly handed to the LLM as a tool_result.
        raise RuntimeError(
            "stash_records handler invoked without redis/flow_run_id in call_context"
        )

    collection = args.get("collection")
    records = args.get("records") or []
    record_ids_raw = args.get("record_ids")

    if not isinstance(collection, str) or not collection:
        return {"error": "collection must be a non-empty string"}
    if not isinstance(records, list):
        return {"error": "records must be an array"}

    record_ids: list[str | None] | None
    if record_ids_raw is None:
        record_ids = None
    else:
        if not isinstance(record_ids_raw, list):
            return {"error": "record_ids must be an array if provided"}
        # Treat an empty list the same as "omitted entirely". The schema
        # marks record_ids optional, so an LLM that has no stable ids
        # for the batch will naturally send `[]` instead of dropping the
        # key. Without this branch we'd later try to index into a
        # 0-length list (or fail the same-length check downstream),
        # turning a reasonable agent choice into an opaque error.
        if not record_ids_raw:
            record_ids = None
        else:
            record_ids = [None if r is None else str(r) for r in record_ids_raw]

    # Per-node record_schema is injected by the runtime via call_context
    # when declared on node.config.stash.record_schema. Validate every
    # record against it before writing — soft errors so the agent can
    # self-correct on retry. When the node doesn't declare a schema we
    # stay fully permissive (back-compat with flows predating the
    # declarative shape).
    record_schema = call_context.get("stash_record_schema")
    if record_schema:
        validation_errors = _validate_records_against_schema(records, record_schema)
        if validation_errors:
            return {
                "error": "records failed schema validation",
                "details": validation_errors,
            }

    # Per-node dedup is injected as a *bound callable* by the runtime
    # when declared on node.config.stash.dedup. The callable already
    # knows which checker to call and how to project each record into
    # the checker's canonical fields, so this handler stays
    # checker-agnostic. Verdicts come back position-aligned with the
    # input list; entries that are duplicates are skipped and reported
    # in the response so the agent can pick another candidate without
    # burning iteration quota on a doomed retry.
    dedup_checker = call_context.get("stash_dedup_checker")
    duplicates: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    survivor_ids: list[str | None] | None = None
    if dedup_checker is not None and records:
        verdicts = await dedup_checker(records, call_context=call_context)
        if len(verdicts) != len(records):
            # Same shape of programming error as the missing-redis
            # branch — raise so it surfaces to the dispatcher loud
            # instead of being silently handed to the LLM.
            raise RuntimeError(
                "stash_dedup_checker returned "
                f"{len(verdicts)} verdicts for {len(records)} records"
            )
        survivor_ids_buf: list[str | None] = []
        for index, (record, verdict) in enumerate(zip(records, verdicts)):
            if verdict is None:
                survivors.append(record)
                survivor_ids_buf.append(
                    record_ids[index] if record_ids is not None else None
                )
                continue
            entry: dict[str, Any] = {
                "record_index": index,
                "match_kind": verdict.match_kind,
                "reason": verdict.reason,
            }
            if verdict.existing_id is not None:
                entry["existing_id"] = verdict.existing_id
            duplicates.append(entry)
        survivor_ids = survivor_ids_buf if record_ids is not None else None
    else:
        survivors = records
        survivor_ids = record_ids

    store = ScratchpadStore(redis)

    # Idempotency layer: detect records whose id is already present in
    # this collection (either from a previous turn or a previous call
    # within the same turn) and report them under ``already_stashed``
    # without re-issuing the SET. This stops a confused agent from
    # piling up identical re-stashes when it second-guesses its own
    # tool_result and calls stash_records again.
    # ScratchpadError (which subclasses ValueError) can come from either
    # call — bad collection name in list_ids, or bad record shape in
    # compute_target_ids — and both should surface as a soft error the
    # agent can react to, not a hard handler crash.
    try:
        target_ids = ScratchpadStore.compute_target_ids(survivors, survivor_ids)
        existing_in_collection = set(
            await store.list_ids(flow_run_id, collection)
        )
    except ValueError as exc:
        return {"error": str(exc)}

    new_records: list[dict[str, Any]] = []
    new_record_ids: list[str | None] | None = (
        [] if survivor_ids is not None else None
    )
    already_stashed: list[dict[str, Any]] = []
    seen_in_this_call: set[str] = set()
    for index, (record, target_id) in enumerate(zip(survivors, target_ids)):
        if target_id in existing_in_collection or target_id in seen_in_this_call:
            already_stashed.append(
                {
                    "id": target_id,
                    "reason": (
                        "id already present in this collection — this call "
                        "had no effect, no need to retry"
                    ),
                }
            )
            continue
        seen_in_this_call.add(target_id)
        new_records.append(record)
        if new_record_ids is not None:
            new_record_ids.append(
                survivor_ids[index] if survivor_ids is not None else None
            )

    try:
        ids = await store.stash_batch(
            flow_run_id, collection, new_records, record_ids=new_record_ids
        )
    except ValueError as exc:
        # ScratchpadError subclasses ValueError — translate to a tool
        # result the agent can react to rather than crashing the run.
        return {"error": str(exc)}
    response: dict[str, Any] = {
        "stashed": len(ids),
        "collection": collection,
        "ids": ids,
    }
    if already_stashed:
        response["already_stashed"] = already_stashed
    if duplicates:
        response["duplicates"] = duplicates
    return response


def _validate_records_against_schema(
    records: list[Any], record_schema: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return a list of per-record validation issues (empty if all OK).

    The error shape is structured (record_index + field + reason) so the
    agent can fix specific problems instead of having to re-derive what
    went wrong from a free-text message. Validation rules mirror the
    field types declared in carry/record schemas — narrow primitives
    that the editor's row-based UI supports today.
    """
    errors: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(
                {
                    "record_index": record_index,
                    "reason": f"record must be an object, got {type(record).__name__}",
                }
            )
            continue
        for field in record_schema:
            name = field["name"]
            expected_type = field.get("type", "string")
            nullable = bool(field.get("nullable", False))
            if name not in record:
                errors.append(
                    {
                        "record_index": record_index,
                        "field": name,
                        "reason": "required field missing",
                    }
                )
                continue
            value = record[name]
            if value is None:
                if not nullable:
                    errors.append(
                        {
                            "record_index": record_index,
                            "field": name,
                            "reason": "field is null but schema declares it non-nullable",
                        }
                    )
                continue
            if not _value_matches_type(value, expected_type):
                errors.append(
                    {
                        "record_index": record_index,
                        "field": name,
                        "reason": (
                            f"expected {expected_type}, got "
                            f"{type(value).__name__}"
                        ),
                    }
                )
    return errors


def _value_matches_type(value: Any, expected: str) -> bool:
    """Lightweight type check matching the primitives the editor offers.

    Mirrors ``CARRY_PRIMITIVE_TYPES`` semantics — narrow on purpose so
    a JSON-Schema-rich check isn't dragged in. ``bool`` is explicitly
    not an integer here even though Python treats ``bool`` as ``int``;
    the agent declaring a boolean field meant boolean, not 0/1.
    """
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    # Unknown declared type — be permissive rather than fail loud, the
    # schema validator upstream already gates which types reach us.
    return True


def register(registry: SystemToolRegistry) -> None:
    from luna_core.mcp.system_tools.registry import SystemTool

    registry.register(
        SystemTool(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=INPUT_SCHEMA,
            handler=handler,
            scope="catalog",
            terminal=False,
        )
    )


__all__ = ["DESCRIPTION", "INPUT_SCHEMA", "TOOL_NAME", "handler", "register"]
