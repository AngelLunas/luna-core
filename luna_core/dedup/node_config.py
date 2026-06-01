"""Per-node ``stash.dedup`` config resolution and dispatch glue.

This module is the bridge between (a) the user-declared
``node.config.stash.dedup`` blob the flow editor writes and (b) the
runtime-bound callable the ``stash_records`` handler reads from
``call_context["stash_dedup_checker"]``.

Two responsibilities, each in its own helper:

  - ``resolve_stash_dedup_binding`` — validates the config shape, looks
    up the named checker in the registry, validates the field mapping
    against the checker's required fields, and returns a "binding"
    dataclass the dispatcher can use to build a per-call callable.
    Raises ``StashDedupConfigError`` (subclass of ValueError) with the
    node id baked in so callers can wrap into ``NodeExecutionError``
    without re-formatting.

  - ``build_call_context_checker`` — takes a binding + the per-call
    ``call_context`` the AgentRunner threads through, returns the
    async callable the ``stash_records`` handler invokes. The
    projection of each agent-supplied record through the field
    mapping happens here, so the checker never has to know what the
    raw record dict looked like.

Why split: resolution can happen once per node (cheap config check)
and is what the editor sees errors from; binding-to-call_context
happens once per dispatch and needs runtime values (db, redis, etc.).
Keeping the steps separate makes it easy to test each in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luna_core.dedup.registry import (
    DedupChecker,
    DedupCheckerRegistry,
    DedupVerdict,
    get_default_registry,
)


class StashDedupConfigError(ValueError):
    """Raised when ``node.config.stash.dedup`` is malformed.

    Subclasses ``ValueError`` so callers can ``except ValueError`` when
    they don't care to distinguish (mirrors ``CarrySchemaError``).
    """


@dataclass(frozen=True)
class StashDedupBinding:
    """The resolved dedup config for one node — checker + field mapping.

    ``field_map`` keys are the checker's canonical field names; values
    are the agent-record field names to read for each. The dispatcher
    uses this to project records before handing them to the checker.
    """

    checker: DedupChecker
    field_map: dict[str, str]


def resolve_stash_dedup_binding(
    stash_cfg: Any,
    *,
    node_id: str,
    registry: DedupCheckerRegistry | None = None,
) -> StashDedupBinding | None:
    """Resolve ``stash_cfg["dedup"]`` against the registry.

    Returns ``None`` when no dedup is configured (the absence path is
    explicit so the dispatcher can branch cleanly on "this node opts
    in or not"). Raises ``StashDedupConfigError`` for malformed configs
    so a misconfigured node fails loudly at run time rather than
    silently bypassing dedup.

    Validation rules:
      - ``dedup.checker`` must be a non-empty string naming a
        registered checker.
      - ``dedup.fields`` must be an object mapping every
        ``required_field`` of the checker to a non-empty record field
        name. Extra keys are dropped (forward compatibility — a
        checker may expose more required fields later without
        invalidating existing configs).
    """
    if not isinstance(stash_cfg, dict):
        return None
    dedup_cfg = stash_cfg.get("dedup")
    if not dedup_cfg:
        return None
    if not isinstance(dedup_cfg, dict):
        raise StashDedupConfigError(
            f"ai_agent node {node_id}: stash.dedup must be an object"
        )

    name = dedup_cfg.get("checker")
    if not isinstance(name, str) or not name:
        raise StashDedupConfigError(
            f"ai_agent node {node_id}: stash.dedup.checker must be a "
            "non-empty string"
        )

    target = registry if registry is not None else get_default_registry()
    checker = target.get(name)
    if checker is None:
        raise StashDedupConfigError(
            f"ai_agent node {node_id}: stash.dedup.checker {name!r} is "
            "not a registered dedup checker"
        )

    raw_fields = dedup_cfg.get("fields") or {}
    if not isinstance(raw_fields, dict):
        raise StashDedupConfigError(
            f"ai_agent node {node_id}: stash.dedup.fields must be an "
            "object mapping checker fields to record fields"
        )

    field_map: dict[str, str] = {}
    for spec in checker.required_fields:
        mapped = raw_fields.get(spec.name)
        if mapped in (None, ""):
            if spec.optional:
                continue
            raise StashDedupConfigError(
                f"ai_agent node {node_id}: stash.dedup.fields is missing "
                f"a mapping for required field {spec.name!r} of checker "
                f"{checker.name!r}"
            )
        if not isinstance(mapped, str):
            raise StashDedupConfigError(
                f"ai_agent node {node_id}: stash.dedup.fields[{spec.name!r}] "
                f"must be a string (the record field to read), got "
                f"{type(mapped).__name__}"
            )
        field_map[spec.name] = mapped

    return StashDedupBinding(checker=checker, field_map=field_map)


def build_call_context_checker(
    binding: StashDedupBinding,
):
    """Build the async callable the ``stash_records`` handler invokes.

    The returned callable matches the contract the handler reads from
    ``call_context["stash_dedup_checker"]``::

        async def checker(
            records: list[dict],
            *,
            call_context: dict,
        ) -> list[DedupVerdict | None]: ...

    Projection happens here: each agent record is reshaped into the
    checker's canonical field names before reaching the checker's
    handler. That keeps checkers agnostic to the record shape the
    flow author chose, and makes it possible to reuse the same
    checker across nodes that produce subtly different record
    schemas.

    Records missing one of the mapped fields produce a verdict that
    tells the LLM exactly which field was missing — surfaced through
    the same return shape so the handler doesn't need a second code
    path for "couldn't even check". The LLM treats it as "this record
    is invalid, try another", same as a duplicate.
    """

    field_map = binding.field_map
    checker = binding.checker
    optional_canonicals = {
        spec.name for spec in checker.required_fields if spec.optional
    }

    async def _bound(
        records: list[dict[str, Any]],
        *,
        call_context: dict[str, Any],
    ) -> list[DedupVerdict | None]:
        # Project every record up front so the checker handler sees a
        # clean list of dicts keyed by its own canonical field names.
        projected: list[dict[str, Any] | None] = []
        missing_verdicts: list[DedupVerdict | None] = []
        for record in records:
            if not isinstance(record, dict):
                projected.append(None)
                missing_verdicts.append(
                    DedupVerdict(
                        match_kind="invalid_record",
                        existing_id=None,
                        reason="record is not an object — cannot extract "
                        "dedup fields",
                    )
                )
                continue
            slim: dict[str, Any] = {}
            missing: list[str] = []
            for canonical, source in field_map.items():
                if source not in record:
                    # Optional fields that the agent didn't include just
                    # drop out of the projection — the checker handler
                    # sees an absent key, not an error.
                    if canonical in optional_canonicals:
                        continue
                    missing.append(source)
                    continue
                slim[canonical] = record[source]
            if missing:
                projected.append(None)
                missing_verdicts.append(
                    DedupVerdict(
                        match_kind="invalid_record",
                        existing_id=None,
                        reason=(
                            "record is missing field(s) required for "
                            f"dedup: {missing!r}. Cannot check for "
                            "duplicates without them."
                        ),
                    )
                )
                continue
            projected.append(slim)
            missing_verdicts.append(None)

        # Only forward the records the projection succeeded for. We
        # keep the position-to-position relationship intact by passing
        # the full list (with placeholders for misses) and merging
        # results back so the handler can map verdicts to original
        # records by index.
        forward_records = [p for p in projected if p is not None]
        forward_results: list[DedupVerdict | None]
        if forward_records:
            forward_results = await checker.handler(
                forward_records, call_context=call_context
            )
            if len(forward_results) != len(forward_records):
                # A misbehaving checker would silently drop verdicts;
                # raise so the bug surfaces at the source instead of
                # confusing the LLM with a half-checked batch.
                raise RuntimeError(
                    f"dedup checker {checker.name!r} returned "
                    f"{len(forward_results)} verdicts for "
                    f"{len(forward_records)} records"
                )
        else:
            forward_results = []

        # Splice forward verdicts back into the per-record list
        # alongside the projection-failure verdicts.
        out: list[DedupVerdict | None] = []
        forward_iter = iter(forward_results)
        for projected_record, miss_verdict in zip(projected, missing_verdicts):
            if projected_record is None:
                out.append(miss_verdict)
            else:
                out.append(next(forward_iter))
        return out

    return _bound


def format_stash_dedup_addendum(
    binding: StashDedupBinding | None,
) -> str | None:
    """Render the per-node dedup config as a system-prompt block.

    Mirrors ``format_stash_schema_addendum``: the ``stash_records``
    tool advertises a generic input_schema, so the LLM has no way to
    know *this* node also dedupes against an external store. Without
    a prompt addendum it would learn about it only the first time
    it tries to stash a duplicate.

    Telling the model up front that duplicates come back as a soft
    rejection (not an iteration-count hit) is the load-bearing piece —
    otherwise an LLM that sees `was_duplicate` once may give up
    on the loop thinking it failed.

    Returns ``None`` when no binding is configured so the caller can
    branch on "do I have anything to append?" without inspecting.
    """
    if binding is None:
        return None
    checker = binding.checker
    field_lines: list[str] = []
    for spec in checker.required_fields:
        mapped = binding.field_map.get(spec.name, "?")
        suffix = f" — {spec.description}" if spec.description else ""
        field_lines.append(f"- record.{mapped} → {spec.name}{suffix}")

    lines = [
        "# Stash dedup (this node only)",
        "",
        f"Records passed to `stash_records` are deduplicated against "
        f"`{checker.display_label()}` before they land in the collection. "
        f"The runtime uses these fields from each record to look up matches:",
        "",
        *field_lines,
        "",
        "When the response includes a `duplicates` array, those records were "
        "rejected as already-seen. Each entry has `record_index`, "
        "`match_kind`, and a `reason` explaining why. Duplicates do NOT "
        "count against your iteration quota — find a different candidate "
        "and try again. Records that succeed appear in `ids` and `stashed`.",
    ]
    return "\n".join(lines)


__all__ = [
    "StashDedupBinding",
    "StashDedupConfigError",
    "build_call_context_checker",
    "format_stash_dedup_addendum",
    "resolve_stash_dedup_binding",
]
