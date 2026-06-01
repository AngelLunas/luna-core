"""Iteration runtime config helpers for ai_agent nodes.

This module owns the per-node iteration knobs — carry schema validation,
max_iterations clamping, on_no_yield policy — but does NOT own the
``yield_iteration`` tool itself. That tool is a system tool registered
in ``luna_core.mcp.system_tools.yield_iteration`` (context scope) and
auto-injected into the agent's tool list by ``nodes._run_ai_agent_iterative``
via ``AgentRunner.run(context_tool_names=["yield_iteration"])``.

Keeping config helpers here and the tool definition in the system tools
package means: one place to find the tool's schema + handler; one place
to find the per-node config validation + safety caps; no inline tool
synthesis at dispatch time.
"""
from __future__ import annotations

from typing import Any

from luna_core.core.config import settings

# Hard cap on per-node iterations regardless of user config — protects
# against a misconfigured loop devouring quota in production. The
# user-declared ``max_iterations`` is further clamped by this ceiling.
ITERATION_HARD_CEILING = 200
DEFAULT_MAX_ITERATIONS = 50

# Concurrency defaults for scratchpad iteration's parallel mode. The
# *absolute* ceiling lives in settings.iteration_concurrency_max so an
# operator can lift it via env var on bigger hardware without touching
# the codebase. The default applies when the node config sets
# ``execution=parallel`` but omits ``concurrency`` — kept conservative
# so an inattentive flow author doesn't surprise themselves with a
# rate-limit storm.
DEFAULT_ITERATION_CONCURRENCY = 4

# JSON Schema primitives the carry schema supports. Kept narrow on
# purpose so the system tool's ``next_carry`` validation (when added)
# stays trivial for any LLM provider to honor.
CARRY_PRIMITIVE_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "object", "array"}
)

ON_NO_YIELD_TREAT_AS_DONE = "treat_as_done"
ON_NO_YIELD_ERROR = "error"
_VALID_ON_NO_YIELD = (ON_NO_YIELD_TREAT_AS_DONE, ON_NO_YIELD_ERROR)

# Iteration sources decide where each turn's "item" comes from.
#
# - ``agent_yield``: the agent itself drives the loop by calling the
#   ``yield_iteration`` context tool each turn (carry_schema declares
#   the shape that travels between turns). This is the pattern used by
#   fetcher-style agents that paginate an external API.
# - ``scratchpad``: the runtime drives the loop by walking a snapshot
#   of record ids from a ``ScratchpadStore`` collection. Each turn the
#   record is injected as ``${iteration.item}`` and the agent runs with
#   whatever tools it has (typically a normal MCP tool to persist).
#   After the agent's turn ends the runtime drops the record from the
#   scratchpad implicitly — the loop consumes the collection.
ITERATION_SOURCE_AGENT_YIELD = "agent_yield"
ITERATION_SOURCE_SCRATCHPAD = "scratchpad"
_VALID_SOURCES = (ITERATION_SOURCE_AGENT_YIELD, ITERATION_SOURCE_SCRATCHPAD)

# Execution modes for scratchpad iteration. ``sequential`` is the legacy
# behaviour and the default for ``agent_yield`` (which is inherently
# sequential — each turn's carry depends on the previous turn). ``parallel``
# only applies to ``scratchpad`` source and runs up to ``concurrency``
# items concurrently inside the worker's asyncio loop.
ITERATION_EXECUTION_SEQUENTIAL = "sequential"
ITERATION_EXECUTION_PARALLEL = "parallel"
_VALID_EXECUTION = (ITERATION_EXECUTION_SEQUENTIAL, ITERATION_EXECUTION_PARALLEL)

# What to do when one parallel iteration raises. ``continue`` keeps the
# rest of the in-flight tasks running (matches the sequential default of
# letting one bad item fail the run without already-done work being
# wasted, but applied per-iteration); ``cancel_siblings`` cancels every
# pending/running sibling and re-raises the first error, useful when the
# items are part of a single logical batch that should be all-or-nothing.
ITERATION_ON_ERROR_CONTINUE = "continue"
ITERATION_ON_ERROR_CANCEL_SIBLINGS = "cancel_siblings"
_VALID_ON_ITERATION_ERROR = (
    ITERATION_ON_ERROR_CONTINUE,
    ITERATION_ON_ERROR_CANCEL_SIBLINGS,
)


class IterationSourceError(ValueError):
    """Raised when a node's iteration.source / source_config is malformed."""


def resolve_iteration_source(raw: Any) -> str:
    """Normalize the iteration.source token. Defaults to ``agent_yield``.

    Unknown values fall back to the default — keeps old flows (which
    predate the field entirely) and typos behaving the way they used
    to instead of crashing the run.
    """
    if raw in _VALID_SOURCES:
        return raw
    return ITERATION_SOURCE_AGENT_YIELD


def resolve_scratchpad_collection(source_config: Any, *, node_id: str) -> str:
    """Pull and validate the collection name from a scratchpad-source config.

    The scratchpad source needs to know *which* collection to read. We
    require it explicitly via ``iteration.source_config.collection`` so
    misconfiguration surfaces at run time as a clear error instead of
    silently iterating over the wrong (or empty) bucket.
    """
    if not isinstance(source_config, dict):
        raise IterationSourceError(
            f"ai_agent node {node_id}: iteration.source_config must be an "
            f"object when source is {ITERATION_SOURCE_SCRATCHPAD!r}"
        )
    collection = source_config.get("collection")
    if not isinstance(collection, str) or not collection:
        raise IterationSourceError(
            f"ai_agent node {node_id}: iteration.source_config.collection "
            "must be a non-empty string"
        )
    return collection


class CarrySchemaError(ValueError):
    """Raised when a user-declared carry_schema is malformed."""


def resolve_stash_record_schema(
    stash_cfg: Any, *, node_id: str
) -> list[dict[str, Any]] | None:
    """Pull and validate ``node.config.stash.record_schema`` if declared.

    Returns the validated schema (same row shape as carry_schema —
    we reuse the validator), or ``None`` when the node doesn't
    declare a stash schema. ``None`` means "no validation"; the
    stash_records handler stays permissive in that case for
    backward compatibility with flows that predate this feature.

    Lives in iteration.py because it reuses ``validate_carry_schema``
    and both helpers serve the same "per-node schema validation"
    concern; not iteration-specific despite the file name.
    """
    if not isinstance(stash_cfg, dict):
        return None
    raw_schema = stash_cfg.get("record_schema")
    if not raw_schema:
        return None
    # Surface validation errors with a "record_schema" wording instead of
    # "carry_schema" so the message matches the config the user is
    # actually looking at in the inspector.
    try:
        return validate_carry_schema(raw_schema, node_id=node_id)
    except CarrySchemaError as exc:
        raise CarrySchemaError(
            str(exc).replace("carry_schema", "stash.record_schema")
        ) from exc


def format_stash_schema_addendum(
    record_schema: list[dict[str, Any]] | None,
) -> str | None:
    """Render the per-node stash record_schema as a system-prompt block.

    The ``stash_records`` system tool exposes a *generic* input_schema
    (collection, records, record_ids) because it's registered once
    process-wide. The actual per-record shape lives in the node config
    and is enforced at handler time via ``call_context``. That means
    the model never sees the required record shape in the tool
    advertisement — it learns it only by trying and reading the error.

    This formatter closes that gap by producing a stable text block the
    caller can append to the agent's system prompt for the current node.
    The agent then knows the shape on its first call instead of burning
    a turn on a validation error.

    Returns ``None`` when the schema is missing or empty so the caller
    can branch on "do I have anything to append?" without inspecting.
    """
    if not record_schema:
        return None
    lines: list[str] = [
        "# Stash records contract (this node only)",
        "",
        "When you call `stash_records` on this node, every record in the "
        "`records` array must match this exact shape:",
        "",
    ]
    for field in record_schema:
        name = field.get("name", "?")
        type_ = field.get("type", "?")
        nullable = bool(field.get("nullable"))
        default = field.get("default")
        flags: list[str] = []
        if nullable:
            flags.append("nullable")
        else:
            flags.append("required")
        if default not in (None, ""):
            flags.append(f"default={default!r}")
        lines.append(f"- {name}: {type_} ({', '.join(flags)})")
    lines.extend(
        [
            "",
            "Missing required fields, null where not allowed, or wrong types "
            "come back as a tool error pointing at the offending record. Fix "
            "just that record and retry the affected chunk.",
        ]
    )
    return "\n".join(lines)


def validate_carry_schema(carry_schema: Any, *, node_id: str) -> list[dict[str, Any]]:
    """Validate and return the carry_schema for an iterative ai_agent node.

    Returns the same list (after type-checks) so callers can use it
    directly without re-walking. Raises ``CarrySchemaError`` with the
    node id baked into the message so the runner can wrap it into a
    NodeExecutionError without losing the context.
    """
    if not isinstance(carry_schema, list):
        raise CarrySchemaError(
            f"ai_agent node {node_id}: iteration.carry_schema must be a list"
        )
    for field in carry_schema:
        if not isinstance(field, dict):
            raise CarrySchemaError(
                f"ai_agent node {node_id}: carry_schema entries must be objects"
            )
        name = field.get("name")
        if not isinstance(name, str) or not name:
            raise CarrySchemaError(
                f"ai_agent node {node_id}: carry_schema entry missing 'name'"
            )
        ftype = field.get("type", "string")
        if ftype not in CARRY_PRIMITIVE_TYPES:
            raise CarrySchemaError(
                f"ai_agent node {node_id}: carry_schema field {name!r} "
                f"has unsupported type {ftype!r}"
            )
    return carry_schema


def resolve_max_iterations(raw: Any) -> int:
    """Coerce a user-supplied max_iterations to a safe integer.

    Non-integers, zero, and negatives fall back to ``DEFAULT_MAX_ITERATIONS``;
    everything is then clamped to ``ITERATION_HARD_CEILING``.
    """
    value = raw if isinstance(raw, int) and raw > 0 else DEFAULT_MAX_ITERATIONS
    return min(value, ITERATION_HARD_CEILING)


def resolve_on_no_yield(raw: Any) -> str:
    """Normalize the on_no_yield policy to one of the supported tokens."""
    return raw if raw in _VALID_ON_NO_YIELD else ON_NO_YIELD_TREAT_AS_DONE


def resolve_iteration_execution(raw: Any) -> str:
    """Normalize iteration.execution; unknowns fall back to sequential.

    Defaulting to sequential keeps every flow predating this field
    behaving exactly as before — opting into parallel must be deliberate.
    """
    return raw if raw in _VALID_EXECUTION else ITERATION_EXECUTION_SEQUENTIAL


def resolve_iteration_concurrency(raw: Any) -> int:
    """Coerce a user-supplied concurrency to a safe integer.

    Non-positive integers / non-ints fall back to
    ``DEFAULT_ITERATION_CONCURRENCY``; the result is clamped to the
    operator-controlled ``settings.iteration_concurrency_max`` ceiling
    so a flow author can't run away from the host hardware budget.
    """
    value = raw if isinstance(raw, int) and raw > 0 else DEFAULT_ITERATION_CONCURRENCY
    ceiling = max(1, settings.iteration_concurrency_max)
    return min(value, ceiling)


def resolve_on_iteration_error(raw: Any) -> str:
    """Normalize the on_iteration_error policy for parallel mode."""
    return (
        raw if raw in _VALID_ON_ITERATION_ERROR else ITERATION_ON_ERROR_CONTINUE
    )


__all__ = [
    "CARRY_PRIMITIVE_TYPES",
    "CarrySchemaError",
    "DEFAULT_ITERATION_CONCURRENCY",
    "DEFAULT_MAX_ITERATIONS",
    "ITERATION_EXECUTION_PARALLEL",
    "ITERATION_EXECUTION_SEQUENTIAL",
    "ITERATION_HARD_CEILING",
    "ITERATION_ON_ERROR_CANCEL_SIBLINGS",
    "ITERATION_ON_ERROR_CONTINUE",
    "ITERATION_SOURCE_AGENT_YIELD",
    "ITERATION_SOURCE_SCRATCHPAD",
    "IterationSourceError",
    "ON_NO_YIELD_ERROR",
    "ON_NO_YIELD_TREAT_AS_DONE",
    "resolve_iteration_concurrency",
    "resolve_iteration_execution",
    "resolve_iteration_source",
    "resolve_max_iterations",
    "resolve_on_iteration_error",
    "resolve_on_no_yield",
    "resolve_scratchpad_collection",
    "format_stash_schema_addendum",
    "resolve_stash_record_schema",
    "validate_carry_schema",
]
