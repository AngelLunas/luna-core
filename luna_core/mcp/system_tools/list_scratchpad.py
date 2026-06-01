"""``list_scratchpad`` — catalog system tool.

Returns every record currently stashed in a named ``ScratchpadStore``
collection for the current flow run. The mirror image of
``stash_records``: producer agents stash, downstream consumers either
iterate (via ``iteration.source=scratchpad``, one record per turn) or
read the whole collection in one shot via this tool.

The catalog scope means it's toggleable per agent via
``AgentSystemToolGrant``. The input_schema is generic — only the
collection name varies per call — so the same tool serves every flow.

Per-call state (``flow_run_id``, ``redis``) arrives via the
``call_context`` argument that the AgentRunner threads in, same as
``stash_records``.
"""
from __future__ import annotations

from typing import Any

from luna_core.mcp.system_tools.registry import SystemToolRegistry
from luna_core.services.scratchpad import ScratchpadStore

TOOL_NAME = "list_scratchpad"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "collection": {
            "type": "string",
            "description": (
                "Name of the scratchpad collection to read. Must match "
                "the name a producer node stashed into — the flow author "
                "tells you which name in the prompt."
            ),
        },
    },
    "required": ["collection"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "Return every record currently in the named scratchpad collection "
    "for this flow run. Records are opaque to the runtime — the prompt "
    "tells you the shape. Returns {collection, count, records} where "
    "records is a list of {id, record} pairs. Order is not guaranteed; "
    "sort by a field inside the record if you need stable order. Not "
    "terminal."
)


async def handler(
    args: dict[str, Any], *, call_context: dict[str, Any]
) -> dict[str, Any]:
    redis = call_context.get("redis")
    flow_run_id = call_context.get("flow_run_id")
    if redis is None or flow_run_id is None:
        raise RuntimeError(
            "list_scratchpad handler invoked without redis/flow_run_id in call_context"
        )

    collection = args.get("collection")
    if not isinstance(collection, str) or not collection:
        return {"error": "collection must be a non-empty string"}

    store = ScratchpadStore(redis)
    try:
        pairs = await store.list_records(flow_run_id, collection)
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "collection": collection,
        "count": len(pairs),
        "records": [{"id": rid, "record": rec} for rid, rec in pairs],
    }


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
