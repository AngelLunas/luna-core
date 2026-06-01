"""``yield_iteration`` — context system tool.

Hands carry/append/done state back to the iteration runtime, ending
the agent's turn. Context scope: not visible in the Agents view, not
toggleable, auto-injected by ``engine/iteration.py`` when a node opts
into iteration mode. Terminal: after a successful call, the agent's
tool-calling loop exits and the captured arguments become the
AgentRunner's return value.

The input_schema is generic — ``next_carry`` is an open object whose
exact shape is declared by the per-node carry_schema and enforced at
the iteration runtime level (not in this handler). Validating here
would couple the handler to per-node config that it doesn't otherwise
need to know about.

The handler itself is a thin acknowledgement; the actual state
transfer happens out-of-band when the AgentRunner sees a terminal
system tool fire and returns its captured args to the caller.
"""
from __future__ import annotations

from typing import Any

from luna_core.mcp.system_tools.registry import SystemToolRegistry

TOOL_NAME = "yield_iteration"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "next_carry": {
            "type": "object",
            "description": (
                "Updated values for the iteration carry. The shape "
                "must match the carry_schema declared on this node — "
                "the prompt tells you the exact fields. Pass values "
                "through unchanged if they don't apply to this turn."
            ),
            "additionalProperties": True,
        },
        "append": {
            "type": "array",
            "description": (
                "Items produced this iteration to accumulate into the "
                "node's final items output. Omit or pass [] when this "
                "turn produced nothing."
            ),
            "items": {},
        },
        "done": {
            "type": "boolean",
            "description": (
                "Set to true when no further iterations are needed "
                "(quota met, cursor exhausted, etc). The flow stops "
                "looping immediately after this call."
            ),
        },
    },
    "required": ["next_carry", "done"],
    "additionalProperties": False,
}

DESCRIPTION = (
    "Close this iteration of the loop and hand state back to the "
    "flow. Pass the updated carry for the next turn, any items to "
    "accumulate, and done=true when the loop should stop. Calling "
    "this ends your turn — do all other tool work first."
)


async def handler(
    args: dict[str, Any], *, call_context: dict[str, Any]
) -> dict[str, Any]:
    iteration_index = call_context.get("iteration_index")
    return {
        "ok": True,
        "iteration": iteration_index,
        "done": bool(args.get("done", False)),
    }


def register(registry: SystemToolRegistry) -> None:
    from luna_core.mcp.system_tools.registry import SystemTool

    registry.register(
        SystemTool(
            name=TOOL_NAME,
            description=DESCRIPTION,
            input_schema=INPUT_SCHEMA,
            handler=handler,
            scope="context",
            terminal=True,
        )
    )


__all__ = ["DESCRIPTION", "INPUT_SCHEMA", "TOOL_NAME", "handler", "register"]
