"""Agent execution loop.

AgentRunner orchestrates one ai_agent node's worth of work:
  - rebuild canonical message history from AgentMessages
  - assemble the agent's tool list from three sources, in this order of
    priority for name collisions:
      1. context system tools (injected by the caller for this run only —
         e.g. ``yield_iteration`` for iterative nodes)
      2. catalog system tools (in-process handlers registered globally;
         eventually filtered per agent via ``AgentOperation``)
      3. MCP-advertised connector tools
  - call the LLM with the agent's system instructions
  - on tool_use blocks: dispatch each — system tools resolve to a local
    handler short-circuiting the MCP HTTP call; everything else hits the
    MCPClient. Persist results as user-role AgentMessages carrying
    tool_result blocks, loop.
  - on a terminal system tool firing successfully: exit the tool loop
    early and return the tool's input args as the agent's output.
  - on terminal text: optionally validate against agent.output_schema, return

Every message and tool interaction is persisted to AgentMessage. The provider
itself saves the assistant turn (partial-on-abort included); the runner only
saves the user-role envelopes around it (the incoming user message and any
tool_result responses).
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import jsonschema
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.engine.emitter import EventEmitter
from luna_core.llm.base import ToolDefinition as LLMToolDefinition
from luna_core.llm.router import LLMRouter
from luna_core.mcp.client import MCPClient
from luna_core.mcp.schemas import ToolDefinition as MCPToolDefinition
from luna_core.mcp.system_tools import SystemTool, SystemToolRegistry, get_default_registry
from luna_core.models.agent import Agent, AgentOperation, AgentSystemToolGrant
from luna_core.models.connector import Operation
from luna_core.models.event import AgentMessageRole, RunEventType

logger = logging.getLogger(__name__)


class AgentRunnerError(RuntimeError):
    pass


class OutputSchemaValidationError(AgentRunnerError):
    pass


# Cap iterations defensively — a runaway agent that keeps emitting tool_use
# blocks should fail fast rather than loop forever.
MAX_TOOL_ITERATIONS = 16


# Operators flip this env var to dump every LLM round-trip (resolved
# system prompt, full message history, advertised tools, response
# blocks) at WARNING level so it surfaces in docker logs without
# touching log-level config. Off by default — the payloads are big.
_DEBUG_LLM_CALLS_ENV = "LUNA_DEBUG_LLM_CALLS"


def _debug_llm_calls_enabled() -> bool:
    return os.environ.get(_DEBUG_LLM_CALLS_ENV, "").lower() in (
        "1", "true", "yes", "on",
    )


def _truncate_for_log(value: Any, *, limit: int) -> str:
    """JSON-dump (or str-fallback) + truncate for one-line log readability."""
    if value is None:
        return "(none)"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} bytes)"


class AgentRunner:
    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_client: MCPClient,
        *,
        system_tool_registry: SystemToolRegistry | None = None,
    ):
        self._llm = llm_router
        self._mcp = mcp_client
        # Default to the process-wide registry. Tests pass an isolated
        # instance so they can register/unregister tools without leaking
        # into the global state.
        self._system_tools = system_tool_registry or get_default_registry()

    async def run(
        self,
        agent: Agent,
        history: list[dict[str, Any]],
        new_message: str | None,
        flow_run_id: uuid.UUID,
        node_id: str,
        emitter: EventEmitter,
        db: AsyncSession,
        redis: Redis,
        system_prompt: str | None = None,
        context_tool_names: list[str] | None = None,
        extra_call_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str:
        # ----- prepare tools ------------------------------------------------
        allowed_names = await self._allowed_tool_names(db, agent.id)

        # Catalog system tools are eligible for the same per-agent filter
        # MCP tools go through. Until the DB-seeding work lands, agents
        # without explicit assignments see all catalog tools (matches the
        # existing unfiltered-defaults behavior for MCP tools).
        catalog_tools = self._system_tools.list_catalog()
        if allowed_names is not None:
            catalog_tools = [t for t in catalog_tools if t.name in allowed_names]

        # Context tools are injected by the caller for this run only and
        # bypass the allowed-names filter — they're intrinsic to the
        # runtime context (e.g. yield_iteration in iterative nodes), not
        # a per-agent capability the user toggles.
        context_tools = self._system_tools.get_many(context_tool_names or [])

        # MCP tools are listed last and filtered by the allowed-names set
        # (when one exists). Their names compete with system tool names
        # in the dispatcher — system tools win because we check the
        # registry first.
        mcp_tools = await self._mcp.list_tools()
        if allowed_names is not None:
            mcp_tools = [t for t in mcp_tools if t.name in allowed_names]

        # Combine for LLM advertisement. Order matters only for human
        # readability of the resulting list; the dispatcher is name-keyed.
        llm_tools = [
            *(_system_to_llm_tool(t) for t in context_tools),
            *(_system_to_llm_tool(t) for t in catalog_tools),
            *(_to_llm_tool(t) for t in mcp_tools),
        ]
        system_by_name: dict[str, SystemTool] = {
            t.name: t for t in (*context_tools, *catalog_tools)
        }

        # ----- build canonical message history -----------------------------
        messages: list[dict[str, Any]] = list(history)
        if new_message:
            user_block = {"type": "text", "text": new_message}
            await emitter.save_agent_message(
                node_id=node_id,
                role=AgentMessageRole.user,
                content=[user_block],
            )
            messages.append({"role": "user", "content": [user_block]})

        await emitter.emit(
            RunEventType.agent_thinking,
            node_id=node_id,
            payload={
                "agent_id": str(agent.id),
                "tools_available": len(llm_tools),
            },
        )

        # Call context threaded into every system tool handler. The
        # dispatcher merges per-call deltas (extra_call_context) so
        # callers like the iteration runtime can pass iteration_index
        # without us having to know about it here.
        call_context: dict[str, Any] = {
            "flow_run_id": flow_run_id,
            "node_id": node_id,
            "redis": redis,
            "db": db,
        }
        if extra_call_context:
            call_context.update(extra_call_context)

        # ----- tool-calling loop -------------------------------------------
        output_schema = agent.output_schema or None
        resolved_system = (
            system_prompt if system_prompt is not None else build_system_prompt(agent)
        )
        for iteration in range(MAX_TOOL_ITERATIONS):
            if _debug_llm_calls_enabled():
                logger.warning(
                    "LLM call agent=%s model=%s node=%s iter=%d/%d\n"
                    "  system:   %s\n"
                    "  history:  %s\n"
                    "  tools:    %s\n"
                    "  tool_defs: %s",
                    agent.name,
                    agent.model,
                    node_id,
                    iteration + 1,
                    MAX_TOOL_ITERATIONS,
                    _truncate_for_log(resolved_system, limit=4000),
                    _truncate_for_log(messages, limit=8000),
                    [t.name for t in llm_tools],
                    _truncate_for_log(
                        [
                            {
                                "name": t.name,
                                "description": t.description,
                                "input_schema": t.input_schema,
                            }
                            for t in llm_tools
                        ],
                        limit=4000,
                    ),
                )

            response_blocks = await self._llm.complete(
                provider_id=agent.llm_provider_id,
                messages=messages,
                system=resolved_system,
                tools=llm_tools,
                temperature=agent.temperature,
                model=agent.model,
                output_schema=output_schema if not llm_tools else None,
                run_id=flow_run_id,
                node_id=node_id,
            )

            if _debug_llm_calls_enabled():
                tool_uses_preview = [
                    {"name": b.get("name"), "input": b.get("input")}
                    for b in response_blocks
                    if b.get("type") == "tool_use"
                ]
                logger.warning(
                    "LLM response agent=%s node=%s iter=%d\n"
                    "  blocks_count: %d\n"
                    "  tool_uses:    %s\n"
                    "  full_blocks:  %s",
                    agent.name,
                    node_id,
                    iteration + 1,
                    len(response_blocks),
                    _truncate_for_log(tool_uses_preview, limit=2000)
                    if tool_uses_preview
                    else "(none — model produced text-only response)",
                    _truncate_for_log(response_blocks, limit=4000),
                )

            messages.append({"role": "assistant", "content": response_blocks})

            tool_uses = [b for b in response_blocks if b.get("type") == "tool_use"]
            if not tool_uses:
                return await self._finalize_output(response_blocks, output_schema)

            tool_result_blocks: list[dict[str, Any]] = []
            terminal_value: Any = None
            terminal_called = False
            for tool_use in tool_uses:
                tool_name = tool_use.get("name", "")
                tool_input = tool_use.get("input", {}) or {}
                tool_call_id = tool_use.get("id", "")

                await emitter.emit(
                    RunEventType.tool_called,
                    node_id=node_id,
                    payload={
                        "tool_use_id": tool_call_id,
                        "name": tool_name,
                        "input": tool_input,
                    },
                )

                system_tool = system_by_name.get(tool_name)
                if system_tool is not None:
                    payload_content, is_error = await _invoke_system_tool(
                        system_tool, tool_input, call_context
                    )
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": payload_content,
                            **({"is_error": True} if is_error else {}),
                        }
                    )
                    await emitter.emit(
                        RunEventType.tool_result,
                        node_id=node_id,
                        payload={
                            "tool_use_id": tool_call_id,
                            "name": tool_name,
                            "is_error": is_error,
                            "output_preview": _preview(payload_content),
                        },
                    )
                    if system_tool.terminal and not is_error:
                        terminal_called = True
                        terminal_value = tool_input
                    continue

                result = await self._mcp.call_tool(tool_name, tool_input)
                payload_content = (
                    result.error_message if result.is_error else result.output
                )
                if not isinstance(payload_content, str):
                    payload_content = json.dumps(payload_content, default=str)

                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": payload_content,
                        **({"is_error": True} if result.is_error else {}),
                    }
                )

                await emitter.emit(
                    RunEventType.tool_result,
                    node_id=node_id,
                    payload={
                        "tool_use_id": tool_call_id,
                        "name": tool_name,
                        "is_error": result.is_error,
                        "output_preview": _preview(payload_content),
                    },
                )

            await emitter.save_agent_message(
                node_id=node_id,
                role=AgentMessageRole.user,
                content=tool_result_blocks,
            )
            messages.append({"role": "user", "content": tool_result_blocks})

            if terminal_called:
                # A terminal system tool short-circuits the LLM loop: the
                # agent has explicitly handed control back to whoever
                # injected the tool. We return the captured arguments so
                # the caller (e.g. the iteration runtime) can act on them.
                return terminal_value if isinstance(terminal_value, dict) else {}

        raise AgentRunnerError(
            f"agent exceeded MAX_TOOL_ITERATIONS={MAX_TOOL_ITERATIONS}"
        )

    # ------------------------------------------------------------------ helpers
    async def _allowed_tool_names(
        self, db: AsyncSession, agent_id: uuid.UUID
    ) -> set[str] | None:
        """Return the set of tool names the agent may use, or ``None`` if
        the agent has no assignments of either kind (in which case all
        advertised tools are allowed — useful for trusted internal
        agents during development).

        Unions two sources:
          - ``AgentOperation`` rows → connector operation names (HTTP
            tools advertised by the MCP server)
          - ``AgentSystemToolGrant`` rows → in-process system tool names
            (catalog tools registered in
            ``luna_core.mcp.system_tools``)

        Both feed the same name-based filter the runner applies before
        building ``llm_tools``. Names are flat (no namespacing) because
        the catalog already disallows collisions across providers.
        """
        op_result = await db.execute(
            select(Operation.name)
            .join(
                AgentOperation,
                AgentOperation.operation_id == Operation.id,
            )
            .where(AgentOperation.agent_id == agent_id)
        )
        operation_names = {row[0] for row in op_result.all()}

        grant_result = await db.execute(
            select(AgentSystemToolGrant.tool_name).where(
                AgentSystemToolGrant.agent_id == agent_id
            )
        )
        system_tool_names = {row[0] for row in grant_result.all()}

        union = operation_names | system_tool_names
        return union or None

    async def _finalize_output(
        self,
        blocks: list[dict[str, Any]],
        output_schema: dict[str, Any] | None,
    ) -> dict[str, Any] | str:
        text = "".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        ).strip()
        if not output_schema:
            return text

        parsed: Any
        try:
            parsed = json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise OutputSchemaValidationError(
                f"agent output was not valid JSON: {exc}"
            ) from exc
        try:
            jsonschema.validate(parsed, output_schema)
        except jsonschema.ValidationError as exc:
            raise OutputSchemaValidationError(
                f"agent output failed schema validation: {exc.message}"
            ) from exc
        return parsed


def build_system_prompt(
    agent: Agent,
    *,
    role: str | None = None,
    instructions: str | None = None,
) -> str:
    """Build the agent's system prompt.

    ``role`` and ``instructions`` overrides let the caller pass in
    template-resolved strings (e.g. with ``${context.profile.name}`` already
    substituted) without mutating the ORM record. ``None`` means "fall back
    to the agent record's verbatim field".
    """
    pieces = []
    actual_role = role if role is not None else agent.role
    actual_instructions = (
        instructions if instructions is not None else agent.instructions
    )
    if actual_role:
        pieces.append(f"Role: {actual_role}")
    if actual_instructions:
        pieces.append(actual_instructions)
    if agent.output_schema:
        pieces.append(
            "When you are finished using tools, return ONLY a JSON object "
            "matching this schema:\n" + json.dumps(agent.output_schema)
        )
    return "\n\n".join(pieces)


# Back-compat alias: previously private; kept so anything still importing the
# underscore name continues to work during the rollout.
_build_system_prompt = build_system_prompt


def _to_llm_tool(tool: MCPToolDefinition) -> LLMToolDefinition:
    return LLMToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )


def _system_to_llm_tool(tool: SystemTool) -> LLMToolDefinition:
    return LLMToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
    )


async def _invoke_system_tool(
    tool: SystemTool,
    args: dict[str, Any],
    call_context: dict[str, Any],
) -> tuple[str, bool]:
    """Run a system tool handler, returning ``(payload_content, is_error)``.

    Agent-visible errors (handler returns an ``{"error": ...}`` dict OR
    raises an exception we treat as a domain error) become tool_result
    blocks with ``is_error=True`` so the agent can react. Programming
    errors raised by handlers (e.g. RuntimeError for missing call
    context) propagate up — those should never be silenced.
    """
    try:
        result = await tool.handler(args, call_context=call_context)
    except RuntimeError:
        # Programming errors propagate so the dispatcher's misuse
        # surfaces during development instead of being masked as a
        # tool_result the LLM treats as input.
        raise
    except Exception as exc:  # noqa: BLE001 — handler errors are domain errors
        return f"{type(exc).__name__}: {exc}", True

    if isinstance(result, dict) and "error" in result and len(result) == 1:
        # Single-key {error: "..."} is the conventional "soft failure"
        # shape — surface as is_error=True so the agent retries.
        return str(result["error"]), True

    if isinstance(result, str):
        return result, False
    return json.dumps(result, default=str), False


def _preview(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


__all__ = [
    "AgentRunner",
    "AgentRunnerError",
    "OutputSchemaValidationError",
    "build_system_prompt",
]
