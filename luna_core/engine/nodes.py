from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.connectors.registry import ConnectorRegistry
from luna_core.core.db import AsyncSessionLocal
from luna_core.dedup.node_config import (
    StashDedupBinding,
    StashDedupConfigError,
    build_call_context_checker,
    format_stash_dedup_addendum,
    resolve_stash_dedup_binding,
)
from luna_core.engine.agent import AgentRunner, build_system_prompt
from luna_core.engine.emitter import EventEmitter
from luna_core.engine.template_paths import resolve_path
from luna_core.engine.iteration import (
    ITERATION_EXECUTION_PARALLEL,
    ITERATION_ON_ERROR_CANCEL_SIBLINGS,
    ITERATION_SOURCE_AGENT_YIELD,
    ITERATION_SOURCE_SCRATCHPAD,
    CarrySchemaError,
    IterationSourceError,
    format_stash_schema_addendum,
    resolve_iteration_concurrency,
    resolve_iteration_execution,
    resolve_iteration_source,
    resolve_max_iterations,
    resolve_on_iteration_error,
    resolve_on_no_yield,
    resolve_scratchpad_collection,
    resolve_stash_record_schema,
    validate_carry_schema,
)
from luna_core.engine.iteration_context import iteration_scope
from luna_core.llm.base import AbortSignalError, abort_key
from luna_core.services.scratchpad import ScratchpadStore
from luna_core.llm.router import LLMRouter
from luna_core.mcp.client import MCPClient
from luna_core.mcp.system_tools.registry import get_default_registry
from luna_core.mcp.system_tools.yield_iteration import TOOL_NAME as YIELD_ITERATION_TOOL_NAME
from luna_core.models.agent import Agent
from luna_core.models.event import AgentMessage, RunEventType
from luna_core.schemas.flow import FlowDefinition, FlowNode
from luna_core.services.context_sources import (
    SourceLoadContext,
    UnknownSourceError,
    get_context_source,
)

logger = logging.getLogger(__name__)


class NodeExecutionError(RuntimeError):
    """Raised when a node handler fails irrecoverably."""


class HumanCheckpointInterrupt(Exception):
    """Raised inside a human_checkpoint node to pause LangGraph execution.

    The FlowRunner catches this, marks the FlowRun as paused, and persists the
    current LangGraph state via its checkpointer. A subsequent resume() call
    appends the human response as an AgentMessage(role=user) and continues from
    the checkpoint.
    """

    def __init__(self, node_id: str, message: str = ""):
        super().__init__(message or f"human checkpoint reached at {node_id}")
        self.node_id = node_id
        self.message = message


DEFAULT_TOOL_RESULT_PREVIEW_LIMIT = 480


class NodeExecutor:
    """Dispatches a flow node to the correct handler and wraps emit/lifecycle.

    Heavy collaborators (LLMRouter, MCPClient, ConnectorRegistry) are injected
    by the host application on startup so luna-core remains a pure library.
    When a collaborator is None and a node needs it, execution raises so the
    misconfiguration surfaces immediately instead of producing fake outputs.

    ``tool_result_preview_limit`` caps the length (in characters) of the
    ``output_preview`` string included on action-node ``tool_result`` events.
    Hosts that surface very large connector responses can raise it; those that
    want tighter streams can shrink it. The full response is always returned
    to the caller of ``execute()`` — the limit only governs what's streamed
    over the event channel for UI consumption.
    """

    def __init__(
        self,
        emitter: EventEmitter,
        db: AsyncSession,
        redis: Redis,
        *,
        llm_router: LLMRouter | None = None,
        mcp_client: MCPClient | None = None,
        connector_registry: ConnectorRegistry | None = None,
        flow_definition: FlowDefinition | None = None,
        tool_result_preview_limit: int = DEFAULT_TOOL_RESULT_PREVIEW_LIMIT,
    ):
        self._emitter = emitter
        self._db = db
        self._redis = redis
        self._llm_router = llm_router
        self._mcp_client = mcp_client
        self._connector_registry = connector_registry
        self._flow_definition = flow_definition
        self._tool_result_preview_limit = tool_result_preview_limit

    async def execute(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        await self._emitter.emit(
            RunEventType.node_started,
            node_id=node.id,
            payload={"type": node.type, "name": node.name},
        )
        try:
            output = await self._dispatch(node, state)
        except HumanCheckpointInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._emitter.emit(
                RunEventType.node_failed,
                node_id=node.id,
                payload={"error": str(exc), "type": exc.__class__.__name__},
            )
            raise NodeExecutionError(f"node {node.id} failed: {exc}") from exc

        await self._emitter.emit(
            RunEventType.node_completed,
            node_id=node.id,
            payload={"output_keys": sorted(output.keys()) if output else []},
        )
        new_outputs = dict(state.get("outputs", {}))
        if output:
            new_outputs.update(output)
        return {
            "outputs": new_outputs,
            "current_node": node.id,
        }

    async def _dispatch(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        match node.type:
            case "action":
                return await self._run_action(node, state)
            case "ai_agent":
                return await self._run_ai_agent(node, state)
            case "condition":
                return await self._run_condition(node, state)
            case "human_checkpoint":
                return await self._run_human_checkpoint(node, state)
            case "trigger":
                return await self._run_trigger(node, state)
            case "output":
                return await self._run_output(node, state)
            case _:
                raise NodeExecutionError(f"unknown node type: {node.type}")

    # ----- handlers ---------------------------------------------------------
    async def _run_action(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        operation_id_raw = node.config.get("operation_id")
        system_tool_name_raw = node.config.get("system_tool_name")
        if operation_id_raw and system_tool_name_raw:
            raise NodeExecutionError(
                f"action node {node.id} declares both operation_id and "
                "system_tool_name; pick exactly one"
            )
        if not operation_id_raw and not system_tool_name_raw:
            raise NodeExecutionError(
                f"action node {node.id} missing config.operation_id or "
                "config.system_tool_name"
            )

        input_data = _resolve_inputs(node.config.get("input", {}), state)

        if system_tool_name_raw:
            return await self._run_action_system_tool(
                node, state, str(system_tool_name_raw), input_data
            )

        if self._connector_registry is None:
            raise NodeExecutionError(
                "action node requires a ConnectorRegistry; host app must "
                "inject one when constructing NodeExecutor"
            )

        operation_id = uuid.UUID(str(operation_id_raw))

        # Look up operation + connector metadata so the streamed event carries
        # enough info for the UI to render an OperationSummary card without an
        # extra REST round-trip per tool call.
        try:
            operation = self._connector_registry.get_operation(operation_id)
            connector = self._connector_registry.get_connector_for(operation_id)
        except KeyError:
            operation = None
            connector = None

        called_payload: dict[str, Any] = {
            "operation_id": str(operation_id),
            "input": input_data,
        }
        if operation is not None:
            called_payload["name"] = operation.name
            called_payload["operation"] = {
                "id": str(operation.id),
                "name": operation.name,
                "description": operation.description,
                "method": operation.method.value,
                "path": operation.path,
            }
        if connector is not None:
            called_payload["connector"] = {
                "name": connector.name,
                "description": connector.description,
                "auth_type": connector.auth_type.value,
                "base_url": connector.base_url,
            }

        await self._emitter.emit(
            RunEventType.tool_called,
            node_id=node.id,
            payload=called_payload,
        )
        result = await self._connector_registry.execute(operation_id, input_data)

        result_payload: dict[str, Any] = {
            "operation_id": str(operation_id),
            "is_error": False,
            "output_preview": _result_preview(
                result, limit=self._tool_result_preview_limit
            ),
        }
        if operation is not None:
            result_payload["name"] = operation.name
        await self._emitter.emit(
            RunEventType.tool_result,
            node_id=node.id,
            payload=result_payload,
        )
        return {node.id: result}

    async def _run_action_system_tool(
        self,
        node: FlowNode,
        state: dict[str, Any],
        tool_name: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch an action node to a catalog system tool.

        Mirrors the connector-operation path: same template-resolved input,
        same tool_called/tool_result events, same {node.id: result} return
        shape. The only differences are how we resolve the tool (in-process
        registry instead of ConnectorRegistry) and the metadata block we
        emit (``system_tool`` instead of ``operation``/``connector``).

        Context-scope tools (yield_iteration, etc.) are rejected because
        they are owned by a specific runtime that injects them at agent
        dispatch time; surfacing them as a node-level capability would
        violate that ownership.
        """
        tool = get_default_registry().get(tool_name)
        if tool is None:
            raise NodeExecutionError(
                f"action node {node.id}: system tool {tool_name!r} not "
                "registered. Hosts register catalog tools at startup; "
                "ensure install_*_system_tools() has run before triggering "
                "the flow."
            )
        if tool.scope != "catalog":
            raise NodeExecutionError(
                f"action node {node.id}: system tool {tool_name!r} has "
                f"scope={tool.scope!r}; only catalog-scope tools may be "
                "invoked from a flow node."
            )

        called_payload: dict[str, Any] = {
            "system_tool_name": tool_name,
            "name": tool_name,
            "input": input_data,
            "system_tool": {
                "name": tool.name,
                "description": tool.description,
                "scope": tool.scope,
            },
        }
        await self._emitter.emit(
            RunEventType.tool_called,
            node_id=node.id,
            payload=called_payload,
        )

        # Build the same call_context shape the AgentRunner threads to
        # system tool handlers; without this, tools that need redis /
        # trigger_user_id (the common case for sentinel tools) would
        # blow up with "missing key" errors that are confusing because
        # they look like agent bugs.
        call_context: dict[str, Any] = {
            "flow_run_id": self._emitter.flow_run_id,
            "node_id": node.id,
            "redis": self._redis,
            "db": self._db,
        }
        extras = _with_call_context_extras(
            stash_record_schema=None,
            stash_dedup_binding=None,
            trigger_user_id=_trigger_user_id_from_state(state),
        )
        if extras:
            call_context.update(extras)

        result = await tool.handler(input_data, call_context=call_context)

        result_payload: dict[str, Any] = {
            "system_tool_name": tool_name,
            "name": tool_name,
            "is_error": False,
            "output_preview": _result_preview(
                result, limit=self._tool_result_preview_limit
            ),
        }
        await self._emitter.emit(
            RunEventType.tool_result,
            node_id=node.id,
            payload=result_payload,
        )
        return {node.id: result}

    async def _run_ai_agent(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        iteration_cfg = node.config.get("iteration")
        if isinstance(iteration_cfg, dict) and iteration_cfg.get("enabled"):
            return await self._run_ai_agent_iterative(node, state, iteration_cfg)

        agent_id_raw = node.config.get("agent_id")
        if not agent_id_raw:
            raise NodeExecutionError(
                f"ai_agent node {node.id} missing config.agent_id"
            )
        if self._llm_router is None or self._mcp_client is None:
            raise NodeExecutionError(
                "ai_agent node requires LLMRouter + MCPClient; host app must "
                "inject both when constructing NodeExecutor"
            )

        agent_id = uuid.UUID(str(agent_id_raw))
        agent = await self._db.get(Agent, agent_id)
        if agent is None:
            raise NodeExecutionError(f"agent {agent_id} not found")

        inherit_from = node.config.get("inherit_history_from") or []
        if not isinstance(inherit_from, list) or not all(
            isinstance(x, str) for x in inherit_from
        ):
            raise NodeExecutionError(
                f"ai_agent node {node.id}: inherit_history_from must be a list of node ids"
            )
        include_tool_interactions = bool(
            node.config.get("inherit_tool_interactions", True)
        )
        if inherit_from:
            self._validate_inheritable_nodes(node.id, inherit_from)

        history = await self._load_history(
            own_node_id=node.id,
            inherited_node_ids=inherit_from,
            include_tool_interactions=include_tool_interactions,
        )

        # Resolve context bindings -> load each required source -> inject into
        # an enriched state copy so template substitution can see them.
        loaded_context = await self._load_agent_context(agent, node, state)
        enriched_state = dict(state)
        merged_context = dict(state.get("context") or {})
        merged_context.update(loaded_context)
        enriched_state["context"] = merged_context

        new_message = _resolve_prompt(node.config, enriched_state)
        resolved_role = (
            _format_template(agent.role, enriched_state) if agent.role else ""
        )
        resolved_instructions = (
            _format_template(agent.instructions, enriched_state)
            if agent.instructions
            else ""
        )
        system_prompt = build_system_prompt(
            agent,
            role=resolved_role,
            instructions=resolved_instructions,
        )

        stash_record_schema = self._resolve_stash_schema(node)
        stash_dedup_binding = self._resolve_stash_dedup(node)
        # Append the per-node stash record_schema / dedup as text addenda
        # to the system prompt so the agent knows the exact shape and
        # the dedup rules on its first `stash_records` call. Without
        # these it would learn them only by trying and reading a soft
        # error — one wasted turn per node.
        for addendum in (
            format_stash_schema_addendum(stash_record_schema),
            format_stash_dedup_addendum(stash_dedup_binding),
        ):
            if addendum:
                system_prompt = f"{system_prompt}\n\n{addendum}"

        runner = AgentRunner(self._llm_router, self._mcp_client)
        output = await runner.run(
            agent=agent,
            history=history,
            new_message=new_message,
            scope_id=self._emitter.flow_run_id,
            node_id=node.id,
            emitter=self._emitter,
            db=self._db,
            redis=self._redis,
            system_prompt=system_prompt,
            extra_call_context=_with_call_context_extras(
                stash_record_schema=stash_record_schema,
                stash_dedup_binding=stash_dedup_binding,
                trigger_user_id=_trigger_user_id_from_state(state),
            ),
        )
        result: dict[str, Any] = {node.id: output}
        if loaded_context:
            # Persist loaded context into the flow state so downstream nodes
            # can reference ${context.<name>...} without re-loading.
            result["context"] = loaded_context
        return result

    def _resolve_stash_schema(self, node: FlowNode) -> list[dict[str, Any]] | None:
        """Pull the per-node stash record_schema, ready for call_context injection.

        Returns ``None`` if the node didn't declare one — that's a
        signal to the stash_records handler to stay permissive (no
        validation). Wrapping the iteration.py helper here translates
        the schema-error into a NodeExecutionError with the node id
        baked in, so the failure surfaces at the same layer as other
        per-node config issues.
        """
        try:
            return resolve_stash_record_schema(node.config.get("stash"), node_id=node.id)
        except CarrySchemaError as exc:
            raise NodeExecutionError(str(exc)) from exc

    def _resolve_stash_dedup(self, node: FlowNode) -> StashDedupBinding | None:
        """Pull the per-node stash.dedup binding, ready for call_context injection.

        Returns ``None`` if the node didn't declare a dedup config —
        the stash_records handler then skips the dedup branch entirely.
        Wrapping the dedup.node_config helper here translates the
        config-error into a NodeExecutionError so misconfigurations
        surface at the same layer as other per-node config issues.
        """
        try:
            return resolve_stash_dedup_binding(
                node.config.get("stash"), node_id=node.id
            )
        except StashDedupConfigError as exc:
            raise NodeExecutionError(str(exc)) from exc

    async def _run_ai_agent_iterative(
        self,
        node: FlowNode,
        state: dict[str, Any],
        iteration_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Run an ai_agent node in batched mode.

        The agent is re-invoked once per batch with a *fresh* context window
        (no own history carried forward) so each turn only sees the data
        relevant to that batch. The carry — declared by the user via
        ``iteration.carry_schema`` — is the small piece of state that
        survives between turns: the agent reads it from inputs and writes
        the next value back via the synthesized ``yield_iteration`` tool.
        Appended items accumulate into a single list emitted as
        ``items`` once the loop exits.
        """
        agent_id_raw = node.config.get("agent_id")
        if not agent_id_raw:
            raise NodeExecutionError(
                f"ai_agent node {node.id} missing config.agent_id"
            )
        if self._llm_router is None or self._mcp_client is None:
            raise NodeExecutionError(
                "ai_agent node requires LLMRouter + MCPClient; host app must "
                "inject both when constructing NodeExecutor"
            )

        max_iterations = resolve_max_iterations(iteration_cfg.get("max_iterations"))

        agent_id = uuid.UUID(str(agent_id_raw))
        agent = await self._db.get(Agent, agent_id)
        if agent is None:
            raise NodeExecutionError(f"agent {agent_id} not found")

        inherit_from = node.config.get("inherit_history_from") or []
        if not isinstance(inherit_from, list) or not all(
            isinstance(x, str) for x in inherit_from
        ):
            raise NodeExecutionError(
                f"ai_agent node {node.id}: inherit_history_from must be a list of node ids"
            )
        include_tool_interactions = bool(
            node.config.get("inherit_tool_interactions", True)
        )
        if inherit_from:
            self._validate_inheritable_nodes(node.id, inherit_from)

        loaded_context = await self._load_agent_context(agent, node, state)

        # Dispatch on the iteration source. Both branches share the agent
        # fetch, inheritance validation, and context loading above; what
        # diverges is how each turn's "item" is produced and how the
        # loop terminates.
        source = resolve_iteration_source(iteration_cfg.get("source"))
        # Parallel execution only makes sense for scratchpad (independent
        # items). For agent_yield each turn's carry depends on the
        # previous turn's return, so log and ignore the misconfiguration
        # rather than running something the user didn't actually request.
        if (
            source == ITERATION_SOURCE_AGENT_YIELD
            and resolve_iteration_execution(iteration_cfg.get("execution"))
            == ITERATION_EXECUTION_PARALLEL
        ):
            logger.warning(
                "ai_agent node %s: iteration.execution='parallel' is "
                "incompatible with source='agent_yield' (each turn depends "
                "on the previous carry). Falling back to sequential.",
                node.id,
            )
        if source == ITERATION_SOURCE_SCRATCHPAD:
            result_payload = await self._iterate_with_scratchpad(
                node=node,
                state=state,
                iteration_cfg=iteration_cfg,
                agent=agent,
                inherit_from=inherit_from,
                include_tool_interactions=include_tool_interactions,
                loaded_context=loaded_context,
                max_iterations=max_iterations,
            )
        else:
            result_payload = await self._iterate_with_agent_yield(
                node=node,
                state=state,
                iteration_cfg=iteration_cfg,
                agent=agent,
                inherit_from=inherit_from,
                include_tool_interactions=include_tool_interactions,
                loaded_context=loaded_context,
                max_iterations=max_iterations,
            )

        result: dict[str, Any] = {node.id: result_payload}
        if loaded_context:
            result["context"] = loaded_context
        return result

    async def _iterate_with_agent_yield(
        self,
        *,
        node: FlowNode,
        state: dict[str, Any],
        iteration_cfg: dict[str, Any],
        agent: Agent,
        inherit_from: list[str],
        include_tool_interactions: bool,
        loaded_context: dict[str, dict[str, Any]],
        max_iterations: int,
    ) -> dict[str, Any]:
        """Agent-driven loop: the agent calls ``yield_iteration`` each turn
        to hand carry/append/done state back to the runtime. The fetcher
        pattern (paginated API → normalize → yield) uses this source.
        """
        try:
            carry_schema = validate_carry_schema(
                iteration_cfg.get("carry_schema") or [], node_id=node.id
            )
        except CarrySchemaError as exc:
            raise NodeExecutionError(str(exc)) from exc

        on_no_yield = resolve_on_no_yield(iteration_cfg.get("on_no_yield"))

        # Resolve initial carry: defaults from schema, then template-overrides
        # from initial_carry. ${inputs.x} / ${context.y} are honored so the
        # caller can seed the loop from flow inputs.
        carry: dict[str, Any] = {}
        for field in carry_schema:
            carry[field["name"]] = field.get("default")
        initial_overrides = iteration_cfg.get("initial_carry") or {}
        if isinstance(initial_overrides, dict):
            for key, raw_value in initial_overrides.items():
                if key not in carry:
                    # Unknown keys are ignored — declaring them in carry_schema
                    # is the source of truth; initial_carry only overrides.
                    continue
                carry[key] = _resolve_value(raw_value, state)

        accumulator: list[Any] = []
        exit_reason = "max_iterations"
        iteration_index = 0

        for iteration_index in range(max_iterations):
            # Each turn rebuilds the enriched state so the prompt template
            # sees the *current* carry, not a stale snapshot.
            enriched_state = dict(state)
            merged_context = dict(state.get("context") or {})
            merged_context.update(loaded_context)
            enriched_state["context"] = merged_context
            enriched_state["iteration"] = {
                "carry": dict(carry),
                "index": iteration_index,
            }

            new_message = _resolve_prompt(node.config, enriched_state)
            resolved_role = (
                _format_template(agent.role, enriched_state) if agent.role else ""
            )
            resolved_instructions = (
                _format_template(agent.instructions, enriched_state)
                if agent.instructions
                else ""
            )
            system_prompt = build_system_prompt(
                agent,
                role=resolved_role,
                instructions=resolved_instructions,
            )
            stash_record_schema = self._resolve_stash_schema(node)
            stash_dedup_binding = self._resolve_stash_dedup(node)
            for addendum in (
                format_stash_schema_addendum(stash_record_schema),
                format_stash_dedup_addendum(stash_dedup_binding),
            ):
                if addendum:
                    system_prompt = f"{system_prompt}\n\n{addendum}"

            # include_own=False is the key to "fresh context per iteration":
            # the agent never sees its own past turns even though they were
            # persisted (audit + resume still works).
            history = await self._load_history(
                own_node_id=node.id,
                inherited_node_ids=inherit_from,
                include_tool_interactions=include_tool_interactions,
                include_own=False,
            )

            runner = AgentRunner(self._llm_router, self._mcp_client)
            captured_args = await runner.run(
                agent=agent,
                history=history,
                new_message=new_message,
                scope_id=self._emitter.flow_run_id,
                node_id=node.id,
                emitter=self._emitter,
                db=self._db,
                redis=self._redis,
                system_prompt=system_prompt,
                context_tool_names=[YIELD_ITERATION_TOOL_NAME],
                extra_call_context=_with_call_context_extras(
                    stash_record_schema=stash_record_schema,
                    stash_dedup_binding=stash_dedup_binding,
                    trigger_user_id=_trigger_user_id_from_state(state),
                    base={"iteration_index": iteration_index},
                ),
            )

            # AgentRunner returns the args dict from a terminal system tool
            # call as soon as one fires successfully. yield_iteration is
            # the only terminal tool we inject here, so any dict result
            # with a "done" key is its payload; anything else means the
            # agent ended its turn without yielding.
            if not (
                isinstance(captured_args, dict)
                and "done" in captured_args
                and "next_carry" in captured_args
            ):
                if on_no_yield == "error":
                    raise NodeExecutionError(
                        f"ai_agent node {node.id}: agent finished iteration "
                        f"{iteration_index} without calling yield_iteration"
                    )
                exit_reason = "no_yield"
                break

            next_carry = captured_args.get("next_carry") or {}
            if not isinstance(next_carry, dict):
                # Treat malformed yield as "no yield" so the agent's mistake
                # doesn't propagate as a typed runtime crash.
                if on_no_yield == "error":
                    raise NodeExecutionError(
                        f"ai_agent node {node.id}: yield_iteration returned "
                        f"non-object next_carry at iteration {iteration_index}"
                    )
                exit_reason = "no_yield"
                break
            # Per-field validation against the declared carry_schema —
            # missing required fields short-circuit the loop with the
            # same policy as a missing yield.
            missing = [f["name"] for f in carry_schema if f["name"] not in next_carry]
            if missing:
                if on_no_yield == "error":
                    raise NodeExecutionError(
                        f"ai_agent node {node.id}: yield_iteration next_carry "
                        f"missing fields {missing!r} at iteration {iteration_index}"
                    )
                exit_reason = "no_yield"
                break

            appended = captured_args.get("append") or []
            if isinstance(appended, list):
                accumulator.extend(appended)

            carry = next_carry

            if bool(captured_args.get("done")):
                exit_reason = "done_signal"
                break
        else:
            exit_reason = "max_iterations"

        return {
            "items": accumulator,
            "carry_final": carry,
            "iterations": iteration_index + 1 if exit_reason != "no_yield" else iteration_index,
            "exit_reason": exit_reason,
        }

    async def _iterate_with_scratchpad(
        self,
        *,
        node: FlowNode,
        state: dict[str, Any],
        iteration_cfg: dict[str, Any],
        agent: Agent,
        inherit_from: list[str],
        include_tool_interactions: bool,
        loaded_context: dict[str, dict[str, Any]],
        max_iterations: int,
    ) -> dict[str, Any]:
        """Runtime-driven loop over a scratchpad collection.

        Snapshots the collection's record ids at start, then for each one:
        loads the record, injects it as ``${iteration.item}`` + ids, runs
        the agent with whatever MCP/system tools it has (no synthetic
        terminator — the agent ends its turn naturally, calling whatever
        persistence tool fits), then drops the record from the
        scratchpad. The loop exits when the snapshot is exhausted or
        ``max_iterations`` is hit, whichever comes first.

        Items stashed *during* the loop are NOT picked up — the snapshot
        is a point-in-time view. If a host needs continuous drain
        semantics, that's a different pattern (and out of scope for v1).

        Execution mode is read from ``iteration_cfg.execution``:
        ``sequential`` (the default and only mode for legacy flows) walks
        the snapshot one item at a time; ``parallel`` spawns up to
        ``concurrency`` items concurrently via asyncio. Parallel mode
        loses ordering guarantees — use sequential when downstream
        consumers depend on item order.
        """
        try:
            collection = resolve_scratchpad_collection(
                iteration_cfg.get("source_config"), node_id=node.id
            )
        except IterationSourceError as exc:
            raise NodeExecutionError(str(exc)) from exc

        scratchpad = ScratchpadStore(self._redis)
        run_id = self._emitter.flow_run_id
        snapshot_ids = await scratchpad.list_ids(run_id, collection)
        # Sorting gives a deterministic processing order in sequential
        # mode and a deterministic *dispatch* order in parallel mode
        # (which still helps when replaying logs — the iteration_index
        # values match across runs even if completion times don't).
        snapshot_ids.sort()
        bounded_ids = snapshot_ids[:max_iterations]

        execution = resolve_iteration_execution(iteration_cfg.get("execution"))
        if execution == ITERATION_EXECUTION_PARALLEL:
            outcomes = await self._run_scratchpad_iterations_parallel(
                node=node,
                state=state,
                iteration_cfg=iteration_cfg,
                agent=agent,
                inherit_from=inherit_from,
                include_tool_interactions=include_tool_interactions,
                loaded_context=loaded_context,
                collection=collection,
                bounded_ids=bounded_ids,
                scratchpad=scratchpad,
                run_id=run_id,
            )
        else:
            outcomes = await self._run_scratchpad_iterations_sequential(
                node=node,
                state=state,
                agent=agent,
                inherit_from=inherit_from,
                include_tool_interactions=include_tool_interactions,
                loaded_context=loaded_context,
                collection=collection,
                bounded_ids=bounded_ids,
                scratchpad=scratchpad,
                run_id=run_id,
            )

        processed = sum(1 for o in outcomes if o == "processed")
        skipped_missing = sum(1 for o in outcomes if o == "skipped_missing")
        skipped_aborted = sum(1 for o in outcomes if o == "skipped_aborted")
        failed = sum(1 for o in outcomes if o == "failed")
        exit_reason = (
            "max_iterations"
            if len(snapshot_ids) > max_iterations
            else "exhausted"
        )
        return {
            "processed": processed,
            "skipped_missing": skipped_missing,
            "skipped_aborted": skipped_aborted,
            "failed": failed,
            "snapshot_size": len(snapshot_ids),
            "collection": collection,
            "exit_reason": exit_reason,
            "execution": execution,
        }

    async def _run_scratchpad_iterations_sequential(
        self,
        *,
        node: FlowNode,
        state: dict[str, Any],
        agent: Agent,
        inherit_from: list[str],
        include_tool_interactions: bool,
        loaded_context: dict[str, dict[str, Any]],
        collection: str,
        bounded_ids: list[str],
        scratchpad: ScratchpadStore,
        run_id: uuid.UUID,
    ) -> list[str]:
        """Sequential walk over ``bounded_ids``. Preserves legacy semantics
        exactly: a failure in one iteration propagates up immediately,
        already-processed items keep their state (records dropped, events
        emitted), pending items stay in the scratchpad for the next run.
        """
        outcomes: list[str] = []
        for iteration_index, record_id in enumerate(bounded_ids):
            outcome = await self._execute_single_scratchpad_iteration(
                node=node,
                state=state,
                agent=agent,
                inherit_from=inherit_from,
                include_tool_interactions=include_tool_interactions,
                loaded_context=loaded_context,
                collection=collection,
                record_id=record_id,
                iteration_index=iteration_index,
                scratchpad=scratchpad,
                run_id=run_id,
            )
            outcomes.append(outcome)
        return outcomes

    async def _run_scratchpad_iterations_parallel(
        self,
        *,
        node: FlowNode,
        state: dict[str, Any],
        iteration_cfg: dict[str, Any],
        agent: Agent,
        inherit_from: list[str],
        include_tool_interactions: bool,
        loaded_context: dict[str, dict[str, Any]],
        collection: str,
        bounded_ids: list[str],
        scratchpad: ScratchpadStore,
        run_id: uuid.UUID,
    ) -> list[str]:
        """Concurrent execution over ``bounded_ids``.

        Spawns one asyncio task per record id, bounded by a semaphore
        sized to ``iteration_cfg.concurrency`` (clamped server-side to
        ``settings.iteration_concurrency_max``). Each task runs inside
        its own iteration_scope so the events it emits carry the right
        ``iteration_id`` tag — asyncio's per-task Context copy keeps the
        scopes from leaking into each other.

        Failure policy depends on ``iteration_cfg.on_iteration_error``:
        ``continue`` lets siblings keep running (the failed task is
        recorded in the returned outcomes); ``cancel_siblings`` cancels
        every other pending/in-flight task and re-raises the first error
        wrapped as ``NodeExecutionError``.
        """
        concurrency = resolve_iteration_concurrency(iteration_cfg.get("concurrency"))
        on_error = resolve_on_iteration_error(
            iteration_cfg.get("on_iteration_error")
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(idx: int, record_id: str) -> str:
            async with semaphore:
                return await self._execute_single_scratchpad_iteration(
                    node=node,
                    state=state,
                    agent=agent,
                    inherit_from=inherit_from,
                    include_tool_interactions=include_tool_interactions,
                    loaded_context=loaded_context,
                    collection=collection,
                    record_id=record_id,
                    iteration_index=idx,
                    scratchpad=scratchpad,
                    run_id=run_id,
                )

        tasks = [
            asyncio.create_task(_bounded(idx, rid))
            for idx, rid in enumerate(bounded_ids)
        ]

        async def _cancel_and_drain() -> None:
            """Cancel every still-pending or in-flight task and wait for them.

            Cancellation propagates ``CancelledError`` into each iteration
            body, which (when it lands inside an LLM stream's
            ``async for chunk``) closes the underlying httpx connection
            so OpenRouter actually stops generating tokens — not just
            our reader stopping reading. The drain with
            ``return_exceptions=True`` swallows the CancelledErrors /
            late-raised AbortSignalErrors without triggering asyncio
            "task exception was never retrieved" warnings.
            """
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if on_error == ITERATION_ON_ERROR_CANCEL_SIBLINGS:
            # gather without return_exceptions=True will surface the
            # first exception; we then cancel the rest and re-raise as a
            # NodeExecutionError so the run fails with the same shape
            # the sequential path produces. AbortSignalError gets the
            # same fast-path treatment as in continue mode below — it's
            # a flow-level signal and should propagate unchanged.
            try:
                results = await asyncio.gather(*tasks)
            except AbortSignalError:
                await _cancel_and_drain()
                raise
            except Exception as exc:  # noqa: BLE001
                await _cancel_and_drain()
                raise NodeExecutionError(
                    f"ai_agent node {node.id}: iteration failed in "
                    f"cancel_siblings mode ({type(exc).__name__}: {exc})"
                ) from exc
            return list(results)

        # continue mode: collect outcomes individually; per-item failures
        # become "failed" tags without cancelling siblings.
        #
        # Exception to the policy: AbortSignalError is a flow-level kill
        # switch, not a per-item failure. Treating it as "failed" would
        # mean the abort cascade keeps starting fresh LLM streams as the
        # semaphore frees up — each one would hit the abort key after a
        # round trip and die a few hundred ms in, burning tokens for no
        # output. So as soon as any task raises AbortSignalError, we
        # cancel everything else (the pre-check at the top of
        # _execute_single_scratchpad_iteration handles the not-yet-started
        # ones already, but in-flight LLM calls need explicit cancellation
        # to actually close the httpx stream) and re-raise so the outer
        # runner records the run as aborted.
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        abort_exc: AbortSignalError | None = None
        for result in gathered:
            if isinstance(result, AbortSignalError):
                abort_exc = result
                break
        if abort_exc is not None:
            await _cancel_and_drain()
            raise abort_exc

        outcomes: list[str] = []
        for idx, result in enumerate(gathered):
            if isinstance(result, BaseException):
                # exc_info=(type, value, tb) tells logging to render the
                # full traceback even though we're not inside the
                # except block — without it, the message above only
                # shows repr(exc) and we lose the stack we need to
                # diagnose what cancelled / killed the iteration.
                logger.error(
                    "ai_agent node %s iteration %d failed in parallel "
                    "continue mode: %s: %s",
                    node.id,
                    idx,
                    type(result).__name__,
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                outcomes.append("failed")
            else:
                outcomes.append(result)
        return outcomes

    async def _execute_single_scratchpad_iteration(
        self,
        *,
        node: FlowNode,
        state: dict[str, Any],
        agent: Agent,
        inherit_from: list[str],
        include_tool_interactions: bool,
        loaded_context: dict[str, dict[str, Any]],
        collection: str,
        record_id: str,
        iteration_index: int,
        scratchpad: ScratchpadStore,
        run_id: uuid.UUID,
    ) -> str:
        """Run one item through the agent and return an outcome tag.

        Returns ``"processed"`` on success, ``"skipped_missing"`` when
        the record disappeared between snapshot and fetch (another
        consumer or a TTL got there first — best-effort over a transient
        store), ``"skipped_aborted"`` when the flow was aborted before
        this task got the chance to start its LLM call, or ``"failed"``
        when on_iteration_error=continue and the agent run raised.
        ``cancel_siblings`` mode lets exceptions propagate so
        ``asyncio.gather`` can cancel the rest. ``AbortSignalError`` is
        always re-raised regardless of policy — abort is a flow-level
        signal, not a per-item failure.

        Emits the per-iteration lifecycle (``iteration_started`` /
        ``_completed`` / ``_failed``) and binds the iteration_id to the
        current task's ContextVar so the AgentRunner's nested events
        carry the same tag automatically.
        """
        # Cheap Redis EXISTS before we burn a slot on the LLM. Without
        # this, a parallel batch with concurrency=8 and 50 items will
        # keep starting fresh LLM streams after the user clicked Abort
        # (each new task acquires the semaphore freed by a sibling
        # raising AbortSignalError, sees the abort key mid-stream, and
        # dies a few hundred ms in). The pre-check turns those wasted
        # starts into instant no-ops.
        if await self._redis.exists(abort_key(run_id)):
            return "skipped_aborted"

        record = await scratchpad.get(run_id, collection, record_id)
        if record is None:
            return "skipped_missing"

        iteration_id = uuid.uuid4()

        # Each iteration body opens its own AsyncSession so concurrent
        # siblings never share an aborted transaction. SQLAlchemy's
        # AsyncSession is explicitly NOT safe for concurrent use — two
        # parallel iterations hitting the same session race on the
        # statement buffer, and the first one to error leaves the
        # connection in a "current transaction is aborted, commands
        # ignored until end of transaction block" state that takes down
        # every sibling that tries to commit after it. Sequential mode
        # also benefits (slightly more isolation, no semantic change).
        # The session is returned to the pool on exit.
        async with AsyncSessionLocal() as iter_db:
            iter_emitter = EventEmitter(iter_db, self._redis, run_id)
            await iter_emitter.emit(
                RunEventType.iteration_started,
                node_id=node.id,
                payload={
                    "iteration_id": str(iteration_id),
                    "iteration_index": iteration_index,
                    "item_id": record_id,
                    "collection": collection,
                },
            )
            started_at = time.monotonic()

            try:
                with iteration_scope(iteration_id):
                    enriched_state = dict(state)
                    merged_context = dict(state.get("context") or {})
                    merged_context.update(loaded_context)
                    enriched_state["context"] = merged_context
                    enriched_state["iteration"] = {
                        "item": record,
                        "item_id": record_id,
                        "index": iteration_index,
                        "collection": collection,
                    }

                    new_message = _resolve_prompt(node.config, enriched_state)
                    resolved_role = (
                        _format_template(agent.role, enriched_state)
                        if agent.role
                        else ""
                    )
                    resolved_instructions = (
                        _format_template(agent.instructions, enriched_state)
                        if agent.instructions
                        else ""
                    )
                    system_prompt = build_system_prompt(
                        agent,
                        role=resolved_role,
                        instructions=resolved_instructions,
                    )
                    # Same addendum injection as the other two ai_agent
                    # paths: if a scratchpad-mode scorer also happens to
                    # have stash_records granted (e.g. it stages a
                    # second-stage collection), give it the per-node
                    # shape and dedup contract up front. No-op when
                    # neither is configured.
                    stash_record_schema = self._resolve_stash_schema(node)
                    stash_dedup_binding = self._resolve_stash_dedup(node)
                    for addendum in (
                        format_stash_schema_addendum(stash_record_schema),
                        format_stash_dedup_addendum(stash_dedup_binding),
                    ):
                        if addendum:
                            system_prompt = f"{system_prompt}\n\n{addendum}"

                    history = await self._load_history(
                        own_node_id=node.id,
                        inherited_node_ids=inherit_from,
                        include_tool_interactions=include_tool_interactions,
                        include_own=False,
                        db=iter_db,
                    )

                    runner = AgentRunner(self._llm_router, self._mcp_client)
                    await runner.run(
                        agent=agent,
                        history=history,
                        new_message=new_message,
                        scope_id=run_id,
                        node_id=node.id,
                        emitter=iter_emitter,
                        db=iter_db,
                        redis=self._redis,
                        system_prompt=system_prompt,
                        # No context tools injected: the scorer pattern
                        # doesn't use yield_iteration; persistence (when
                        # appropriate) is the agent's own MCP tool call.
                        extra_call_context=_with_call_context_extras(
                            stash_record_schema=stash_record_schema,
                            stash_dedup_binding=stash_dedup_binding,
                            trigger_user_id=_trigger_user_id_from_state(state),
                            base={
                                "iteration_index": iteration_index,
                                "iteration_item_id": record_id,
                                "iteration_collection": collection,
                                "iteration_id": str(iteration_id),
                            },
                        ),
                    )

                    # Drop is implicit: processing the item consumes it.
                    await scratchpad.drop(run_id, collection, record_id)
            except BaseException as exc:  # noqa: BLE001
                # Catch BaseException (not Exception) so CancelledError
                # also produces an iteration_failed event. Before this
                # change, a cancelled iteration emitted iteration_started
                # but neither completed nor failed, leaving the UI
                # "running" forever and the scratchpad record undrained.
                duration_ms = int((time.monotonic() - started_at) * 1000)
                # Log the full traceback BEFORE the emit. If the
                # original exception left iter_db in a rollback-needed
                # state, the emit will raise a second exception — the
                # primary stack must already be in logs by then.
                logger.exception(
                    "ai_agent node %s iteration %d (item=%s) died with %s",
                    node.id,
                    iteration_index,
                    record_id,
                    type(exc).__name__,
                )
                # Guard the emit so a secondary failure (e.g. session
                # rolled back, redis dropped) does not mask the
                # primary exception that the outer gather/outcomes
                # loop needs to classify the iteration. First, force a
                # rollback so iter_db is in a usable state — the
                # primary exception almost certainly left it in
                # PendingRollbackError, which would make every SQL
                # query inside emit() raise InvalidRequestError
                # instead of letting the INSERT happen.
                try:
                    await iter_db.rollback()
                except BaseException as rb_exc:  # noqa: BLE001
                    logger.exception(
                        "ai_agent node %s iteration %d: rollback before "
                        "iteration_failed emit raised %s",
                        node.id,
                        iteration_index,
                        type(rb_exc).__name__,
                    )
                try:
                    await iter_emitter.emit(
                        RunEventType.iteration_failed,
                        node_id=node.id,
                        payload={
                            "iteration_id": str(iteration_id),
                            "iteration_index": iteration_index,
                            "item_id": record_id,
                            "collection": collection,
                            "duration_ms": duration_ms,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                except BaseException as emit_exc:  # noqa: BLE001
                    logger.exception(
                        "ai_agent node %s iteration %d: emit("
                        "iteration_failed) itself raised %s — UI will "
                        "show stale 'running'; primary error stays above",
                        node.id,
                        iteration_index,
                        type(emit_exc).__name__,
                    )
                raise

            duration_ms = int((time.monotonic() - started_at) * 1000)
            await iter_emitter.emit(
                RunEventType.iteration_completed,
                node_id=node.id,
                payload={
                    "iteration_id": str(iteration_id),
                    "iteration_index": iteration_index,
                    "item_id": record_id,
                    "collection": collection,
                    "duration_ms": duration_ms,
                },
            )
            return "processed"

    async def _load_agent_context(
        self,
        agent: Agent,
        node: FlowNode,
        state: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Resolve every source named in ``agent.required_sources`` against
        ``node.config.context_bindings`` and return ``{name: loaded_dict}``.

        Failures (unknown source, missing/unresolvable binding, loader raising)
        are converted to ``NodeExecutionError`` so the run halts with a clear
        message — declared bindings are treated as load-bearing.
        """
        required = list(getattr(agent, "required_sources", None) or [])
        if not required:
            return {}

        bindings_raw = node.config.get("context_bindings") or {}
        if not isinstance(bindings_raw, dict):
            raise NodeExecutionError(
                f"ai_agent node {node.id}: context_bindings must be an object"
            )

        loaded: dict[str, dict[str, Any]] = {}
        load_ctx = SourceLoadContext(db=self._db, redis=self._redis, state=state)
        for name in required:
            try:
                source = get_context_source(name)
            except UnknownSourceError as exc:
                raise NodeExecutionError(
                    f"ai_agent node {node.id}: agent requires context source "
                    f"{name!r} but no such source is registered"
                ) from exc

            if source.id_implicit:
                source_id: str | None = None
            else:
                binding = bindings_raw.get(name)
                if not isinstance(binding, dict):
                    raise NodeExecutionError(
                        f"ai_agent node {node.id}: missing context_bindings entry "
                        f"for required source {name!r}"
                    )
                source_id = _resolve_binding_id(binding, state)
                if source_id is None or source_id == "":
                    raise NodeExecutionError(
                        f"ai_agent node {node.id}: binding for source "
                        f"{name!r} resolved to no id (binding={binding!r})"
                    )

            try:
                data = await source.loader(load_ctx, source_id)
            except Exception as exc:  # noqa: BLE001 — loader errors are domain errors
                raise NodeExecutionError(
                    f"ai_agent node {node.id}: loader for context source "
                    f"{name!r} failed: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise NodeExecutionError(
                    f"ai_agent node {node.id}: loader for {name!r} returned "
                    f"{type(data).__name__}, expected dict"
                )
            loaded[name] = data
        return loaded

    async def _run_condition(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        # routing is performed at the edge level; the node itself is a no-op.
        return {}

    async def _run_human_checkpoint(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        message = node.config.get("message", "")
        await self._emitter.emit(
            RunEventType.human_checkpoint,
            node_id=node.id,
            payload={"message": message, "name": node.name},
        )
        raise HumanCheckpointInterrupt(node_id=node.id, message=message)

    async def _run_trigger(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    async def _run_output(
        self, node: FlowNode, state: dict[str, Any]
    ) -> dict[str, Any]:
        # Bubble selected outputs into the canonical `state.outputs` map. If
        # the node config names specific keys to surface, copy only those;
        # otherwise echo every existing output keyed under this node id.
        select_keys = node.config.get("select")
        outputs = state.get("outputs", {}) or {}
        if isinstance(select_keys, list) and select_keys:
            return {key: outputs.get(key) for key in select_keys}
        return {}

    async def _load_history(
        self,
        *,
        own_node_id: str,
        inherited_node_ids: list[str],
        include_tool_interactions: bool,
        include_own: bool = True,
        db: AsyncSession | None = None,
    ) -> list[dict[str, Any]]:
        """Load canonical message history for an ai_agent node run.

        Own messages: only non-partial (existing behavior — protects resume).
        Inherited messages: non-partial pass through; partials included too so
        interrupted reasoning isn't lost. After loading, tool_use / tool_result
        blocks whose ids don't have a matching pair are dropped, which prevents
        provider validation errors from orphaned partial tool calls. When
        ``include_tool_interactions`` is False, all tool_use / tool_result
        blocks from inherited messages are stripped (text only). When
        ``include_own`` is False, the node's own past messages are excluded
        — used by the iteration runtime so each batch starts with a fresh
        context window even though previous iterations persisted messages.

        ``db`` overrides the executor's shared session. Parallel iteration
        bodies pass their own isolated session here so concurrent siblings
        don't share an aborted transaction (SQLAlchemy ``AsyncSession``
        is not safe for concurrent use).
        """
        session = db or self._db
        node_ids = (
            [own_node_id, *inherited_node_ids] if include_own else list(inherited_node_ids)
        )
        if not node_ids:
            return []
        result = await session.execute(
            select(AgentMessage)
            .where(AgentMessage.flow_run_id == self._emitter.flow_run_id)
            .where(AgentMessage.node_id.in_(node_ids))
            .order_by(AgentMessage.sequence.asc())
        )
        rows = list(result.scalars().all())

        rows = [m for m in rows if not (m.node_id == own_node_id and m.is_partial)]

        # Tool-pair survivors: only tool_use_ids that appear in BOTH an
        # assistant tool_use block and a user tool_result block survive.
        use_ids: set[str] = set()
        result_ids: set[str] = set()
        for msg in rows:
            for block in msg.content or []:
                btype = block.get("type")
                if btype == "tool_use":
                    bid = block.get("id")
                    if bid:
                        use_ids.add(bid)
                elif btype == "tool_result":
                    bid = block.get("tool_use_id")
                    if bid:
                        result_ids.add(bid)
        valid_pair_ids = use_ids & result_ids

        canonical: list[dict[str, Any]] = []
        for msg in rows:
            is_own = msg.node_id == own_node_id
            blocks: list[dict[str, Any]] = []
            for block in msg.content or []:
                btype = block.get("type")
                if btype == "tool_use":
                    if not is_own and not include_tool_interactions:
                        continue
                    if block.get("id") not in valid_pair_ids:
                        continue
                    blocks.append(block)
                elif btype == "tool_result":
                    if not is_own and not include_tool_interactions:
                        continue
                    if block.get("tool_use_id") not in valid_pair_ids:
                        continue
                    blocks.append(block)
                else:
                    blocks.append(block)
            if not blocks:
                continue
            canonical.append({"role": msg.role.value, "content": blocks})
        return canonical

    def _validate_inheritable_nodes(
        self, own_node_id: str, inherited: list[str]
    ) -> None:
        if self._flow_definition is None:
            raise NodeExecutionError(
                f"ai_agent node {own_node_id}: inherit_history_from set but "
                "NodeExecutor was constructed without a flow_definition"
            )
        nodes_by_id = {n.id: n for n in self._flow_definition.nodes}
        order_by_id = {n.id: i for i, n in enumerate(self._flow_definition.nodes)}
        own_order = order_by_id.get(own_node_id)
        for nid in inherited:
            if nid == own_node_id:
                raise NodeExecutionError(
                    f"ai_agent node {own_node_id}: cannot inherit from itself"
                )
            ref = nodes_by_id.get(nid)
            if ref is None:
                raise NodeExecutionError(
                    f"ai_agent node {own_node_id}: inherit_history_from "
                    f"references unknown node {nid!r}"
                )
            if ref.type != "ai_agent":
                raise NodeExecutionError(
                    f"ai_agent node {own_node_id}: inherit_history_from "
                    f"references {nid!r} which is type {ref.type!r}, expected ai_agent"
                )
            if own_order is not None and order_by_id.get(nid, 0) >= own_order:
                raise NodeExecutionError(
                    f"ai_agent node {own_node_id}: inherit_history_from "
                    f"references {nid!r} which is not declared before this node"
                )


def _trigger_user_id_from_state(state: dict[str, Any]) -> str | None:
    """Extract the triggering user's id from the flow state.

    Mirrors how context source loaders read it (see _load_profile): the
    flow runner stamps ``state.trigger.user_id`` from the authenticated
    CurrentUser when seeding a run. System tool handlers need this when
    they persist user-owned records — without it they'd have to guess
    (e.g. "first profile"), which silently mis-attributes data.

    Returns ``None`` when the run wasn't stamped with a trigger user
    (e.g. background reseed scripts); handlers that require it should
    raise on None rather than fall back.
    """
    trigger = state.get("trigger") or {}
    raw = trigger.get("user_id")
    if not raw:
        return None
    return str(raw)


def _with_call_context_extras(
    *,
    stash_record_schema: list[dict[str, Any]] | None,
    stash_dedup_binding: StashDedupBinding | None,
    trigger_user_id: str | None,
    base: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the ``extra_call_context`` dict the AgentRunner forwards to
    every system tool handler this turn.

    Keeps the "only set keys that have values" rule so the handler can
    distinguish "operator didn't configure it" from "operator set it
    explicitly to null".

    Dedup is materialized into a *callable* here, not passed as the raw
    binding, so the stash_records handler doesn't have to reach back
    into the dedup module — its contract is just "call this if it's
    present". Building the callable per dispatch is cheap (a closure
    over the binding); the per-node resolution that produced the
    binding is the expensive part and already cached upstream.
    """
    out: dict[str, Any] = dict(base or {})
    if stash_record_schema is not None:
        out["stash_record_schema"] = stash_record_schema
    if stash_dedup_binding is not None:
        out["stash_dedup_checker"] = build_call_context_checker(
            stash_dedup_binding
        )
    if trigger_user_id is not None:
        out["trigger_user_id"] = trigger_user_id
    return out or None


def _result_preview(result: Any, *, limit: int) -> str:
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(result)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _resolve_binding_id(binding: dict[str, Any], state: dict[str, Any]) -> str | None:
    """Resolve a ``context_bindings`` entry to a string id.

    Two supported shapes:
      - ``{"from": "inputs.foo"}`` — resolve via dot-path against state
      - ``{"static_id": "<literal>"}`` — pass-through literal

    Returns ``None`` if the dynamic ``from`` path resolves to None (caller
    treats that as a missing binding).
    """
    if "static_id" in binding:
        sid = binding.get("static_id")
        return None if sid in (None, "") else str(sid)
    from_path = binding.get("from")
    if not isinstance(from_path, str) or not from_path:
        raise NodeExecutionError(
            f"context binding must have either 'from' or 'static_id': {binding!r}"
        )
    value = _resolve_value("${" + from_path + "}", state)
    if value is None:
        return None
    return str(value)


def _resolve_inputs(spec: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Resolve a node's input mapping against the current flow state.

    Supports string template values shaped like `${outputs.foo.bar}` or
    `${inputs.field}`; everything else is passed through unchanged.
    """
    if not isinstance(spec, dict):
        return {}
    return {key: _resolve_value(value, state) for key, value in spec.items()}


def _resolve_value(value: Any, state: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    if not (value.startswith("${") and value.endswith("}")):
        return value
    return resolve_path(state, value[2:-1])


def _resolve_prompt(config: dict[str, Any], state: dict[str, Any]) -> str | None:
    prompt = config.get("prompt")
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return _format_template(prompt, state)
    return str(prompt)


def _format_template(text: str, state: dict[str, Any]) -> str:
    # Lightweight ${path} substitution; missing paths become empty strings.
    result = text
    cursor = 0
    while True:
        start = result.find("${", cursor)
        if start == -1:
            return result
        end = result.find("}", start)
        if end == -1:
            return result
        path = result[start + 2 : end]
        replacement = _resolve_value("${" + path + "}", state)
        replacement_str = "" if replacement is None else str(replacement)
        result = result[:start] + replacement_str + result[end + 1 :]
        cursor = start + len(replacement_str)
