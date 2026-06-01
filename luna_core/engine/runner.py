from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.connectors.registry import ConnectorRegistry
from luna_core.core.config import settings
from luna_core.engine.emitter import EventEmitter
from luna_core.engine.nodes import HumanCheckpointInterrupt, NodeExecutor
from luna_core.llm.base import AbortSignalError, abort_key, run_state_key, stream_key
from luna_core.llm.router import LLMRouter
from luna_core.mcp.client import MCPClient
from luna_core.models.event import AgentMessageRole, RunEventType
from luna_core.models.flow import FlowRunStatus
from luna_core.schemas.flow import (
    FlowDefinition,
    FlowEdge,
    FlowEdgeCondition,
)
from luna_core.services.flow import (
    get_flow,
    get_flow_run,
    set_run_status,
)

logger = logging.getLogger(__name__)

END_TOKEN = "__end__"


def _merge_dict_channel(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for accumulating dict channels (outputs / context) across node
    updates. Without this, returning ``{"outputs": {"x": 1}}`` from one node
    would clobber outputs written by earlier nodes.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class FlowState(TypedDict, total=False):
    # Single-valued metadata channels. Default LangGraph behavior is
    # "last-writer-wins"; nodes never write these after the initial state,
    # so they survive untouched for the whole run.
    run_id: str
    flow_id: str
    inputs: dict[str, Any]
    trigger: dict[str, Any]
    current_node: str
    error: str | None
    # Accumulating channels — every node may contribute keys; the reducer
    # merges them with whatever was there before so prior contributions are
    # preserved.
    outputs: Annotated[dict[str, Any], _merge_dict_channel]
    context: Annotated[dict[str, Any], _merge_dict_channel]


CheckpointerFactory = Callable[[], Any]


def _get_path(data: Any, path: str) -> Any:
    cursor: Any = data
    for part in path.split("."):
        if cursor is None:
            return None
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        else:
            cursor = getattr(cursor, part, None)
    return cursor


def _evaluate_condition(condition: FlowEdgeCondition, state: dict[str, Any]) -> bool:
    actual = _get_path(state, condition.field)
    expected = condition.value
    match condition.operator:
        case "eq":
            return actual == expected
        case "ne":
            return actual != expected
        case "gt":
            return actual is not None and actual > expected
        case "gte":
            return actual is not None and actual >= expected
        case "lt":
            return actual is not None and actual < expected
        case "lte":
            return actual is not None and actual <= expected
        case "in":
            return expected is not None and actual in expected
        case "contains":
            return actual is not None and expected in actual
    return False


def _merge_state(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """LangGraph reducer for the entire state dict (last-writer-wins per key).

    ``outputs`` and ``context`` are merged dict-by-dict so concurrent or
    sequential nodes don't clobber each other's keys.
    """
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if key in ("outputs", "context") and isinstance(value, dict):
            merged_branch = dict(merged.get(key) or {})
            merged_branch.update(value)
            merged[key] = merged_branch
        else:
            merged[key] = value
    return merged


class FlowRunner:
    """Compiles flow definitions into LangGraph state machines and executes them.

    The runner is library-shaped — it accepts a DB session and a Redis client
    per call so the host application owns lifecycle. The host injects shared
    collaborators (LLMRouter, MCPClient, ConnectorRegistry) once at
    construction time so every NodeExecutor created by this runner receives
    the same instances.

    Run-state persistence strategy:
      - During execution the LangGraph snapshot lives in Redis at
        `run_state:{run_id}` (refreshed on every node completion).
      - On pause/complete/fail/abort the final snapshot is mirrored back into
        `FlowRun.state` in Postgres and the Redis stream keys are deleted.
      - On startup, if `run_state:{run_id}` already exists we use it as the
        initial state — this is how a worker that crashed mid-run picks up
        where it left off.
    """

    def __init__(
        self,
        checkpointer_factory: CheckpointerFactory | None = None,
        *,
        llm_router: LLMRouter | None = None,
        mcp_client: MCPClient | None = None,
        connector_registry: ConnectorRegistry | None = None,
        tool_result_preview_limit: int | None = None,
    ):
        self._checkpointer_factory = checkpointer_factory or MemorySaver
        self._llm_router = llm_router
        self._mcp_client = mcp_client
        self._connector_registry = connector_registry
        self._tool_result_preview_limit = tool_result_preview_limit

    # ---- public API ---------------------------------------------------------
    def compile(self, definition: FlowDefinition, executor: NodeExecutor):
        return self._build_graph(definition, executor)

    async def run(
        self,
        db: AsyncSession,
        redis: Redis,
        flow_run_id: uuid.UUID,
    ) -> uuid.UUID:
        """Execute an already-persisted FlowRun.

        The trigger payload (inputs/source/metadata) was stored on the row by
        whoever called `create_flow_run` upstream (typically the HTTP
        trigger endpoint), so callers here just hand in the run id. This
        avoids the historical footgun where the API created one FlowRun and
        the Celery task quietly created a second — the UI would navigate to
        the first (which stayed pending forever) while the worker ran the
        second.
        """
        run = await get_flow_run(db, flow_run_id)
        flow = await get_flow(db, run.flow_id)
        definition = FlowDefinition.model_validate(flow.definition)
        await self._execute(db, redis, run.id, definition, resume=False)
        return run.id

    async def resume(
        self,
        db: AsyncSession,
        redis: Redis,
        flow_run_id: uuid.UUID,
        human_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        run = await get_flow_run(db, flow_run_id)
        if run.status != FlowRunStatus.paused:
            raise ValueError(
                f"flow run {flow_run_id} is not paused (status={run.status})"
            )

        flow = await get_flow(db, run.flow_id)
        definition = FlowDefinition.model_validate(flow.definition)

        # persist the human response as an AgentMessage(role=user) before
        # continuing so reconstructed agent context sees the approval inline.
        emitter = EventEmitter(db, redis, flow_run_id)
        current_node = (run.state or {}).get("current_node") or "human_checkpoint"
        await emitter.save_agent_message(
            node_id=current_node,
            role=AgentMessageRole.user,
            content=[{"type": "text", "text": human_response}],
        )
        await emitter.emit(
            RunEventType.human_response,
            node_id=current_node,
            payload={"response": human_response, "metadata": metadata or {}},
        )

        await self._execute(
            db,
            redis,
            flow_run_id,
            definition,
            resume=True,
            extra_state={"human_response": human_response},
        )
        return flow_run_id

    # ---- internals ----------------------------------------------------------
    async def _execute(
        self,
        db: AsyncSession,
        redis: Redis,
        flow_run_id: uuid.UUID,
        definition: FlowDefinition,
        *,
        resume: bool,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        emitter = EventEmitter(db, redis, flow_run_id)
        executor_kwargs: dict[str, Any] = {
            "llm_router": self._llm_router,
            "mcp_client": self._mcp_client,
            "connector_registry": self._connector_registry,
            "flow_definition": definition,
        }
        if self._tool_result_preview_limit is not None:
            executor_kwargs["tool_result_preview_limit"] = (
                self._tool_result_preview_limit
            )
        executor = NodeExecutor(emitter, db, redis, **executor_kwargs)
        graph = self._build_graph(definition, executor)
        checkpointer = self._checkpointer_factory()
        compiled = graph.compile(checkpointer=checkpointer)

        await set_run_status(db, flow_run_id, FlowRunStatus.running, redis=redis)
        if not resume:
            await emitter.emit(
                RunEventType.flow_started,
                node_id=None,
                payload={"entry_point": definition.entry_point},
            )

        initial_state = await self._build_initial_state(
            db, redis, flow_run_id, extra_state
        )
        await self._persist_run_state(redis, flow_run_id, initial_state)

        config = {"configurable": {"thread_id": str(flow_run_id)}}
        final_state: dict[str, Any] = initial_state
        try:
            async for event in compiled.astream(initial_state, config=config):
                if await redis.exists(abort_key(flow_run_id)):
                    raise AbortSignalError(flow_run_id, final_state.get("current_node", ""))
                for node_update in event.values():
                    if isinstance(node_update, dict):
                        final_state = _merge_state(final_state, node_update)
                await self._persist_run_state(redis, flow_run_id, final_state)
        except HumanCheckpointInterrupt as interrupt:
            final_state["current_node"] = interrupt.node_id
            await set_run_status(
                db,
                flow_run_id,
                FlowRunStatus.paused,
                state=final_state,
                redis=redis,
            )
            await self._persist_run_state(redis, flow_run_id, final_state)
            return
        except AbortSignalError:
            logger.info("flow run %s aborted", flow_run_id)
            final_state["error"] = "aborted"
            await emitter.emit(
                RunEventType.flow_failed,
                node_id=None,
                payload={"reason": "aborted"},
            )
            await set_run_status(
                db, flow_run_id, FlowRunStatus.failed, state=final_state, redis=redis
            )
            await self._cleanup_run_keys(redis, flow_run_id, definition)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("flow run %s failed", flow_run_id)
            final_state["error"] = str(exc)
            await emitter.emit(
                RunEventType.flow_failed,
                node_id=None,
                payload={"error": str(exc), "type": exc.__class__.__name__},
            )
            await set_run_status(
                db, flow_run_id, FlowRunStatus.failed, state=final_state, redis=redis
            )
            await self._cleanup_run_keys(redis, flow_run_id, definition)
            return

        await emitter.emit(
            RunEventType.flow_completed,
            node_id=None,
            payload={
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_keys": sorted((final_state.get("outputs") or {}).keys()),
            },
        )
        await set_run_status(
            db, flow_run_id, FlowRunStatus.completed, state=final_state, redis=redis
        )
        await self._cleanup_run_keys(redis, flow_run_id, definition)

    async def _build_initial_state(
        self,
        db: AsyncSession,
        redis: Redis,
        flow_run_id: uuid.UUID,
        extra_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cached = await redis.get(run_state_key(flow_run_id))
        if cached:
            try:
                state = json.loads(cached)
                if isinstance(state, dict):
                    if extra_state:
                        state.update(extra_state)
                    return state
            except json.JSONDecodeError:
                logger.warning(
                    "ignoring malformed run_state cache for %s", flow_run_id
                )

        run = await get_flow_run(db, flow_run_id)
        trigger = run.trigger or {}
        trigger_metadata = trigger.get("metadata") or {}
        state: dict[str, Any] = {
            "run_id": str(flow_run_id),
            "flow_id": str(run.flow_id),
            "inputs": trigger.get("inputs", {}),
            "trigger": {
                "source": trigger.get("source", "manual"),
                "timestamp": (
                    run.created_at.isoformat() if run.created_at else None
                ),
                "user_id": trigger_metadata.get("user_id"),
                "run_id": str(flow_run_id),
                **{k: v for k, v in trigger_metadata.items() if k != "user_id"},
            },
            "context": run.state.get("context", {}) if run.state else {},
            "outputs": run.state.get("outputs", {}) if run.state else {},
            "current_node": run.state.get("current_node") if run.state else "",
            "error": None,
        }
        if extra_state:
            state.update(extra_state)
        return state

    async def _persist_run_state(
        self, redis: Redis, flow_run_id: uuid.UUID, state: dict[str, Any]
    ) -> None:
        try:
            payload = json.dumps(state, default=str)
        except (TypeError, ValueError):
            logger.exception("run state for %s is not JSON-serializable", flow_run_id)
            return
        await redis.set(
            run_state_key(flow_run_id),
            payload,
            ex=settings.run_stream_key_ttl_seconds,
        )

    async def _cleanup_run_keys(
        self,
        redis: Redis,
        flow_run_id: uuid.UUID,
        definition: FlowDefinition,  # kept for signature compat
    ) -> None:
        # ``definition`` was used to enumerate per-node stream keys; now
        # stream keys are per-message_id (parallel iteration fix) so we
        # SCAN with the run-scoped prefix instead. Param stays so call
        # sites don't need to change.
        _ = definition
        keys: list[str] = [
            run_state_key(flow_run_id),
            abort_key(flow_run_id),
        ]
        try:
            async for raw_key in redis.scan_iter(
                match=stream_key(flow_run_id, "*")
            ):
                if isinstance(raw_key, bytes):
                    raw_key = raw_key.decode("utf-8")
                keys.append(raw_key)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to scan stream keys for run %s", flow_run_id
            )
        try:
            await redis.delete(*keys)
        except Exception:  # noqa: BLE001
            logger.exception("failed to clean Redis keys for run %s", flow_run_id)

    def _build_graph(
        self, definition: FlowDefinition, executor: NodeExecutor
    ) -> StateGraph:
        # StateGraph(FlowState) gives every declared TypedDict field its own
        # channel — so non-returned fields (trigger, inputs, run_id, ...)
        # persist across nodes instead of being clobbered when a node returns
        # only its own contribution.
        graph = StateGraph(FlowState)

        nodes_by_id = {node.id: node for node in definition.nodes}
        if definition.entry_point not in nodes_by_id:
            raise ValueError(
                f"entry_point '{definition.entry_point}' is not in nodes"
            )

        for node in definition.nodes:
            graph.add_node(node.id, _make_node_runner(node, executor))

        graph.add_edge(START, definition.entry_point)

        # group edges by source node
        outgoing: dict[str, list[FlowEdge]] = {}
        for edge in definition.edges:
            outgoing.setdefault(edge.from_, []).append(edge)

        for source, edges in outgoing.items():
            if source not in nodes_by_id:
                raise ValueError(f"edge.from references unknown node: {source}")
            conditional = [e for e in edges if e.condition is not None]
            unconditional = [e for e in edges if e.condition is None]

            if conditional:
                mapping: dict[str, Any] = {}
                for edge in edges:
                    target = END if edge.to == END_TOKEN else edge.to
                    mapping[edge.to] = target
                # ensure the router always has a halt path when nothing matches
                if not unconditional:
                    mapping.setdefault(END_TOKEN, END)
                default_target = (
                    unconditional[0].to if unconditional else END_TOKEN
                )
                graph.add_conditional_edges(
                    source,
                    _make_router(edges, default=default_target),
                    mapping,
                )
            else:
                for edge in unconditional:
                    target = END if edge.to == END_TOKEN else edge.to
                    graph.add_edge(source, target)

        # any terminal node without outgoing edges flows to END so the graph
        # always halts.
        nodes_with_outgoing = set(outgoing.keys())
        for node_id in nodes_by_id:
            if node_id not in nodes_with_outgoing:
                graph.add_edge(node_id, END)

        return graph


def _make_node_runner(
    node, executor: NodeExecutor
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def runner(state: dict[str, Any]) -> dict[str, Any]:
        return await executor.execute(node, state)

    return runner


def _make_router(
    edges: list[FlowEdge], default: str
) -> Callable[[dict[str, Any]], str]:
    """Evaluate conditional edges first; fall back to the first unconditional
    edge if nothing matches; otherwise return the static `default`."""

    def router(state: dict[str, Any]) -> str:
        for edge in edges:
            if edge.condition is not None and _evaluate_condition(
                edge.condition, state
            ):
                return edge.to
        for edge in edges:
            if edge.condition is None:
                return edge.to
        return default

    return router
