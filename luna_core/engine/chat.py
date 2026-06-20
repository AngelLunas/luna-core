"""Conversation-native agent IO.

``ConversationIO`` is the chat counterpart of the flow ``EventEmitter``: it
satisfies the ``streaming.AgentIO`` protocol so the very same ``AgentRunner``
and streaming provider can drive a persistent ``Conversation`` instead of a
flow run — no synthetic flow, no special-casing in the engine.

Two deliberate differences from the flow implementation:

  - **Events are pub/sub only.** A chat timeline reconstructs fully from its
    ``ConversationMessage`` rows (tool_use / tool_result blocks live inside
    message content) plus the live delta stream, so lifecycle events are
    ephemeral signals, not an audit log. There is no per-event table.
  - **Sequencing has no DB allocator.** Event ordinals come from the same
    Redis high-water mark (``max_seq``) the streaming provider already bumps
    for every transient delta, so a persisted-after event like
    ``agent_message_completed`` still sorts above the deltas it follows.

Transcript turns persist as ``ConversationMessage`` rows. ``node_id`` and
``thinking`` are flow-only fields; chat has no sub-scope and keeps the
thinking block inside ``content``, so both arguments are accepted (to honor
the protocol) and ignored.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.engine.agent import AgentRunner, SuspendedForApproval
from luna_core.engine.emitter import (
    _EMIT_MAX_RETRIES,
    max_seq_key,
    publish_run_event,
)
from luna_core.engine.streaming import SupportsSequence
from luna_core.llm.router import LLMRouter
from luna_core.mcp.client import MCPClient
from luna_core.mcp.system_tools import SystemToolRegistry
from luna_core.models.agent import Agent
from luna_core.models.conversation import (
    ConversationMessage,
    ConversationMessageRole,
)
from luna_core.models.event import AgentMessageRole, RunEventType

# Event-stream segment label for a chat turn. Chat has no flow nodes; this is
# a stable tag the streaming events ride under, mirroring how flow events
# carry their node id. It never keys persistence (ConversationIO ignores it).
_CHAT_NODE = "chat"


class _Sequenced:
    """Lightweight ``SupportsSequence`` for the pub/sub-only chat event —
    chat never persists an event row, so there is nothing heavier to return."""

    __slots__ = ("sequence",)

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence


async def _redis_event_sequence(redis: Redis, scope_id: uuid.UUID) -> int:
    """Next event ordinal for a pub/sub-only scope: one above the Redis
    high-water mark (``max_seq``) the streaming provider bumps for every
    transient delta. No DB allocator — these events aren't persisted, so a
    persisted-after event like ``agent_message_completed`` still sorts above
    the deltas it follows and the client keeps one ordered timeline."""
    raw = await redis.get(max_seq_key(scope_id))
    current = 0
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            current = int(raw)
        except (TypeError, ValueError):
            current = 0
    return current + 1


class ConversationIO:
    """``streaming.AgentIO`` backed by ``Conversation`` / ``ConversationMessage``."""

    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
        conversation_id: uuid.UUID,
    ) -> None:
        self._db = db
        self._redis = redis
        self._conversation_id = conversation_id

    @property
    def scope_id(self) -> uuid.UUID:
        return self._conversation_id

    def for_session(self, db: AsyncSession) -> "ConversationIO":
        """Sibling bound to ``db``, same redis + conversation. The streaming
        provider opens its own short-lived sessions and persists through this."""
        return ConversationIO(db, self._redis, self._conversation_id)

    async def emit(
        self,
        event_type: RunEventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SupportsSequence:
        sequence = await _redis_event_sequence(self._redis, self._conversation_id)
        await publish_run_event(
            self._redis,
            self._conversation_id,
            event_type,
            node_id,
            payload or {},
            sequence,
        )
        return _Sequenced(sequence)

    async def save_message(
        self,
        node_id: str | None,
        role: AgentMessageRole,
        content: list[dict[str, Any]],
        is_partial: bool = False,
        thinking: str | None = None,
        message_id: uuid.UUID | None = None,
    ) -> ConversationMessage:
        # Bounded retry on the per-conversation sequence unique constraint,
        # mirroring EventEmitter: the runner's session and the provider's
        # short-lived session can both compute MAX+1 for the same
        # conversation and race to INSERT. The loser rolls back and recomputes.
        last_error: IntegrityError | None = None
        message: ConversationMessage | None = None
        for _attempt in range(_EMIT_MAX_RETRIES):
            sequence = await self._next_message_sequence()
            kwargs: dict[str, Any] = dict(
                conversation_id=self._conversation_id,
                sequence=sequence,
                role=ConversationMessageRole(role.value),
                content=content,
                is_partial=is_partial,
            )
            # Honor a caller-supplied id so the row persisted at end-of-stream
            # shares the UUID already broadcast on the *_delta frames — the
            # client keys its rendered bubble off one stable id across REST+WS.
            if message_id is not None:
                kwargs["id"] = message_id
            message = ConversationMessage(**kwargs)
            self._db.add(message)
            try:
                await self._db.commit()
                await self._db.refresh(message)
                break
            except IntegrityError as exc:
                last_error = exc
                await self._db.rollback()
                continue
        else:
            assert last_error is not None  # bounded loop guarantees this
            raise last_error
        assert message is not None  # set on the successful break
        return message

    # ----------------------------------------------------------------- internals
    async def _next_message_sequence(self) -> int:
        current = await self._db.execute(
            select(
                func.coalesce(func.max(ConversationMessage.sequence), 0)
            ).where(ConversationMessage.conversation_id == self._conversation_id)
        )
        return int(current.scalar() or 0) + 1


class ChatRunner:
    """Drives one agent turn over a persistent ``Conversation`` — the
    conversation-native counterpart of the flow ``FlowRunner`` path.

    It loads prior turns as canonical history, binds a ``ConversationIO``
    scope to the request's session + redis, and hands both to the very same
    ``AgentRunner`` the flow engine uses. There is no flow run, no LangGraph,
    no synthetic scope — just the agent loop streaming over a conversation.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_client: MCPClient,
        *,
        system_tool_registry: SystemToolRegistry | None = None,
    ) -> None:
        self._runner = AgentRunner(
            llm_router, mcp_client, system_tool_registry=system_tool_registry
        )

    async def send(
        self,
        *,
        agent: Agent,
        conversation_id: uuid.UUID,
        new_message: str,
        db: AsyncSession,
        redis: Redis,
        system_prompt: str | None = None,
        extra_call_context: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | str | SuspendedForApproval:
        """Run one assistant turn: stream deltas + lifecycle events on the
        conversation's channel, persist the turn as ``ConversationMessage``
        rows, and return the agent's final output (text, or a structured dict
        when the agent declares an output schema). ``attachments`` are extra
        content blocks for the user turn (e.g. ``{"type": "image", "media_id":
        ...}``) the model/provider renders per its vision capability. If the turn
        calls a tool the agent marks ``requires_approval``, the turn suspends and
        returns ``SuspendedForApproval`` — resume via :meth:`resume`."""
        history = await self._load_history(db, conversation_id)
        io = ConversationIO(db, redis, conversation_id)
        return await self._runner.run(
            agent=agent,
            history=history,
            new_message=new_message,
            scope_id=conversation_id,
            node_id=_CHAT_NODE,
            emitter=io,
            db=db,
            redis=redis,
            system_prompt=system_prompt,
            extra_call_context=extra_call_context,
            approval_enabled=True,
            attachments=attachments,
        )

    async def resume(
        self,
        *,
        agent: Agent,
        conversation_id: uuid.UUID,
        db: AsyncSession,
        redis: Redis,
        system_prompt: str | None = None,
        extra_call_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | str | SuspendedForApproval:
        """Resume a turn suspended for tool approval. Loads the history (ending
        with the assistant message holding the gated ``tool_use`` blocks) and
        hands off to the runner, which executes the decided tools and either
        re-invokes the LLM or ends the turn."""
        history = await self._load_history(db, conversation_id)
        io = ConversationIO(db, redis, conversation_id)
        return await self._runner.resume(
            agent=agent,
            history=history,
            scope_id=conversation_id,
            node_id=_CHAT_NODE,
            emitter=io,
            db=db,
            redis=redis,
            system_prompt=system_prompt,
            extra_call_context=extra_call_context,
        )

    async def _load_history(
        self, db: AsyncSession, conversation_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Prior conversation turns as canonical ``{role, content}`` dicts in
        sequence order. Partial rows (an interrupted assistant turn) are
        skipped — they aren't durable context for the next turn."""
        result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence)
        )
        return [
            {"role": row.role.value, "content": row.content}
            for row in result.scalars().all()
            if not row.is_partial
        ]


# Event-stream segment label for a sub-agent run (see _CHAT_NODE).
_SUBAGENT_NODE = "subagent"


class _EphemeralIO:
    """``streaming.AgentIO`` for a sub-agent run.

    Streams lifecycle events + token deltas on a fresh sub-scope channel — so
    a caller can nest the sub-agent's live progress under the tool call that
    triggered it — but persists no durable transcript. A sub-agent's internal
    reasoning is a means to its returned result, not conversation history; the
    agent loop keeps its working messages in memory, so a no-op transcript is
    correct and intentional.
    """

    def __init__(self, redis: Redis, scope_id: uuid.UUID) -> None:
        self._redis = redis
        self._scope_id = scope_id

    @property
    def scope_id(self) -> uuid.UUID:
        return self._scope_id

    def for_session(self, _db: AsyncSession) -> "_EphemeralIO":
        return self

    async def emit(
        self,
        event_type: RunEventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SupportsSequence:
        sequence = await _redis_event_sequence(self._redis, self._scope_id)
        await publish_run_event(
            self._redis,
            self._scope_id,
            event_type,
            node_id,
            payload or {},
            sequence,
        )
        return _Sequenced(sequence)

    async def save_message(
        self,
        node_id: str | None,
        role: AgentMessageRole,
        content: list[dict[str, Any]],
        is_partial: bool = False,
        thinking: str | None = None,
        message_id: uuid.UUID | None = None,
    ) -> None:
        # Intentionally not persisted — see the class docstring.
        return None


@dataclass(slots=True)
class SubAgentResult:
    """What ``run_sub_agent`` hands back: the sub-agent's final output (text,
    or the structured dict from its output schema) and the fresh scope its
    live events streamed on — so the caller can point a nested UI panel at it
    via ``run_event_channel(scope_id)``."""

    output: dict[str, Any] | str
    scope_id: uuid.UUID


async def run_sub_agent(
    *,
    llm_router: LLMRouter,
    mcp_client: MCPClient,
    agent: Agent,
    prompt: str,
    db: AsyncSession,
    redis: Redis,
    system_prompt: str | None = None,
    system_tool_registry: SystemToolRegistry | None = None,
    attachments: list[dict[str, Any]] | None = None,
    image_resolver: Any | None = None,
) -> SubAgentResult:
    """Run ``agent`` as a sub-agent to completion and return its output.

    This is the generic agent-composition primitive: a cheap orchestrator
    agent delegates a focused sub-task (e.g. an agronomist "doctor" that runs
    its own multi-step tool loop and returns a structured diagnosis) and gets
    the result back, while keeping ownership of the conversation. The host app
    wires it into a system tool (e.g. ``consult_doctor``) whose handler closes
    over the router + client and passes the sub-agent record.

    The sub-agent runs on its own fresh scope, so its events and token deltas
    fan out on ``run_event_channel(result.scope_id)`` for a nested live view;
    its transcript is ephemeral and only the final output flows back.
    """
    scope_id = uuid.uuid4()
    io = _EphemeralIO(redis, scope_id)
    runner = AgentRunner(
        llm_router, mcp_client, system_tool_registry=system_tool_registry
    )
    output = await runner.run(
        agent=agent,
        history=[],
        new_message=prompt,
        scope_id=scope_id,
        node_id=_SUBAGENT_NODE,
        emitter=io,
        db=db,
        redis=redis,
        system_prompt=system_prompt,
        attachments=attachments,
        image_resolver=image_resolver,
    )
    return SubAgentResult(output=output, scope_id=scope_id)


__all__ = ["ChatRunner", "ConversationIO", "SubAgentResult", "run_sub_agent"]
