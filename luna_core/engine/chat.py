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

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
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
from luna_core.llm.context import COMPACTION_BLOCK, ContextPolicy, compaction_block, public_compaction
from luna_core.llm.router import LLMRouter
from luna_core.mcp.client import MCPClient
from luna_core.mcp.system_tools import SystemToolRegistry
from luna_core.models.agent import Agent
from luna_core.models.conversation import (
    ConversationMessage,
    ConversationMessageRole,
)
from luna_core.models.event import AgentMessageRole, RunEventType
from luna_core.models.usage import LLMUsage

logger = logging.getLogger(__name__)

# The attachment kinds a provider labels (img-N / vid-N); a compaction counts
# the ones it summarizes so later labels keep their numbers.
_MEDIA_KINDS = ("image", "video")

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
        agent_name: str | None = None,
    ) -> None:
        self._db = db
        self._redis = redis
        self._conversation_id = conversation_id
        # Stamped on assistant rows so a reloaded transcript keeps per-bubble
        # agent identity (the live stream already carries it in events).
        self._agent_name = agent_name

    @property
    def scope_id(self) -> uuid.UUID:
        return self._conversation_id

    def for_session(self, db: AsyncSession) -> "ConversationIO":
        """Sibling bound to ``db``, same redis + conversation. The streaming
        provider opens its own short-lived sessions and persists through this."""
        return ConversationIO(
            db, self._redis, self._conversation_id, agent_name=self._agent_name
        )

    async def emit(
        self,
        event_type: RunEventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SupportsSequence:
        payload = dict(payload or {})
        # Stamp the speaking agent on the turn-start event so a multi-agent
        # client can label the live bubble; the provider that emits it only
        # knows the model, not the agent — this scope does.
        if (
            event_type is RunEventType.agent_message_started
            and self._agent_name
            and "agent_name" not in payload
        ):
            payload["agent_name"] = self._agent_name
        sequence = await _redis_event_sequence(self._redis, self._conversation_id)
        await publish_run_event(
            self._redis,
            self._conversation_id,
            event_type,
            node_id,
            payload,
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
            if role is AgentMessageRole.assistant and self._agent_name:
                kwargs["agent_name"] = self._agent_name
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

    def live_only(self) -> "_EphemeralIO":
        """An IO on this conversation's channel that streams but persists
        nothing — for a completion whose text is written in by hand after."""
        return _EphemeralIO(self._redis, self._conversation_id)

    async def add_text_to_last_assistant(self, text: str) -> uuid.UUID | None:
        """Put ``text`` into the conversation's latest assistant message, after
        its thinking and before its tool calls. Used when a turn pauses for
        approval with no words of its own (they streamed live already)."""
        row = (
            await self._db.execute(
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id == self._conversation_id,
                    ConversationMessage.role == ConversationMessageRole.assistant,
                )
                .order_by(ConversationMessage.sequence.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        blocks = list(row.content or [])
        at = 0
        while at < len(blocks) and isinstance(blocks[at], dict) and (
            blocks[at].get("type") == "thinking"
            or (blocks[at].get("native") or {}).get("type") in ("thinking", "redacted_thinking")
        ):
            at += 1
        blocks.insert(at, {"type": "text", "text": text})
        row.content = blocks
        await self._db.commit()
        return row.id

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
        new_message: str | None,
        db: AsyncSession,
        redis: Redis,
        system_prompt: str | None = None,
        extra_call_context: dict[str, Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        image_resolver: Any | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> dict[str, Any] | str | SuspendedForApproval:
        """Run one assistant turn: stream deltas + lifecycle events on the
        conversation's channel, persist the turn as ``ConversationMessage``
        rows, and return the agent's final output (text, or a structured dict
        when the agent declares an output schema). ``attachments`` are extra
        content blocks for the user turn (e.g. ``{"type": "image", "media_id":
        ...}``) the model/provider renders per its vision capability.

        ``new_message=None`` runs the agent over the existing history with no new
        user turn — the continuation path a multi-agent host uses to let a second
        agent answer after a handoff. If the turn calls a tool the agent marks
        ``requires_approval``, the turn suspends and returns
        ``SuspendedForApproval`` — resume via :meth:`resume`.

        ``context_policy`` bounds what the model is sent (see
        ``luna_core.llm.context``). When the last call of the conversation
        reached its ``compact_at``, a new user turn first compacts the earlier
        ones — ``compaction_started`` / ``compaction_completed`` on the channel
        — and runs over the shorter history."""
        rows = await self._history_rows(db, conversation_id)
        io = ConversationIO(db, redis, conversation_id, agent_name=agent.name)
        if new_message is not None or attachments:
            rows = await self._compact_if_due(
                agent, conversation_id, rows, io, db,
                system_prompt=system_prompt,
                timezone=(extra_call_context or {}).get("timezone"),
                image_resolver=image_resolver,
                policy=context_policy,
            )
        history = [_canonical(row) for row in rows]
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
            image_resolver=image_resolver,
            context_policy=context_policy,
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
        image_resolver: Any | None = None,
        context_policy: ContextPolicy | None = None,
    ) -> dict[str, Any] | str | SuspendedForApproval:
        """Resume a turn suspended for tool approval. Loads the history (ending
        with the assistant message holding the gated ``tool_use`` blocks) and
        hands off to the runner, which executes the decided tools and either
        re-invokes the LLM or ends the turn."""
        history = await self._load_history(db, conversation_id)
        io = ConversationIO(db, redis, conversation_id, agent_name=agent.name)
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
            image_resolver=image_resolver,
            context_policy=context_policy,
        )

    async def _load_history(
        self, db: AsyncSession, conversation_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """The history the model is sent, as canonical ``{role, content}``
        dicts (see :meth:`_history_rows`)."""
        return [_canonical(row) for row in await self._history_rows(db, conversation_id)]

    async def _history_rows(
        self, db: AsyncSession, conversation_id: uuid.UUID
    ) -> list[ConversationMessage]:
        """The stored turns the model is sent, in sequence order. Partial rows
        (an interrupted assistant turn) are skipped — they aren't durable
        context for the next turn. After a compaction the history starts at
        its marker, followed by the turns it kept and everything since; the
        summarized turns stay stored but are no longer sent."""
        result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.sequence)
        )
        rows = [row for row in result.scalars().all() if not row.is_partial]
        marker = next(
            (row for row in reversed(rows) if compaction_block(_canonical(row)) is not None), None
        )
        if marker is None:
            return rows
        through = int(compaction_block(_canonical(marker)).get("compacted_through_sequence") or 0)
        return [marker, *(
            row for row in rows
            if row.sequence > through and row is not marker and compaction_block(_canonical(row)) is None
        )]

    async def _compact_if_due(
        self,
        agent: Agent,
        conversation_id: uuid.UUID,
        rows: list[ConversationMessage],
        io: ConversationIO,
        db: AsyncSession,
        *,
        system_prompt: str | None,
        timezone: str | None,
        image_resolver: Any | None,
        policy: ContextPolicy | None,
    ) -> list[ConversationMessage]:
        """Compact before this turn when the conversation's last call reached
        ``policy.compact_at`` tokens. The summary covers exactly the messages
        of that call — everything but the last reply, which stays after the
        marker word for word. Any failure leaves the history as it was: the
        turn still answers, at full length."""
        if not policy or not policy.compact_at or len(rows) < 3:
            return rows
        if await self._last_call_tokens(db, conversation_id, rows) < policy.compact_at:
            return rows
        summarized = rows[:-1] if rows[-1].role is ConversationMessageRole.assistant else rows
        kept = rows[len(summarized):]
        await io.emit(RunEventType.compaction_started, node_id=_CHAT_NODE, payload={})
        native = await self._runner.compact(
            agent,
            [_canonical(row) for row in summarized],
            conversation_id,
            db,
            system_prompt=system_prompt,
            image_resolver=image_resolver,
            timezone=timezone,
            instructions=policy.compaction_instructions,
        )
        if native is None:
            await io.emit(RunEventType.compaction_completed, node_id=_CHAT_NODE, payload={"compacted": False})
            return rows
        block = {
            "type": COMPACTION_BLOCK,
            "native": native,
            "model": agent.model,
            "media": _media_counts(summarized),
            "compacted_through_sequence": summarized[-1].sequence,
            "at": datetime.now(UTC).isoformat(),
        }
        marker = await io.save_message(
            node_id=_CHAT_NODE, role=AgentMessageRole.system, content=[block]
        )
        logger.info(
            "conversation compacted — conversation=%s agent=%s through=%s",
            conversation_id, agent.name, block["compacted_through_sequence"],
        )
        await io.emit(
            RunEventType.compaction_completed,
            node_id=_CHAT_NODE,
            payload={
                "compacted": True,
                "message_id": str(marker.id),
                "sequence": marker.sequence,
                **public_compaction(block),
            },
        )
        return [marker, *kept]

    async def _last_call_tokens(
        self, db: AsyncSession, conversation_id: uuid.UUID, rows: list[ConversationMessage]
    ) -> int:
        """Prompt + reply of the latest model call that wrote a message of the
        history now sent — what the next call starts from. Calls whose words
        live elsewhere (a title, a sub-agent) don't count."""
        # A reply kept past a compaction was billed at the old length: only
        # calls since the marker measure the history as it is now.
        since = rows[0].sequence if rows and compaction_block(_canonical(rows[0])) else 0
        ids = [
            row.id for row in rows
            if row.role is ConversationMessageRole.assistant and row.sequence > since
        ]
        if not ids:
            return 0
        total = await db.scalar(
            select(LLMUsage.total_tokens)
            .where(LLMUsage.scope_id == conversation_id, LLMUsage.message_id.in_(ids))
            .order_by(LLMUsage.created_at.desc())
            .limit(1)
        )
        return int(total or 0)


def _canonical(row: ConversationMessage) -> dict[str, Any]:
    return {"role": row.role.value, "content": row.content}


def _media_counts(rows: list[ConversationMessage]) -> dict[str, int]:
    """Labeled attachments up to the end of ``rows`` — counting those an
    earlier compaction already summarized."""
    counts = {kind: 0 for kind in _MEDIA_KINDS}
    for row in rows:
        earlier = compaction_block(_canonical(row))
        if earlier is not None:
            for kind, n in (earlier.get("media") or {}).items():
                if kind in counts:
                    counts[kind] = int(n)
            continue
        if row.role is not ConversationMessageRole.user:
            continue
        for block in row.content or []:
            if isinstance(block, dict) and block.get("type") in counts:
                counts[block["type"]] += 1
    return counts


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

    def __init__(
        self,
        redis: Redis,
        scope_id: uuid.UUID,
        usage_scope_id: uuid.UUID | None = None,
    ) -> None:
        self._redis = redis
        self._scope_id = scope_id
        self._usage_scope_id = usage_scope_id

    @property
    def scope_id(self) -> uuid.UUID:
        return self._scope_id

    @property
    def usage_scope_id(self) -> uuid.UUID | None:
        """Where the sub-agent's token usage is ledgered (the caller's scope),
        apart from the channel its events stream on."""
        return self._usage_scope_id

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
    usage_scope_id: uuid.UUID | None = None,
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

    ``usage_scope_id`` is the scope its token usage counts against — the
    caller's conversation or run, so what a delegated task cost is not lost
    under a scope nobody owns. Without it, the sub-agent's own scope.
    """
    scope_id = uuid.uuid4()
    io = _EphemeralIO(redis, scope_id, usage_scope_id=usage_scope_id)
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
