from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.engine.iteration_context import inject_iteration_tag
from luna_core.models.event import (
    AgentMessage,
    AgentMessageRole,
    RunEvent,
    RunEventType,
)


# Bounded retry count for ``EventEmitter.emit`` when an INSERT collides
# on the per-run sequence unique constraint. With parallel iteration
# concurrency capped at ``settings.iteration_concurrency_max`` (Pi
# default 8, ceiling 20 in the UI), this cap is comfortably larger
# than the worst-case number of concurrent emitters racing to allocate
# the next sequence. Hitting the cap means something is wrong with
# sequence allocation, not just contention — we want the exception to
# surface in that case rather than spinning forever.
_EMIT_MAX_RETRIES = 32


def max_seq_key(run_id: uuid.UUID | str) -> str:
    """High-water mark key for any sequence number ever published for a run.

    Combines two writers into one observable bound: the streaming
    provider's per-message INCR (for transient deltas) and
    ``EventEmitter.emit`` (for persisted events). Lets ``_next_event_sequence``
    skip past sequences already used by live deltas so persisted events
    (e.g. ``agent_message_completed``, ``node_completed``) never collide
    with them and the client reducer doesn't split a single node across
    two timeline blocks. Lives here rather than in ``llm/base.py`` to
    keep the emitter module free of any ``luna_core.llm`` import — the
    LLM package eagerly loads providers that import the emitter, so
    pulling helpers from ``llm.base`` here triggers a circular import.
    """
    return f"max_seq:{run_id}"


def run_event_channel(run_id: uuid.UUID | str) -> str:
    return f"{settings.run_event_channel_prefix}:{run_id}"


def flow_run_channel(flow_id: uuid.UUID | str) -> str:
    """Pub/sub channel for FlowRun lifecycle notifications on a specific flow.

    Distinct from `run_event_channel`, which carries per-run timeline events.
    This channel only fans out coarse "a run was created / changed status"
    messages so the FlowDetail page can update its run history live without
    polling. Pub/sub only — nothing is persisted in Redis; the canonical
    record is the FlowRun row.
    """
    return f"{settings.run_event_channel_prefix}:flow:{flow_id}"


async def publish_flow_run_event(
    redis: Redis,
    flow_id: uuid.UUID,
    event: str,
    run_payload: dict[str, Any],
) -> None:
    """Fire-and-forget broadcast of a FlowRun lifecycle change.

    `event` is one of `"run_created"` or `"run_status_changed"`. `run_payload`
    is the JSON-serializable FlowRunRead dump. Subscribers (the FlowDetail
    WebSocket) upsert into their local runs list by `run.id`.
    """
    message = {"event": event, "run": run_payload}
    await redis.publish(flow_run_channel(flow_id), json.dumps(message, default=str))


async def publish_run_event(
    redis: Redis,
    flow_run_id: uuid.UUID,
    event_type: RunEventType,
    node_id: str | None,
    payload: dict[str, Any],
    sequence: int,
    event_id: uuid.UUID | None = None,
) -> None:
    """Broadcast a run event over Redis pub/sub without persisting it.

    Used for high-frequency streaming chunks (agent_text_delta /
    agent_thinking_delta) where the canonical record is the eventual
    AgentMessage row — persisting every chunk would multiply the
    run_events table by the chunk count without adding any recoverable
    information. The REST events endpoint rehydrates one synthetic delta
    per AgentMessage at read time so historical timelines look identical
    to live ones.

    ``event_id`` lets callers pin the published id to a deterministic
    value (e.g. ``delta_event_id`` for streamed chunks) so a reconnecting
    client's mid-stream snapshot can dedupe against any matching live
    frame by id alone. The published ``sequence`` is also recorded as the
    per-run high-water mark so subsequent DB-allocated events sort after
    every transient delta — without that, ``agent_message_completed`` and
    ``node_completed`` (allocated from MAX(DB.sequence)+1, oblivious to
    the in-Redis INCR counter) would collide with delta sequences and
    leave the client reducer splitting a single node across two blocks.
    """
    if event_id is None:
        event_id = uuid.uuid4()
    # Inject the iteration tag (when the current asyncio task is inside
    # an iteration scope) so the UI can route this delta to the right
    # iteration block. No-op outside iteration bodies.
    tagged_payload = inject_iteration_tag(payload)
    message = {
        "id": str(event_id),
        "flow_run_id": str(flow_run_id),
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type.value,
        "node_id": node_id,
        "payload": tagged_payload,
    }
    await redis.publish(run_event_channel(flow_run_id), json.dumps(message))
    await _bump_max_seq(redis, flow_run_id, sequence)


async def _bump_max_seq(
    redis: Redis, flow_run_id: uuid.UUID, sequence: int
) -> None:
    """Track the highest sequence number observed for a run.

    Plain SET (not a SET-if-greater) is safe because the only writers are
    (a) the streaming provider's per-message INCR — monotonic by design —
    and (b) ``EventEmitter.emit`` for persisted events, which itself
    allocates ``max(DB_max, redis_max) + 1`` so each call writes a value
    strictly greater than what was there. The TTL just keeps the key from
    surviving forever after the run finishes.
    """
    await redis.set(
        max_seq_key(flow_run_id),
        sequence,
        ex=settings.run_stream_key_ttl_seconds,
    )


class EventEmitter:
    """Persists RunEvents / AgentMessages and fans them out over Redis pub/sub.

    Sequence numbers are monotonic per (flow_run_id, kind). Allocation reads
    MAX(sequence) from the DB on every call rather than caching in-process,
    because multiple short-lived emitter instances (e.g. the streaming LLM
    provider creates its own emitter with a fresh session to persist the
    assistant turn) can interleave writes for the same flow_run_id.
    """

    def __init__(self, db: AsyncSession, redis: Redis, flow_run_id: uuid.UUID):
        self._db = db
        self._redis = redis
        self._flow_run_id = flow_run_id

    @property
    def flow_run_id(self) -> uuid.UUID:
        return self._flow_run_id

    @property
    def scope_id(self) -> uuid.UUID:
        """Satisfies ``streaming.EventSink.scope_id``. For the flow
        implementation the execution scope *is* the flow run."""
        return self._flow_run_id

    def for_session(self, db: AsyncSession) -> "EventEmitter":
        """Satisfies ``streaming.AgentIO.for_session``: a sibling bound to
        ``db`` with the same redis + flow run. Lets the streaming provider
        persist on its own short-lived sessions without referencing this
        concrete class."""
        return EventEmitter(db, self._redis, self._flow_run_id)

    async def _next_event_sequence(self) -> int:
        current = await self._db.execute(
            select(func.coalesce(func.max(RunEvent.sequence), 0)).where(
                RunEvent.flow_run_id == self._flow_run_id
            )
        )
        db_max = int(current.scalar() or 0)
        # Live deltas published via ``publish_run_event`` never hit the DB but
        # do consume sequence numbers; reading their high-water mark from
        # Redis keeps persisted events strictly above them.
        redis_raw = await self._redis.get(max_seq_key(self._flow_run_id))
        redis_max = 0
        if redis_raw is not None:
            if isinstance(redis_raw, bytes):
                redis_raw = redis_raw.decode("utf-8")
            try:
                redis_max = int(redis_raw)
            except (TypeError, ValueError):
                redis_max = 0
        return max(db_max, redis_max) + 1

    async def _next_message_sequence(self) -> int:
        current = await self._db.execute(
            select(func.coalesce(func.max(AgentMessage.sequence), 0)).where(
                AgentMessage.flow_run_id == self._flow_run_id
            )
        )
        return int(current.scalar() or 0) + 1

    async def emit(
        self,
        event_type: RunEventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        # Auto-tag with iteration_id when emitted from inside an
        # iteration_scope; lifecycle events emitted by the iteration
        # executor itself already set their own iteration_id (which
        # inject_iteration_tag preserves) — this covers all the sub-events
        # (tool_called, tool_result, agent_message_*) that fire from the
        # nested AgentRunner without touching their call sites.
        tagged_payload = inject_iteration_tag(payload)
        # Sequence allocation does a SELECT MAX + 1 plus a Redis
        # high-water-mark read, then we INSERT and COMMIT. Under
        # parallel iteration, two emitters with their own sessions can
        # both observe sequence=N as the next free slot and race to
        # INSERT N+1, with one losing on the uq_run_events_run_sequence
        # unique constraint. The race window is tiny (microseconds) so
        # a small bounded retry handles it cleanly: roll back the
        # failed transaction and recompute the next sequence — the
        # winner's commit is now visible so the retry sees a higher
        # MAX.
        last_error: IntegrityError | None = None
        for _attempt in range(_EMIT_MAX_RETRIES):
            sequence = await self._next_event_sequence()
            timestamp = datetime.now(timezone.utc)
            event = RunEvent(
                flow_run_id=self._flow_run_id,
                sequence=sequence,
                timestamp=timestamp,
                event_type=event_type,
                node_id=node_id,
                payload=tagged_payload,
            )
            self._db.add(event)
            try:
                await self._db.commit()
                await self._db.refresh(event)
                break
            except IntegrityError as exc:
                last_error = exc
                # rollback() puts the session back in a usable state
                # so the next SELECT MAX query (in the loop's next
                # iteration) doesn't fail with
                # "current transaction is aborted".
                await self._db.rollback()
                continue
        else:
            assert last_error is not None  # bounded loop guarantees this
            raise last_error

        # event.payload already carries the iteration tag (we mutated /
        # returned the same dict above); publish what was persisted.
        message = {
            "id": str(event.id),
            "flow_run_id": str(event.flow_run_id),
            "sequence": event.sequence,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "node_id": event.node_id,
            "payload": event.payload,
        }
        await self._redis.publish(
            run_event_channel(self._flow_run_id), json.dumps(message)
        )
        await _bump_max_seq(self._redis, self._flow_run_id, event.sequence)
        return event

    async def save_message(
        self,
        node_id: str,
        role: AgentMessageRole,
        content: list[dict[str, Any]],
        is_partial: bool = False,
        thinking: str | None = None,
        message_id: uuid.UUID | None = None,
    ) -> AgentMessage:
        # Bounded retry on uq_agent_messages_run_sequence violations, same
        # pattern as ``emit`` above. Under parallel iteration, two
        # AgentRunners with independent sessions can both observe
        # MAX(sequence)=N and race to INSERT N+1 — one loses on the unique
        # constraint. Roll back, recompute (the winner's commit is now
        # visible), retry. Without this the losing iteration dies with an
        # IntegrityError that is not retried anywhere upstream.
        last_error: IntegrityError | None = None
        message: AgentMessage | None = None
        for _attempt in range(_EMIT_MAX_RETRIES):
            sequence = await self._next_message_sequence()
            # Honor a caller-supplied id so the assistant message persisted
            # at the end of a stream shares the same UUID we already
            # broadcast via the *_delta events — the frontend can then key
            # the rendered bubble off one stable identifier across REST+WS.
            kwargs: dict[str, Any] = dict(
                flow_run_id=self._flow_run_id,
                node_id=node_id,
                sequence=sequence,
                role=role,
                content=content,
                is_partial=is_partial,
                thinking=thinking,
            )
            if message_id is not None:
                kwargs["id"] = message_id
            message = AgentMessage(**kwargs)
            self._db.add(message)
            try:
                await self._db.commit()
                await self._db.refresh(message)
                break
            except IntegrityError as exc:
                last_error = exc
                # rollback() returns the session to a usable state so the
                # next _next_message_sequence() doesn't fail with
                # "current transaction is aborted".
                await self._db.rollback()
                continue
        else:
            assert last_error is not None  # bounded loop guarantees this
            raise last_error
        assert message is not None  # set on the successful break
        return message

    # Back-compat alias for external hosts (and tests) still calling the
    # flow-named method. ``save_message`` is the canonical name declared on
    # ``streaming.TranscriptStore``; both names are the same coroutine.
    save_agent_message = save_message
