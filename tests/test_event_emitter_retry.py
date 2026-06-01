"""Tests for ``EventEmitter.emit``'s retry-on-IntegrityError loop.

When parallel iterations each hold their own ``AsyncSession`` and emit
concurrently, two emitters can both observe ``max(sequence)=N`` and race
to INSERT ``sequence=N+1`` — the loser hits the
``uq_run_events_run_sequence`` unique constraint. The emitter must roll
back the failed transaction and recompute the next sequence so the
caller never sees the collision.

We exercise that loop against an in-memory fake session that returns a
controllable MAX value and raises ``IntegrityError`` on the first commit.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError

from luna_core.engine.emitter import EventEmitter
from luna_core.models.event import RunEvent, RunEventType


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {}

    async def get(self, key):
        return self.strings.get(key)

    async def set(self, key, value, ex=None):
        self.strings[key] = str(value)

    async def publish(self, _channel, _payload):
        return 0


class _FakeDB:
    """Async session double that the emitter can drive.

    ``max_sequence_seen`` is what the next SELECT MAX returns; tests
    bump it between calls to simulate a sibling's commit landing
    between our SELECT and our INSERT. ``fail_commits`` is the number
    of times commit() should raise IntegrityError before succeeding.
    """

    def __init__(
        self,
        *,
        max_sequence_seen: int = 0,
        fail_commits: int = 0,
    ) -> None:
        self.max_sequence_seen = max_sequence_seen
        self.fail_commits = fail_commits
        self.commit_calls = 0
        self.rollback_calls = 0
        self.added: list[Any] = []
        # Sequences the emitter actually used in successful commits.
        # When the constraint collides on commit, the sequence the
        # emitter tried is recorded too so tests can assert the retry
        # loop advanced past it.
        self.attempted_sequences: list[int] = []

    async def execute(self, _stmt):
        # The emitter's _next_event_sequence runs a SELECT MAX query;
        # we don't need to interpret it — just return the controllable
        # scalar.
        max_value = self.max_sequence_seen
        return _FakeScalar(max_value)

    def add(self, obj):
        self.added.append(obj)
        # Both RunEvent and AgentMessage carry a ``sequence`` field that
        # the emitter sets per attempt; recording it here lets tests
        # assert the retry loop advanced past each collision regardless
        # of which table the emitter was targeting.
        sequence = getattr(obj, "sequence", None)
        if sequence is not None:
            self.attempted_sequences.append(sequence)

    async def commit(self):
        self.commit_calls += 1
        if self.fail_commits > 0:
            self.fail_commits -= 1
            # Mimic asyncpg's path: the next emit's SELECT would
            # fail with "transaction aborted" until rollback is
            # called. Tests assert rollback is invoked between
            # attempts, which the emitter does inside the except.
            raise IntegrityError("INSERT", {}, Exception("uq violation"))

    async def rollback(self):
        self.rollback_calls += 1

    async def refresh(self, _obj):
        return None


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar(self) -> int:
        return self._value


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.asyncio
async def test_emit_retries_when_sequence_collision_then_succeeds(run_id):
    """First commit raises IntegrityError, second succeeds. The emit
    call must return cleanly; the caller never sees the collision."""
    db = _FakeDB(max_sequence_seen=5, fail_commits=1)
    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    event = await emitter.emit(
        RunEventType.node_started,
        node_id="some_node",
        payload={"name": "x"},
    )

    assert event is not None
    assert db.commit_calls == 2  # one failed, one succeeded
    assert db.rollback_calls == 1
    # Two RunEvent additions: the colliding first attempt and the
    # successful retry.
    assert len(db.attempted_sequences) == 2


@pytest.mark.asyncio
async def test_emit_eventually_raises_when_collisions_exceed_retry_cap(run_id):
    """If every retry fails, the last IntegrityError must surface so the
    caller can act on it rather than swallowing the bug silently."""
    from luna_core.engine.emitter import _EMIT_MAX_RETRIES

    db = _FakeDB(max_sequence_seen=5, fail_commits=_EMIT_MAX_RETRIES)
    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    with pytest.raises(IntegrityError):
        await emitter.emit(
            RunEventType.node_started,
            node_id="some_node",
            payload={"name": "x"},
        )

    # Every attempt rolled back to leave the session usable for the next
    # retry; the final raise propagates the most recent IntegrityError.
    assert db.commit_calls == _EMIT_MAX_RETRIES
    assert db.rollback_calls == _EMIT_MAX_RETRIES


@pytest.mark.asyncio
async def test_emit_does_not_retry_on_unrelated_exceptions(run_id):
    """Only IntegrityError triggers the retry. Other exceptions bubble
    up on the first attempt — we don't want to mask programming bugs
    behind a retry loop."""
    db = _FakeDB(max_sequence_seen=5)

    async def explode_on_commit():
        raise RuntimeError("not an integrity error")

    db.commit = explode_on_commit  # type: ignore[assignment]

    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    with pytest.raises(RuntimeError, match="not an integrity error"):
        await emitter.emit(
            RunEventType.node_started,
            node_id="some_node",
            payload={"name": "x"},
        )

    # Only one rollback path — the exception didn't take the retry path.
    assert db.rollback_calls == 0


# ---- save_agent_message retry loop ---------------------------------------
#
# Same race condition (parallel iterations both calculating MAX+1 against
# core.agent_messages and racing on uq_agent_messages_run_sequence) — the
# fix is the same retry pattern as emit(). These tests mirror the ones
# above so any future refactor that re-breaks the retry surfaces here.


@pytest.mark.asyncio
async def test_save_agent_message_retries_on_sequence_collision(run_id):
    """First commit hits a unique-violation, second succeeds. The caller
    gets a persisted AgentMessage and never sees the collision."""
    from luna_core.models.event import AgentMessageRole

    db = _FakeDB(max_sequence_seen=10, fail_commits=1)
    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    message = await emitter.save_agent_message(
        node_id="job_scorer",
        role=AgentMessageRole.assistant,
        content=[{"type": "text", "text": "hello"}],
    )

    assert message is not None
    assert db.commit_calls == 2  # one failed, one succeeded
    assert db.rollback_calls == 1
    assert len(db.attempted_sequences) == 2


@pytest.mark.asyncio
async def test_save_agent_message_eventually_raises_when_retries_exhausted(run_id):
    """If every retry collides, the IntegrityError must propagate so the
    iteration's except handler can surface it as iteration_failed."""
    from luna_core.engine.emitter import _EMIT_MAX_RETRIES
    from luna_core.models.event import AgentMessageRole

    db = _FakeDB(max_sequence_seen=10, fail_commits=_EMIT_MAX_RETRIES)
    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    with pytest.raises(IntegrityError):
        await emitter.save_agent_message(
            node_id="job_scorer",
            role=AgentMessageRole.assistant,
            content=[{"type": "text", "text": "hello"}],
        )

    assert db.commit_calls == _EMIT_MAX_RETRIES
    assert db.rollback_calls == _EMIT_MAX_RETRIES


@pytest.mark.asyncio
async def test_save_agent_message_does_not_retry_on_unrelated_exceptions(run_id):
    """Only IntegrityError should trigger the retry. Other failures must
    surface immediately so real bugs aren't masked by the loop."""
    from luna_core.models.event import AgentMessageRole

    db = _FakeDB(max_sequence_seen=10)

    async def explode_on_commit():
        raise RuntimeError("not an integrity error")

    db.commit = explode_on_commit  # type: ignore[assignment]

    redis = _FakeRedis()
    emitter = EventEmitter(db, redis, run_id)

    with pytest.raises(RuntimeError, match="not an integrity error"):
        await emitter.save_agent_message(
            node_id="job_scorer",
            role=AgentMessageRole.assistant,
            content=[{"type": "text", "text": "hello"}],
        )

    assert db.rollback_calls == 0
