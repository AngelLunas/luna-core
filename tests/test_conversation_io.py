"""``ConversationIO`` is the chat implementation of ``streaming.AgentIO``.

Two invariants matter for it to slot under the unchanged AgentRunner +
streaming provider:

  1. ``emit`` allocates its ordinal one above the Redis high-water mark the
     provider bumps for every transient delta, and publishes on the
     conversation's channel — so a persisted-after event never collides with
     the deltas it follows.
  2. ``save_message`` persists a ``ConversationMessage`` with the role mapped
     from the flow-named ``AgentMessageRole`` and the next per-conversation
     sequence.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from luna_core.engine.chat import ConversationIO
from luna_core.engine.emitter import max_seq_key, run_event_channel
from luna_core.models.conversation import (
    ConversationMessage,
    ConversationMessageRole,
)
from luna_core.models.event import AgentMessageRole, RunEventType


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.published: list[tuple[str, str]] = []

    async def get(self, key):
        return self.strings.get(key)

    async def set(self, key, value, ex=None):
        self.strings[key] = str(value)

    async def publish(self, channel, payload):
        self.published.append((channel, payload))


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self._value = value

    def scalar(self) -> int:
        return self._value


class _FakeDB:
    def __init__(self, *, max_sequence_seen: int = 0) -> None:
        self.max_sequence_seen = max_sequence_seen
        self.added: list[Any] = []

    async def execute(self, _stmt):
        return _FakeScalar(self.max_sequence_seen)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def refresh(self, _obj):
        return None


@pytest.mark.asyncio
async def test_emit_allocates_above_delta_highwater_and_publishes():
    conv = uuid.uuid4()
    redis = _FakeRedis()
    await redis.set(max_seq_key(conv), 7)  # deltas already rode up to 7
    io = ConversationIO(_FakeDB(), redis, conv)

    event = await io.emit(RunEventType.agent_message_completed, payload={})

    assert event.sequence == 8
    channel, _payload = redis.published[0]
    assert channel == run_event_channel(conv)
    # publish_run_event bumped the high-water mark to the new ordinal.
    assert redis.strings[max_seq_key(conv)] == "8"


@pytest.mark.asyncio
async def test_save_message_maps_role_and_allocates_sequence():
    conv = uuid.uuid4()
    db = _FakeDB(max_sequence_seen=3)
    io = ConversationIO(db, _FakeRedis(), conv)

    message = await io.save_message(
        node_id="ignored-by-chat",
        role=AgentMessageRole.assistant,
        content=[{"type": "text", "text": "hi"}],
    )

    assert isinstance(message, ConversationMessage)
    assert message.role == ConversationMessageRole.assistant
    assert message.sequence == 4
    assert message.conversation_id == conv
    assert message.content == [{"type": "text", "text": "hi"}]
    assert len(db.added) == 1


@pytest.mark.asyncio
async def test_save_message_honors_supplied_message_id():
    conv = uuid.uuid4()
    fixed = uuid.uuid4()
    io = ConversationIO(_FakeDB(), _FakeRedis(), conv)

    message = await io.save_message(
        node_id=None,
        role=AgentMessageRole.user,
        content=[{"type": "text", "text": "hello"}],
        message_id=fixed,
    )

    assert message.id == fixed
    assert message.role == ConversationMessageRole.user
