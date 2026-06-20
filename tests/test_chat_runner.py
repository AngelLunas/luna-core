"""``ChatRunner`` drives the unchanged AgentRunner over a Conversation.

The send() path delegates to AgentRunner (covered by the flow tests); the
logic unique to ChatRunner is history reconstruction — turning stored
``ConversationMessage`` rows back into the canonical ``{role, content}``
shape the agent loop consumes, in order, dropping partial (interrupted)
turns.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from luna_core.engine.agent import AgentRunner
from luna_core.engine.chat import ChatRunner, _EphemeralIO, run_sub_agent
from luna_core.engine.emitter import run_event_channel
from luna_core.models.conversation import ConversationMessageRole
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


class _Row:
    def __init__(self, role, content, is_partial=False):
        self.role = role
        self.content = content
        self.is_partial = is_partial


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _HistoryDB:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return _Result(self._rows)


@pytest.mark.asyncio
async def test_load_history_maps_rows_in_order_and_drops_partials():
    rows = [
        _Row(ConversationMessageRole.user, [{"type": "text", "text": "hi"}]),
        _Row(
            ConversationMessageRole.assistant,
            [{"type": "text", "text": "hello"}],
        ),
        _Row(
            ConversationMessageRole.assistant,
            [{"type": "text", "text": "interrupted"}],
            is_partial=True,
        ),
    ]
    runner = ChatRunner(llm_router=None, mcp_client=None)

    history: list[dict[str, Any]] = await runner._load_history(
        _HistoryDB(rows), uuid.uuid4()
    )

    assert history == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]


@pytest.mark.asyncio
async def test_ephemeral_io_streams_on_its_scope_and_skips_persistence():
    scope = uuid.uuid4()
    redis = _FakeRedis()
    io = _EphemeralIO(redis, scope)

    event = await io.emit(RunEventType.tool_called, payload={"name": "x"})

    assert event.sequence == 1
    assert redis.published[0][0] == run_event_channel(scope)
    # A sub-agent's transcript is not persisted; the loop keeps it in memory.
    assert (
        await io.save_message(
            node_id=None, role=AgentMessageRole.assistant, content=[]
        )
        is None
    )
    # No DB binding needed — the same ephemeral IO serves every session.
    assert io.for_session(None) is io


@pytest.mark.asyncio
async def test_run_sub_agent_uses_fresh_scope_and_returns_output(monkeypatch):
    captured: dict[str, Any] = {}

    async def fake_run(self, **kwargs):
        captured.update(kwargs)
        return {"diagnosis": "leaf rust", "severity": "high"}

    monkeypatch.setattr(AgentRunner, "run", fake_run)

    result = await run_sub_agent(
        llm_router=None,
        mcp_client=None,
        agent=object(),
        prompt="analyze this plant",
        db=None,
        redis=_FakeRedis(),
    )

    assert result.output == {"diagnosis": "leaf rust", "severity": "high"}
    # The sub-agent ran on a fresh scope, propagated to the runner and back,
    # streaming through an ephemeral IO bound to that same scope.
    assert captured["scope_id"] == result.scope_id
    assert captured["new_message"] == "analyze this plant"
    assert captured["history"] == []
    assert captured["emitter"].scope_id == result.scope_id
