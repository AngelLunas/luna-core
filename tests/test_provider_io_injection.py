"""The streaming provider must persist its assistant turn through the
injected ``make_io`` factory rather than a hardcoded flow ``EventEmitter``.

This is the seam that lets the same provider stream into a chat
conversation: the caller passes an ``AgentIO`` backed by
``ConversationMessage`` instead of ``AgentMessage``. We drive
``complete()`` with a fake OpenAI stream and a capturing IO, and assert
the assembled assistant turn landed on the injected IO.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from luna_core.llm.providers.generic import GenericProvider
from luna_core.models.event import AgentMessageRole


def _chunk(content: str | None = None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """Async-iterable over a fixed list of chunks (mimics the SDK stream)."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def create(self, **_kwargs):
        return _FakeStream(self._chunks)


class _FakeRedis:
    """Minimal async Redis double covering every op the stream path hits."""

    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.lists: dict[str, list] = {}
        self.published: list[tuple[str, str]] = []

    async def exists(self, *_keys):
        return 0

    async def rpush(self, key, *vals):
        self.lists.setdefault(key, []).extend(vals)
        return len(self.lists[key])

    async def expire(self, *_a, **_k):
        return True

    async def incr(self, key):
        n = int(self.strings.get(key, 0)) + 1
        self.strings[key] = n
        return n

    async def set(self, key, value, ex=None):
        self.strings[key] = value

    async def get(self, key):
        return self.strings.get(key)

    async def publish(self, channel, payload):
        self.published.append((channel, payload))

    async def delete(self, *keys):
        for k in keys:
            self.strings.pop(k, None)
            self.lists.pop(k, None)


class _CapturingIO:
    """Stands in for a chat ``AgentIO``. Records persisted messages so the
    test can prove the provider routed persistence through the injected
    factory instead of a flow ``EventEmitter``."""

    def __init__(self, scope_id: uuid.UUID) -> None:
        self._scope_id = scope_id
        self.saved: list[dict[str, Any]] = []
        self.events: list[Any] = []
        self._seq = 0

    @property
    def scope_id(self) -> uuid.UUID:
        return self._scope_id

    def for_session(self, _db):
        return self

    async def emit(self, event_type, node_id=None, payload=None):
        self._seq += 1
        self.events.append(event_type)
        return SimpleNamespace(sequence=self._seq)

    async def save_message(
        self,
        node_id,
        role,
        content,
        is_partial=False,
        thinking=None,
        message_id=None,
    ):
        self.saved.append(
            {"role": role, "content": content, "is_partial": is_partial}
        )
        return SimpleNamespace(id=message_id or uuid.uuid4())


async def _noop(*_a, **_k):
    return None


@asynccontextmanager
async def _fake_session():
    yield SimpleNamespace(commit=_noop, refresh=_noop)


@pytest.mark.asyncio
async def test_complete_persists_through_injected_make_io():
    scope_id = uuid.uuid4()
    captured = _CapturingIO(scope_id)
    redis = _FakeRedis()

    provider = GenericProvider(
        api_key="x",
        base_url="http://localhost",
        session_factory=_fake_session,
    )
    # Swap the real SDK client for a fake streaming two text chunks.
    provider._chat_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_FakeCompletions([_chunk("Hello "), _chunk("world")])
        )
    )

    blocks = await provider.complete(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        system="be nice",
        tools=[],
        temperature=0.0,
        model="fake",
        output_schema=None,
        run_id=scope_id,
        node_id="chat",
        redis=redis,
        make_io=lambda _db: captured,
    )

    # The provider assembled the streamed deltas into canonical blocks...
    assert blocks == [{"type": "text", "text": "Hello world"}]
    # ...and persisted exactly one assistant turn THROUGH the injected IO,
    # proving persistence is no longer welded to the flow EventEmitter.
    assert len(captured.saved) == 1
    saved = captured.saved[0]
    assert saved["role"] == AgentMessageRole.assistant
    assert saved["is_partial"] is False
    assert saved["content"] == [{"type": "text", "text": "Hello world"}]


@pytest.mark.asyncio
async def test_complete_streams_text_deltas_over_pubsub():
    """Live token frames still fan out on the run/conversation channel —
    the injection changes only persistence, not streaming."""
    scope_id = uuid.uuid4()
    redis = _FakeRedis()
    provider = GenericProvider(
        api_key="x", base_url="http://localhost", session_factory=_fake_session
    )
    provider._chat_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=_FakeCompletions([_chunk("hi")])
        )
    )

    await provider.complete(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        system="",
        tools=[],
        temperature=0.0,
        model="fake",
        output_schema=None,
        run_id=scope_id,
        node_id="chat",
        redis=redis,
        make_io=lambda _db: _CapturingIO(scope_id),
    )

    # At least one text delta was published live.
    assert any("agent_text_delta" in payload for _ch, payload in redis.published)
