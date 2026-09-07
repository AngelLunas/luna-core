"""The conversation stream WebSocket replays in-flight chunks on connect.

A chat client that reconnects mid-turn (network blip, app foregrounded)
gets no persisted assistant row to rehydrate from — the row lands at the
END of the turn — so without a snapshot it lost the prefix it had already
painted. The providers write every in-flight turn's chunks to the stream
cache under the conversation id, and the conversations WebSocket manager
now serves them as synthetic delta frames, exactly like the runs channel.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import pytest

from luna_core.llm.base import delta_event_id, inflight_meta_key, stream_key
from luna_core.models.event import RunEventType
from luna_core.routers import conversations as conversations_router


class FakeRedis:
    """Minimal async Redis double — just what the snapshot path calls."""

    def __init__(self) -> None:
        self._strings: dict[str, bytes] = {}
        self._lists: dict[str, list[bytes]] = {}

    async def set(self, key: str, value: str) -> None:
        self._strings[key] = value.encode("utf-8")

    async def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value.encode("utf-8"))

    async def get(self, key: str) -> bytes | None:
        return self._strings.get(key)

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        items = self._lists.get(key, [])
        if end == -1:
            return list(items[start:])
        return list(items[start : end + 1])

    async def scan_iter(self, match: str) -> AsyncIterator[bytes]:
        assert match.endswith(":*")
        prefix = match[:-1]
        for key in list(self._strings.keys()):
            if key.startswith(prefix):
                yield key.encode("utf-8")


@pytest.fixture
def manager(monkeypatch):
    """A fresh conversations WS manager bound to a fake Redis; the module
    caches it lazily, so reset the global around each test."""
    redis = FakeRedis()
    monkeypatch.setattr(conversations_router, "get_redis_client", lambda: redis)
    monkeypatch.setattr(conversations_router, "_ws_manager", None)
    return conversations_router.get_ws_manager(), redis


def test_conversation_manager_has_a_snapshot(manager):
    ws_manager, _ = manager
    assert ws_manager._snapshot_fn is not None


@pytest.mark.asyncio
async def test_snapshot_is_empty_when_no_turn_is_in_flight(manager):
    ws_manager, _ = manager
    conversation_id = uuid.uuid4()
    assert await ws_manager._snapshot_fn(conversation_id) == []


@pytest.mark.asyncio
async def test_snapshot_replays_in_flight_chunks_for_the_conversation(manager):
    ws_manager, redis = manager
    conversation_id = uuid.uuid4()
    message_id = str(uuid.uuid4())
    started_seq = 12

    # What the provider leaves behind mid-turn on the chat path: meta + chunks
    # keyed by the conversation id (its ``run_id``) and the message id.
    await redis.set(
        inflight_meta_key(conversation_id, message_id),
        json.dumps(
            {
                "message_id": message_id,
                "node_id": "chat",
                "started_seq": started_seq,
                "timestamp": "2026-09-06T10:00:00+00:00",
            }
        ),
    )
    s_key = stream_key(conversation_id, message_id)
    await redis.rpush(s_key, json.dumps({"kind": "thinking", "text": "hmm"}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "Riega "}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "hoy"}))

    # A different conversation's cache must not leak into this snapshot.
    other = uuid.uuid4()
    await redis.set(
        inflight_meta_key(other, message_id),
        json.dumps({"message_id": message_id, "node_id": "chat", "started_seq": 1}),
    )
    await redis.rpush(
        stream_key(other, message_id), json.dumps({"kind": "text", "text": "no"})
    )

    frames = [json.loads(f) for f in await ws_manager._snapshot_fn(conversation_id)]

    assert [f["event_type"] for f in frames] == [
        RunEventType.agent_thinking_delta.value,
        RunEventType.agent_text_delta.value,
        RunEventType.agent_text_delta.value,
    ]
    assert [f["payload"]["text"] for f in frames] == ["hmm", "Riega ", "hoy"]
    assert all(f["payload"]["message_id"] == message_id for f in frames)
    # Same scope id + deterministic ids as the live frames, so the client
    # dedupes any overlap between the replay and what it already painted.
    assert all(f["flow_run_id"] == str(conversation_id) for f in frames)
    assert [f["id"] for f in frames] == [
        str(delta_event_id(message_id, "thinking", 0)),
        str(delta_event_id(message_id, "text", 0)),
        str(delta_event_id(message_id, "text", 1)),
    ]
    assert [f["sequence"] for f in frames] == [
        started_seq + 1,
        started_seq + 2,
        started_seq + 3,
    ]
