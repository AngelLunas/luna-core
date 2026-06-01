"""Tests for the in-flight snapshot helpers powering the WebSocket
zero-loss reconnect path.

Covers the chunk-cache reader and the snapshot frame builder via a tiny
in-memory fake — keeps the suite hermetic without spinning up a real
Redis or matching the full async-redis surface area. The dedup table
that used to live here is gone: deterministic delta ids make every
snapshot frame collapse with its live counterpart at the client
reducer, so there's nothing to test on the server side.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import pytest

from luna_core.llm.base import (
    delta_event_id,
    inflight_meta_key,
    stream_key,
)
from luna_core.models.event import RunEventType
from luna_core.services.flow import (
    _read_inflight_chunks,
    build_run_stream_snapshot,
)


class FakeRedis:
    """Minimal async Redis double — just the four operations the snapshot
    path actually calls. Keys/values are plain strings; lists are plain
    Python lists. Encoding mirrors redis-py's default (``decode_responses``
    is False) so the helpers exercise their bytes-decode branches.
    """

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
        assert match.endswith(":*"), "snapshot only uses suffix-* patterns"
        prefix = match[:-1]
        for key in list(self._strings.keys()):
            if key.startswith(prefix):
                yield key.encode("utf-8")


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def message_id() -> str:
    return "11111111-1111-1111-1111-111111111111"


# -------------------------------------------------------- chunk reader

@pytest.mark.asyncio
async def test_read_inflight_chunks_records_each_chunk(run_id, message_id):
    redis = FakeRedis()
    s_key = stream_key(run_id, message_id)
    await redis.rpush(s_key, json.dumps({"kind": "thinking", "text": "ponder"}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "Hello "}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "world"}))
    await redis.rpush(s_key, json.dumps({"kind": "thinking", "text": "ing"}))

    chunks = await _read_inflight_chunks(redis, run_id, message_id)
    assert chunks == [
        {
            "kind": "thinking",
            "text": "ponder",
            "chunk_index": 0,
            "sequence_offset": 1,
        },
        {
            "kind": "text",
            "text": "Hello ",
            "chunk_index": 0,
            "sequence_offset": 2,
        },
        {
            "kind": "text",
            "text": "world",
            "chunk_index": 1,
            "sequence_offset": 3,
        },
        {
            "kind": "thinking",
            "text": "ing",
            "chunk_index": 1,
            "sequence_offset": 4,
        },
    ]


@pytest.mark.asyncio
async def test_read_inflight_chunks_empty_when_no_cache(run_id):
    redis = FakeRedis()
    assert await _read_inflight_chunks(redis, run_id, "missing") == []


@pytest.mark.asyncio
async def test_read_inflight_chunks_skips_malformed_entries(run_id, message_id):
    redis = FakeRedis()
    s_key = stream_key(run_id, message_id)
    await redis.rpush(s_key, "not json")
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "ok"}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": 42}))

    chunks = await _read_inflight_chunks(redis, run_id, message_id)
    assert len(chunks) == 1
    assert chunks[0]["kind"] == "text"
    assert chunks[0]["text"] == "ok"
    assert chunks[0]["chunk_index"] == 0


# ---------------------------------------------------------- snapshot

@pytest.mark.asyncio
async def test_build_run_stream_snapshot_empty_when_nothing_in_flight(run_id):
    redis = FakeRedis()
    assert await build_run_stream_snapshot(redis, run_id) == []


@pytest.mark.asyncio
async def test_build_run_stream_snapshot_emits_one_frame_per_chunk(
    run_id, message_id
):
    redis = FakeRedis()
    node_id = "writer"
    started_seq = 7

    # Meta is keyed by message_id (per-turn isolation); node_id rides
    # in the payload because the snapshot scanner no longer parses it
    # out of the key.
    await redis.set(
        inflight_meta_key(run_id, message_id),
        json.dumps(
            {
                "message_id": message_id,
                "node_id": node_id,
                "started_seq": started_seq,
                "timestamp": "2026-05-22T10:00:00+00:00",
            }
        ),
    )
    s_key = stream_key(run_id, message_id)
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "Hola"}))
    await redis.rpush(s_key, json.dumps({"kind": "thinking", "text": "hmm"}))
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": " mundo"}))

    frames = await build_run_stream_snapshot(redis, run_id)
    assert len(frames) == 3

    decoded = [json.loads(f) for f in frames]
    # Frames come back in stream-cache (= live publish) order, with
    # sequences contiguous from started_seq + 1.
    assert [f["sequence"] for f in decoded] == [
        started_seq + 1,
        started_seq + 2,
        started_seq + 3,
    ]
    assert [f["event_type"] for f in decoded] == [
        RunEventType.agent_text_delta.value,
        RunEventType.agent_thinking_delta.value,
        RunEventType.agent_text_delta.value,
    ]
    assert [f["payload"]["text"] for f in decoded] == [
        "Hola",
        "hmm",
        " mundo",
    ]
    # Kind-specific chunk_index matches what the live publisher allocates.
    assert [f["payload"]["chunk_index"] for f in decoded] == [0, 0, 1]
    # Deterministic ids — every frame must match what the live publisher
    # would have produced for the same (message_id, kind, chunk_index).
    assert decoded[0]["id"] == str(delta_event_id(message_id, "text", 0))
    assert decoded[1]["id"] == str(delta_event_id(message_id, "thinking", 0))
    assert decoded[2]["id"] == str(delta_event_id(message_id, "text", 1))


@pytest.mark.asyncio
async def test_build_run_stream_snapshot_skips_meta_without_chunks(
    run_id, message_id
):
    """A bare stream_meta with an empty stream list means the turn just
    started but no token has landed yet — nothing useful to snapshot."""
    redis = FakeRedis()
    node_id = "writer"
    await redis.set(
        inflight_meta_key(run_id, message_id),
        json.dumps(
            {
                "message_id": message_id,
                "node_id": node_id,
                "started_seq": 3,
                "timestamp": "2026-05-22T10:00:00+00:00",
            }
        ),
    )
    assert await build_run_stream_snapshot(redis, run_id) == []


@pytest.mark.asyncio
async def test_build_run_stream_snapshot_handles_multiple_in_flight_nodes(
    run_id,
):
    redis = FakeRedis()
    msg_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    msg_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    for node, msg, started_seq, text in [
        ("node_a", msg_a, 5, "from A"),
        ("node_b", msg_b, 12, "from B"),
    ]:
        await redis.set(
            inflight_meta_key(run_id, msg),
            json.dumps(
                {
                    "message_id": msg,
                    "node_id": node,
                    "started_seq": started_seq,
                    "timestamp": "2026-05-22T10:00:00+00:00",
                }
            ),
        )
        await redis.rpush(
            stream_key(run_id, msg),
            json.dumps({"kind": "text", "text": text}),
        )

    frames = await build_run_stream_snapshot(redis, run_id)
    assert len(frames) == 2
    decoded = sorted(
        (json.loads(f) for f in frames), key=lambda f: f["sequence"]
    )
    assert decoded[0]["payload"]["message_id"] == msg_a
    assert decoded[0]["payload"]["text"] == "from A"
    assert decoded[0]["sequence"] == 6
    assert decoded[0]["id"] == str(delta_event_id(msg_a, "text", 0))
    assert decoded[1]["payload"]["message_id"] == msg_b
    assert decoded[1]["payload"]["text"] == "from B"
    assert decoded[1]["sequence"] == 13
    assert decoded[1]["id"] == str(delta_event_id(msg_b, "text", 0))


# ---------------------------------------------------- iteration tagging

@pytest.mark.asyncio
async def test_snapshot_propagates_iteration_id_when_present(
    run_id, message_id
):
    """When the LLM provider wrote the stream_meta from inside an
    iteration scope, it stamps ``iteration_id`` into the meta payload.
    The snapshot helper must propagate that tag into every synthesized
    delta frame so the WebSocket filter routes them to the correct
    iteration block (otherwise the historical in-flight text bubble
    falls through to the parent NodeBlock and the iteration accordion
    looks empty)."""
    redis = FakeRedis()
    node_id = "writer"
    iteration_id = "iter-abc-123"

    await redis.set(
        inflight_meta_key(run_id, message_id),
        json.dumps(
            {
                "message_id": message_id,
                "node_id": node_id,
                "started_seq": 4,
                "timestamp": "2026-05-22T10:00:00+00:00",
                "iteration_id": iteration_id,
            }
        ),
    )
    s_key = stream_key(run_id, message_id)
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "partial"}))

    frames = await build_run_stream_snapshot(redis, run_id)
    assert len(frames) == 1
    decoded = json.loads(frames[0])
    assert decoded["payload"]["iteration_id"] == iteration_id


@pytest.mark.asyncio
async def test_snapshot_omits_iteration_id_for_non_iterative_runs(
    run_id, message_id
):
    """Backward compatibility: when the meta has no iteration_id (the
    common case — a non-iterative ai_agent node), the synthesized frame
    must NOT carry an iteration_id key. Otherwise the wire shape
    diverges from what the live publisher emits and the client reducer
    routes the frame inconsistently."""
    redis = FakeRedis()
    node_id = "writer"

    await redis.set(
        inflight_meta_key(run_id, message_id),
        json.dumps(
            {
                "message_id": message_id,
                "node_id": node_id,
                "started_seq": 4,
                "timestamp": "2026-05-22T10:00:00+00:00",
            }
        ),
    )
    s_key = stream_key(run_id, message_id)
    await redis.rpush(s_key, json.dumps({"kind": "text", "text": "partial"}))

    frames = await build_run_stream_snapshot(redis, run_id)
    decoded = json.loads(frames[0])
    assert "iteration_id" not in decoded["payload"]


# --------------------------------- parallel-iteration cache isolation


@pytest.mark.asyncio
async def test_snapshot_isolates_parallel_iterations_of_same_node(run_id):
    """Regression: parallel iterations of the same ai_agent node each
    write their own per-message meta + stream cache, so the snapshot
    must emit one set of frames per iteration without leaking chunks
    between them. Before the per-message_id key fix, both iterations
    pushed to the same ``stream:{run}:{node}`` list and the meta key
    overwrote on every started, so the snapshot returned the wrong
    chunks for any iteration whose meta got overwritten and lost all
    history the moment the first sibling completed and DELETEd the
    shared cache."""
    redis = FakeRedis()
    node_id = "scorer"
    msg_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    msg_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    # Iteration A's meta + chunks under msg_a's key.
    await redis.set(
        inflight_meta_key(run_id, msg_a),
        json.dumps(
            {
                "message_id": msg_a,
                "node_id": node_id,
                "started_seq": 10,
                "timestamp": "2026-05-26T10:00:00+00:00",
                "iteration_id": "iter-A",
            }
        ),
    )
    await redis.rpush(
        stream_key(run_id, msg_a),
        json.dumps({"kind": "text", "text": "A chunk 0"}),
    )
    await redis.rpush(
        stream_key(run_id, msg_a),
        json.dumps({"kind": "text", "text": "A chunk 1"}),
    )

    # Iteration B's meta + chunks under msg_b's key — same node, but
    # totally isolated cache.
    await redis.set(
        inflight_meta_key(run_id, msg_b),
        json.dumps(
            {
                "message_id": msg_b,
                "node_id": node_id,
                "started_seq": 20,
                "timestamp": "2026-05-26T10:00:00+00:00",
                "iteration_id": "iter-B",
            }
        ),
    )
    await redis.rpush(
        stream_key(run_id, msg_b),
        json.dumps({"kind": "text", "text": "B chunk 0"}),
    )

    frames = [json.loads(f) for f in await build_run_stream_snapshot(redis, run_id)]
    by_msg: dict[str, list[str]] = {}
    for f in frames:
        by_msg.setdefault(f["payload"]["message_id"], []).append(
            f["payload"]["text"]
        )
    assert by_msg[msg_a] == ["A chunk 0", "A chunk 1"]
    assert by_msg[msg_b] == ["B chunk 0"]
    # And the iteration tag follows each set into its own frames.
    iters_by_msg = {
        f["payload"]["message_id"]: f["payload"]["iteration_id"]
        for f in frames
    }
    assert iters_by_msg[msg_a] == "iter-A"
    assert iters_by_msg[msg_b] == "iter-B"


# ---------------------------------------------------- deterministic id

def test_delta_event_id_is_stable_per_triple(message_id):
    assert delta_event_id(message_id, "text", 0) == delta_event_id(
        message_id, "text", 0
    )
    # Distinct chunk_index → distinct id.
    assert delta_event_id(message_id, "text", 0) != delta_event_id(
        message_id, "text", 1
    )
    # Distinct kind → distinct id (same chunk_index doesn't collide
    # between text and thinking streams of the same message).
    assert delta_event_id(message_id, "text", 0) != delta_event_id(
        message_id, "thinking", 0
    )
