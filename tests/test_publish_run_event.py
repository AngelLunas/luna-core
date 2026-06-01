"""Tests for the publish-side contract of ``publish_run_event``.

Pinning two invariants the snapshot + reconnect path relies on:

  1. When the caller supplies ``event_id``, it lands verbatim on the
     wire — that's what lets snapshot frames and live frames collapse
     to a single timeline entry at the client via deterministic ids.
  2. Every publish bumps ``max_seq:{run_id}`` so the DB-backed sequence
     allocator (``EventEmitter._next_event_sequence``) skips past
     transient delta sequences when it later emits
     ``agent_message_completed`` / ``node_completed``.
"""
from __future__ import annotations

import json
import uuid

import pytest

from luna_core.engine.emitter import (
    max_seq_key,
    publish_run_event,
    run_event_channel,
)
from luna_core.models.event import RunEventType


class FakeRedis:
    """Minimal async Redis double covering just publish + set."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.strings: dict[str, str] = {}

    async def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))

    async def set(self, key: str, value, ex: int | None = None) -> None:
        # The real client stringifies non-str values via the wire protocol —
        # do the same so assertions can be string-typed without caring how
        # the caller spelled the value.
        self.strings[key] = str(value)


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000042")


@pytest.mark.asyncio
async def test_publish_uses_supplied_event_id(run_id):
    redis = FakeRedis()
    msg_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    fixed_id = uuid.uuid5(uuid.NAMESPACE_OID, "delta:test:text:0")

    await publish_run_event(
        redis,
        run_id,
        RunEventType.agent_text_delta,
        "node_a",
        {"message_id": str(msg_id), "chunk_index": 0, "text": "hi"},
        sequence=5,
        event_id=fixed_id,
    )

    assert len(redis.published) == 1
    channel, payload = redis.published[0]
    assert channel == run_event_channel(run_id)
    event = json.loads(payload)
    assert event["id"] == str(fixed_id)
    assert event["sequence"] == 5
    assert event["event_type"] == RunEventType.agent_text_delta.value


@pytest.mark.asyncio
async def test_publish_defaults_to_random_event_id(run_id):
    redis = FakeRedis()
    await publish_run_event(
        redis,
        run_id,
        RunEventType.tool_called,
        None,
        {},
        sequence=1,
    )
    event = json.loads(redis.published[0][1])
    # Default is uuid4 — just check it parses and isn't the all-zero id.
    parsed = uuid.UUID(event["id"])
    assert parsed.int != 0


@pytest.mark.asyncio
async def test_publish_bumps_max_seq_high_water_mark(run_id):
    redis = FakeRedis()
    await publish_run_event(
        redis,
        run_id,
        RunEventType.agent_text_delta,
        "node_a",
        {},
        sequence=42,
    )
    assert redis.strings[max_seq_key(run_id)] == "42"

    # Later publish at a higher sequence overwrites — fine because each
    # writer (per-message INCR, or emit() itself) only ever calls this
    # with a value strictly greater than what it just wrote.
    await publish_run_event(
        redis,
        run_id,
        RunEventType.agent_text_delta,
        "node_a",
        {},
        sequence=99,
    )
    assert redis.strings[max_seq_key(run_id)] == "99"
