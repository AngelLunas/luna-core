"""Lifecycle tests for run_transcription_session using fake transports.

The fakes implement the ClientTransport / UpstreamTransport protocols with
scripted queues — no network, no FastAPI app. Timeouts are shrunk to
milliseconds so failure paths (idle, max duration) run in test time.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from luna_core.services.voice_transcription import run_transcription_session

HANG = object()  # sentinel: the fake blocks forever at this point


def _text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def _audio(data: bytes) -> dict[str, Any]:
    return {"type": "websocket.receive", "bytes": data}


DISCONNECT = {"type": "websocket.disconnect"}


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.sent: list[dict[str, Any]] = []
        self.close_code: int | None = None

    async def receive(self) -> dict[str, Any]:
        if not self._script or self._script[0] is HANG:
            await asyncio.Event().wait()
        return self._script.pop(0)

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000) -> None:
        self.close_code = code

    def frames(self, frame_type: str) -> list[dict[str, Any]]:
        return [f for f in self.sent if f.get("type") == frame_type]


class FakeUpstream:
    def __init__(self, events: list[Any] | None = None, *, hang_after: bool = True) -> None:
        self._events = list(events or [])
        self._hang_after = hang_after
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, data: str) -> None:
        if self.closed:
            raise ConnectionError("upstream closed")
        self.sent.append(json.loads(data))

    async def recv(self) -> str:
        # Let queued client frames flow first so event ordering is stable.
        await asyncio.sleep(0.01)
        if self._events:
            item = self._events.pop(0)
            if item is HANG:
                await asyncio.Event().wait()
            return json.dumps(item)
        if self._hang_after:
            await asyncio.Event().wait()
        raise ConnectionError("upstream closed")

    async def close(self) -> None:
        self.closed = True

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.sent if e.get("type") == event_type]


def _run(client: FakeClient, upstream: FakeUpstream, **overrides: Any):
    defaults: dict[str, Any] = {
        "model": "test-model",
        "language": "es",
        "max_seconds": 5.0,
        "idle_timeout": 5.0,
        "finalize_timeout": 0.05,
    }
    defaults.update(overrides)
    return run_transcription_session(client, upstream, **defaults)


@pytest.mark.asyncio
async def test_happy_path_stop() -> None:
    client = FakeClient(
        [_audio(b"\x00\x01"), _audio(b"\x02\x03"), _audio(b"\x04\x05"), _text({"type": "stop"}), HANG]
    )
    upstream = FakeUpstream(
        [
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "i1",
                "delta": "hola ",
            },
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "i1",
                "delta": "mundo",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "i1",
                "transcript": "hola mundo",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 40,
                    "input_token_details": {"audio_tokens": 35},
                    "output_tokens": 3,
                    "total_tokens": 43,
                },
            },
        ]
    )

    result = await _run(client, upstream)

    assert result.reason == "stopped"
    assert result.finals == 1
    assert len(result.turn_usages) == 1
    assert result.turn_usages[0].audio_input_tokens == 35

    # Upstream saw: session config, three appends, the stop commit.
    assert upstream.sent[0]["type"] == "session.update"
    assert len(upstream.events("input_audio_buffer.append")) == 3
    assert len(upstream.events("input_audio_buffer.commit")) == 1
    assert upstream.closed

    # Client saw: ready → partials (accumulated) → final → session_closed.
    assert client.sent[0]["type"] == "ready"
    partials = client.frames("partial")
    assert [p["text"] for p in partials] == ["hola ", "hola mundo"]
    finals = client.frames("final")
    assert finals[0]["text"] == "hola mundo"
    closed = client.frames("session_closed")
    assert closed[0]["reason"] == "stopped"
    assert closed[0]["usage_total"]["total_tokens"] == 43
    assert client.sent[-1]["type"] == "session_closed"
    assert client.close_code == 1000


@pytest.mark.asyncio
async def test_idle_timeout() -> None:
    client = FakeClient([HANG])
    upstream = FakeUpstream()

    result = await _run(client, upstream, idle_timeout=0.05)

    assert result.reason == "idle_timeout"
    assert client.frames("session_closed")[0]["reason"] == "idle_timeout"
    assert upstream.closed


@pytest.mark.asyncio
async def test_max_duration() -> None:
    client = FakeClient([HANG])
    upstream = FakeUpstream()

    result = await _run(client, upstream, max_seconds=0.05)

    assert result.reason == "max_duration"
    assert client.frames("session_closed")[0]["reason"] == "max_duration"
    assert upstream.closed


@pytest.mark.asyncio
async def test_client_disconnect_closes_upstream_silently() -> None:
    client = FakeClient([_audio(b"\x00\x01"), DISCONNECT])
    upstream = FakeUpstream()

    result = await _run(client, upstream)

    assert result.reason == "client_disconnect"
    assert upstream.closed  # the cost-control invariant
    assert client.frames("session_closed") == []  # nobody left to tell


@pytest.mark.asyncio
async def test_upstream_error_event_forwarded() -> None:
    client = FakeClient([HANG])
    upstream = FakeUpstream(
        [{"type": "error", "error": {"code": "boom", "message": "it broke"}}, HANG]
    )

    result = await _run(client, upstream, idle_timeout=0.2)

    assert result.reason == "idle_timeout"
    errors = client.frames("error")
    assert errors and errors[0]["code"] == "upstream_error"


@pytest.mark.asyncio
async def test_upstream_close_ends_session() -> None:
    client = FakeClient([HANG])
    upstream = FakeUpstream([], hang_after=False)

    result = await _run(client, upstream)

    assert result.reason == "upstream_closed"
    assert client.frames("session_closed")[0]["reason"] == "upstream_closed"


@pytest.mark.asyncio
async def test_stop_with_nothing_pending_closes_immediately() -> None:
    client = FakeClient([_text({"type": "stop"}), HANG])
    upstream = FakeUpstream()

    result = await _run(client, upstream, finalize_timeout=5.0)

    assert result.reason == "stopped"
    assert result.duration_seconds < 1.0  # no blind finalize wait


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_final() -> None:
    client = FakeClient([_audio(b"\x00\x01"), _text({"type": "stop"}), HANG])
    upstream = FakeUpstream(
        [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "i1",
                "transcript": "tarde pero llega",
            }
        ]
    )

    result = await _run(client, upstream, finalize_timeout=5.0)

    assert result.reason == "stopped"
    finals = client.frames("final")
    assert finals and finals[0]["text"] == "tarde pero llega"
    assert result.duration_seconds < 1.0  # closed as soon as the final landed


@pytest.mark.asyncio
async def test_bad_client_frame_is_nonfatal() -> None:
    client = FakeClient(
        [
            {"type": "websocket.receive", "text": "not json"},
            _audio(b"\x00\x01"),
            _text({"type": "stop"}),
            HANG,
        ]
    )
    upstream = FakeUpstream()

    result = await _run(client, upstream)

    assert result.reason == "stopped"
    errors = client.frames("error")
    assert errors and errors[0]["code"] == "bad_frame"
    assert len(upstream.events("input_audio_buffer.append")) == 1
