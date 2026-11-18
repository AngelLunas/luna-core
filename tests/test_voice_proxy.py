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
HANG_UNTIL_COMMIT = object()  # sentinel: the fake upstream waits for our commit


def _text(payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


def _sleep(seconds: float) -> dict[str, Any]:
    """Pause the client script — lets queued upstream events land first, the
    way a real user's stop arrives seconds after the utterance's final."""
    return {"type": "__sleep__", "seconds": seconds}


def _audio(data: bytes) -> dict[str, Any]:
    return {"type": "websocket.receive", "bytes": data}


DISCONNECT = {"type": "websocket.disconnect"}


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.sent: list[dict[str, Any]] = []
        self.close_code: int | None = None

    async def receive(self) -> dict[str, Any]:
        while True:
            if not self._script or self._script[0] is HANG:
                await asyncio.Event().wait()
            item = self._script.pop(0)
            if item.get("type") == "__sleep__":
                await asyncio.sleep(item["seconds"])
                continue
            return item

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
            if item is HANG_UNTIL_COMMIT:
                while not self.events("input_audio_buffer.commit"):
                    await asyncio.sleep(0.01)
                item = self._events.pop(0)
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
        "vad_settle": 0.3,
    }
    defaults.update(overrides)
    return run_transcription_session(client, upstream, **defaults)


@pytest.mark.asyncio
async def test_happy_path_stop() -> None:
    client = FakeClient(
        [_audio(b"\x00\x01"), _audio(b"\x02\x03"), _audio(b"\x04\x05"), _sleep(0.15), _text({"type": "stop"}), HANG]
    )
    upstream = FakeUpstream(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
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

    # Upstream saw: session config, three appends — and NO stop commit: VAD
    # had already segmented the utterance (speech_stopped), so the buffer
    # held only trailing silence.
    assert upstream.sent[0]["type"] == "session.update"
    assert len(upstream.events("input_audio_buffer.append")) == 3
    assert len(upstream.events("input_audio_buffer.commit")) == 0
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
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "i1",
                "transcript": "tarde pero llega",
            },
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


@pytest.mark.asyncio
async def test_stop_after_final_plus_trailing_silence_closes_immediately() -> None:
    """The app keeps streaming mic silence after the user stops talking. Once
    the utterance's final has landed, a stop must NOT wait the finalize grace
    (silence is not a pending turn) and must NOT force-commit the silence."""
    client = FakeClient(
        [
            _audio(b"\x00\x01"),
            _sleep(0.15),  # the utterance's final lands here
            _audio(b"\x00\x00"),  # trailing silence keeps streaming…
            _audio(b"\x00\x00"),
            _sleep(1.2),  # …for longer than the VAD settle window
            _text({"type": "stop"}),
            HANG,
        ]
    )
    upstream = FakeUpstream(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "i1",
                "transcript": "ya está todo dicho",
            },
        ]
    )

    result = await _run(client, upstream, finalize_timeout=5.0)

    assert result.reason == "stopped"
    assert result.finals == 1
    assert result.duration_seconds < 1.6  # ~1.35s of scripted client time, no grace wait
    assert len(upstream.events("input_audio_buffer.commit")) == 0


@pytest.mark.asyncio
async def test_stop_mid_utterance_commits_and_waits_for_that_final() -> None:
    """User taps stop while still talking: VAD saw speech start but no stop →
    we force a commit and wait (only) for its final."""
    client = FakeClient([_audio(b"\x00\x01"), _text({"type": "stop"}), HANG])
    upstream = FakeUpstream(
        [
            {"type": "input_audio_buffer.speech_started"},
            HANG_UNTIL_COMMIT,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "i1",
                "transcript": "cortado a media frase",
            },
        ]
    )

    result = await _run(client, upstream, finalize_timeout=5.0)

    assert result.reason == "stopped"
    assert len(upstream.events("input_audio_buffer.commit")) == 1
    assert client.frames("final")[0]["text"] == "cortado a media frase"
    assert result.duration_seconds < 1.0


@pytest.mark.asyncio
async def test_stop_mid_utterance_commit_empty_closes() -> None:
    """Speech started, stop, but the forced commit finds nothing upstream:
    the commit_empty error ends the wait instead of burning the grace."""
    client = FakeClient([_audio(b"\x00\x01"), _text({"type": "stop"}), HANG])
    upstream = FakeUpstream(
        [
            {"type": "input_audio_buffer.speech_started"},
            HANG_UNTIL_COMMIT,
            {
                "type": "error",
                "error": {"code": "input_audio_buffer_commit_empty", "message": "x"},
            },
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "i1",
                "error": {"message": "no audio"},
            },
        ]
    )

    result = await _run(client, upstream, finalize_timeout=5.0)

    assert result.reason == "stopped"
    assert client.frames("error") == []  # commit_empty never reaches the client
    assert result.duration_seconds < 1.0


# ── Dictation mode: a VAD draft session plus a one-pass session ─────────────

from luna_core.services.voice_context import VoiceContext  # noqa: E402
from luna_core.services.voice_transcription import run_dictation_session  # noqa: E402


def _usage(total: int) -> dict[str, Any]:
    return {
        "type": "tokens",
        "input_tokens": total,
        "input_token_details": {"audio_tokens": total},
        "output_tokens": 0,
        "total_tokens": total,
    }


def _draft_turns() -> list[Any]:
    """Two VAD turns, each transcribed as a sentence of its own."""
    return [
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "input_audio_buffer.committed", "item_id": "d1"},
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "d1", "delta": "Cambio el timer"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "d1",
            "transcript": "Cambio el timer del LED.",
            "usage": _usage(10),
        },
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "input_audio_buffer.committed", "item_id": "d2"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "d2",
            "transcript": "A doce horas de luz.",
            "usage": _usage(8),
        },
    ]


def _dictation_client() -> FakeClient:
    return FakeClient([_audio(b"\x00\x01"), _audio(b"\x02\x03"), _sleep(0.25), _text({"type": "stop"}), HANG])


def _run_dictation(client: FakeClient, draft: FakeUpstream, full: FakeUpstream, **overrides: Any):
    defaults: dict[str, Any] = {
        "model": "gpt-transcribe",
        "language": "es",
        "context": VoiceContext(prompt="Diario de cultivo.", keywords=("LED",)),
        "max_seconds": 5.0,
        "idle_timeout": 5.0,
        "finalize_timeout": 0.3,
    }
    defaults.update(overrides)
    return run_dictation_session(client, draft, full, **defaults)


@pytest.mark.asyncio
async def test_dictation_drafts_then_one_pass_transcript() -> None:
    client = _dictation_client()
    draft = FakeUpstream(_draft_turns())
    full = FakeUpstream(
        [
            HANG_UNTIL_COMMIT,
            {"type": "input_audio_buffer.committed", "item_id": "f1"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "f1",
                "transcript": "Cambio el timer del LED a doce horas de luz.",
                "usage": _usage(20),
            },
        ]
    )
    result = await _run_dictation(client, draft, full)

    assert result.reason == "stopped"
    assert [f["text"] for f in client.frames("draft")] == [
        "Cambio el timer",
        "Cambio el timer del LED.",
        "Cambio el timer del LED a doce horas de luz.",
    ]
    assert client.frames("transcript") == [
        {"type": "transcript", "text": "Cambio el timer del LED a doce horas de luz.", "fallback": False}
    ]
    assert client.sent[-1]["type"] == "session_closed"
    assert client.sent[-1]["usage_total"]["total_tokens"] == 38  # both sessions are billed
    # Same audio to both; only the draft session cuts turns.
    assert len(draft.events("input_audio_buffer.append")) == len(full.events("input_audio_buffer.append")) == 2
    assert draft.events("session.update")[0]["session"]["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert full.events("session.update")[0]["session"]["audio"]["input"]["turn_detection"] is None
    assert full.events("input_audio_buffer.commit")
    assert draft.closed and full.closed


@pytest.mark.asyncio
async def test_dictation_falls_back_to_draft_when_one_pass_fails() -> None:
    client = _dictation_client()
    draft = FakeUpstream(_draft_turns())
    full = FakeUpstream(
        [HANG_UNTIL_COMMIT, {"type": "conversation.item.input_audio_transcription.failed", "item_id": "f1"}]
    )
    await _run_dictation(client, draft, full)

    assert client.frames("transcript") == [
        {"type": "transcript", "text": "Cambio el timer del LED a doce horas de luz.", "fallback": True}
    ]
    assert draft.closed and full.closed


@pytest.mark.asyncio
async def test_dictation_falls_back_when_one_pass_is_too_slow() -> None:
    client = _dictation_client()
    draft = FakeUpstream(_draft_turns())
    full = FakeUpstream([HANG])
    await _run_dictation(client, draft, full, finalize_timeout=0.1)

    transcript = client.frames("transcript")
    assert transcript[0]["fallback"] is True
    assert transcript[0]["text"] == "Cambio el timer del LED a doce horas de luz."


@pytest.mark.asyncio
async def test_dictation_keeps_drafting_when_one_pass_session_drops() -> None:
    client = _dictation_client()
    draft = FakeUpstream(_draft_turns())
    full = FakeUpstream([], hang_after=False)
    result = await _run_dictation(client, draft, full)

    assert result.reason == "stopped"
    assert client.frames("draft")
    assert client.frames("transcript")[0]["fallback"] is True


@pytest.mark.asyncio
async def test_dictation_with_nothing_said() -> None:
    client = FakeClient([_sleep(0.05), _text({"type": "stop"}), HANG])
    draft = FakeUpstream([])
    full = FakeUpstream(
        [HANG_UNTIL_COMMIT, {"type": "error", "error": {"code": "input_audio_buffer_commit_empty"}}]
    )
    await _run_dictation(client, draft, full)

    assert client.frames("transcript") == [{"type": "transcript", "text": "", "fallback": False}]


@pytest.mark.asyncio
async def test_dictation_idle_timeout_sends_no_transcript() -> None:
    client = FakeClient([HANG])
    draft = FakeUpstream([])
    full = FakeUpstream([])
    result = await _run_dictation(client, draft, full, idle_timeout=0.05)

    assert result.reason == "idle_timeout"
    assert not client.frames("transcript")
    assert client.sent[-1] == {"type": "session_closed", "reason": "idle_timeout", "usage_total": None}
    assert draft.closed and full.closed


@pytest.mark.asyncio
async def test_dictation_max_duration_still_transcribes() -> None:
    client = FakeClient([_audio(b"\x00\x01"), HANG])
    draft = FakeUpstream([])
    full = FakeUpstream(
        [
            HANG_UNTIL_COMMIT,
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "f1",
                "transcript": "Hola.",
            },
        ]
    )
    result = await _run_dictation(client, draft, full, max_seconds=0.1)

    assert result.reason == "max_duration"
    assert client.frames("transcript") == [{"type": "transcript", "text": "Hola.", "fallback": False}]


@pytest.mark.asyncio
async def test_dictation_client_disconnect_closes_upstreams_silently() -> None:
    client = FakeClient([_audio(b"\x00\x01"), DISCONNECT])
    draft = FakeUpstream([])
    full = FakeUpstream([])
    result = await _run_dictation(client, draft, full)

    assert result.reason == "client_disconnect"
    assert not client.frames("transcript")
    assert not client.frames("session_closed")
    assert draft.closed and full.closed


def _tone(seconds: float, amplitude: int) -> bytes:
    import array as _array

    return _array.array("h", [amplitude] * int(24000 * seconds)).tobytes()


@pytest.mark.asyncio
async def test_dictation_cuts_the_draft_at_a_quiet_chunk_during_long_speech() -> None:
    loud, quiet = _tone(0.1, 8000), _tone(0.1, 500)
    # 2.4 s of steady speech, then a quiet chunk: past the soft limit, cut there.
    script = [_sleep(0.05)] + [_audio(loud)] * 24 + [_audio(quiet), _audio(loud), _sleep(0.1), _text({"type": "stop"}), HANG]
    client = FakeClient(script)
    draft = FakeUpstream([{"type": "input_audio_buffer.speech_started"}])
    full = FakeUpstream([HANG_UNTIL_COMMIT, {"type": "error", "error": {"code": "input_audio_buffer_commit_empty"}}])
    await _run_dictation(client, draft, full, draft_cut_seconds=(2.0, 4.0))

    commits = [i for i, e in enumerate(draft.sent) if e["type"] == "input_audio_buffer.commit"]
    appends_before_cut = [e for e in draft.sent[: commits[0]] if e["type"] == "input_audio_buffer.append"]
    assert len(appends_before_cut) == 25  # the cut lands right after the quiet chunk


@pytest.mark.asyncio
async def test_dictation_cuts_the_draft_at_the_hard_limit() -> None:
    loud = _tone(0.1, 8000)
    script = [_sleep(0.05)] + [_audio(loud)] * 12 + [_sleep(0.1), _text({"type": "stop"}), HANG]
    client = FakeClient(script)
    draft = FakeUpstream([{"type": "input_audio_buffer.speech_started"}])
    full = FakeUpstream(
        [HANG_UNTIL_COMMIT, {"type": "conversation.item.input_audio_transcription.completed", "item_id": "f1", "transcript": "x"}]
    )
    await _run_dictation(client, draft, full, draft_cut_seconds=(0.5, 1.0))

    commits = [i for i, e in enumerate(draft.sent) if e["type"] == "input_audio_buffer.commit"]
    assert len(commits) == 1  # 1.2 s of steady speech: one cut at 1.0 s
    appends_before_cut = [e for e in draft.sent[: commits[0]] if e["type"] == "input_audio_buffer.append"]
    assert len(appends_before_cut) == 10


@pytest.mark.asyncio
async def test_dictation_never_cuts_silence() -> None:
    quiet = _tone(0.1, 100)
    script = [_audio(quiet)] * 12 + [_sleep(0.1), _text({"type": "stop"}), HANG]
    client = FakeClient(script)
    draft = FakeUpstream([])  # VAD never hears speech
    full = FakeUpstream([HANG_UNTIL_COMMIT, {"type": "error", "error": {"code": "input_audio_buffer_commit_empty"}}])
    await _run_dictation(client, draft, full, draft_cut_seconds=(0.2, 0.5))

    assert not draft.events("input_audio_buffer.commit")
