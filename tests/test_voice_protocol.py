"""Pure-function tests for the voice-transcription protocol layer."""
from __future__ import annotations

import base64
import json

import pytest

from luna_core.schemas.voice import (
    InvalidVoiceFrame,
    VoiceAudioJson,
    VoiceError,
    VoiceFinal,
    VoicePartial,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
    VoiceStart,
    VoiceStop,
    VoiceUsage,
    parse_client_frame,
)
from luna_core.services.voice_transcription import (
    audio_frame_to_append,
    build_session_update,
    total_usage,
    translate_upstream_event,
)


class TestParseClientFrame:
    def test_start_with_language(self) -> None:
        frame = parse_client_frame('{"type": "start", "language": "es"}')
        assert isinstance(frame, VoiceStart)
        assert frame.language == "es"

    def test_start_without_language(self) -> None:
        frame = parse_client_frame('{"type": "start"}')
        assert isinstance(frame, VoiceStart)
        assert frame.language is None

    def test_stop(self) -> None:
        assert isinstance(parse_client_frame('{"type": "stop"}'), VoiceStop)

    def test_audio_json_fallback(self) -> None:
        frame = parse_client_frame('{"type": "audio", "data": "AAAA"}')
        assert isinstance(frame, VoiceAudioJson)
        assert frame.data == "AAAA"

    @pytest.mark.parametrize(
        "raw",
        ["not json", "{}", '{"type": "unknown"}', '{"type": "audio"}', ""],
    )
    def test_invalid_frames_raise(self, raw: str) -> None:
        with pytest.raises(InvalidVoiceFrame):
            parse_client_frame(raw)


class TestBuildSessionUpdate:
    def test_shape(self) -> None:
        payload = build_session_update("gpt-4o-mini-transcribe", "es")
        assert payload["type"] == "session.update"
        session = payload["session"]
        assert session["type"] == "transcription"
        audio_input = session["audio"]["input"]
        assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
        assert audio_input["transcription"] == {
            "model": "gpt-4o-mini-transcribe",
            "language": "es",
        }
        assert audio_input["turn_detection"] == {"type": "server_vad"}

    def test_language_omitted_when_none(self) -> None:
        payload = build_session_update("gpt-4o-mini-transcribe", None)
        transcription = payload["session"]["audio"]["input"]["transcription"]
        assert "language" not in transcription


def test_audio_frame_to_append_roundtrip() -> None:
    raw = b"\x01\x02\x03\x04"
    event = json.loads(audio_frame_to_append(raw))
    assert event["type"] == "input_audio_buffer.append"
    assert base64.b64decode(event["audio"]) == raw


class TestTranslateUpstreamEvent:
    def test_deltas_accumulate_into_partials(self) -> None:
        acc: dict[str, str] = {}
        first = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "hola ",
            },
            acc,
        )
        second = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "mundo",
            },
            acc,
        )
        assert isinstance(first, VoicePartial)
        assert first.text == "hola "
        assert isinstance(second, VoicePartial)
        assert second.text == "hola mundo"

    def test_completed_uses_transcript_and_pops_accumulator(self) -> None:
        acc = {"item_1": "hola mun"}
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "hola mundo",
            },
            acc,
        )
        assert isinstance(frame, VoiceFinal)
        assert frame.text == "hola mundo"
        assert frame.usage is None
        assert acc == {}

    def test_completed_falls_back_to_accumulated_text(self) -> None:
        acc = {"item_1": "hola mundo"}
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
            },
            acc,
        )
        assert isinstance(frame, VoiceFinal)
        assert frame.text == "hola mundo"

    def test_completed_extracts_token_usage(self) -> None:
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "hola",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 40,
                    "input_token_details": {"text_tokens": 5, "audio_tokens": 35},
                    "output_tokens": 3,
                    "total_tokens": 43,
                },
            },
            {},
        )
        assert isinstance(frame, VoiceFinal)
        assert frame.usage == VoiceUsage(
            input_tokens=40, audio_input_tokens=35, output_tokens=3, total_tokens=43
        )

    def test_duration_usage_is_ignored(self) -> None:
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "hola",
                "usage": {"type": "duration", "seconds": 3.2},
            },
            {},
        )
        assert isinstance(frame, VoiceFinal)
        assert frame.usage is None

    def test_vad_events(self) -> None:
        assert isinstance(
            translate_upstream_event({"type": "input_audio_buffer.speech_started"}, {}),
            VoiceSpeechStarted,
        )
        assert isinstance(
            translate_upstream_event({"type": "input_audio_buffer.speech_stopped"}, {}),
            VoiceSpeechStopped,
        )

    def test_error_event_maps_to_voice_error(self) -> None:
        frame = translate_upstream_event(
            {"type": "error", "error": {"code": "boom", "message": "it broke"}}, {}
        )
        assert isinstance(frame, VoiceError)
        assert frame.code == "upstream_error"
        assert frame.message == "it broke"

    def test_empty_commit_error_is_swallowed(self) -> None:
        frame = translate_upstream_event(
            {
                "type": "error",
                "error": {"code": "input_audio_buffer_commit_empty", "message": "x"},
            },
            {},
        )
        assert frame is None

    def test_unknown_event_dropped(self) -> None:
        assert translate_upstream_event({"type": "session.updated"}, {}) is None


class TestTotalUsage:
    def test_empty_is_none(self) -> None:
        assert total_usage([]) is None

    def test_sums_turns(self) -> None:
        total = total_usage(
            [
                VoiceUsage(input_tokens=40, audio_input_tokens=35, output_tokens=3, total_tokens=43),
                VoiceUsage(input_tokens=10, audio_input_tokens=None, output_tokens=2, total_tokens=12),
            ]
        )
        assert total == VoiceUsage(
            input_tokens=50, audio_input_tokens=35, output_tokens=5, total_tokens=55
        )

    def test_audio_none_when_never_reported(self) -> None:
        total = total_usage([VoiceUsage(input_tokens=1, output_tokens=1, total_tokens=2)])
        assert total is not None
        assert total.audio_input_tokens is None
