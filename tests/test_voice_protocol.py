"""Pure-function tests for the voice-transcription protocol layer."""
from __future__ import annotations

import base64
import json
import uuid

import pytest

from luna_core.schemas.voice import (
    InvalidVoiceFrame,
    VoiceAudioJson,
    VoiceError,
    VoiceFinal,
    VoiceMode,
    VoicePartial,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
    VoiceStart,
    VoiceStop,
    VoiceUsage,
    parse_client_frame,
)
from luna_core.services.voice_context import (
    VoiceContext,
    resolve_voice_context,
    set_voice_context_provider,
)
from luna_core.services.voice_transcription import (
    audio_frame_to_append,
    build_session_update,
    join_draft,
    total_usage,
    translate_upstream_event,
)


ServerFrameTypes = (VoicePartial, VoiceFinal, VoiceSpeechStarted, VoiceSpeechStopped, VoiceError)


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

    def test_duration_usage_becomes_audio_tokens(self) -> None:
        """Per-minute models report seconds; the ledger gets 10 tokens a second."""
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
        assert frame.usage == VoiceUsage(
            input_tokens=32, audio_input_tokens=32, output_tokens=0, total_tokens=32
        )

    def test_unknown_usage_type_is_ignored(self) -> None:
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "hola",
                "usage": {"type": "credits", "amount": 1},
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

    def test_empty_commit_error_is_not_forwarded(self) -> None:
        # Not a client frame: the session loop consumes it as "nothing pending".
        frame = translate_upstream_event(
            {
                "type": "error",
                "error": {"code": "input_audio_buffer_commit_empty", "message": "x"},
            },
            {},
        )
        assert not isinstance(frame, ServerFrameTypes)
        assert frame is not None and repr(frame) == "<commit_empty>"

    def test_transcription_failed_is_not_forwarded(self) -> None:
        acc = {"item_1": "partial text"}
        frame = translate_upstream_event(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "item_1",
                "error": {"message": "nope"},
            },
            acc,
        )
        assert not isinstance(frame, ServerFrameTypes)
        assert frame is not None and repr(frame) == "<turn_failed>"
        assert acc == {}

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


class TestStartMode:
    def test_mode_defaults_to_turns(self) -> None:
        frame = parse_client_frame('{"type": "start", "language": "es"}')
        assert isinstance(frame, VoiceStart)
        assert frame.mode == VoiceMode.turns

    def test_dictation_mode(self) -> None:
        frame = parse_client_frame('{"type": "start", "mode": "dictation"}')
        assert isinstance(frame, VoiceStart)
        assert frame.mode == VoiceMode.dictation

    def test_unknown_mode_is_invalid(self) -> None:
        with pytest.raises(InvalidVoiceFrame):
            parse_client_frame('{"type": "start", "mode": "karaoke"}')


class TestSessionUpdateByModel:
    CONTEXT = VoiceContext(prompt="Diario de cultivo.", keywords=("EC", "perlita", "bad<term", "EC"))

    def _input(self, payload: dict) -> dict:
        return payload["session"]["audio"]["input"]

    def test_new_models_take_language_list_and_keywords(self) -> None:
        audio_input = self._input(build_session_update("gpt-transcribe", "es", context=self.CONTEXT))
        assert audio_input["transcription"] == {
            "model": "gpt-transcribe",
            "languages": ["es"],
            "keywords": ["EC", "perlita"],
            "prompt": "Diario de cultivo.",
        }

    def test_older_models_carry_keywords_in_the_prompt(self) -> None:
        audio_input = self._input(
            build_session_update("gpt-4o-mini-transcribe", "es", context=self.CONTEXT)
        )
        assert audio_input["transcription"] == {
            "model": "gpt-4o-mini-transcribe",
            "language": "es",
            "prompt": "Diario de cultivo. EC, perlita.",
        }

    def test_without_vad_turn_detection_is_off(self) -> None:
        audio_input = self._input(build_session_update("gpt-transcribe", "es", vad=False))
        assert audio_input["turn_detection"] is None

    def test_noise_reduction(self) -> None:
        audio_input = self._input(
            build_session_update("gpt-transcribe", "es", noise_reduction="near_field")
        )
        assert audio_input["noise_reduction"] == {"type": "near_field"}
        assert "noise_reduction" not in self._input(build_session_update("gpt-transcribe", "es"))


class TestJoinDraft:
    def test_pause_period_and_capital_go(self) -> None:
        assert (
            join_draft(["Cambio el timer del LED.", "A doce horas de luz."])
            == "Cambio el timer del LED a doce horas de luz."
        )

    def test_period_before_a_number(self) -> None:
        assert join_draft(["La humedad bajó a.", "45%, entonces prendí el humidificador."]) == (
            "La humedad bajó a 45%, entonces prendí el humidificador."
        )

    def test_question_and_exclamation_stay(self) -> None:
        assert join_draft(["¿Riego hoy?", "Mañana no puedo."]) == "¿Riego hoy? Mañana no puedo."
        assert join_draft(["¡Ya floreció!", "Qué bien."]) == "¡Ya floreció! Qué bien."

    def test_ellipsis_stays(self) -> None:
        assert join_draft(["Le eché...", "No sé cuánto."]) == "Le eché... No sé cuánto."

    def test_acronyms_mixed_case_and_I_keep_their_spelling(self) -> None:
        assert join_draft(["Medí.", "EC uno punto cuatro."]) == "Medí EC uno punto cuatro."
        assert join_draft(["Medí.", "pH seis."]) == "Medí pH seis."
        assert join_draft(["Then.", "I watered."]) == "Then I watered."

    def test_host_terms_keep_their_capital(self) -> None:
        assert (
            join_draft(["Compré un frasco de.", "Trichoderma ayer."], keep_case=["Trichoderma"])
            == "Compré un frasco de Trichoderma ayer."
        )

    def test_empty_turns_are_skipped(self) -> None:
        assert join_draft(["", "Hola.", "  "]) == "Hola."


class TestVoiceContext:
    @pytest.mark.asyncio
    async def test_no_provider_gives_empty_context(self) -> None:
        set_voice_context_provider(None)
        assert await resolve_voice_context(uuid.uuid4(), "es") == VoiceContext()

    @pytest.mark.asyncio
    async def test_failing_provider_gives_empty_context(self) -> None:
        async def boom(user_id: uuid.UUID, language: str | None) -> VoiceContext:
            raise RuntimeError("db down")

        set_voice_context_provider(boom)
        try:
            assert await resolve_voice_context(uuid.uuid4(), "es") == VoiceContext()
        finally:
            set_voice_context_provider(None)

    @pytest.mark.asyncio
    async def test_provider_gets_user_and_language(self) -> None:
        seen = []

        async def provider(user_id: uuid.UUID, language: str | None) -> VoiceContext:
            seen.append((user_id, language))
            return VoiceContext(keywords=("perlita",))

        user = uuid.uuid4()
        set_voice_context_provider(provider)
        try:
            assert (await resolve_voice_context(user, "es")).keywords == ("perlita",)
        finally:
            set_voice_context_provider(None)
        assert seen == [(user, "es")]
