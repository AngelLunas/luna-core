"""Client↔server protocol for the realtime voice-transcription WebSocket.

Control travels as JSON text frames; audio travels as raw binary WS frames
(PCM16LE mono 24kHz) with a base64-in-JSON fallback for clients that can't
send binary. The server accumulates upstream deltas per turn and sends the
full running text in each ``partial``, so clients just replace — no delta
assembly on their side. ``session_closed`` is always the last frame.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class InvalidVoiceFrame(ValueError):
    """A client text frame that isn't valid protocol JSON."""


# ── Client → server ─────────────────────────────────────────────────────────


class VoiceStart(BaseModel):
    """Must be the first frame. Only ``language`` is client-settable — model
    and provider are server policy (cost control)."""

    type: Literal["start"]
    language: str | None = Field(default=None, min_length=2, max_length=16)


class VoiceAudioJson(BaseModel):
    """Fallback for clients without binary WS frames."""

    type: Literal["audio"]
    data: str  # base64-encoded PCM16LE mono 24kHz


class VoiceStop(BaseModel):
    type: Literal["stop"]


ClientFrame = Annotated[
    Union[VoiceStart, VoiceAudioJson, VoiceStop], Field(discriminator="type")
]
_client_frame_adapter: TypeAdapter[ClientFrame] = TypeAdapter(ClientFrame)


def parse_client_frame(text: str) -> VoiceStart | VoiceAudioJson | VoiceStop:
    try:
        return _client_frame_adapter.validate_json(text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise InvalidVoiceFrame(str(exc)) from exc


# ── Server → client ─────────────────────────────────────────────────────────


class VoiceUsage(BaseModel):
    """Token counts for one transcribed turn (or a session total). Mirrors
    what lands in the ``core.llm_usage`` ledger."""

    input_tokens: int = 0
    audio_input_tokens: int | None = None
    output_tokens: int = 0
    total_tokens: int = 0


class VoiceReady(BaseModel):
    type: Literal["ready"] = "ready"


class VoiceSpeechStarted(BaseModel):
    type: Literal["speech_started"] = "speech_started"


class VoiceSpeechStopped(BaseModel):
    type: Literal["speech_stopped"] = "speech_stopped"


class VoicePartial(BaseModel):
    """Accumulated text of the in-flight turn — replace, don't append."""

    type: Literal["partial"] = "partial"
    item_id: str
    text: str


class VoiceFinal(BaseModel):
    """One per VAD turn."""

    type: Literal["final"] = "final"
    item_id: str
    text: str
    usage: VoiceUsage | None = None


class VoiceError(BaseModel):
    type: Literal["error"] = "error"
    code: Literal["stt_unavailable", "upstream_error", "bad_frame", "start_timeout"]
    message: str


class VoiceSessionClosed(BaseModel):
    """Always the last frame before the server closes the socket."""

    type: Literal["session_closed"] = "session_closed"
    reason: Literal[
        "stopped", "max_duration", "idle_timeout", "upstream_closed", "error"
    ]
    usage_total: VoiceUsage | None = None


ServerFrame = Union[
    VoiceReady,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
    VoicePartial,
    VoiceFinal,
    VoiceError,
    VoiceSessionClosed,
]
