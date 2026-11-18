"""Client↔server protocol for the realtime voice-transcription WebSocket.

Control travels as JSON text frames; audio travels as raw binary WS frames
(PCM16LE mono 24kHz) with a base64-in-JSON fallback for clients that can't
send binary. ``session_closed`` is always the last frame.

Two modes, chosen by the client in ``start``:

- ``turns`` (default): one ``partial`` per in-flight voice turn (the full
  running text of that turn — replace, don't append) and one ``final`` per
  turn. The client joins the turns itself.
- ``dictation``: the server sends ``draft`` with the whole session's running
  text, already joined, while the user talks; after ``stop`` it sends one
  ``transcript`` with the whole recording transcribed in one pass, which
  replaces the draft. Clients just paint what arrives.
"""
from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class InvalidVoiceFrame(ValueError):
    """A client text frame that isn't valid protocol JSON."""


# ── Client → server ─────────────────────────────────────────────────────────


class VoiceMode(StrEnum):
    turns = "turns"
    dictation = "dictation"


class VoiceStart(BaseModel):
    """Must be the first frame. Only ``language`` and ``mode`` are
    client-settable — model and provider are server policy (cost control)."""

    type: Literal["start"]
    language: str | None = Field(default=None, min_length=2, max_length=16)
    mode: VoiceMode = VoiceMode.turns


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


class VoiceDraft(BaseModel):
    """Dictation mode: the whole session's running text — replace, don't
    append. Provisional: the ``transcript`` after ``stop`` supersedes it."""

    type: Literal["draft"] = "draft"
    text: str


class VoiceTranscript(BaseModel):
    """Dictation mode: the definitive text of the whole recording, sent once
    after ``stop``. ``fallback`` is true when the one-pass transcription did
    not land in time and the text is the joined draft instead."""

    type: Literal["transcript"] = "transcript"
    text: str
    fallback: bool = False


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
    VoiceDraft,
    VoiceTranscript,
    VoiceError,
    VoiceSessionClosed,
]
