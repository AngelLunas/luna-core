"""Realtime voice-transcription proxy (client WS ↔ OpenAI Realtime WS).

Generic infrastructure: nothing app-specific lives here. The client speaks the
protocol in ``luna_core.schemas.voice`` (JSON control + binary PCM16 frames);
upstream is OpenAI's Realtime API in transcription mode. The proxy is two
concurrent pumps plus a hard session deadline — the idle timeout rides on the
client pump's ``wait_for`` so any client frame resets it.

Cost containment is structural: the upstream socket is ALWAYS closed in the
``finally`` (a vanished client can't leave a billing session open), sessions
are capped at ``voice_stt_max_session_seconds``, and per-turn token usage is
returned to the caller for the ``core.llm_usage`` ledger. Voice sessions have
no run/conversation, so callers record usage with the authenticated user id
as ``scope_id``.

The upstream wire shapes (session config, event names, usage payload) are
isolated in ``build_session_update`` / ``translate_upstream_event`` — if the
Realtime API drifts, the fix lands in those two functions only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import websockets
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.schemas.voice import (
    InvalidVoiceFrame,
    ServerFrame,
    VoiceAudioJson,
    VoiceError,
    VoiceFinal,
    VoicePartial,
    VoiceReady,
    VoiceSessionClosed,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
    VoiceStop,
    VoiceUsage,
    parse_client_frame,
)
from luna_core.services.llm_provider import (
    LLMProviderNotFound,
    get_decrypted_api_key,
    get_llm_provider_by_name,
)

logger = logging.getLogger(__name__)


class SttUnavailable(RuntimeError):
    """No usable STT provider: row missing, inactive, or keyless."""


@dataclass(frozen=True)
class SttSessionConfig:
    api_key: str
    model: str
    url: str


async def resolve_stt_session(db: AsyncSession) -> SttSessionConfig:
    """Resolve credentials from the LLM-provider registry. The provider row
    (named ``settings.voice_stt_provider_name``) supplies only the key; model
    and realtime URL are luna-core settings."""
    try:
        provider = await get_llm_provider_by_name(
            db, settings.voice_stt_provider_name
        )
    except LLMProviderNotFound as exc:
        raise SttUnavailable(
            f"no LLM provider named {settings.voice_stt_provider_name!r}"
        ) from exc
    if not provider.is_active:
        raise SttUnavailable(f"provider {provider.name!r} is inactive")
    api_key = get_decrypted_api_key(provider)
    if not api_key:
        raise SttUnavailable(f"provider {provider.name!r} has no API key")
    return SttSessionConfig(
        api_key=api_key,
        model=settings.voice_stt_model,
        url=settings.voice_stt_realtime_url,
    )


# ── Pure translation layer (unit-testable, no I/O) ──────────────────────────


def build_session_update(model: str, language: str | None) -> dict[str, Any]:
    """GA Realtime transcription-session config (PCM16 mono 24kHz, server
    VAD). Verified against the 2026 docs: ``session.update`` with
    ``session.type: "transcription"`` and audio config nested under
    ``audio.input``."""
    transcription: dict[str, Any] = {"model": model}
    if language:
        transcription["language"] = language
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": transcription,
                    "turn_detection": {"type": "server_vad"},
                }
            },
        },
    }


def audio_frame_to_append(data: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(data).decode("ascii"),
        }
    )


def _usage_from_event(event: dict[str, Any]) -> VoiceUsage | None:
    usage = event.get("usage")
    if not isinstance(usage, dict) or usage.get("type") != "tokens":
        return None

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    details = usage.get("input_token_details")
    audio = details.get("audio_tokens") if isinstance(details, dict) else None
    return VoiceUsage(
        input_tokens=_int(usage.get("input_tokens")),
        audio_input_tokens=_int(audio) if audio is not None else None,
        output_tokens=_int(usage.get("output_tokens")),
        total_tokens=_int(usage.get("total_tokens")),
    )


def translate_upstream_event(
    event: dict[str, Any], acc: dict[str, str]
) -> ServerFrame | None:
    """Map one upstream event to a protocol frame (None = drop). ``acc``
    accumulates transcription deltas per item so each ``partial`` carries the
    full running text of its turn."""
    etype = event.get("type")
    if etype == "conversation.item.input_audio_transcription.delta":
        item_id = str(event.get("item_id") or "")
        acc[item_id] = acc.get(item_id, "") + str(event.get("delta") or "")
        return VoicePartial(item_id=item_id, text=acc[item_id])
    if etype == "conversation.item.input_audio_transcription.completed":
        item_id = str(event.get("item_id") or "")
        transcript = event.get("transcript")
        text = transcript if isinstance(transcript, str) else acc.get(item_id, "")
        acc.pop(item_id, None)
        return VoiceFinal(item_id=item_id, text=text, usage=_usage_from_event(event))
    if etype == "input_audio_buffer.speech_started":
        return VoiceSpeechStarted()
    if etype == "input_audio_buffer.speech_stopped":
        return VoiceSpeechStopped()
    if etype == "error":
        error = event.get("error")
        error = error if isinstance(error, dict) else {}
        # Expected when `stop` lands with an empty/already-committed buffer —
        # the commit we send on stop is best-effort (see _pump_client).
        if error.get("code") == "input_audio_buffer_commit_empty":
            return None
        return VoiceError(
            code="upstream_error",
            message=str(error.get("message") or "upstream error"),
        )
    return None


def total_usage(usages: list[VoiceUsage]) -> VoiceUsage | None:
    if not usages:
        return None
    audio_parts = [u.audio_input_tokens for u in usages if u.audio_input_tokens is not None]
    return VoiceUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        audio_input_tokens=sum(audio_parts) if audio_parts else None,
        output_tokens=sum(u.output_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
    )


# ── Transports (structural — FastAPI WebSocket / websockets both conform) ───


class ClientTransport(Protocol):
    async def receive(self) -> dict[str, Any]: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000) -> None: ...


class UpstreamTransport(Protocol):
    async def send(self, data: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


async def open_upstream(config: SttSessionConfig) -> UpstreamTransport:
    return await websockets.connect(
        config.url,
        additional_headers={"Authorization": f"Bearer {config.api_key}"},
    )


# ── Session orchestration ────────────────────────────────────────────────────


@dataclass
class SessionResult:
    """What the session cost and how it ended — the router flushes
    ``turn_usages`` to the ``core.llm_usage`` ledger after the socket work
    is done."""

    reason: str = "upstream_closed"
    turn_usages: list[VoiceUsage] = field(default_factory=list)
    finals: int = 0
    duration_seconds: float = 0.0


async def run_transcription_session(
    client: ClientTransport,
    upstream: UpstreamTransport,
    *,
    model: str,
    language: str | None = None,
    max_seconds: float | None = None,
    idle_timeout: float | None = None,
    finalize_timeout: float | None = None,
) -> SessionResult:
    """Bridge one client session to one upstream transcription session.
    Assumes the client socket is already accepted and the ``start`` frame
    consumed. Always closes the upstream socket before returning."""
    max_seconds = max_seconds if max_seconds is not None else float(settings.voice_stt_max_session_seconds)
    idle_timeout = idle_timeout if idle_timeout is not None else float(settings.voice_stt_idle_timeout_seconds)
    finalize_timeout = finalize_timeout if finalize_timeout is not None else settings.voice_stt_finalize_timeout_seconds

    started = time.monotonic()
    result = SessionResult()
    acc: dict[str, str] = {}
    # True while audio has been appended upstream that no `final` has covered
    # yet. Drives the post-stop grace: if nothing is pending the session
    # closes immediately; if a turn is in flight we wait for its final (up to
    # finalize_timeout) instead of blindly burning the whole timeout.
    pending_audio = False

    async def pump_client() -> str:
        """Client → upstream. Returns the end reason. The ``wait_for`` IS the
        idle timeout — any client frame resets it."""
        nonlocal pending_audio
        while True:
            try:
                message = await asyncio.wait_for(client.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                return "idle_timeout"
            if message.get("type") == "websocket.disconnect":
                return "client_disconnect"
            data = message.get("bytes")
            if data:
                pending_audio = True
                await upstream.send(audio_frame_to_append(data))
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                frame = parse_client_frame(text)
            except InvalidVoiceFrame as exc:
                await client.send_text(
                    VoiceError(code="bad_frame", message=str(exc)).model_dump_json()
                )
                continue
            if isinstance(frame, VoiceAudioJson):
                pending_audio = True
                await upstream.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": frame.data}
                    )
                )
            elif isinstance(frame, VoiceStop):
                # Force transcription of any speech VAD hasn't segmented yet
                # (user tapped stop mid-utterance). Empty-buffer errors from
                # this commit are swallowed in translate_upstream_event.
                try:
                    await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception:  # noqa: BLE001 — commit is best-effort
                    pass
                return "stopped"
            # a duplicate `start` is ignored

    async def pump_upstream() -> str:
        """Upstream → client. Returns when the upstream socket closes."""
        nonlocal pending_audio
        while True:
            try:
                raw = await upstream.recv()
            except Exception:  # noqa: BLE001 — any close/teardown ends the pump
                return "upstream_closed"
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            frame = translate_upstream_event(event, acc)
            if frame is None:
                continue
            if isinstance(frame, VoiceFinal):
                result.finals += 1
                if frame.usage is not None:
                    result.turn_usages.append(frame.usage)
                if not acc:
                    # No other turn in flight — everything appended so far
                    # is transcribed (drives the post-stop grace period).
                    pending_audio = False
            await client.send_text(frame.model_dump_json())

    client_task: asyncio.Task[str] = asyncio.create_task(pump_client())
    upstream_task: asyncio.Task[str] = asyncio.create_task(pump_upstream())
    deadline_task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(max_seconds))
    tasks: set[asyncio.Task[Any]] = {client_task, upstream_task, deadline_task}

    try:
        try:
            await upstream.send(json.dumps(build_session_update(model, language)))
            await client.send_text(VoiceReady().model_dump_json())
        except Exception:  # noqa: BLE001
            result.reason = "error"
            await _send_best_effort(
                client,
                VoiceError(code="upstream_error", message="could not configure upstream session"),
            )
            return result

        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if deadline_task in done:
            result.reason = "max_duration"
        elif client_task in done:
            exc = client_task.exception()
            if exc is not None:
                # sends to a dead upstream raise here — surface as upstream loss
                result.reason = "upstream_closed"
            else:
                result.reason = client_task.result()
                if result.reason == "stopped":
                    # Post-stop grace: only if a turn is still untranscribed,
                    # and only until its final lands — a stop with nothing
                    # pending closes immediately.
                    grace_deadline = time.monotonic() + finalize_timeout
                    while pending_audio and time.monotonic() < grace_deadline:
                        if upstream_task.done():
                            break
                        await asyncio.sleep(0.05)
        else:
            result.reason = upstream_task.result() if upstream_task.exception() is None else "upstream_closed"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            pass
        result.duration_seconds = time.monotonic() - started
        if result.reason != "client_disconnect":
            closed_reason = (
                result.reason
                if result.reason in ("stopped", "max_duration", "idle_timeout", "upstream_closed", "error")
                else "error"
            )
            await _send_best_effort(
                client,
                VoiceSessionClosed(
                    reason=closed_reason,  # type: ignore[arg-type]
                    usage_total=total_usage(result.turn_usages),
                ),
            )
            try:
                await client.close(1000)
            except Exception:  # noqa: BLE001
                pass
    return result


async def _send_best_effort(client: ClientTransport, frame: ServerFrame) -> None:
    try:
        await client.send_text(frame.model_dump_json())
    except Exception:  # noqa: BLE001 — client may already be gone
        pass
