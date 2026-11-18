"""Realtime voice-transcription proxy (client WS ↔ OpenAI Realtime WS).

Generic infrastructure: nothing app-specific lives here (domain vocabulary
comes from the host through ``services.voice_context``). The client speaks the
protocol in ``luna_core.schemas.voice`` (JSON control + binary PCM16 frames);
upstream is OpenAI's Realtime API in transcription mode. Each session is
concurrent pumps plus a hard session deadline — the idle timeout rides on the
client pump's ``wait_for`` so any client frame resets it.

``turns`` mode bridges one upstream session cut by server VAD. ``dictation``
mode feeds the same audio to two upstream sessions: a VAD one whose turns are
joined into a live draft, and one without turn detection that transcribes the
whole recording in a single pass when the user stops — per-turn transcripts
punctuate every pause as a sentence end, the single pass does not.

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

import array
import asyncio
import base64
import json
import logging
import math
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import websockets
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.schemas.voice import (
    InvalidVoiceFrame,
    ServerFrame,
    VoiceAudioJson,
    VoiceDraft,
    VoiceError,
    VoiceFinal,
    VoicePartial,
    VoiceReady,
    VoiceSessionClosed,
    VoiceSpeechStarted,
    VoiceSpeechStopped,
    VoiceStop,
    VoiceTranscript,
    VoiceUsage,
    parse_client_frame,
)
from luna_core.services.voice_context import VoiceContext
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
    noise_reduction: str | None = None


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
        noise_reduction=settings.voice_stt_noise_reduction or None,
    )


# ── Pure translation layer (unit-testable, no I/O) ──────────────────────────


# Models of the gpt-4o / whisper generation take a single ``language`` and no
# ``keywords``; their successors (gpt-transcribe, gpt-live-transcribe) take a
# ``languages`` list and ``keywords``.
_SINGLE_LANGUAGE_MODEL_PREFIXES = ("gpt-4o", "whisper")
# The Realtime API rejects a session whose keyword contains any of these.
_KEYWORD_FORBIDDEN = re.compile(r"[<>\r\n]")


def _clean_keywords(keywords: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for keyword in keywords:
        keyword = keyword.strip()
        if keyword and not _KEYWORD_FORBIDDEN.search(keyword):
            seen.setdefault(keyword, None)
    return list(seen)


def build_session_update(
    model: str,
    language: str | None,
    *,
    vad: bool = True,
    vad_silence_ms: int | None = None,
    context: VoiceContext | None = None,
    noise_reduction: str | None = None,
) -> dict[str, Any]:
    """GA Realtime transcription-session config (PCM16 mono 24kHz).
    ``session.update`` with ``session.type: "transcription"`` and audio config
    nested under ``audio.input``. ``vad=False`` leaves turn detection off: the
    buffer is transcribed only when the proxy commits it. ``vad_silence_ms``
    is how long a silence closes a turn (the API's default when None)."""
    single_language = model.startswith(_SINGLE_LANGUAGE_MODEL_PREFIXES)
    transcription: dict[str, Any] = {"model": model}
    if language:
        if single_language:
            transcription["language"] = language
        else:
            transcription["languages"] = [language]
    keywords = _clean_keywords(context.keywords) if context else []
    prompt = (context.prompt or "").strip() if context else ""
    if keywords and single_language:
        # No keyword field on these models: the terms ride in the prompt.
        prompt = " ".join(filter(None, [prompt, ", ".join(keywords) + "."]))
    elif keywords:
        transcription["keywords"] = keywords
    if prompt:
        transcription["prompt"] = prompt
    audio_input: dict[str, Any] = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": transcription,
        "turn_detection": None,
    }
    if vad:
        audio_input["turn_detection"] = {"type": "server_vad"}
        if vad_silence_ms:
            audio_input["turn_detection"]["silence_duration_ms"] = vad_silence_ms
    if noise_reduction:
        audio_input["noise_reduction"] = {"type": noise_reduction}
    return {
        "type": "session.update",
        "session": {"type": "transcription", "audio": {"input": audio_input}},
    }


def audio_frame_to_append(data: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(data).decode("ascii"),
        }
    )


# Per-minute models (gpt-transcribe) report seconds of audio instead of tokens.
# The ledger and its prices count tokens, so seconds land as audio tokens at the
# Realtime API's own audio rate — one token per 100 ms — and the model's price
# is entered per million of those (USD per minute ÷ 600 × 1e6).
AUDIO_TOKENS_PER_SECOND = 10


def _usage_from_event(event: dict[str, Any]) -> VoiceUsage | None:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    if usage.get("type") == "duration":
        try:
            seconds = float(usage.get("seconds") or 0)
        except (TypeError, ValueError):
            return None
        tokens = math.ceil(seconds * AUDIO_TOKENS_PER_SECOND)
        return VoiceUsage(
            input_tokens=tokens, audio_input_tokens=tokens, output_tokens=0, total_tokens=tokens
        )
    if usage.get("type") != "tokens":
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


# Internal sentinels ``translate_upstream_event`` returns for events that are
# not forwarded to the client but change the session's "anything still to
# transcribe?" bookkeeping. Never sent on the wire.
class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


_COMMIT_EMPTY = _Sentinel("commit_empty")
_TURN_FAILED = _Sentinel("turn_failed")


def translate_upstream_event(
    event: dict[str, Any], acc: dict[str, str]
) -> ServerFrame | _Sentinel | None:
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
    if etype == "conversation.item.input_audio_transcription.failed":
        # The turn ended without text; still a finalisation for bookkeeping.
        acc.pop(str(event.get("item_id") or ""), None)
        return _TURN_FAILED
    if etype == "input_audio_buffer.speech_started":
        return VoiceSpeechStarted()
    if etype == "input_audio_buffer.speech_stopped":
        return VoiceSpeechStopped()
    if etype == "error":
        error = event.get("error")
        error = error if isinstance(error, dict) else {}
        # Expected when `stop` lands with an empty/already-committed buffer —
        # the commit we send on stop is best-effort (see _pump_client). The
        # session loop treats it as "nothing left to transcribe".
        if error.get("code") == "input_audio_buffer_commit_empty":
            return _COMMIT_EMPTY
        return VoiceError(
            code="upstream_error",
            message=str(error.get("message") or "upstream error"),
        )
    return None


# A single trailing period — not an ellipsis, which marks a real hesitation.
_TRAILING_PERIOD = re.compile(r"(?<![.…])\.\s*$")


def _lowercase_opening(fragment: str, keep: set[str]) -> str:
    """Undo the capital a per-turn transcript gives its first word. Acronyms
    (LED), mixed case (pH), the English "I" and the host's own capitalised
    terms keep their spelling."""
    first = fragment.split(maxsplit=1)[0]
    letters = re.sub(r"\W", "", first)
    if not letters or not letters[0].isupper() or letters == "I":
        return fragment
    if letters.casefold() in keep:
        return fragment
    if len(letters) > 1 and not letters[1:].islower():
        return fragment
    index = fragment.index(letters[0])
    return fragment[:index] + letters[0].lower() + fragment[index + 1 :]


def join_draft(fragments: Iterable[str], keep_case: Iterable[str] = ()) -> str:
    """Join per-turn transcripts into one running text. Each turn arrives as a
    sentence of its own ("bajó a. 45%"); a pause is not a sentence end, so the
    period between turns goes and the next turn's capital with it. Questions
    and exclamations keep their marks."""
    keep = {
        re.sub(r"\W", "", term.split(maxsplit=1)[0]).casefold()
        for term in keep_case
        if term.strip() and term.strip()[0].isupper()
    }
    text = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if not text:
            text = fragment
            continue
        if _TRAILING_PERIOD.search(text):
            text = _TRAILING_PERIOD.sub("", text)
            fragment = _lowercase_opening(fragment, keep)
        text = f"{text} {fragment}"
    return text


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
        # Bound the close handshake: the client has already been told the
        # session is over by the time we close upstream.
        close_timeout=3,
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
    context: VoiceContext | None = None,
    noise_reduction: str | None = None,
    max_seconds: float | None = None,
    idle_timeout: float | None = None,
    finalize_timeout: float | None = None,
    vad_settle: float | None = None,
) -> SessionResult:
    """Bridge one client session to one upstream transcription session.
    Assumes the client socket is already accepted and the ``start`` frame
    consumed. Always closes the upstream socket before returning."""
    max_seconds = max_seconds if max_seconds is not None else float(settings.voice_stt_max_session_seconds)
    idle_timeout = idle_timeout if idle_timeout is not None else float(settings.voice_stt_idle_timeout_seconds)
    finalize_timeout = finalize_timeout if finalize_timeout is not None else settings.voice_stt_finalize_timeout_seconds
    vad_settle = vad_settle if vad_settle is not None else settings.voice_stt_vad_settle_seconds

    started = time.monotonic()
    result = SessionResult()
    acc: dict[str, str] = {}
    # Post-stop bookkeeping is driven by SPEECH, not by audio bytes: the client
    # keeps streaming silence after the user stops talking, so "audio appended
    # since the last final" is almost always true and would make every stop
    # wait the whole finalize_timeout for a final that never comes (the forced
    # commit of silence yields nothing — or worse, hallucinated words).
    # ``turns_open`` counts VAD ``speech_started`` events not yet balanced by a
    # transcription completed/failed; ``speech_open`` is True between
    # ``speech_started`` and ``speech_stopped`` (the user is mid-utterance).
    turns_open = 0
    speech_open = False
    # When audio started arriving that VAD has not (yet) given a verdict on —
    # None once VAD reacts (speech_started/stopped) or a turn finalises. Lets a
    # stop that lands a few hundred ms after the user began talking wait for
    # VAD to catch up, while long-running silence VAD ignored closes at once.
    unjudged_since: float | None = None
    commit_sent = False

    def pending() -> bool:
        return turns_open > 0

    async def send_commit() -> None:
        """Force-transcribe what's in the upstream buffer (best-effort)."""
        nonlocal commit_sent
        if commit_sent:
            return
        commit_sent = True
        try:
            await upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception:  # noqa: BLE001 — commit is best-effort
            pass

    async def pump_client() -> str:
        """Client → upstream. Returns the end reason. The ``wait_for`` IS the
        idle timeout — any client frame resets it."""
        nonlocal unjudged_since
        while True:
            try:
                message = await asyncio.wait_for(client.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                return "idle_timeout"
            if message.get("type") == "websocket.disconnect":
                return "client_disconnect"
            data = message.get("bytes")
            if data:
                if unjudged_since is None:
                    unjudged_since = time.monotonic()
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
                if unjudged_since is None:
                    unjudged_since = time.monotonic()
                await upstream.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": frame.data}
                    )
                )
            elif isinstance(frame, VoiceStop):
                # Only when the user tapped stop MID-UTTERANCE (VAD saw speech
                # start but not stop) force-transcribe what's in the buffer.
                # Otherwise VAD already committed every utterance and the
                # buffer holds only trailing silence — committing that would
                # at best waste a round trip and at worst transcribe silence
                # into invented words. (The post-stop loop also commits if VAD
                # reports speech_started moments AFTER the stop.)
                if speech_open:
                    await send_commit()
                return "stopped"
            # a duplicate `start` is ignored

    async def pump_upstream() -> str:
        """Upstream → client. Returns when the upstream socket closes."""
        nonlocal turns_open, speech_open, unjudged_since
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
            if frame is _COMMIT_EMPTY:
                # Our stop-commit found nothing: no turn will come of it.
                speech_open = False
                continue
            if frame is _TURN_FAILED:
                turns_open = max(0, turns_open - 1)
                continue
            if isinstance(frame, VoiceSpeechStarted):
                turns_open += 1
                speech_open = True
                unjudged_since = None
            elif isinstance(frame, VoiceSpeechStopped):
                speech_open = False
                unjudged_since = None
            elif isinstance(frame, VoiceFinal):
                result.finals += 1
                if frame.usage is not None:
                    result.turn_usages.append(frame.usage)
                turns_open = max(0, turns_open - 1)
                speech_open = False
                unjudged_since = None
            await client.send_text(frame.model_dump_json())

    client_task: asyncio.Task[str] = asyncio.create_task(pump_client())
    upstream_task: asyncio.Task[str] = asyncio.create_task(pump_upstream())
    deadline_task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(max_seconds))
    tasks: set[asyncio.Task[Any]] = {client_task, upstream_task, deadline_task}

    try:
        try:
            await upstream.send(
                json.dumps(
                    build_session_update(
                        model, language, context=context, noise_reduction=noise_reduction
                    )
                )
            )
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
                    # Post-stop grace: only while a spoken turn is still
                    # untranscribed, and only until its final lands — a stop
                    # with nothing pending closes immediately. Fresh audio VAD
                    # hasn't judged yet gets a short settle window; if VAD
                    # then reports speech we force its commit and wait for
                    # that final. Long-ignored silence does not count.
                    now = time.monotonic()
                    grace_deadline = now + finalize_timeout
                    settle_deadline = (
                        unjudged_since + vad_settle
                        if unjudged_since is not None
                        and now - unjudged_since < vad_settle
                        else now
                    )
                    while time.monotonic() < grace_deadline and not upstream_task.done():
                        if speech_open:
                            await send_commit()
                        if not pending() and time.monotonic() >= settle_deadline:
                            break
                        await asyncio.sleep(0.02)
        else:
            result.reason = upstream_task.result() if upstream_task.exception() is None else "upstream_closed"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        result.duration_seconds = time.monotonic() - started
        # Tell the client FIRST: the upstream close handshake can take a
        # couple of seconds and the user is staring at a spinner until
        # ``session_closed`` lands. The upstream is still always closed below
        # (cost containment) before this returns.
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
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            pass
    return result


_PCM_BYTES_PER_SECOND = 24000 * 2


def _rms(pcm: bytes) -> float:
    """Loudness of a PCM16LE mono chunk."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))


@dataclass
class _DraftCutter:
    """Decides when to cut the draft session's buffer while the user talks
    without pausing, so the draft keeps up instead of waiting for a pause.
    Past ``soft`` seconds it cuts at the first chunk clearly quieter than the
    turn so far — usually a gap between words; at ``hard`` seconds it cuts
    regardless. A cut that splits a word only blurs the draft: the one-pass
    transcript replaces it."""

    soft: float
    hard: float
    audio_bytes: int = 0
    energy_sum: float = 0.0
    chunks: int = 0

    def reset(self) -> None:
        self.audio_bytes = 0
        self.energy_sum = 0.0
        self.chunks = 0

    def should_cut(self, pcm: bytes) -> bool:
        level = _rms(pcm)
        # Bytes, not summed float seconds: ten 100 ms chunks make exactly 1 s.
        self.audio_bytes += len(pcm)
        mean = self.energy_sum / self.chunks if self.chunks else level
        self.energy_sum += level
        self.chunks += 1
        if self.audio_bytes >= self.hard * _PCM_BYTES_PER_SECOND:
            return True
        return self.audio_bytes >= self.soft * _PCM_BYTES_PER_SECOND and level < 0.5 * mean


@dataclass
class _DraftTurns:
    """The VAD session's turns in commit order, each with its final text or,
    while it is still being transcribed, the deltas received so far."""

    order: list[str] = field(default_factory=list)
    text: dict[str, str] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)

    def _track(self, item_id: str) -> None:
        if item_id not in self.text:
            self.order.append(item_id)
            self.text[item_id] = ""

    def committed(self, item_id: str) -> None:
        self._track(item_id)

    def delta(self, item_id: str, delta: str) -> None:
        self._track(item_id)
        if item_id not in self.done:
            self.text[item_id] += delta

    def completed(self, item_id: str, transcript: str | None) -> None:
        self._track(item_id)
        if transcript is not None:
            self.text[item_id] = transcript
        self.done.add(item_id)

    def failed(self, item_id: str) -> None:
        self._track(item_id)
        self.done.add(item_id)

    def pending(self) -> bool:
        return any(item_id not in self.done for item_id in self.order)

    def joined(self, keep_case: Iterable[str]) -> str:
        return join_draft((self.text[i] for i in self.order), keep_case)


async def run_dictation_session(
    client: ClientTransport,
    draft_upstream: UpstreamTransport,
    full_upstream: UpstreamTransport,
    *,
    model: str,
    language: str | None = None,
    context: VoiceContext | None = None,
    noise_reduction: str | None = None,
    draft_silence_ms: int | None = None,
    draft_cut_seconds: tuple[float, float] | None = None,
    max_seconds: float | None = None,
    idle_timeout: float | None = None,
    finalize_timeout: float | None = None,
) -> SessionResult:
    """Bridge one dictation session to two upstream sessions fed the same
    audio. While the user talks the client gets ``draft`` frames (the VAD
    session's turns joined); after ``stop`` — or when the session hits its
    maximum duration — it gets one ``transcript`` of the whole recording from
    the session without turn detection, or the draft if that pass fails or
    takes longer than ``finalize_timeout``. Always closes both upstreams."""
    max_seconds = max_seconds if max_seconds is not None else float(settings.voice_stt_max_session_seconds)
    idle_timeout = idle_timeout if idle_timeout is not None else float(settings.voice_stt_idle_timeout_seconds)
    finalize_timeout = finalize_timeout if finalize_timeout is not None else settings.voice_stt_finalize_timeout_seconds
    draft_silence_ms = draft_silence_ms if draft_silence_ms is not None else settings.voice_stt_draft_silence_ms
    soft_cut, hard_cut = draft_cut_seconds or (
        settings.voice_stt_draft_cut_after_seconds,
        settings.voice_stt_draft_cut_max_seconds,
    )
    cutter = _DraftCutter(soft=soft_cut, hard=hard_cut)
    keep_case = context.keywords if context else ()

    started = time.monotonic()
    result = SessionResult()
    turns = _DraftTurns()
    speech_open = False
    last_draft = ""
    full_text: str | None = None
    full_done = asyncio.Event()

    async def send_draft() -> None:
        nonlocal last_draft
        text = turns.joined(keep_case)
        if text != last_draft:
            last_draft = text
            await client.send_text(VoiceDraft(text=text).model_dump_json())

    async def pump_client() -> str:
        while True:
            try:
                message = await asyncio.wait_for(client.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                return "idle_timeout"
            if message.get("type") == "websocket.disconnect":
                return "client_disconnect"
            payload: str | None = None
            pcm: bytes | None = None
            data = message.get("bytes")
            if data:
                pcm = data
                payload = audio_frame_to_append(data)
            else:
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
                if isinstance(frame, VoiceStop):
                    return "stopped"
                if isinstance(frame, VoiceAudioJson):
                    payload = json.dumps({"type": "input_audio_buffer.append", "audio": frame.data})
                    try:
                        pcm = base64.b64decode(frame.data)
                    except (ValueError, TypeError):
                        pcm = None
            if payload is not None:
                await draft_upstream.send(payload)
                # Cut only mid-speech: silence is VAD's to close, and a
                # committed silence can come back as invented words.
                if pcm and speech_open and cutter.should_cut(pcm):
                    cutter.reset()
                    await draft_upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                if not full_done.is_set():
                    try:
                        await full_upstream.send(payload)
                    except Exception:  # noqa: BLE001 — the draft stands in for it
                        full_done.set()

    async def pump_draft() -> str:
        nonlocal speech_open
        while True:
            try:
                raw = await draft_upstream.recv()
            except Exception:  # noqa: BLE001
                return "upstream_closed"
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            etype = event.get("type")
            item_id = str(event.get("item_id") or "")
            if etype == "input_audio_buffer.committed":
                turns.committed(item_id)
                cutter.reset()
            elif etype == "conversation.item.input_audio_transcription.delta":
                turns.delta(item_id, str(event.get("delta") or ""))
                await send_draft()
            elif etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                turns.completed(item_id, transcript if isinstance(transcript, str) else None)
                usage = _usage_from_event(event)
                if usage is not None:
                    result.turn_usages.append(usage)
                await send_draft()
            elif etype == "conversation.item.input_audio_transcription.failed":
                turns.failed(item_id)
            elif etype == "input_audio_buffer.speech_started":
                speech_open = True
                cutter.reset()
                await client.send_text(VoiceSpeechStarted().model_dump_json())
            elif etype == "input_audio_buffer.speech_stopped":
                speech_open = False
                await client.send_text(VoiceSpeechStopped().model_dump_json())
            elif etype == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else {}
                if error.get("code") == "input_audio_buffer_commit_empty":
                    speech_open = False
                else:
                    # The one-pass session still carries the recording.
                    logger.warning("voice: draft session error: %s", error.get("message"))

    async def pump_full() -> None:
        nonlocal full_text
        while True:
            try:
                raw = await full_upstream.recv()
            except Exception:  # noqa: BLE001
                full_done.set()
                return
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            etype = event.get("type")
            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                full_text = transcript if isinstance(transcript, str) else ""
                usage = _usage_from_event(event)
                if usage is not None:
                    result.turn_usages.append(usage)
                full_done.set()
            elif etype == "conversation.item.input_audio_transcription.failed":
                full_done.set()
            elif etype == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else {}
                if error.get("code") == "input_audio_buffer_commit_empty":
                    full_text = ""
                else:
                    logger.warning("voice: one-pass session error: %s", error.get("message"))
                full_done.set()

    async def flush_draft(deadline: float) -> None:
        """Let the draft catch up before it stands in for the transcript."""
        if speech_open:
            try:
                await draft_upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception:  # noqa: BLE001
                return
        while time.monotonic() < deadline and (speech_open or turns.pending()):
            if draft_task.done():
                return
            await asyncio.sleep(0.02)

    client_task: asyncio.Task[str] = asyncio.create_task(pump_client())
    draft_task: asyncio.Task[str] = asyncio.create_task(pump_draft())
    full_task: asyncio.Task[None] = asyncio.create_task(pump_full())
    deadline_task: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(max_seconds))
    tasks: set[asyncio.Task[Any]] = {client_task, draft_task, full_task, deadline_task}

    try:
        try:
            await draft_upstream.send(
                json.dumps(
                    build_session_update(
                        model,
                        language,
                        vad_silence_ms=draft_silence_ms,
                        context=context,
                        noise_reduction=noise_reduction,
                    )
                )
            )
            await full_upstream.send(
                json.dumps(
                    build_session_update(
                        model, language, vad=False, context=context, noise_reduction=noise_reduction
                    )
                )
            )
            await client.send_text(VoiceReady().model_dump_json())
        except Exception:  # noqa: BLE001
            result.reason = "error"
            await _send_best_effort(
                client,
                VoiceError(code="upstream_error", message="could not configure upstream session"),
            )
            return result

        done, _pending = await asyncio.wait(
            {client_task, draft_task, deadline_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if deadline_task in done:
            result.reason = "max_duration"
        elif client_task in done:
            result.reason = "upstream_closed" if client_task.exception() else client_task.result()
        else:
            result.reason = "upstream_closed"

        if result.reason in ("stopped", "max_duration"):
            deadline = time.monotonic() + finalize_timeout
            if not full_done.is_set():
                try:
                    await full_upstream.send(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception:  # noqa: BLE001
                    full_done.set()
            try:
                await asyncio.wait_for(full_done.wait(), timeout=max(0.0, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                pass
            text = (full_text or "").strip()
            fallback = False
            if not text:
                await flush_draft(deadline)
                text = turns.joined(keep_case)
                fallback = bool(text)
            result.finals += 1
            await _send_best_effort(client, VoiceTranscript(text=text, fallback=fallback))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
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
        for upstream in (draft_upstream, full_upstream):
            try:
                await upstream.close()
            except Exception:  # noqa: BLE001
                pass
    return result


async def _send_best_effort(client: ClientTransport, frame: ServerFrame) -> None:
    try:
        await client.send_text(frame.model_dump_json())
    except Exception:  # noqa: BLE001 — client may already be gone
        pass
