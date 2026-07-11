"""Generic realtime voice-transcription WebSocket.

``WS /voice/transcribe?token=<jwt>`` — authenticate, resolve the STT
credential from the LLM-provider registry, then hand the socket to the
bidirectional proxy in ``services.voice_transcription``. Not tied to any
conversation: transcribing voice is useful for any input surface of any
host app.

DB sessions here are deliberately short-lived (credential lookup before the
proxy, usage flush after) so a minutes-long voice session never pins a
pooled connection.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, status

from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal
from luna_core.core.security import user_id_from_ws_token
from luna_core.schemas.voice import (
    InvalidVoiceFrame,
    VoiceError,
    VoiceStart,
    parse_client_frame,
)
from luna_core.services.usage import record_usage_counts
from luna_core.services.voice_transcription import (
    SttUnavailable,
    open_upstream,
    resolve_stt_session,
    run_transcription_session,
    total_usage,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.websocket("/transcribe")
async def transcribe(websocket: WebSocket, token: str | None = None) -> None:
    # Authenticate before accepting — an invalid token rejects the handshake.
    user_id = user_id_from_ws_token(token)
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    async with AsyncSessionLocal() as db:
        try:
            config = await resolve_stt_session(db)
        except SttUnavailable as exc:
            logger.info("voice: unavailable for user %s: %s", user_id, exc)
            await _send_error_and_close(
                websocket, "stt_unavailable", "voice transcription is not configured"
            )
            return

    # First frame must be `start` (it carries the client's language hint).
    try:
        message = await asyncio.wait_for(
            websocket.receive(), timeout=settings.voice_stt_start_timeout_seconds
        )
    except asyncio.TimeoutError:
        await _send_error_and_close(
            websocket, "start_timeout", "no start frame received"
        )
        return
    if message.get("type") == "websocket.disconnect":
        return
    try:
        frame = parse_client_frame(message.get("text") or "")
    except InvalidVoiceFrame:
        frame = None
    if not isinstance(frame, VoiceStart):
        await _send_error_and_close(websocket, "bad_frame", "expected a start frame")
        return

    try:
        upstream = await open_upstream(config)
    except Exception:  # noqa: BLE001
        logger.exception("voice: upstream connect failed")
        await _send_error_and_close(
            websocket, "upstream_error", "could not reach the transcription service"
        )
        return

    result = await run_transcription_session(
        websocket, upstream, model=config.model, language=frame.language
    )

    # Flush the session's per-turn usage to the generic ledger. Voice has no
    # run/conversation, so the scope IS the authenticated user.
    if result.turn_usages:
        try:
            async with AsyncSessionLocal() as db:
                for usage in result.turn_usages:
                    await record_usage_counts(
                        db,
                        scope_id=user_id,
                        message_id=None,
                        model=config.model,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                        audio_input_tokens=usage.audio_input_tokens,
                    )
                await db.commit()
        except Exception:  # noqa: BLE001 — never let bookkeeping kill the socket path
            logger.exception("voice: failed to record usage for user %s", user_id)

    total = total_usage(result.turn_usages)
    logger.info(
        "voice session closed user=%s reason=%s duration=%.1fs turns=%d total_tokens=%d",
        user_id,
        result.reason,
        result.duration_seconds,
        result.finals,
        total.total_tokens if total else 0,
    )


async def _send_error_and_close(
    websocket: WebSocket,
    code: str,
    message: str,
) -> None:
    try:
        await websocket.send_text(
            VoiceError(code=code, message=message).model_dump_json()  # type: ignore[arg-type]
        )
        await websocket.close(code=1000)
    except Exception:  # noqa: BLE001 — client may already be gone
        pass
