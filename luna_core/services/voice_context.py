"""Vocabulary hints for voice transcription, supplied by the host app.

luna-core knows nothing about a product's domain, so the words its users are
likely to say (product names, units, jargon) come from the host: it registers
one provider at startup, and every voice session asks it for the context of
the user who is talking. The provider receives the user id so a host can mix
per-user terms (their products, their plants) into the shared vocabulary.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceContext:
    """``prompt`` describes what the audio is about; ``keywords`` are literal
    terms to spell the host's way."""

    prompt: str | None = None
    keywords: tuple[str, ...] = ()


VoiceContextProvider = Callable[[uuid.UUID, str | None], Awaitable[VoiceContext]]

_provider: VoiceContextProvider | None = None


def set_voice_context_provider(provider: VoiceContextProvider | None) -> None:
    global _provider
    _provider = provider


async def resolve_voice_context(user_id: uuid.UUID, language: str | None) -> VoiceContext:
    """The host's context for this user, or an empty one. A failing provider
    never blocks dictation: transcribing without hints beats not transcribing."""
    if _provider is None:
        return VoiceContext()
    try:
        return await _provider(user_id, language)
    except Exception:  # noqa: BLE001
        logger.exception("voice: context provider failed for user %s", user_id)
        return VoiceContext()
