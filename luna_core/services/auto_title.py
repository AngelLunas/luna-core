"""Auto-naming of a conversation, off the realtime critical path.

The first turn streams the agent's reply live over the conversation WebSocket;
the *title* is a nicety that must never delay that. So the conversation router
spawns this as a fire-and-forget task AFTER a turn completes (see
``routers/conversations._maybe_spawn_auto_title``):

  1. bail unless the conversation still lacks a title — idempotent, so one title
     per conversation, effectively generated only on the first turn,
  2. ask the caller-supplied ``(provider_id, model)`` for a 3-6 word label in a
     single non-streaming completion — a few dozen tokens, fractions of a cent,
  3. persist it and publish a ``conversation_titled`` event on the conversation
     channel so an already-open chat updates its header + list live, no refetch.

Deliberately generic: it takes a ``provider_id`` + ``model`` rather than any
agent or host concept, so every host reuses it. The host decides *which* model
titles (savia passes the answering agent's own small model). Any failure falls
back to the first user message truncated, and the whole thing is wrapped so a
titling error can never surface to the user or break a turn.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from redis.asyncio import Redis

from luna_core.core.db import AsyncSessionLocal
from luna_core.engine.emitter import max_seq_key, publish_run_event
from luna_core.models.event import RunEventType
from luna_core.services.conversation import (
    ConversationNotFound,
    get_conversation,
    list_messages,
    update_conversation_title,
)

logger = logging.getLogger(__name__)

# Keep titles short and the completion cheap.
_MAX_TITLE_CHARS = 60
_FALLBACK_CHARS = 48

_TITLE_SYSTEM = (
    "You name chat conversations. Given the user's first message, reply with a "
    "concise title of 3 to 6 words that captures the topic. Use the user's own "
    "language. No quotes, no trailing punctuation, no preamble — only the title."
)


def _first_user_text(messages: list) -> str | None:
    """First user turn's plain text, joined across its text blocks (images and
    other block types are ignored). ``None`` if there is no user text yet."""
    for m in messages:
        if m.role.value != "user":
            continue
        parts = [
            b.get("text", "")
            for b in (m.content or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = " ".join(p for p in parts if p).strip()
        if text:
            return text
    return None


def _clean_title(raw: str | None, fallback: str) -> str:
    """Normalize the model's output into a stored title, or fall back to the
    truncated first message when the model gave us nothing usable."""
    text = (raw or "").strip().strip("\"'").strip()
    # Collapse a stray multi-line answer to its first line.
    text = text.splitlines()[0].strip() if text else ""
    if not text:
        text = fallback.strip()
    return text[:_MAX_TITLE_CHARS].strip()


async def _generate_title(
    llm_router: Any, provider_id: uuid.UUID, model: str, first_user: str
) -> str | None:
    """One non-streaming completion on the given provider/model. Returns the
    model's text, or ``None`` on any failure (caller falls back)."""
    try:
        blocks = await llm_router.complete(
            provider_id=provider_id,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": first_user}]}
            ],
            system=_TITLE_SYSTEM,
            tools=[],
            temperature=0.0,
            model=model,
            output_schema=None,
            run_id=uuid.uuid4(),  # fresh scope: no abort/rate-limit key clash with the turn
            node_id="title",
            make_io=None,  # non-streaming — nothing to fan out
        )
    except Exception:  # noqa: BLE001 — titling must never break anything
        logger.exception("auto-title completion failed")
        return None
    text = " ".join(
        b.get("text", "")
        for b in (blocks or [])
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip()
    return text or None


async def _publish_titled(
    redis: Redis, conversation_id: uuid.UUID, title: str
) -> None:
    """Fan the new title out on the conversation channel (pub/sub-only). Its
    sequence sits one above the turn's high-water mark so it sorts after the
    reply's deltas in the client's single ordered timeline."""
    raw = await redis.get(max_seq_key(conversation_id))
    try:
        current = int(raw.decode() if isinstance(raw, bytes) else raw) if raw else 0
    except (TypeError, ValueError):
        current = 0
    await publish_run_event(
        redis,
        conversation_id,
        RunEventType.conversation_titled,
        "chat",
        {"conversation_id": str(conversation_id), "title": title},
        current + 1,
    )


async def maybe_title_conversation(
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    llm_router: Any,
    redis: Redis,
    provider_id: uuid.UUID,
    model: str,
) -> None:
    """Generate + persist + broadcast a title for a freshly-started conversation.

    Idempotent (no-op once a title exists) and fully defensive — any error is
    logged and swallowed. Runs on its OWN session because the request session
    that spawned it is already closing."""
    try:
        async with AsyncSessionLocal() as db:
            try:
                convo = await get_conversation(db, conversation_id, user_id=user_id)
            except ConversationNotFound:
                return
            if convo.title:  # already named (or a later turn already titled it)
                return
            messages = await list_messages(db, conversation_id, user_id=user_id)
            first_user = _first_user_text(messages)
            if not first_user:
                return
            raw = await _generate_title(llm_router, provider_id, model, first_user)
            title = _clean_title(raw, fallback=first_user[:_FALLBACK_CHARS])
            await update_conversation_title(
                db, conversation_id, title=title, user_id=user_id
            )
        await _publish_titled(redis, conversation_id, title)
    except Exception:  # noqa: BLE001 — a titling failure must never surface
        logger.exception("auto-title failed for conversation %s", conversation_id)


__all__ = ["maybe_title_conversation"]
