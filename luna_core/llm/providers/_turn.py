"""Plumbing every streaming chat provider shares, whatever its wire format.

One assistant turn, as the rest of the engine sees it: an
``agent_message_started`` event, live text/thinking deltas on the run channel
(also cached in Redis so a reconnecting client can rebuild them), a final
persisted message, and a partial one when the turn is aborted or fails. This
module owns that lifecycle so a provider only translates its own wire format.

It also holds the attached-media conventions every provider must share — the
``img-N`` / ``vid-N`` labels, numbered the same way a host numbers its stored
media so tools can resolve a label back to the row — and the adapters for
Anthropic-shaped payloads (image blocks, usage) used by more than one
provider.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal
from luna_core.llm.base import inflight_meta_key, stream_key
from luna_core.models.event import AgentMessageRole, RunEventType

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory

logger = logging.getLogger(__name__)

# Resolves a media_id to a URL the provider can put on an image content part —
# in practice a ``data:`` base64 URL (works for both local-dev storage and
# object storage without exposing a public/signed URL). Host-injected (the
# host owns its media table + storage); ``None`` means "could not resolve", in
# which case the image falls back to a text note.
ImageResolver = Callable[[str], Awaitable[str | None]]

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

# Attached media the model is told about, by canonical block type: the label
# prefix, the note's wording, and what "shown" means for it. A video is never
# played — its poster (first frame) is what a vision model sees, and the note
# says so, so the model does not claim to have watched it.
_LABELED_KINDS: dict[str, tuple[str, str, str]] = {
    "image": ("img", "image attached", "shown below"),
    "video": ("vid", "video attached", "first frame shown below"),
}


class _MediaLabels:
    """``img-N`` / ``vid-N`` counters: independent per kind, conversation-wide,
    in order of appearance — the same order a host derives from its stored
    messages, so a tool can resolve a label back to the media row."""

    def __init__(self) -> None:
        self._seq = {kind: 0 for kind in _LABELED_KINDS}

    def next(self, kind: str) -> str:
        self._seq[kind] += 1
        return f"{_LABELED_KINDS[kind][0]}-{self._seq[kind]}"


def _media_note(kind: str, label: str, shown: bool) -> str:
    _prefix, what, shown_text = _LABELED_KINDS[kind]
    return f"[{what}: {label} ({shown_text})]" if shown else f"[{what}: {label}]"


def _data_url_to_image_block(url: str) -> dict[str, Any] | None:
    """``data:<media_type>;base64,<data>`` → an Anthropic image content block.
    Non-data URLs are skipped (nothing here can fetch them)."""
    if not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep or ";base64" not in header:
        return None
    media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _anthropic_usage_shim(usage: dict[str, Any]) -> Any:
    """Adapt Anthropic-shaped usage (input / cache read / cache write / output
    tokens) to the OpenAI-attribute shape ``record_usage`` reads. Anthropic
    counts cached tokens apart from ``input_tokens``; the prompt total is
    their sum."""
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_creation
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
    )


class StreamingTurnProvider:
    """Base for chat providers: the turn lifecycle, none of the wire format.

    ``session_factory`` opens a short-lived session for persisting the turn
    mid-stream. Defaults to luna-core's AsyncSessionLocal so hosts that share
    that engine don't have to wire anything; hosts with custom engines pass
    their own factory.
    """

    def __init__(self, *, session_factory: SessionFactory | None = None) -> None:
        self._session_factory = session_factory or AsyncSessionLocal

    async def _resolve_image_urls(
        self,
        messages: list[dict[str, Any]],
        image_resolver: ImageResolver | None,
    ) -> dict[str, str]:
        """Resolve every distinct attached media ``media_id`` (images AND
        videos — for a video the host answers with its poster frame) to a
        renderable URL via the injected resolver. No resolver → empty map
        (text-note path). Each id is resolved once even if it recurs across
        turns."""
        if image_resolver is None:
            return {}
        urls: dict[str, str] = {}
        for msg in messages:
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") not in _LABELED_KINDS:
                    continue
                media_id = block.get("media_id")
                if media_id is None:
                    continue
                key = str(media_id)
                if key in urls:
                    continue
                url = await image_resolver(key)
                if url:
                    urls[key] = url
        return urls

    async def _push_stream(
        self,
        redis: Redis,
        s_key: str,
        kind: str,
        text: str,
    ) -> None:
        # Append to a Redis LIST cache used both for crash-mid-stream recovery
        # and for the WebSocket snapshot path: when a client reconnects while
        # a turn is still streaming, the snapshot reader rehydrates a
        # synthetic delta from these chunks. The canonical broadcast for live
        # clients is still the pub/sub event emitted alongside each push.
        chunk = json.dumps({"kind": kind, "text": text})
        await redis.rpush(s_key, chunk)
        await redis.expire(s_key, settings.run_stream_key_ttl_seconds)

    async def _write_inflight_meta(
        self,
        redis: Redis,
        run_id: uuid.UUID,
        node_id: str,
        message_id: uuid.UUID,
        started_seq: int,
    ) -> None:
        # Capture the iteration tag of the *task that owns this turn* so
        # the WebSocket snapshot path can route mid-flight synthesized
        # delta frames to the right iteration block on the dashboard.
        # ``get_current_iteration_id`` returns None outside an iteration
        # scope, and we omit the key in that case so the wire shape for
        # non-iterative runs stays unchanged.
        from luna_core.engine.iteration_context import get_current_iteration_id

        # The key is per-message_id (not per-node) so parallel iterations
        # of the same ai_agent node each write their own meta — see
        # docstring on ``stream_key``/``inflight_meta_key``. We carry
        # ``node_id`` in the payload because the snapshot scanner no
        # longer parses it out of the key.
        meta: dict[str, Any] = {
            "message_id": str(message_id),
            "node_id": node_id,
            "started_seq": started_seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        iteration_id = get_current_iteration_id()
        if iteration_id is not None:
            meta["iteration_id"] = str(iteration_id)
        await redis.set(
            inflight_meta_key(run_id, message_id),
            json.dumps(meta),
            ex=settings.run_stream_key_ttl_seconds,
        )

    async def _emit(
        self,
        build_io: IOFactory,
        run_id: uuid.UUID,
        event_type: RunEventType,
        node_id: str,
        payload: dict[str, Any],
    ) -> int | None:
        """Persist + broadcast an event. Returns the assigned sequence so
        the caller can base downstream synthetic-sequence math on it
        (e.g. transient delta events that ride above the same baseline).

        ``run_id`` is retained only for the failure log; persistence goes
        through the injected ``build_io`` factory."""
        # Opens a short-lived session per event so the streaming loop never
        # holds a transaction open while awaiting the next LLM chunk.
        async with self._session_factory() as db:
            emitter = build_io(db)
            try:
                event = await emitter.emit(
                    event_type, node_id=node_id, payload=payload
                )
                return event.sequence
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to persist run event %s for run %s node %s",
                    event_type.value,
                    run_id,
                    node_id,
                )
                return None

    async def _next_delta_sequence(
        self, redis: Redis, key: str, base: int
    ) -> int:
        """Atomically allocate the next per-stream delta sequence.

        Uses INCR on a Redis key keyed by (run_id, message_id) so the
        counter survives any in-stream coordination hiccups and stays
        consistent if the same stream were ever driven by multiple
        coroutines. The returned sequence is ``base + INCR_value`` so
        deltas always sort right after the persisted agent_message_started
        event (whose sequence is ``base``). The key inherits the same TTL
        as the stream cache and is deleted at end-of-stream.
        """
        offset = await redis.incr(key)
        if offset == 1:
            await redis.expire(key, settings.run_stream_key_ttl_seconds)
        return base + int(offset)

    async def _save_partial_blocks(
        self,
        text_parts: list[str],
        thinking_parts: list[str],
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        message_id: uuid.UUID,
        build_io: IOFactory,
    ) -> None:
        """Persist whatever streamed before an abort/error as a partial turn,
        then drop the turn's stream cache. All three keys are per-message:
        deleting them only affects this turn, never a sibling iteration's
        in-flight history."""
        s_key = stream_key(run_id, message_id)
        d_key = f"delta_seq:{run_id}:{message_id}"
        m_key = inflight_meta_key(run_id, message_id)
        thinking = "".join(thinking_parts).strip() or None
        blocks: list[dict[str, Any]] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        text = "".join(text_parts)
        if text:
            blocks.append({"type": "text", "text": text})
        if not blocks:
            await redis.delete(s_key, d_key, m_key)
            return
        try:
            async with self._session_factory() as db:
                emitter = build_io(db)
                await emitter.save_message(
                    node_id=node_id,
                    role=AgentMessageRole.assistant,
                    content=blocks,
                    is_partial=True,
                    thinking=thinking,
                    message_id=message_id,
                )
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist partial turn for run %s node %s",
                run_id,
                node_id,
            )
        await redis.delete(s_key, d_key, m_key)


__all__ = ["ImageResolver", "SessionFactory", "StreamingTurnProvider"]
