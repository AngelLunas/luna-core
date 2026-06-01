"""Semantic deduplication on top of EmbeddingService.

Domain-agnostic: callers supply the canonical text, collection, and any
structured filters. Host apps (e.g. luna-sentinel for Jobs) decide what
makes an entity "the same" by choosing what to embed and what metadata
to attach. There is no per-domain logic baked in here.

The contract: an embedding row is the dedup index. Its metadata MUST
include an `entity_id` pointing back at the domain row (Job, Profile,
Media, etc.) so a positive match returns something the caller can
update in its own table.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from luna_core.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 0.9


class DedupService:
    def __init__(self, embeddings: EmbeddingService):
        self._embeddings = embeddings

    async def find_match(
        self,
        text: str,
        collection: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return the closest hit if similarity >= threshold, else None.

        The returned dict is the same shape as EmbeddingService.search hits:
        id, text, metadata, distance, similarity.
        """
        hits = await self._embeddings.search(
            query=text, collection=collection, k=1, filter=filter
        )
        if not hits:
            return None
        top = hits[0]
        similarity = top.get("similarity")
        if similarity is None or similarity < threshold:
            return None
        return top

    async def upsert_unique(
        self,
        text: str,
        collection: str,
        *,
        entity_id: uuid.UUID,
        metadata: dict[str, Any],
        threshold: float = DEFAULT_THRESHOLD,
        filter: dict[str, Any] | None = None,
    ) -> tuple[uuid.UUID, bool]:
        """Claim an entity_id for `text` in `collection`.

        Returns (entity_id, is_new).
          - If a semantic match exists, returns (existing_entity_id, False)
            and does NOT insert. The caller is expected to refresh that
            existing row in its own domain table.
          - Otherwise inserts an embedding tagged with the caller's
            entity_id and returns (entity_id, True).

        `metadata['entity_id']` is set/overridden automatically — callers
        pass it via `entity_id`, not inside `metadata`.
        """
        match = await self.find_match(
            text=text, collection=collection, threshold=threshold, filter=filter
        )
        if match is not None:
            existing_raw = (match.get("metadata") or {}).get("entity_id")
            if existing_raw:
                try:
                    return uuid.UUID(str(existing_raw)), False
                except ValueError:
                    logger.warning(
                        "dedup match has non-uuid entity_id=%r in collection %s; "
                        "treating as new",
                        existing_raw,
                        collection,
                    )

        await self._embeddings.upsert(
            text=text,
            collection=collection,
            metadata={**metadata, "entity_id": str(entity_id)},
        )
        return entity_id, True


__all__ = ["DEFAULT_THRESHOLD", "DedupService"]
