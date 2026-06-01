"""Retrieval-augmented generation helpers.

Domain-agnostic: the caller picks queries, collections, and the number of
results. RAGService returns either raw hits (for custom formatting) or a
pre-formatted context string ready to inject into a system prompt.
"""
from __future__ import annotations

import logging
from typing import Any

from luna_core.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, embedding_service: EmbeddingService):
        self._embeddings = embedding_service

    async def retrieve(
        self,
        query: str,
        collection: str,
        k: int = 3,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._embeddings.search(
            query=query, collection=collection, k=k, filter=filter
        )

    async def build_context(
        self,
        queries: list[str],
        collections: list[str],
        k_per_query: int = 3,
    ) -> str:
        seen: set[str] = set()
        blocks: list[str] = []
        for query in queries:
            for collection in collections:
                hits = await self.retrieve(
                    query=query, collection=collection, k=k_per_query
                )
                for hit in hits:
                    text = hit.get("text") or ""
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    blocks.append(_format_hit(collection, hit))
        return "\n\n".join(blocks)


def _format_hit(collection: str, hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    similarity = hit.get("similarity")
    header_parts = [f"[{collection}]"]
    if similarity is not None:
        header_parts.append(f"similarity={similarity:.3f}")
    if metadata:
        header_parts.append(f"metadata={metadata}")
    header = " ".join(header_parts)
    return f"{header}\n{hit['text']}"


__all__ = ["RAGService"]
