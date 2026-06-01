"""Embedding service — encodes text via the configured LLM provider and stores
results in the pgvector `embeddings` table.

Generic by design: callers pick the collection name. luna-sentinel (or any
other host app) decides what those collections mean — there is no
domain-specific tagging baked in here.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.llm.router import LLMRouter
from luna_core.models.embedding import EMBEDDING_DIMENSIONS, Embedding

logger = logging.getLogger(__name__)


class EmbeddingDimensionMismatch(ValueError):
    pass


class EmbeddingService:
    def __init__(self, llm_router: LLMRouter, db: AsyncSession):
        self._llm = llm_router
        self._db = db

    async def embed(self, text: str) -> list[float]:
        vector = await self._llm.embed(text)
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingDimensionMismatch(
                f"embedding provider returned {len(vector)} dims, "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        return vector

    async def upsert(
        self,
        text: str,
        collection: str,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        vector = await self.embed(text)
        row = Embedding(
            collection=collection,
            text=text,
            vector=vector,
            embedding_metadata=metadata or {},
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        return row.id

    async def search(
        self,
        query: str,
        collection: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vector = await self.embed(query)
        stmt = (
            select(
                Embedding.id,
                Embedding.text,
                Embedding.embedding_metadata,
                Embedding.vector.cosine_distance(vector).label("distance"),
            )
            .where(Embedding.collection == collection)
            .order_by(text("distance ASC"))
            .limit(k)
        )
        if filter:
            for key, value in filter.items():
                stmt = stmt.where(
                    Embedding.embedding_metadata[key].astext == str(value)
                )

        result = await self._db.execute(stmt)
        out: list[dict[str, Any]] = []
        for row in result.all():
            distance = float(row.distance) if row.distance is not None else None
            similarity = 1.0 - distance if distance is not None else None
            out.append(
                {
                    "id": row.id,
                    "text": row.text,
                    "metadata": row.embedding_metadata,
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return out


__all__ = ["EmbeddingDimensionMismatch", "EmbeddingService"]
