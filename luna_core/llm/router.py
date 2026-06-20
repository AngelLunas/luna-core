"""Per-provider routing, rate limiting, and retries.

Providers are stored in the database (`core.llm_providers`) — the router
opens a short-lived session per call to resolve the row, caches the built
`GenericProvider` keyed by `(provider_id, updated_at)`, and rebuilds on
the first call after any provider edit. Embeddings live on a dedicated
env-configured provider since most installs use a single embedding model.

Aborts are NEVER retried — they propagate as-is.
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.rate_limit import check_rate_limit
from luna_core.llm.base import (
    AbortSignalError,
    BaseLLMProvider,
    LLMRateLimitError,
    ToolDefinition,
)
from luna_core.llm.providers.generic import GenericProvider
from luna_core.models.llm_provider import LLMProvider
from luna_core.services.llm_provider import get_decrypted_api_key

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory

logger = logging.getLogger(__name__)


SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _normalize_chat_base_url(url: str) -> str:
    """Coerce a user-supplied chat URL into the prefix the OpenAI SDK expects.

    The SDK takes a `base_url` and appends `/chat/completions` itself, so if
    the user pasted the full chat-completions URL we strip that trailing
    segment. Anything else is returned untouched.
    """
    stripped = url.rstrip("/")
    suffix = "/chat/completions"
    if stripped.endswith(suffix):
        return stripped[: -len(suffix)]
    return url


class LLMRouter:
    """Resolves dynamic LLM providers, applies rate limiting, retries on 429."""

    def __init__(
        self,
        *,
        redis: Redis,
        session_factory: SessionFactory,
        embedding_provider: BaseLLMProvider | None = None,
        rate_limit_rpm: int | None = None,
        max_retries: int | None = None,
    ):
        self._redis = redis
        self._session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._rpm = rate_limit_rpm or settings.llm_rate_limit_rpm
        self._max_retries = (
            max_retries if max_retries is not None else settings.llm_max_retries
        )
        # Cache of built chat providers keyed by provider_id. Each entry
        # carries the row's `updated_at` so the first call after any edit
        # detects the staleness and rebuilds with the new credentials/URLs.
        self._chat_cache: dict[uuid.UUID, tuple[datetime, BaseLLMProvider]] = {}

    async def resolve_chat_provider(
        self, provider_id: uuid.UUID
    ) -> BaseLLMProvider:
        async with self._session_factory() as db:
            row = await db.get(LLMProvider, provider_id)
            if row is None:
                raise KeyError(f"no llm provider {provider_id}")
            if not row.is_active:
                raise KeyError(f"llm provider {provider_id} is inactive")

            cached = self._chat_cache.get(provider_id)
            if cached is not None and cached[0] == row.updated_at:
                return cached[1]

            provider = self._build_provider(row)
            self._chat_cache[provider_id] = (row.updated_at, provider)
            return provider

    def _build_provider(self, row: LLMProvider) -> BaseLLMProvider:
        api_key = get_decrypted_api_key(row)
        base_url_for_sdk = _normalize_chat_base_url(row.chat_url or row.base_url)
        # Embeddings on chat providers are unused — `embed()` always goes
        # through `self._embedding_provider` — but GenericProvider always
        # builds an embedding client, so we feed it the env defaults to
        # keep the constructor happy.
        return GenericProvider(
            api_key=api_key or "missing",
            base_url=base_url_for_sdk,
        )

    async def complete(
        self,
        *,
        provider_id: uuid.UUID,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        temperature: float,
        model: str,
        output_schema: dict[str, Any] | None,
        run_id: uuid.UUID,
        node_id: str,
        make_io: IOFactory | None = None,
    ) -> list[dict[str, Any]]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._max_retries:
            impl = await self.resolve_chat_provider(provider_id)
            await self._enforce_rate_limit(provider_id, run_id)
            try:
                return await impl.complete(
                    messages=messages,
                    system=system,
                    tools=tools,
                    temperature=temperature,
                    model=model,
                    output_schema=output_schema,
                    run_id=run_id,
                    node_id=node_id,
                    redis=self._redis,
                    make_io=make_io,
                )
            except AbortSignalError:
                raise
            except LLMRateLimitError as exc:
                last_exc = exc
                attempt += 1
                if attempt > self._max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "rate-limited by provider %s (attempt %d/%d); sleeping %.2fs",
                    provider_id,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def embed(self, text: str) -> list[float]:
        if self._embedding_provider is None:
            raise RuntimeError(
                "LLMRouter has no embedding provider configured; "
                "pass `embedding_provider=` when constructing the router."
            )
        return await self._embedding_provider.embed(text)

    # --------------------------------------------------------------- internals
    async def _enforce_rate_limit(
        self, provider_id: uuid.UUID, run_id: uuid.UUID
    ) -> None:
        key = f"llm_ratelimit:{provider_id}"
        result = await check_rate_limit(
            self._redis,
            key,
            limit=self._rpm,
            window_seconds=settings.llm_rate_limit_window_seconds,
        )
        if result.allowed:
            return
        wait_for = max(
            1, min(result.retry_after, settings.llm_rate_limit_window_seconds)
        )
        logger.info(
            "LLM rate limit hit for provider %s (run %s); waiting %ds",
            provider_id,
            run_id,
            wait_for,
        )
        await asyncio.sleep(wait_for)

    def _backoff_delay(self, attempt: int) -> float:
        base = settings.llm_retry_base_delay_seconds
        return base * (2 ** (attempt - 1)) + random.uniform(0, base)


__all__ = ["LLMRouter"]
