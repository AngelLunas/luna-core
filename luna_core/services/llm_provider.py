"""CRUD + upstream model-listing for dynamically-configured LLM providers.

The API key is persisted Fernet-encrypted (wrapped as JSON) and never leaves
the service layer — `get_decrypted_api_key` is for internal callers (router,
upstream model-listing) only. The schema layer exposes `has_api_key: bool`
to clients so they can render "key set / not set" without ever receiving
the secret.
"""
from __future__ import annotations

import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.crypto import decrypt_json, encrypt_json
from luna_core.models.agent import Agent
from luna_core.models.llm_provider import LLMProvider
from luna_core.schemas.llm_provider import (
    LLMProviderCreate,
    LLMProviderModel,
    LLMProviderUpdate,
)


class LLMProviderNotFound(LookupError):
    pass


class DuplicateLLMProvider(ValueError):
    pass


class LLMProviderInUse(RuntimeError):
    """Raised when deleting a provider that's still referenced by agents."""


class LLMProviderUpstreamError(RuntimeError):
    """Raised when the upstream model-listing endpoint refuses or errors."""


def _wrap_api_key(value: str | None) -> str | None:
    """Encrypt the api_key into the on-disk JSON envelope.

    None → leave column null. Empty string → also clear (treat as 'remove
    the key'). Any other string → store {"api_key": value} encrypted.
    """
    if value is None or value == "":
        return None
    return encrypt_json({"api_key": value})


async def create_llm_provider(
    db: AsyncSession, payload: LLMProviderCreate
) -> LLMProvider:
    provider = LLMProvider(
        name=payload.name,
        base_url=payload.base_url,
        chat_url=payload.chat_url,
        models_url=payload.models_url,
        api_key_encrypted=_wrap_api_key(payload.api_key),
        is_active=payload.is_active,
    )
    db.add(provider)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateLLMProvider(payload.name) from exc
    await db.refresh(provider)
    return provider


async def list_llm_providers(db: AsyncSession) -> list[LLMProvider]:
    result = await db.execute(
        select(LLMProvider).order_by(LLMProvider.created_at.desc())
    )
    return list(result.scalars().all())


async def get_llm_provider(
    db: AsyncSession, provider_id: uuid.UUID
) -> LLMProvider:
    provider = await db.get(LLMProvider, provider_id)
    if provider is None:
        raise LLMProviderNotFound(str(provider_id))
    return provider


async def update_llm_provider(
    db: AsyncSession,
    provider_id: uuid.UUID,
    payload: LLMProviderUpdate,
) -> LLMProvider:
    provider = await get_llm_provider(db, provider_id)
    data = payload.model_dump(exclude_unset=True)
    if "api_key" in data:
        provider.api_key_encrypted = _wrap_api_key(data.pop("api_key"))
    for field, value in data.items():
        setattr(provider, field, value)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateLLMProvider(payload.name or provider.name) from exc
    await db.refresh(provider)
    return provider


async def delete_llm_provider(
    db: AsyncSession, provider_id: uuid.UUID
) -> None:
    provider = await get_llm_provider(db, provider_id)
    # Refuse if any agent still points to this provider; the FK is
    # ON DELETE RESTRICT so the DB would also reject, but a pre-check
    # gives the API a clean 409 instead of a 500.
    in_use = await db.execute(
        select(Agent.id).where(Agent.llm_provider_id == provider_id).limit(1)
    )
    if in_use.scalar_one_or_none() is not None:
        raise LLMProviderInUse(str(provider_id))
    await db.delete(provider)
    await db.commit()


def get_decrypted_api_key(provider: LLMProvider) -> str | None:
    payload = decrypt_json(provider.api_key_encrypted)
    if payload is None:
        return None
    value = payload.get("api_key")
    return value if isinstance(value, str) and value else None


async def list_upstream_models(
    provider: LLMProvider,
    *,
    timeout_seconds: float = 10.0,
) -> list[LLMProviderModel]:
    """Hit the upstream `/models` endpoint and return the available models.

    Uses `models_url` when set (full URL override), otherwise
    `<base_url>/models`. Sends `Authorization: Bearer <api_key>` when a key
    is configured. The response is expected to follow the OpenAI shape:
    `{"data": [{"id": "...", "owned_by": "..."}, ...]}` — that's the
    contract every OpenAI-compatible host implements.
    """
    url = provider.models_url or _join_url(provider.base_url, "models")
    headers: dict[str, str] = {}
    api_key = get_decrypted_api_key(provider)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise LLMProviderUpstreamError(str(exc)) from exc

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise LLMProviderUpstreamError(
            "upstream response did not include a 'data' array"
        )

    models: list[LLMProviderModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        owned_by = item.get("owned_by")
        models.append(
            LLMProviderModel(
                id=model_id,
                owned_by=owned_by if isinstance(owned_by, str) else None,
            )
        )
    return models


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"
