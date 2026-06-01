from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Response, status

from luna_core.core.dependencies import DBSession, require_permission
from luna_core.models.llm_provider import LLMProvider
from luna_core.schemas.llm_provider import (
    LLMProviderCreate,
    LLMProviderModelsResponse,
    LLMProviderRead,
    LLMProviderUpdate,
)
from luna_core.services.llm_provider import (
    DuplicateLLMProvider,
    LLMProviderInUse,
    LLMProviderNotFound,
    LLMProviderUpstreamError,
    create_llm_provider,
    delete_llm_provider,
    get_llm_provider,
    list_llm_providers,
    list_upstream_models,
    update_llm_provider,
)

router = APIRouter(prefix="/llm-providers", tags=["llm-providers"])


def _to_read(provider: LLMProvider) -> LLMProviderRead:
    return LLMProviderRead(
        id=provider.id,
        name=provider.name,
        base_url=provider.base_url,
        chat_url=provider.chat_url,
        models_url=provider.models_url,
        has_api_key=provider.api_key_encrypted is not None,
        is_active=provider.is_active,
        created_at=provider.created_at,
        updated_at=provider.updated_at,
    )


@router.post(
    "",
    response_model=LLMProviderRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_permission("llm_providers:create")],
)
async def create(
    payload: LLMProviderCreate, db: DBSession
) -> LLMProviderRead:
    try:
        provider = await create_llm_provider(db, payload)
    except DuplicateLLMProvider as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"llm provider with name '{exc}' already exists",
        ) from exc
    return _to_read(provider)


@router.get(
    "",
    response_model=list[LLMProviderRead],
    dependencies=[require_permission("llm_providers:read")],
)
async def index(db: DBSession) -> list[LLMProviderRead]:
    providers = await list_llm_providers(db)
    return [_to_read(p) for p in providers]


@router.get(
    "/{provider_id}",
    response_model=LLMProviderRead,
    dependencies=[require_permission("llm_providers:read")],
)
async def detail(
    provider_id: uuid.UUID, db: DBSession
) -> LLMProviderRead:
    try:
        provider = await get_llm_provider(db, provider_id)
    except LLMProviderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="llm provider not found",
        ) from exc
    return _to_read(provider)


@router.put(
    "/{provider_id}",
    response_model=LLMProviderRead,
    dependencies=[require_permission("llm_providers:update")],
)
async def update(
    provider_id: uuid.UUID,
    payload: LLMProviderUpdate,
    db: DBSession,
) -> LLMProviderRead:
    try:
        provider = await update_llm_provider(db, provider_id, payload)
    except LLMProviderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="llm provider not found",
        ) from exc
    except DuplicateLLMProvider as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"llm provider with name '{exc}' already exists",
        ) from exc
    return _to_read(provider)


@router.delete(
    "/{provider_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("llm_providers:delete")],
)
async def destroy(provider_id: uuid.UUID, db: DBSession) -> Response:
    try:
        await delete_llm_provider(db, provider_id)
    except LLMProviderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="llm provider not found",
        ) from exc
    except LLMProviderInUse as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="llm provider is still referenced by one or more agents",
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{provider_id}/models",
    response_model=LLMProviderModelsResponse,
    dependencies=[require_permission("llm_providers:read")],
)
async def models(
    provider_id: uuid.UUID, db: DBSession
) -> LLMProviderModelsResponse:
    try:
        provider = await get_llm_provider(db, provider_id)
    except LLMProviderNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="llm provider not found",
        ) from exc
    try:
        models_list = await list_upstream_models(provider)
    except LLMProviderUpstreamError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"failed to list models from upstream: {exc}",
        ) from exc
    return LLMProviderModelsResponse(provider_id=provider.id, models=models_list)
