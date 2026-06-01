from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LLMProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    base_url: str = Field(min_length=1, max_length=1024)
    chat_url: str | None = Field(default=None, max_length=1024)
    models_url: str | None = Field(default=None, max_length=1024)
    api_key: str | None = None
    is_active: bool = True


class LLMProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    base_url: str | None = Field(default=None, min_length=1, max_length=1024)
    chat_url: str | None = Field(default=None, max_length=1024)
    models_url: str | None = Field(default=None, max_length=1024)
    # api_key is write-only: omit → no change; provide a string → replace;
    # explicit empty string → clear the stored key.
    api_key: str | None = None
    is_active: bool | None = None


class LLMProviderRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    base_url: str
    chat_url: str | None
    models_url: str | None
    has_api_key: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime


class LLMProviderModel(BaseModel):
    id: str
    owned_by: str | None = None


class LLMProviderModelsResponse(BaseModel):
    provider_id: uuid.UUID
    models: list[LLMProviderModel]
