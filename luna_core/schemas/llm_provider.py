from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# "openai_compatible": HTTP chat-completions (GenericProvider).
# "claude_cli": local Claude Code binary on subscription auth — base_url is
# the binary path, api_key is unused, models come from a curated alias list.
LLMProviderKind = Literal["openai_compatible", "claude_cli"]


class LLMProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    kind: LLMProviderKind = "openai_compatible"
    base_url: str = Field(min_length=1, max_length=1024)
    chat_url: str | None = Field(default=None, max_length=1024)
    models_url: str | None = Field(default=None, max_length=1024)
    api_key: str | None = None
    is_active: bool = True


class LLMProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    kind: LLMProviderKind | None = None
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
    kind: LLMProviderKind
    # Derived from the kind registry so clients never branch on `kind`:
    # a provider is usable when active and (has_api_key or not
    # requires_api_key); display_name is what to show for it.
    requires_api_key: bool
    display_name: str
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
    # Friendly name when the id is an alias/opaque (e.g. "haiku" →
    # "Claude Haiku (latest)"); None → show the id.
    label: str | None = None


class LLMProviderModelsResponse(BaseModel):
    provider_id: uuid.UUID
    models: list[LLMProviderModel]
