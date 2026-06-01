from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from luna_core.core.db import Base


class LLMProvider(Base):
    __tablename__ = "llm_providers"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    # chat_url / models_url are optional overrides for hosts whose chat or
    # model-listing endpoints live somewhere different than base_url. When
    # null we derive: <base_url> for the SDK and <base_url>/models for the
    # listing call.
    chat_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    models_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # API key persisted as Fernet-encrypted JSON ({"api_key": "..."}). Never
    # returned through the API surface; only updatable.
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
