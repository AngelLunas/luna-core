"""Generic LLM token-usage ledger.

Every completed assistant turn records one row: the real input/output/cached
token counts the provider reported, keyed by the **execution scope** (a
``flow_run_id`` for flows, a ``conversation_id`` for chat — same neutral
``scope_id`` the streaming layer already threads) and the assistant
``message_id``. No FK on ``scope_id`` (it spans two tables) and no owner column:
this is raw infrastructure. A host app maps scope → user/credits from its own
data (e.g. savia joins ``scope_id`` → ``conversations.user_id``). It is the
single source of truth for what an LLM turn actually cost.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from luna_core.core.db import Base


class LLMUsage(Base):
    __tablename__ = "llm_usage"
    __table_args__ = (
        Index("ix_llm_usage_scope", "scope_id"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Execution scope: flow_run_id OR conversation_id (no FK — polymorphic).
    scope_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    # The assistant message this usage is for (nullable for non-message calls).
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    # Cached/prompt-cache read tokens, when the provider reports them.
    cached_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
