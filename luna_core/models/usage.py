"""Generic LLM token-usage ledger.

Every completed assistant turn records one row: the real input/output/cached
token counts the provider reported, keyed by the **execution scope** (a
``flow_run_id`` for flows, a ``conversation_id`` for chat — same neutral
``scope_id`` the streaming layer already threads) and the assistant
``message_id``. No FK on ``scope_id`` (it spans two tables).

Each row also says WHO it was for and WHAT it was doing, from the attribution
the entry point set (``services.usage.usage_context``): ``user_id`` (the
account the call served; NULL when no user is behind it), ``purpose`` (a dotted
path of activities, outermost first — ``chat``, ``chat.title``,
``chat.image`` — whose words the host chooses) and ``agent_name``. That is
what a per-user budget sums and a usage report groups by, without a host
mapping every kind of scope back to an owner. It is the single source of truth
for what an LLM call actually cost; prices live apart (``LLMPrice``) and are
applied when reading.
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
        Index("ix_llm_usage_user_created", "user_id", "created_at"),
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
    # Who and what (see the module docstring). No FK: a deleted user's spend
    # stays readable for the operator's books.
    user_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    purpose: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    input_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    # Cached/prompt-cache read tokens, when the provider reports them.
    cached_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Prompt-cache WRITE tokens (subset of input_tokens), when the provider
    # reports them — a write is priced above a plain input token and a read
    # far below, so cost math needs all three apart.
    cache_write_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Audio input tokens (subset of input_tokens), when the provider reports
    # them — audio and text tokens are priced differently, so cost math needs
    # the split. Populated by audio features (voice STT, audio-input chat).
    audio_input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
