"""What a model's tokens cost, per million, as the operator enters them.

Prices are data, never code: no model name or figure is written here. An admin
copies them from the provider's pricing page (``/llm-prices``). They are applied
when the ledger is READ (``services.usage``), so prices entered late still cost
the history, and a model without a row is charged at the dearest row known — a
forgotten price never makes calls free.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from luna_core.core.db import Base


class LLMPrice(Base):
    __tablename__ = "llm_prices"
    __table_args__ = {"schema": "core"}

    # The model id exactly as the ledger records it (``llm_usage.model``).
    model: Mapped[str] = mapped_column(String(255), primary_key=True)
    # USD per million tokens. Cache read / write and audio are priced apart
    # from a plain input token; NULL = priced as plain input.
    input_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    output_per_mtok: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    cache_read_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    cache_write_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    audio_input_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
