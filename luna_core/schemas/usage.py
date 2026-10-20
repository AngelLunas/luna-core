"""Schemas for model prices and the usage report."""
from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class LLMPriceWrite(BaseModel):
    """USD per million tokens, as the provider's pricing page states them.
    Cache read / write and audio left out = priced as plain input."""

    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)
    cache_read_per_mtok: Decimal | None = Field(default=None, ge=0)
    cache_write_per_mtok: Decimal | None = Field(default=None, ge=0)
    audio_input_per_mtok: Decimal | None = Field(default=None, ge=0)


class LLMPriceRead(LLMPriceWrite):
    model_config = ConfigDict(from_attributes=True)

    model: str
    updated_at: datetime


class UsageReportRow(BaseModel):
    """One group of the report: its keys (whichever were grouped by), then
    calls, tokens and cost (``None`` while no price is entered)."""

    user_id: uuid.UUID | None = None
    purpose: str | None = None
    model: str | None = None
    agent_name: str | None = None
    day: datetime | None = None
    calls: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    cost_usd: Decimal | None = None
    # Models in this group that have no price row (charged at the dearest).
    unpriced: bool = False
