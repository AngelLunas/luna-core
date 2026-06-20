"""Schemas for the tool-approval surface."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolApprovalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    tool_use_id: str
    tool_name: str
    tool_input: dict
    status: str
    reason: str | None
    created_at: datetime
    resolved_at: datetime | None


class ToolApprovalDecision(BaseModel):
    """Approve or reject a pending tool call. ``reason`` is optional on reject
    (a rejection reason, or a "do this instead" instruction). A plain reject of
    the whole turn does not re-invoke the LLM; a reject with a reason does."""

    decision: Literal["approved", "rejected"]
    reason: str | None = Field(default=None, max_length=4000)
