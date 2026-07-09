"""Request/response schemas for the conversation surface."""
from __future__ import annotations

import uuid
from datetime import datetime

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from luna_core.schemas.tool_approval import ToolApprovalRead


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID | None
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    sequence: int
    role: str
    content: list
    is_partial: bool
    created_at: datetime

    @field_validator("role", mode="before")
    @classmethod
    def _enum_to_value(cls, v: Any) -> Any:
        return v.value if isinstance(v, Enum) else v


class SendMessageRequest(BaseModel):
    # Empty is allowed only when media is attached (an image-only turn — "here's a
    # photo, what's wrong?"); the validator below enforces "text or media".
    new_message: str = Field(default="")
    # Media (already uploaded by the app) attached to this turn. The ids are
    # opaque to luna-core — embedded as ``{"type":"image","media_id":...}`` blocks
    # so the agent can pass them to a tool, and a vision-native model can see them.
    media_ids: list[uuid.UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_text_or_media(self) -> "SendMessageRequest":
        if not self.new_message.strip() and not self.media_ids:
            raise ValueError("new_message or media_ids is required")
        return self


class SendMessageResponse(BaseModel):
    """The result of a turn.

    - ``status="completed"``: the agent finished; ``output`` is plain text, or a
      structured object when the agent declares an output schema.
    - ``status="awaiting_approval"``: the turn paused for human tool approval;
      ``pending`` lists the gated calls to approve/reject (also fetchable via the
      tool-approvals endpoint, so the buttons survive a reload).
    - ``status="aborted"``: the user stopped the turn mid-stream; whatever
      streamed so far was persisted as a partial assistant message."""

    conversation_id: uuid.UUID
    status: Literal["completed", "awaiting_approval", "aborted"]
    output: str | dict | None = None
    pending: list[ToolApprovalRead] | None = None
