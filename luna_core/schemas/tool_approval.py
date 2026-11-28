"""Schemas for the tool-approval surface."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolApprovalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    tool_use_id: str
    tool_name: str
    tool_input: dict
    #: Canonical name of the agent that proposed the call; None on legacy rows.
    agent_name: str | None
    status: str
    reason: str | None
    created_at: datetime
    resolved_at: datetime | None


class ToolApprovalDecision(BaseModel):
    """Approve or reject a pending tool call. ``reason`` is optional on reject
    (a rejection reason, or a "do this instead" instruction). A plain reject of
    the whole turn does not re-invoke the LLM; a reject with a reason does.
    The reason's length is capped by the route (``chat_max_message_chars``),
    which refuses with a code the client can explain."""

    decision: Literal["approved", "rejected"]
    reason: str | None = None
    #: Approve the call with these inputs instead of the proposed ones — the
    #: user corrected it in the app's own form before saying yes. Approve only.
    tool_input: dict | None = None

    @model_validator(mode="after")
    def _edited_only_when_approved(self) -> ToolApprovalDecision:
        if self.tool_input is not None and self.decision != "approved":
            raise ValueError("tool_input goes with an approval only")
        return self


class ToolApprovalBulkApprove(BaseModel):
    """Approve several pending calls of the suspended turn at once.

    The ids are the cards the user was looking at, not "whatever is pending":
    a call that arrived after they tapped is never approved on their behalf.
    Kept out of the list by the client (destructive calls, say) stays pending.
    """

    approval_ids: list[uuid.UUID] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def _unique(self) -> ToolApprovalBulkApprove:
        self.approval_ids = list(dict.fromkeys(self.approval_ids))
        return self
