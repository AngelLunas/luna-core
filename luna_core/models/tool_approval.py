"""Human-in-the-loop tool approval (chat-scoped).

When an agent emits a ``tool_use`` for a tool its grant marks
``requires_approval``, the runner does **not** execute it: it persists this row
(the durable "intent to execute") and suspends the turn. The row survives page
reloads and arbitrary delays — the frontend lists pending rows over REST and
renders approve/reject. On resolution the conversation resumes.

Keyed by ``(conversation_id, tool_use_id)``: the tool_use_id is the id of the
``tool_use`` block inside the assistant message, so the frontend anchors the
buttons under the right bubble without needing the message id.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from luna_core.core.db import Base


class ToolApprovalStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ToolApproval(Base):
    __tablename__ = "tool_approvals"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "tool_use_id", name="uq_tool_approvals_tool_use"
        ),
        Index("ix_tool_approvals_conv_status", "conversation_id", "status"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_use_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tool_input: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    # pending / approved / rejected (stored as text; values from ToolApprovalStatus).
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending"
    )
    # Optional rejection reason / "do this instead" instruction.
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.users.id", ondelete="SET NULL"),
        nullable=True,
    )


__all__ = ["ToolApproval", "ToolApprovalStatus"]
