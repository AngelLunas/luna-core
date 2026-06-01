"""Conversation primitive — persistent chat thread.

Unlike ``AgentMessage`` (run-scoped audit log, purged on compaction),
a ``Conversation`` survives the flow that created it. Any domain table
that needs persistent user-agent chat (sentinel cover letters today,
debug threads tomorrow) holds an FK to ``Conversation.id`` instead of
embedding messages itself.

The shape is naked on purpose: no FK back to agents or runs. The
owning domain table is the one that knows what the conversation is
*about*. Keeping that out of core lets the same primitive serve every
future agent without dragging domain entities into the engine.

``ConversationMessage.content`` mirrors the JSONB array-of-blocks shape
``AgentMessage.content`` already uses (text + tool_use + tool_result),
so renderers and serializers transfer 1:1.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base


class ConversationMessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
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

    messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationMessage.sequence",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence", name="uq_conversation_messages_sequence"
        ),
        Index(
            "ix_conversation_messages_conv_seq",
            "conversation_id",
            "sequence",
        ),
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
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[ConversationMessageRole] = mapped_column(
        Enum(
            ConversationMessageRole,
            name="conversation_message_role",
            schema="core",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    content: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    is_partial: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


__all__ = [
    "Conversation",
    "ConversationMessage",
    "ConversationMessageRole",
]
