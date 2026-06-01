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
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base
from luna_core.models.flow import FlowRun


class RunEventType(str, enum.Enum):
    flow_started = "flow_started"
    flow_completed = "flow_completed"
    flow_failed = "flow_failed"
    node_started = "node_started"
    node_completed = "node_completed"
    node_failed = "node_failed"
    # Legacy single-shot "agent is about to think"; superseded for streaming
    # by agent_message_started / *_delta / agent_message_completed, but kept
    # for backwards compatibility with older runs.
    agent_thinking = "agent_thinking"
    # Streaming lifecycle of one assistant turn. Every delta carries the same
    # message_id so the UI can group chunks; the persisted AgentMessage row
    # uses the same id as its primary key.
    agent_message_started = "agent_message_started"
    agent_text_delta = "agent_text_delta"
    agent_thinking_delta = "agent_thinking_delta"
    agent_message_completed = "agent_message_completed"
    tool_called = "tool_called"
    tool_result = "tool_result"
    human_checkpoint = "human_checkpoint"
    human_response = "human_response"
    # Per-iteration lifecycle for ai_agent nodes running in scratchpad mode.
    # Emitted once per item the runtime processes (whether sequential or
    # parallel). All `agent_*`, `tool_*` events emitted from inside an
    # iteration carry the same `iteration_id` in their payload so the UI
    # can group them under the iteration block they belong to.
    iteration_started = "iteration_started"
    iteration_completed = "iteration_completed"
    iteration_failed = "iteration_failed"
    # Emitted once when a terminal run is soft-compacted; the events themselves
    # are then deleted so this is the last surviving row.
    run_cleared = "run_cleared"


class AgentMessageRole(str, enum.Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class RunEvent(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint(
            "flow_run_id", "sequence", name="uq_run_events_run_sequence"
        ),
        Index("ix_run_events_run_sequence", "flow_run_id", "sequence"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flow_run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.flow_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    event_type: Mapped[RunEventType] = mapped_column(
        Enum(
            RunEventType,
            name="run_event_type",
            schema="core",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    node_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    flow_run: Mapped[FlowRun] = relationship(back_populates="events")


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    __table_args__ = (
        UniqueConstraint(
            "flow_run_id", "sequence", name="uq_agent_messages_run_sequence"
        ),
        Index(
            "ix_agent_messages_run_node",
            "flow_run_id",
            "node_id",
            "sequence",
        ),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flow_run_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.flow_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[AgentMessageRole] = mapped_column(
        Enum(
            AgentMessageRole,
            name="agent_message_role",
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
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    flow_run: Mapped[FlowRun] = relationship(back_populates="messages")
