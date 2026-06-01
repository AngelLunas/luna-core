from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base

if TYPE_CHECKING:
    from luna_core.models.event import AgentMessage, RunEvent


class FlowRunStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


class Flow(Base):
    __tablename__ = "flows"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
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

    runs: Mapped[list["FlowRun"]] = relationship(
        back_populates="flow",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class FlowRun(Base):
    __tablename__ = "flow_runs"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flow_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.flows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[FlowRunStatus] = mapped_column(
        Enum(
            FlowRunStatus,
            name="flow_run_status",
            schema="core",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default=FlowRunStatus.pending.value,
        index=True,
    )
    trigger: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    state: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Set when the user soft-compacts the run: events + assistant messages are
    # purged but the row itself survives so metrics (duration, status) stay
    # intact.
    cleared_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    flow: Mapped[Flow] = relationship(back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="flow_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="flow_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
