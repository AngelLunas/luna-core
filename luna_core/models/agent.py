from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base
from luna_core.models.connector import Operation
from luna_core.models.llm_provider import LLMProvider


class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    role: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    instructions: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    llm_provider_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.llm_providers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.7")
    output_schema: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    required_sources: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
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

    llm_provider: Mapped[LLMProvider] = relationship()
    agent_operations: Mapped[list["AgentOperation"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    agent_system_tool_grants: Mapped[list["AgentSystemToolGrant"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AgentOperation(Base):
    __tablename__ = "agent_operations"
    __table_args__ = (
        UniqueConstraint("agent_id", "operation_id", name="uq_agent_operations_pair"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    operation_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.operations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # When true, the agent must get human approval before this tool runs.
    requires_approval: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    agent: Mapped[Agent] = relationship(back_populates="agent_operations")
    operation: Mapped[Operation] = relationship(back_populates="agent_operations")


class AgentSystemToolGrant(Base):
    """Per-agent grant for a system tool by name.

    System tools live in the in-process registry (``luna_core/mcp/system_tools``),
    not in the ``Operation`` table — they have no connector, no HTTP method,
    no URL. This table just records which agent has access to which tool
    by name, mirroring how ``AgentOperation`` records access to connector
    operations by id. Both are unioned at run time by the AgentRunner to
    decide the agent's effective tool set.

    Tool names are stored as strings (not FKs to a system_tools table)
    because the catalog is authoritative in code: adding/removing a tool
    is a code change, not a DB change. A grant for a tool that no longer
    exists in the registry is a harmless dangling reference — the run-time
    filter intersects with whatever the registry actually advertises.
    """

    __tablename__ = "agent_system_tool_grants"
    __table_args__ = (
        UniqueConstraint(
            "agent_id", "tool_name", name="uq_agent_system_tool_grants_pair"
        ),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # When true, the agent must get human approval before this tool runs.
    requires_approval: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    agent: Mapped[Agent] = relationship(back_populates="agent_system_tool_grants")
