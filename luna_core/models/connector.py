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
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base

if TYPE_CHECKING:
    from luna_core.models.agent import AgentOperation


class AuthType(str, enum.Enum):
    none = "none"
    api_key = "api_key"
    oauth2 = "oauth2"
    basic = "basic"


class HTTPMethod(str, enum.Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class Connector(Base):
    __tablename__ = "connectors"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    auth_type: Mapped[AuthType] = mapped_column(
        Enum(
            AuthType,
            name="connector_auth_type",
            schema="core",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default=AuthType.none.value,
    )
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    credentials_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    operations: Mapped[list["Operation"]] = relationship(
        back_populates="connector",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Operation(Base):
    __tablename__ = "operations"
    __table_args__ = (
        UniqueConstraint("connector_id", "name", name="uq_operations_connector_name"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    connector_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.connectors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    method: Mapped[HTTPMethod] = mapped_column(
        Enum(
            HTTPMethod,
            name="http_method",
            schema="core",
            native_enum=True,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    input_schema: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    output_schema: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    # Visual source of truth: list of ParameterDef dicts. `input_schema` above
    # is derived from this on save so MCP tool definitions keep working.
    parameters: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    # Constants always merged into the outgoing request. Values may contain
    # `{param}` placeholders that resolve from input at call time.
    fixed_headers: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    fixed_body: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Optional per-operation retry policy for transient HTTP failures
    # (e.g. flaky edges that return spurious 404s/5xx). See
    # luna_core.connectors.retry.RetryPolicy for the expected shape.
    # NULL = single attempt, no retries (default behavior).
    retry_policy: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    connector: Mapped[Connector] = relationship(back_populates="operations")
    agent_operations: Mapped[list["AgentOperation"]] = relationship(
        back_populates="operation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
