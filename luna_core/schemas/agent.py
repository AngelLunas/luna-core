from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from luna_core.schemas.connector import OperationRead


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    role: str = ""
    instructions: str = ""
    llm_provider_id: uuid.UUID
    model: str = Field(min_length=1, max_length=255)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class AgentUpdate(BaseModel):
    role: str | None = None
    instructions: str | None = None
    llm_provider_id: uuid.UUID | None = None
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    output_schema: dict[str, Any] | None = None


class AgentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    role: str
    instructions: str
    llm_provider_id: uuid.UUID
    model: str
    temperature: float
    output_schema: dict[str, Any] = Field(default_factory=dict)
    required_sources: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    # Populated only when the caller asks for `?include=operations`; left as
    # None otherwise so clients can tell "not loaded" from "loaded, empty".
    operations: list["AgentOperationRead"] | None = None
    # Same convention as `operations`: populated only when the detail
    # endpoint is called with `?include=system_tools`. Carries the grant
    # rows (each one is just the agent_id + tool_name pair); the catalog
    # entry the name resolves to is fetched separately by the client.
    system_tools: list["AgentSystemToolGrantRead"] | None = None

    @field_validator("output_schema", mode="before")
    @classmethod
    def coerce_none_to_dict(cls, v: Any) -> dict[str, Any]:
        return v if v is not None else {}

    @field_validator("required_sources", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v: Any) -> list[str]:
        return v if v is not None else []


class AgentOperationAssign(BaseModel):
    operation_ids: list[uuid.UUID] = Field(min_length=1)


class AgentOperationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    operation_id: uuid.UUID
    operation: OperationRead | None = None


class AgentSystemToolGrantAssign(BaseModel):
    """Bulk-replace request for an agent's system-tool grants.

    Empty list is allowed — that's the "unassign everything" path. (This
    deliberately differs from ``AgentOperationAssign.operation_ids``
    which requires min_length=1; the operations endpoint inherits a
    historical constraint we don't want to propagate to the new
    surface.)
    """

    tool_names: list[str] = Field(default_factory=list)


class AgentSystemToolGrantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    tool_name: str


class InstructionsPreviewIn(BaseModel):
    """Request body for the preview endpoint.

    ``instructions`` is the raw template text the UI currently has in the
    editor. ``source_bindings`` is an optional map of explicit source ids —
    used only for non-implicit sources (where the loader needs a target id
    that doesn't come from the session). Implicit sources resolve from the
    authenticated user automatically.
    """

    instructions: str = ""
    source_bindings: dict[str, str] = Field(default_factory=dict)


class InstructionsPreviewSourceDiag(BaseModel):
    """Per-source diagnostic for the preview run."""

    name: str
    status: str  # "ok" | "missing-binding" | "loader-error" | "unknown-source"
    detail: str = ""


class InstructionsPreviewOut(BaseModel):
    """Response from the preview endpoint."""

    resolved: str
    required_sources: list[str] = Field(default_factory=list)
    diagnostics: list[InstructionsPreviewSourceDiag] = Field(default_factory=list)


AgentRead.model_rebuild()
