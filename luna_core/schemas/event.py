from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from luna_core.models.event import AgentMessageRole, RunEventType


class RunEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    flow_run_id: uuid.UUID
    sequence: int
    timestamp: datetime
    event_type: RunEventType
    node_id: str | None
    payload: dict[str, Any]


class AgentMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    flow_run_id: uuid.UUID
    node_id: str
    sequence: int
    role: AgentMessageRole
    content: list[dict[str, Any]]
    created_at: datetime


class ResumeRequest(BaseModel):
    response: str = Field(min_length=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
