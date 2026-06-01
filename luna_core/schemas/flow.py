from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from luna_core.models.flow import FlowRunStatus

NodeType = Literal["action", "ai_agent", "condition", "human_checkpoint", "trigger", "output"]
ConditionOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]
FlowInputType = Literal["string", "integer", "number", "boolean", "object", "array"]


class FlowEdgeCondition(BaseModel):
    field: str
    operator: ConditionOperator
    value: Any


class FlowNode(BaseModel):
    id: str
    type: NodeType
    name: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class FlowEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    condition: FlowEdgeCondition | None = None

    model_config = ConfigDict(populate_by_name=True)


class ScheduleRule(BaseModel):
    """A single recurring rule. ``cron`` is the 5-field expression evaluated in
    ``tz`` (IANA), ``value`` is the rich representation the frontend uses to
    rebuild its picker without lossy cron parsing, and ``user_id`` identifies
    the user the run should be executed as — context sources like ``profile``
    resolve from ``state.trigger.user_id``, so a flow without an owner only
    has meaning when each rule says who's behind it. Schedules created before
    this field existed have ``user_id=None`` and are skipped by the dispatcher.
    """

    cron: str
    tz: str = "UTC"
    value: dict[str, Any] = Field(default_factory=dict)
    user_id: uuid.UUID | None = None
    # Inputs to feed into ``state.inputs`` whenever this rule fires. Each
    # value is validated against ``FlowDefinition.inputs`` at dispatch time
    # via the same path as a manual run, so missing keys fall back to the
    # input's declared ``default``.
    inputs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cron")
    @classmethod
    def _valid_cron(cls, v: str) -> str:
        if not croniter.is_valid(v):
            raise ValueError(f"invalid cron expression: {v!r}")
        return v

    @field_validator("tz")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {v!r}") from exc
        return v


class FlowTrigger(BaseModel):
    type: Literal["manual", "schedule", "webhook"] = "manual"
    # Legacy single-cron field kept for read-side back-compat. New writes go
    # through ``schedules``; ``_normalize_legacy_cron`` below promotes any
    # incoming payload that still uses the old shape.
    cron: str | None = None
    schedules: list[ScheduleRule] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_legacy_cron(self) -> "FlowTrigger":
        if self.type == "schedule" and not self.schedules and self.cron:
            self.schedules = [ScheduleRule(cron=self.cron, tz="UTC")]
        return self


class FlowInputDef(BaseModel):
    """Declares a parameter that a flow expects from its caller.

    When the flow is triggered manually, the trigger payload is validated
    against the list of FlowInputDef and the resulting values are made
    available to nodes as ``state.inputs.<name>``.
    """

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: FlowInputType = "string"
    required: bool = False
    default: Any = None
    description: str = ""


class FlowNodePosition(BaseModel):
    """Visual position of a node in the editor canvas. Persisted alongside the
    definition so the visualizer can reproduce the user's layout instead of
    falling back to auto-layout every time the flow is opened.
    """

    x: float
    y: float


class FlowDefinition(BaseModel):
    entry_point: str
    trigger: FlowTrigger | None = None
    nodes: list[FlowNode]
    edges: list[FlowEdge] = Field(default_factory=list)
    inputs: list[FlowInputDef] = Field(default_factory=list)
    layout: dict[str, FlowNodePosition] = Field(default_factory=dict)

    @field_validator("inputs")
    @classmethod
    def _no_duplicate_input_names(cls, inputs: list[FlowInputDef]) -> list[FlowInputDef]:
        seen: set[str] = set()
        for spec in inputs:
            if spec.name in seen:
                raise ValueError(f"duplicate flow input name: {spec.name!r}")
            seen.add(spec.name)
        return inputs

    @model_validator(mode="after")
    def _layout_keys_reference_nodes(self) -> "FlowDefinition":
        node_ids = {n.id for n in self.nodes}
        stray = [nid for nid in self.layout if nid not in node_ids]
        if stray:
            raise ValueError(
                f"layout has positions for unknown nodes: {sorted(stray)!r}"
            )
        return self


class FlowCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    definition: FlowDefinition
    is_active: bool = True


class FlowUpdate(BaseModel):
    description: str | None = None
    definition: FlowDefinition | None = None
    is_active: bool | None = None


class FlowRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    definition: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FlowRunTrigger(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    source: Literal["manual", "schedule", "webhook", "api"] = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    flow_id: uuid.UUID
    status: FlowRunStatus
    trigger: dict[str, Any]
    state: dict[str, Any]
    started_at: datetime | None
    completed_at: datetime | None
    cleared_at: datetime | None = None
    created_at: datetime


class SchedulePreviewIn(BaseModel):
    cron: str
    tz: str = "UTC"
    count: int = Field(default=5, ge=1, le=20)


class SchedulePreviewOut(BaseModel):
    next_runs: list[datetime]
