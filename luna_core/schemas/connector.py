from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from luna_core.models.connector import AuthType, HTTPMethod


class ParameterType(str, enum.Enum):
    string = "string"
    integer = "integer"
    number = "number"
    boolean = "boolean"
    array = "array"
    object = "object"


class ParameterIn(str, enum.Enum):
    """Where in the outgoing HTTP request a parameter is rendered."""

    path = "path"
    query = "query"
    body = "body"
    header = "header"


class ParameterDef(BaseModel):
    """Visual definition of one input parameter.

    The list of these on an Operation is the source of truth — `input_schema`
    is derived from it on save. Each parameter declares where it goes in the
    outgoing request via `in_`.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, max_length=255)
    type: ParameterType = ParameterType.string
    description: str = ""
    required: bool = False
    # The JSON field is `in` (reserved word in Python) — alias keeps the wire
    # contract clean while letting us use `in_` in Python code.
    in_: ParameterIn = Field(default=ParameterIn.body, alias="in")
    # Restricts type=string to one of these values (becomes JSON Schema enum).
    enum_values: list[str] | None = None
    # For type=array: element type (kept simple — no nested arrays of arrays).
    item_type: ParameterType | None = None
    # For type=object: nested parameters.
    properties: list[ParameterDef] | None = None
    # Optional fallback value applied at dispatch time when the caller's
    # input omits this parameter entirely. The value is also surfaced as
    # the JSON Schema `default` keyword so the LLM sees it as
    # documentation in the tool definition. Applies for omitted keys
    # only — an explicit ``null``/``""`` from the caller is honored
    # as their intent, not silently swapped for the default. Common
    # use: paginated endpoints whose first-page cursor is a server-
    # specific sentinel like ``"0"`` that the LLM keeps forgetting
    # to send on turn 1.
    default: Any | None = None


# Recursive model needs explicit rebuild so `properties: list[ParameterDef]`
# resolves to the final class.
ParameterDef.model_rebuild()


class ConnectorCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    auth_type: AuthType = AuthType.none
    base_url: str = Field(min_length=1, max_length=1024)
    credentials: dict[str, Any] | None = None
    is_active: bool = True


class ConnectorUpdate(BaseModel):
    description: str | None = None
    auth_type: AuthType | None = None
    base_url: str | None = None
    credentials: dict[str, Any] | None = None
    is_active: bool | None = None


class ConnectorRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    auth_type: AuthType
    base_url: str
    has_credentials: bool
    # `None` for non-OAuth2 conectors. `True` once the OAuth2 handshake
    # finished and an access_token is on file; `False` when the conector is
    # OAuth2-configured but the user hasn't clicked "Connect" yet (or the
    # tokens were wiped by a credentials replacement).
    oauth2_connected: bool | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class OAuth2ConfigResponse(BaseModel):
    """Public OAuth2 config exposed to the dashboard.

    The callback URL is what each OAuth2 provider (Upwork, Slack, etc.)
    expects registered in its developer console — the dashboard surfaces
    this so admins know exactly what to paste.
    """

    callback_url: str


class OAuth2StartResponse(BaseModel):
    """Result of starting an OAuth2 handshake — the URL the popup opens."""

    authorize_url: str


class OAuth2CallbackRequest(BaseModel):
    """Body of the callback endpoint — the dashboard sends what came on the URL."""

    code: str
    state: str


class RetryPolicyModel(BaseModel):
    """Per-operation retry policy for transient HTTP failures.

    Listed auth statuses (401/403) are dropped — those route through the
    OAuth2 refresh path, not the generic retry loop.
    """

    max_attempts: int = Field(ge=1, le=10)
    retry_on_status: list[int] = Field(default_factory=list)
    initial_delay_ms: int = Field(default=200, ge=0, le=30_000)
    multiplier: float = Field(default=3.0, ge=1.0, le=10.0)
    jitter: bool = True


class OperationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    method: HTTPMethod
    path: str = Field(min_length=1, max_length=1024)
    # New canonical shape — the backend derives `input_schema` from this.
    parameters: list[ParameterDef] = Field(default_factory=list)
    fixed_headers: dict[str, str] = Field(default_factory=dict)
    fixed_body: dict[str, Any] | None = None
    retry_policy: RetryPolicyModel | None = None
    # Escape hatch: when supplied AND `parameters` is empty, the backend
    # stores this verbatim as input_schema (legacy/advanced flow). Ignored
    # when `parameters` is non-empty.
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class OperationUpdate(BaseModel):
    description: str | None = None
    method: HTTPMethod | None = None
    path: str | None = Field(default=None, min_length=1, max_length=1024)
    parameters: list[ParameterDef] | None = None
    fixed_headers: dict[str, str] | None = None
    fixed_body: dict[str, Any] | None = None
    retry_policy: RetryPolicyModel | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    is_active: bool | None = None


class OperationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    connector_id: uuid.UUID
    name: str
    description: str
    method: HTTPMethod
    path: str
    parameters: list[ParameterDef]
    fixed_headers: dict[str, str]
    fixed_body: dict[str, Any] | None
    retry_policy: RetryPolicyModel | None = None
    # Derived from `parameters` — kept on the read so existing consumers
    # (MCP builder, agent tool list) don't need to know about the visual shape.
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    is_active: bool
    created_at: datetime


class ConnectorSummary(BaseModel):
    """Lightweight, non-sensitive view of a connector for embedding in other reads."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    auth_type: AuthType
    base_url: str
    is_active: bool


class OperationWithConnector(OperationRead):
    connector: ConnectorSummary


class OperationTestRequest(BaseModel):
    """Inputs supplied when manually testing an operation from the UI."""

    input: dict[str, Any] = Field(default_factory=dict)


class OperationDraftTestRequest(BaseModel):
    """Payload for testing an unsaved operation (the create-flow Test button).

    Carries the full operation definition the user has typed into the form,
    so the backend can run a real HTTP call against the connector's stored
    credentials without persisting anything.
    """

    method: HTTPMethod
    path: str = Field(min_length=1, max_length=1024)
    parameters: list[ParameterDef] = Field(default_factory=list)
    fixed_headers: dict[str, str] = Field(default_factory=dict)
    fixed_body: dict[str, Any] | None = None
    retry_policy: RetryPolicyModel | None = None
    input: dict[str, Any] = Field(default_factory=dict)


class OperationTestResponse(BaseModel):
    """Result of running an operation against its real upstream service.

    Mirrors the actual HTTP call (status, body, latency) without raising on
    4xx/5xx so the UI can render error responses verbatim.
    """

    ok: bool
    status_code: int | None = None
    latency_ms: int
    request_method: str
    request_url: str
    response: dict[str, Any] | None = None
    error: str | None = None
