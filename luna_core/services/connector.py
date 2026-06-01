from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from luna_core.core.crypto import decrypt_json, encrypt_json
from luna_core.models.connector import Connector, Operation
from luna_core.schemas.connector import (
    ConnectorCreate,
    ConnectorUpdate,
    OperationCreate,
    OperationUpdate,
    ParameterDef,
    ParameterType,
)


@dataclass(frozen=True)
class OperationSnapshot:
    """Detached read-only snapshot of an Operation.

    Used by update/delete hooks so they can reference the previous state
    after the ORM row has been mutated, expired, or removed.
    """

    id: uuid.UUID
    connector_id: uuid.UUID
    name: str


OperationCreatedHook = Callable[[Operation], Awaitable[None]]
OperationUpdatedHook = Callable[[OperationSnapshot, Operation], Awaitable[None]]
OperationDeletedHook = Callable[[OperationSnapshot], Awaitable[None]]
ConnectorDeletedHook = Callable[[uuid.UUID, list[str]], Awaitable[None]]


class ConnectorNotFound(LookupError):
    pass


class OperationNotFound(LookupError):
    pass


class DuplicateConnector(ValueError):
    pass


class DuplicateOperation(ValueError):
    pass


# ---------------------------------------------------------------------------
# parameters → JSON Schema
# ---------------------------------------------------------------------------


def parameters_to_input_schema(parameters: list[ParameterDef]) -> dict[str, Any]:
    """Derive a JSON Schema object from a list of parameters.

    The result follows the shape MCP / the agent tool layer expect:
    `{type: 'object', properties: {...}, required: [...]}`. Each parameter's
    `in_` is intentionally not encoded into the schema — `in_` is an
    execution-layer concern (where the value goes at HTTP time) and the AI
    just needs to know "this field exists and has this type".
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in parameters:
        properties[p.name] = _parameter_to_schema(p)
        if p.required:
            required.append(p.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parameter_to_schema(param: ParameterDef) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": param.type.value}
    if param.description:
        schema["description"] = param.description
    if param.type is ParameterType.string and param.enum_values:
        schema["enum"] = list(param.enum_values)
    if param.type is ParameterType.array:
        # Default to string items when the form leaves it blank so the
        # rendered JSON Schema is still valid for an MCP tool.
        schema["items"] = {"type": (param.item_type or ParameterType.string).value}
    if param.type is ParameterType.object:
        sub_props: dict[str, Any] = {}
        sub_required: list[str] = []
        for sub in param.properties or []:
            sub_props[sub.name] = _parameter_to_schema(sub)
            if sub.required:
                sub_required.append(sub.name)
        schema["properties"] = sub_props
        if sub_required:
            schema["required"] = sub_required
    # JSON Schema's `default` keyword is informational only; the LLM
    # uses it as documentation, the actual fallback happens in
    # ``_distribute_input`` at dispatch time. Both surfaces stay in
    # sync so the agent and the runtime never disagree about what
    # "missing" should mean.
    if param.default is not None:
        schema["default"] = param.default
    return schema


def _serialize_parameters(parameters: list[ParameterDef]) -> list[dict[str, Any]]:
    """Convert parameters to plain dicts for JSONB storage.

    `by_alias=True` keeps the wire key `in` (not `in_`) so reads round-trip
    cleanly through Pydantic.
    """
    return [p.model_dump(mode="json", by_alias=True) for p in parameters]


async def create_connector(db: AsyncSession, payload: ConnectorCreate) -> Connector:
    connector = Connector(
        name=payload.name,
        description=payload.description,
        auth_type=payload.auth_type,
        base_url=payload.base_url,
        credentials_encrypted=encrypt_json(payload.credentials),
        is_active=payload.is_active,
    )
    db.add(connector)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateConnector(payload.name) from exc
    await db.refresh(connector)
    return connector


async def list_connectors(db: AsyncSession) -> list[Connector]:
    result = await db.execute(select(Connector).order_by(Connector.created_at.desc()))
    return list(result.scalars().all())


async def get_connector(db: AsyncSession, connector_id: uuid.UUID) -> Connector:
    connector = await db.get(Connector, connector_id)
    if connector is None:
        raise ConnectorNotFound(str(connector_id))
    return connector


async def update_connector(
    db: AsyncSession, connector_id: uuid.UUID, payload: ConnectorUpdate
) -> Connector:
    connector = await get_connector(db, connector_id)
    data = payload.model_dump(exclude_unset=True)
    if "credentials" in data:
        connector.credentials_encrypted = encrypt_json(data.pop("credentials"))
    for field, value in data.items():
        setattr(connector, field, value)
    await db.commit()
    await db.refresh(connector)
    return connector


async def delete_connector(
    db: AsyncSession,
    connector_id: uuid.UUID,
    *,
    on_deleted: ConnectorDeletedHook | None = None,
) -> None:
    """Delete a connector and (via DB cascade) all of its operations.

    `on_deleted` runs after the DB commit succeeds with (connector_id,
    operation_names) so hosts can detach the corresponding MCP tools and
    drop the registry entries without re-fetching anything.
    """
    connector = await get_connector(db, connector_id)
    # Capture operation names before delete so the hook can detach MCP
    # tools by name (those rows are gone after the cascade fires).
    op_names = [op.name for op in await list_operations(db, connector_id)]
    await db.delete(connector)
    await db.commit()
    if on_deleted is not None:
        await on_deleted(connector_id, op_names)


async def get_decrypted_credentials(
    connector: Connector,
) -> dict[str, Any] | None:
    return decrypt_json(connector.credentials_encrypted)


async def create_operation(
    db: AsyncSession,
    connector_id: uuid.UUID,
    payload: OperationCreate,
    *,
    on_created: OperationCreatedHook | None = None,
) -> Operation:
    """Persist a new Operation and optionally notify a hook after commit.

    `on_created` is invoked with the refreshed Operation row only after the
    DB commit succeeds. Hosts use this to register the operation as an MCP
    tool without coupling the service layer to FastMCP.
    """
    await get_connector(db, connector_id)  # raises if missing
    # Source-of-truth precedence: if the caller supplied `parameters`, the
    # backend derives `input_schema` from them. The bare `input_schema` field
    # on the payload is an escape hatch for legacy/advanced consumers — only
    # honored when `parameters` is empty.
    if payload.parameters:
        derived_schema = parameters_to_input_schema(payload.parameters)
    elif payload.input_schema is not None:
        derived_schema = payload.input_schema
    else:
        derived_schema = {}
    operation = Operation(
        connector_id=connector_id,
        name=payload.name,
        description=payload.description,
        method=payload.method,
        path=payload.path,
        parameters=_serialize_parameters(payload.parameters),
        fixed_headers=dict(payload.fixed_headers),
        fixed_body=payload.fixed_body,
        retry_policy=(
            payload.retry_policy.model_dump(mode="json")
            if payload.retry_policy is not None
            else None
        ),
        input_schema=derived_schema,
        output_schema=payload.output_schema,
        is_active=payload.is_active,
    )
    db.add(operation)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateOperation(payload.name) from exc
    await db.refresh(operation)
    if on_created is not None:
        await on_created(operation)
    return operation


async def list_operations(
    db: AsyncSession, connector_id: uuid.UUID
) -> list[Operation]:
    await get_connector(db, connector_id)
    result = await db.execute(
        select(Operation)
        .where(Operation.connector_id == connector_id)
        .order_by(Operation.created_at.desc())
    )
    return list(result.scalars().all())


async def get_operation(db: AsyncSession, operation_id: uuid.UUID) -> Operation:
    result = await db.execute(
        select(Operation)
        .where(Operation.id == operation_id)
        .options(selectinload(Operation.connector))
    )
    operation = result.scalar_one_or_none()
    if operation is None:
        raise OperationNotFound(str(operation_id))
    return operation


async def update_operation(
    db: AsyncSession,
    operation_id: uuid.UUID,
    payload: OperationUpdate,
    *,
    on_updated: OperationUpdatedHook | None = None,
) -> Operation:
    """Patch an operation and notify a hook with (previous_snapshot, current).

    Hooks compare names to decide whether to detach the old MCP tool before
    re-attaching the new one (renaming an operation has to drop the old
    tool entry so clients don't see ghosts).
    """
    operation = await get_operation(db, operation_id)
    previous = OperationSnapshot(
        id=operation.id,
        connector_id=operation.connector_id,
        name=operation.name,
    )
    data = payload.model_dump(exclude_unset=True, by_alias=False)
    # Re-derive input_schema whenever the caller sent `parameters` — even an
    # empty list is meaningful ("this operation now takes no inputs"). The raw
    # `input_schema` key on the patch is an escape hatch that only applies
    # when `parameters` is omitted from the payload entirely.
    if "parameters" in data:
        params = payload.parameters or []
        operation.parameters = _serialize_parameters(params)
        operation.input_schema = parameters_to_input_schema(params)
        data.pop("parameters", None)
        data.pop("input_schema", None)
    for field, value in data.items():
        setattr(operation, field, value)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateOperation(operation.name) from exc
    await db.refresh(operation)
    if on_updated is not None:
        await on_updated(previous, operation)
    return operation


async def delete_operation(
    db: AsyncSession,
    operation_id: uuid.UUID,
    *,
    on_deleted: OperationDeletedHook | None = None,
) -> None:
    operation = await get_operation(db, operation_id)
    snapshot = OperationSnapshot(
        id=operation.id,
        connector_id=operation.connector_id,
        name=operation.name,
    )
    await db.delete(operation)
    await db.commit()
    if on_deleted is not None:
        await on_deleted(snapshot)
