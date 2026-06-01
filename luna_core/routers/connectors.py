from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status

from luna_core.connectors.registry import ConnectorRegistry
from luna_core.connectors.oauth2_flow import (
    OAuth2StateError,
    build_authorize_url,
    exchange_code_for_tokens,
    make_state,
    verify_state,
)
from luna_core.core.config import settings
from luna_core.core.crypto import decrypt_json, encrypt_json
from luna_core.core.dependencies import DBSession, require_permission
from luna_core.models.connector import AuthType, Operation
from luna_core.schemas.connector import (
    ConnectorCreate,
    ConnectorRead,
    ConnectorSummary,
    ConnectorUpdate,
    OAuth2CallbackRequest,
    OAuth2ConfigResponse,
    OAuth2StartResponse,
    OperationCreate,
    OperationDraftTestRequest,
    OperationRead,
    OperationTestRequest,
    OperationTestResponse,
    OperationUpdate,
    OperationWithConnector,
)
from luna_core.services.connector import (
    ConnectorNotFound,
    DuplicateConnector,
    DuplicateOperation,
    OperationNotFound,
    OperationSnapshot,
    create_connector,
    create_operation,
    delete_connector,
    delete_operation,
    get_connector,
    get_operation,
    list_connectors,
    list_operations,
    update_connector,
    update_operation,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/connectors", tags=["connectors"])


def _connector_to_read(connector) -> ConnectorRead:  # noqa: ANN001
    oauth2_connected: bool | None = None
    if connector.auth_type == AuthType.oauth2:
        # `has_credentials` is True as soon as the user fills out client_id,
        # client_secret, token_url — but we only consider the conector
        # "connected" once the popup handshake actually produced an
        # access_token. Decrypt and peek; the encrypted blob is small.
        decrypted = decrypt_json(connector.credentials_encrypted) or {}
        oauth2_connected = bool(decrypted.get("access_token"))
    return ConnectorRead(
        id=connector.id,
        name=connector.name,
        description=connector.description,
        auth_type=connector.auth_type,
        base_url=connector.base_url,
        has_credentials=connector.credentials_encrypted is not None,
        oauth2_connected=oauth2_connected,
        is_active=connector.is_active,
        created_at=connector.created_at,
        updated_at=connector.updated_at,
    )


def _get_registry(request: Request) -> ConnectorRegistry | None:
    """Return the host-wired ConnectorRegistry, or None if absent.

    The router stays usable in test setups that don't bother wiring the
    registry — only the test endpoint hard-requires it.
    """
    return getattr(request.app.state, "connector_registry", None)


@router.post(
    "",
    response_model=ConnectorRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_permission("connectors:create")],
)
async def create(
    payload: ConnectorCreate,
    db: DBSession,
) -> ConnectorRead:
    try:
        connector = await create_connector(db, payload)
    except DuplicateConnector as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"connector with name '{exc}' already exists",
        ) from exc
    return _connector_to_read(connector)


@router.get(
    "",
    response_model=list[ConnectorRead],
    dependencies=[require_permission("connectors:read")],
)
async def index(db: DBSession) -> list[ConnectorRead]:
    connectors = await list_connectors(db)
    return [_connector_to_read(c) for c in connectors]


@router.get(
    "/{connector_id}",
    response_model=ConnectorRead,
    dependencies=[require_permission("connectors:read")],
)
async def show(connector_id: uuid.UUID, db: DBSession) -> ConnectorRead:
    try:
        connector = await get_connector(db, connector_id)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc
    return _connector_to_read(connector)


@router.patch(
    "/{connector_id}",
    response_model=ConnectorRead,
    dependencies=[require_permission("connectors:update")],
)
async def update(
    connector_id: uuid.UUID,
    payload: ConnectorUpdate,
    db: DBSession,
    request: Request,
) -> ConnectorRead:
    try:
        connector = await update_connector(db, connector_id, payload)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc

    # Re-sync the in-memory registry so subsequent execute() calls see the
    # new base_url / credentials / is_active state without a restart.
    registry = _get_registry(request)
    if registry is not None:
        operations = await list_operations(db, connector_id)
        # `register()` is the public idempotent path — passes a fresh
        # snapshot of the connector + its operations into the cache.
        registry.register(connector, list(operations))

    return _connector_to_read(connector)


@router.delete(
    "/{connector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("connectors:delete")],
)
async def destroy(
    connector_id: uuid.UUID,
    db: DBSession,
    request: Request,
) -> Response:
    registry = _get_registry(request)
    mcp_server = getattr(request.app.state, "mcp_server", None)
    mcp_builder = getattr(request.app.state, "mcp_builder", None)

    async def _on_deleted(cid: uuid.UUID, op_names: list[str]) -> None:
        if registry is not None:
            registry.unregister_connector(cid)
        if mcp_server is None or mcp_builder is None:
            return
        for name in op_names:
            try:
                mcp_builder.detach(mcp_server, name)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to detach MCP tool '%s' for deleted connector %s",
                    name,
                    cid,
                )

    try:
        await delete_connector(db, connector_id, on_deleted=_on_deleted)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/oauth2/config",
    response_model=OAuth2ConfigResponse,
)
async def oauth2_config() -> OAuth2ConfigResponse:
    """Public OAuth2 settings the dashboard needs to drive the popup flow.

    The callback URL is whatever the deployment registered with each OAuth2
    provider (Upwork, Slack, etc.). Surface it so admins can copy-paste into
    those provider consoles without guessing.
    """
    return OAuth2ConfigResponse(callback_url=settings.oauth2_callback_url)


@router.post(
    "/{connector_id}/oauth2/start",
    response_model=OAuth2StartResponse,
    dependencies=[require_permission("connectors:update")],
)
async def oauth2_start(
    connector_id: uuid.UUID,
    db: DBSession,
) -> OAuth2StartResponse:
    """Begin an OAuth2 authorization_code handshake for `connector_id`.

    Returns the URL the dashboard should open in a popup. The user logs
    into the IdP, approves, and the IdP redirects to our configured
    callback URL with `?code=...&state=...`. The dashboard's callback
    page then POSTs those to `/connectors/oauth2/callback`.
    """
    try:
        connector = await get_connector(db, connector_id)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc

    if connector.auth_type != AuthType.oauth2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="connector auth_type is not oauth2",
        )

    creds = decrypt_json(connector.credentials_encrypted) or {}
    authorize_url = creds.get("authorize_url")
    client_id = creds.get("client_id")
    if not authorize_url or not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "oauth2 connector is missing authorize_url and/or client_id "
                "in credentials — finish configuring it first"
            ),
        )

    state = make_state(connector.id)
    url = build_authorize_url(
        authorize_url=str(authorize_url),
        client_id=str(client_id),
        redirect_uri=settings.oauth2_callback_url,
        scope=creds.get("scope"),
        state=state,
    )
    return OAuth2StartResponse(authorize_url=url)


@router.post(
    "/oauth2/callback",
    response_model=ConnectorRead,
)
async def oauth2_callback(
    payload: OAuth2CallbackRequest,
    db: DBSession,
    request: Request,
) -> ConnectorRead:
    """Complete an OAuth2 handshake — exchange the `code` for tokens.

    The dashboard's `/connectors/oauth2/callback` page forwards the IdP's
    query params (`code`, `state`) here. We verify the state signature to
    recover the target conector id, swap the code for tokens at
    `token_url`, and persist the result encrypted.

    No app-level auth: the signed `state` JWT IS the authentication for
    this endpoint — it was minted by the auth-gated /oauth2/start, embeds
    the target connector_id, and expires in minutes. App-level auth would
    also fight cross-tab/new-tab flows where the Bearer in localStorage
    isn't necessarily hydrated when this fires.
    """
    try:
        connector_id = verify_state(payload.state)
    except OAuth2StateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    try:
        connector = await get_connector(db, connector_id)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc

    if connector.auth_type != AuthType.oauth2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="connector auth_type is not oauth2",
        )

    creds = decrypt_json(connector.credentials_encrypted) or {}
    token_url = creds.get("token_url")
    client_id = creds.get("client_id")
    client_secret = creds.get("client_secret")
    if not token_url or not client_id or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "oauth2 connector is missing token_url / client_id / "
                "client_secret — cannot complete the handshake"
            ),
        )

    try:
        patch = await exchange_code_for_tokens(
            token_url=str(token_url),
            client_id=str(client_id),
            client_secret=str(client_secret),
            code=payload.code,
            redirect_uri=settings.oauth2_callback_url,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    merged = {**creds, **patch}
    connector.credentials_encrypted = encrypt_json(merged)
    db.add(connector)
    await db.commit()
    await db.refresh(connector)

    # Re-sync registry so subsequent execute() calls see the new tokens
    # without waiting for a startup load_from_db.
    registry = _get_registry(request)
    if registry is not None:
        operations = await list_operations(db, connector_id)
        registry.register(connector, list(operations))

    return _connector_to_read(connector)


@router.get(
    "/operations/{operation_id}",
    response_model=OperationWithConnector,
    dependencies=[require_permission("connectors:read")],
)
async def show_operation(
    operation_id: uuid.UUID,
    db: DBSession,
) -> OperationWithConnector:
    try:
        operation = await get_operation(db, operation_id)
    except OperationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="operation not found"
        ) from exc
    connector = operation.connector
    return OperationWithConnector(
        id=operation.id,
        connector_id=operation.connector_id,
        name=operation.name,
        description=operation.description,
        method=operation.method,
        path=operation.path,
        input_schema=operation.input_schema,
        output_schema=operation.output_schema,
        is_active=operation.is_active,
        created_at=operation.created_at,
        connector=ConnectorSummary(
            id=connector.id,
            name=connector.name,
            description=connector.description,
            auth_type=connector.auth_type,
            base_url=connector.base_url,
            is_active=connector.is_active,
        ),
    )


@router.post(
    "/{connector_id}/operations",
    response_model=OperationRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_permission("connectors:create")],
)
async def add_operation(
    connector_id: uuid.UUID,
    payload: OperationCreate,
    db: DBSession,
    request: Request,
) -> OperationRead:
    # If the host wired an MCP server + builder into app.state, register the
    # newly-created operation as a live tool so clients see it on their next
    # tools/list call without restarting the server.
    mcp_server = getattr(request.app.state, "mcp_server", None)
    mcp_builder = getattr(request.app.state, "mcp_builder", None)

    async def _attach(op: Operation) -> None:
        if mcp_server is None or mcp_builder is None:
            return
        try:
            mcp_builder.attach_operation(mcp_server, op)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to attach operation '%s' (id=%s) to MCP server",
                op.name,
                op.id,
            )

    try:
        operation = await create_operation(
            db, connector_id, payload, on_created=_attach
        )
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc
    except DuplicateOperation as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"operation '{exc}' already exists for this connector",
        ) from exc
    return OperationRead.model_validate(operation)


@router.get(
    "/{connector_id}/operations",
    response_model=list[OperationRead],
    dependencies=[require_permission("connectors:read")],
)
async def list_ops(
    connector_id: uuid.UUID,
    db: DBSession,
) -> list[OperationRead]:
    try:
        operations = await list_operations(db, connector_id)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc
    return [OperationRead.model_validate(op) for op in operations]


@router.patch(
    "/{connector_id}/operations/{operation_id}",
    response_model=OperationRead,
    dependencies=[require_permission("connectors:update")],
)
async def update_op(
    connector_id: uuid.UUID,
    operation_id: uuid.UUID,
    payload: OperationUpdate,
    db: DBSession,
    request: Request,
) -> OperationRead:
    registry = _get_registry(request)
    mcp_server = getattr(request.app.state, "mcp_server", None)
    mcp_builder = getattr(request.app.state, "mcp_builder", None)

    async def _on_updated(previous: OperationSnapshot, current: Operation) -> None:
        # Keep the in-memory registry's view in sync so the next execute()
        # sees the patched path / method / schema.
        if registry is not None:
            registry.register(current.connector, [current])
        if mcp_server is None or mcp_builder is None:
            return
        # Renaming an operation needs the old MCP tool detached first so
        # the manager doesn't keep stale entries that point at this op.
        if previous.name != current.name:
            try:
                mcp_builder.detach(mcp_server, previous.name)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to detach old MCP tool '%s' on rename",
                    previous.name,
                )
        try:
            mcp_builder.attach_operation(mcp_server, current)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to re-attach MCP tool '%s' after update",
                current.name,
            )

    try:
        operation = await update_operation(
            db, operation_id, payload, on_updated=_on_updated
        )
    except OperationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="operation not found"
        ) from exc
    except DuplicateOperation as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"operation '{exc}' already exists for this connector",
        ) from exc

    if operation.connector_id != connector_id:
        # URL/body mismatch — the operation exists but belongs to a different
        # connector. Treat as 404 so the path acts as a true scoping check.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="operation not found for this connector",
        )

    return OperationRead.model_validate(operation)


@router.delete(
    "/{connector_id}/operations/{operation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("connectors:delete")],
)
async def destroy_op(
    connector_id: uuid.UUID,
    operation_id: uuid.UUID,
    db: DBSession,
    request: Request,
) -> Response:
    # Pre-check the URL scoping before delete so we can return a clean 404
    # without first running the delete + rolling back.
    try:
        operation = await get_operation(db, operation_id)
    except OperationNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="operation not found"
        ) from exc
    if operation.connector_id != connector_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="operation not found for this connector",
        )

    registry = _get_registry(request)
    mcp_server = getattr(request.app.state, "mcp_server", None)
    mcp_builder = getattr(request.app.state, "mcp_builder", None)

    async def _on_deleted(snapshot: OperationSnapshot) -> None:
        if registry is not None:
            registry.unregister_operation(snapshot.id)
        if mcp_server is None or mcp_builder is None:
            return
        try:
            mcp_builder.detach(mcp_server, snapshot.name)
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to detach MCP tool '%s' for deleted op %s",
                snapshot.name,
                snapshot.id,
            )

    await delete_operation(db, operation_id, on_deleted=_on_deleted)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{connector_id}/operations/test-draft",
    response_model=OperationTestResponse,
    dependencies=[require_permission("connectors:test")],
)
async def test_op_draft(
    connector_id: uuid.UUID,
    payload: OperationDraftTestRequest,
    db: DBSession,
    request: Request,
) -> OperationTestResponse:
    """Execute an unsaved operation definition against this connector.

    Used by the Operation create flow's Test button — the form sends the
    full draft (method/path/parameters/fixed_headers/fixed_body) plus an
    input payload; the registry runs the real HTTP call using the
    connector's stored credentials without persisting anything.
    """
    registry = _get_registry(request)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="connector registry not wired into this app",
        )

    try:
        connector = await get_connector(db, connector_id)
    except ConnectorNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="connector not found"
        ) from exc

    # ParameterDef → JSON-compatible dicts (the registry helper expects the
    # storage shape, same as what comes off a saved Operation row).
    parameters_payload = [
        p.model_dump(mode="json", by_alias=True) for p in payload.parameters
    ]

    try:
        result = await registry.perform_draft(
            connector=connector,
            method=payload.method,
            path=payload.path,
            parameters=parameters_payload,
            fixed_headers=dict(payload.fixed_headers),
            fixed_body=payload.fixed_body,
            retry_policy=(
                payload.retry_policy.model_dump(mode="json")
                if payload.retry_policy is not None
                else None
            ),
            input_data=payload.input,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return OperationTestResponse(
        ok=result.ok,
        status_code=result.status_code,
        latency_ms=result.latency_ms,
        request_method=result.request_method,
        request_url=result.request_url,
        response=result.response,
        error=result.error,
    )


@router.post(
    "/{connector_id}/operations/{operation_id}/test",
    response_model=OperationTestResponse,
    dependencies=[require_permission("connectors:test")],
)
async def test_op(
    connector_id: uuid.UUID,
    operation_id: uuid.UUID,
    payload: OperationTestRequest,
    db: DBSession,
    request: Request,
) -> OperationTestResponse:
    """Execute an operation with caller-supplied input and return the raw
    response (including 4xx/5xx) so the UI can render it for inspection.

    The operation is loaded into the registry on demand, so inactive
    connectors / operations can still be tested before being switched on.
    """
    registry = _get_registry(request)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="connector registry not wired into this app",
        )

    try:
        operation = await registry.ensure_registered(db, operation_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="operation not found"
        ) from exc

    if operation.connector_id != connector_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="operation not found for this connector",
        )

    try:
        result = await registry.perform(operation_id, payload.input)
    except Exception as exc:  # noqa: BLE001
        # Programmer errors (missing path param, unknown method) bubble up
        # as 400 so the UI can surface "your input is wrong" distinct from
        # "the upstream said 500".
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return OperationTestResponse(
        ok=result.ok,
        status_code=result.status_code,
        latency_ms=result.latency_ms,
        request_method=result.request_method,
        request_url=result.request_url,
        response=result.response,
        error=result.error,
    )
