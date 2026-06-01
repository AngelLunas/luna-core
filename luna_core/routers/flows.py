from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Response, WebSocket, status

from luna_core.core.dependencies import (
    CurrentUser,
    DBSession,
    RedisClient,
    get_redis_client,
    require_permission,
)
from luna_core.engine.emitter import flow_run_channel
from luna_core.engine.websocket import WebSocketManager
from luna_core.schemas.flow import (
    FlowCreate,
    FlowDefinition,
    FlowRead,
    FlowRunRead,
    FlowRunTrigger,
    FlowUpdate,
    SchedulePreviewIn,
    SchedulePreviewOut,
)
from luna_core.services.flow import (
    DuplicateFlow,
    FlowDefinitionInvalid,
    FlowInputValidationError,
    FlowNotFound,
    create_flow,
    create_flow_run,
    delete_flow,
    flow_for_user,
    get_flow,
    list_flow_runs,
    list_flows,
    update_flow,
    validate_definition,
)
from pydantic import BaseModel
from luna_core.services.scheduling import preview_next_runs
from luna_core.tasks import run_flow_task

router = APIRouter(prefix="/flows", tags=["flows"])


_flow_ws_manager: WebSocketManager | None = None


def get_flow_ws_manager() -> WebSocketManager:
    global _flow_ws_manager
    if _flow_ws_manager is None:
        _flow_ws_manager = WebSocketManager(get_redis_client(), flow_run_channel)
    return _flow_ws_manager


@router.post(
    "",
    response_model=FlowRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_permission("flows:create")],
)
async def create(
    payload: FlowCreate, db: DBSession, current_user: CurrentUser
) -> FlowRead:
    try:
        flow = await create_flow(db, payload, user_id=current_user.id)
    except DuplicateFlow as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"flow with name '{exc}' already exists",
        ) from exc
    return FlowRead.model_validate(flow_for_user(flow, current_user.id))


@router.get(
    "",
    response_model=list[FlowRead],
    dependencies=[require_permission("flows:read")],
)
async def index(db: DBSession, current_user: CurrentUser) -> list[FlowRead]:
    flows = await list_flows(db)
    return [
        FlowRead.model_validate(flow_for_user(f, current_user.id)) for f in flows
    ]


@router.get(
    "/{flow_id}",
    response_model=FlowRead,
    dependencies=[require_permission("flows:read")],
)
async def detail(
    flow_id: uuid.UUID, db: DBSession, current_user: CurrentUser
) -> FlowRead:
    try:
        flow = await get_flow(db, flow_id)
    except FlowNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="flow not found"
        ) from exc
    return FlowRead.model_validate(flow_for_user(flow, current_user.id))


class FlowValidateIn(BaseModel):
    definition: FlowDefinition


class FlowValidateOut(BaseModel):
    ok: bool
    errors: list[str] = []


@router.post(
    "/validate",
    response_model=FlowValidateOut,
    dependencies=[require_permission("flows:read")],
)
async def validate(payload: FlowValidateIn) -> FlowValidateOut:
    """Cheap pre-flight for the editor: Pydantic has already accepted the
    shape, so this only runs the cross-field rules in
    ``services.flow.validate_definition``. Returns 200 with ``ok=False`` for
    structural violations instead of 4xx so the editor can render the errors
    panel without treating it as an HTTP failure.
    """
    try:
        validate_definition(payload.definition)
    except FlowDefinitionInvalid as exc:
        return FlowValidateOut(ok=False, errors=exc.errors)
    return FlowValidateOut(ok=True)


@router.post(
    "/preview-schedule",
    response_model=SchedulePreviewOut,
    dependencies=[require_permission("flows:read")],
)
async def preview_schedule(payload: SchedulePreviewIn) -> SchedulePreviewOut:
    try:
        next_runs = preview_next_runs(payload.cron, payload.tz, payload.count)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return SchedulePreviewOut(next_runs=next_runs)


@router.put("/{flow_id}", response_model=FlowRead)
async def update(
    flow_id: uuid.UUID,
    payload: FlowUpdate,
    db: DBSession,
    current_user: CurrentUser,
) -> FlowRead:
    try:
        flow = await update_flow(db, flow_id, payload, user_id=current_user.id)
    except FlowNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="flow not found"
        ) from exc
    return FlowRead.model_validate(flow_for_user(flow, current_user.id))


@router.delete(
    "/{flow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("flows:delete")],
)
async def remove(flow_id: uuid.UUID, db: DBSession) -> Response:
    try:
        await delete_flow(db, flow_id)
    except FlowNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="flow not found"
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{flow_id}/runs",
    response_model=list[FlowRunRead],
    dependencies=[require_permission("flows:read")],
)
async def runs(
    flow_id: uuid.UUID,
    db: DBSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[FlowRunRead]:
    try:
        items = await list_flow_runs(db, flow_id, limit=limit, offset=offset)
    except FlowNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="flow not found"
        ) from exc
    return [FlowRunRead.model_validate(r) for r in items]


@router.post(
    "/{flow_id}/run",
    response_model=FlowRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger(
    flow_id: uuid.UUID,
    db: DBSession,
    current_user: CurrentUser,
    redis: RedisClient,
    payload: FlowRunTrigger | None = None,
) -> FlowRunRead:
    # Stamp the triggering user onto the payload so the runner can expose it
    # as state.trigger.user_id (used by id_implicit context sources like `user`).
    if payload is None:
        payload = FlowRunTrigger()
    metadata = dict(payload.metadata or {})
    metadata.setdefault("user_id", str(current_user.id))
    payload = payload.model_copy(update={"metadata": metadata})

    try:
        run = await create_flow_run(db, flow_id, payload, redis=redis)
    except FlowNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="flow not found"
        ) from exc
    except FlowInputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "invalid flow inputs", "errors": exc.errors},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # Hand the worker the *existing* run id — trigger payload, inputs and
    # metadata are already persisted on the row. The worker does not create
    # a second FlowRun; doing so was the root cause of pending-forever rows
    # in the UI while the worker silently executed a different one.
    run_flow_task.delay(str(run.id))
    return FlowRunRead.model_validate(run)


@router.websocket("/{flow_id}/stream")
async def stream(websocket: WebSocket, flow_id: uuid.UUID) -> None:
    """Live FlowRun lifecycle updates for a single flow.

    Frames are `{event: "run_created" | "run_status_changed", run: FlowRunRead}`.
    Subscribers upsert into their local runs list keyed by `run.id`.
    """
    manager = get_flow_ws_manager()
    await manager.connect(flow_id, websocket)
