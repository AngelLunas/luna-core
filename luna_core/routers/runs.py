from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, WebSocket, status

from luna_core.core.config import settings
from luna_core.core.dependencies import CurrentUser, DBSession, RedisClient, get_redis_client
from luna_core.engine.emitter import EventEmitter
from luna_core.engine.websocket import WebSocketManager
from luna_core.llm.base import abort_key
from luna_core.models.event import RunEventType
from luna_core.models.flow import FlowRunStatus
from luna_core.schemas.event import AgentMessageRead, ResumeRequest, RunEventRead
from luna_core.schemas.flow import FlowRunRead
from luna_core.services.flow import (
    FlowRunNotFound,
    RunNotTerminal,
    build_run_stream_snapshot,
    clear_run_data,
    get_flow_run,
    list_run_events,
    list_run_messages,
    set_run_status,
)
from luna_core.tasks import resume_flow_task

router = APIRouter(prefix="/runs", tags=["runs"])

_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    global _ws_manager
    if _ws_manager is None:
        redis_client = get_redis_client()
        _ws_manager = WebSocketManager(
            redis_client,
            snapshot_fn=lambda run_id: build_run_stream_snapshot(
                redis_client, run_id
            ),
        )
    return _ws_manager


@router.get("/{run_id}", response_model=FlowRunRead)
async def detail(
    run_id: uuid.UUID, db: DBSession, _: CurrentUser
) -> FlowRunRead:
    try:
        run = await get_flow_run(db, run_id)
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc
    return FlowRunRead.model_validate(run)


@router.get("/{run_id}/events", response_model=list[RunEventRead])
async def events(
    run_id: uuid.UUID,
    db: DBSession,
    redis: RedisClient,
    _: CurrentUser,
    since_sequence: int | None = None,
    iteration_id: str | None = None,
) -> list[RunEventRead]:
    """Historical event stream for a run.

    Filters:
    - ``since_sequence``: only events strictly greater than this sequence
      (reconnect backfill).
    - ``iteration_id``: only events tagged with this iteration_id —
      used by the dashboard when expanding one iteration block so the
      panel backfills just that iteration's history instead of the full
      node timeline.
    """
    try:
        items = await list_run_events(
            db,
            run_id,
            since_sequence=since_sequence,
            redis=redis,
            iteration_id=iteration_id,
        )
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc
    return [RunEventRead.model_validate(e) for e in items]


@router.get("/{run_id}/messages", response_model=list[AgentMessageRead])
async def messages(
    run_id: uuid.UUID,
    db: DBSession,
    _: CurrentUser,
    node_id: str | None = None,
) -> list[AgentMessageRead]:
    try:
        items = await list_run_messages(db, run_id, node_id)
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc
    return [AgentMessageRead.model_validate(m) for m in items]


@router.post(
    "/{run_id}/resume",
    response_model=FlowRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume(
    run_id: uuid.UUID,
    payload: ResumeRequest,
    db: DBSession,
    _: CurrentUser,
) -> FlowRunRead:
    try:
        run = await get_flow_run(db, run_id)
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc

    resume_flow_task.delay(str(run_id), payload.response, payload.metadata)
    return FlowRunRead.model_validate(run)


@router.post(
    "/{run_id}/abort",
    response_model=FlowRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def abort(
    run_id: uuid.UUID,
    db: DBSession,
    redis: RedisClient,
    _: CurrentUser,
) -> FlowRunRead:
    try:
        run = await get_flow_run(db, run_id)
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc

    if run.status in (FlowRunStatus.completed, FlowRunStatus.failed):
        return FlowRunRead.model_validate(run)

    await redis.set(
        abort_key(run_id), "1", ex=settings.run_abort_key_ttl_seconds
    )
    emitter = EventEmitter(db, redis, run_id)
    await emitter.emit(
        RunEventType.flow_failed,
        node_id=None,
        payload={"reason": "aborted"},
    )
    updated = await set_run_status(
        db, run_id, FlowRunStatus.failed, state=run.state, redis=redis
    )
    return FlowRunRead.model_validate(updated)


@router.delete("/{run_id}", response_model=FlowRunRead)
async def clear(
    run_id: uuid.UUID, db: DBSession, _: CurrentUser
) -> FlowRunRead:
    """Soft-compact a terminal run: purge events + messages, keep the row.

    Resources created during the run (jobs, media, etc.) are unaffected —
    none of them carry a FK back to flow_runs.
    """
    try:
        run = await clear_run_data(db, run_id)
    except FlowRunNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="run not found"
        ) from exc
    except RunNotTerminal as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return FlowRunRead.model_validate(run)


@router.websocket("/{run_id}/stream")
async def stream(websocket: WebSocket, run_id: uuid.UUID) -> None:
    manager = get_ws_manager()
    await manager.connect(run_id, websocket)
