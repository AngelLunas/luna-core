from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, Response, status

from luna_core.core.dependencies import (
    CurrentUser,
    DBSession,
    RedisClient,
    require_permission,
)
from luna_core.schemas.agent import (
    AgentCreate,
    AgentOperationAssign,
    AgentOperationRead,
    AgentRead,
    AgentSystemToolGrantAssign,
    AgentSystemToolGrantRead,
    AgentUpdate,
    InstructionsPreviewIn,
    InstructionsPreviewOut,
)
from luna_core.services.agent import (
    AgentNotFound,
    DuplicateAgent,
    assign_operations,
    assign_system_tools,
    create_agent,
    delete_agent,
    get_agent,
    list_agents,
    list_assigned_operations,
    list_assigned_system_tools,
    preview_instructions,
    update_agent,
)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post(
    "/preview-instructions",
    response_model=InstructionsPreviewOut,
    dependencies=[require_permission("agents:read")],
)
async def preview(
    payload: InstructionsPreviewIn,
    db: DBSession,
    redis: RedisClient,
    user: CurrentUser,
) -> InstructionsPreviewOut:
    """Resolve an agent's instructions against live context loaders.

    Used by the editor's preview button to show the user the exact text
    the agent would receive at run time. Mirrors the engine's template
    substitution and replaces failing references with `[unresolved <name>]`
    so the gaps are visible instead of silently empty.
    """
    return await preview_instructions(
        db=db,
        redis=redis,
        user=user,
        instructions=payload.instructions,
        source_bindings=payload.source_bindings,
    )


@router.post(
    "",
    response_model=AgentRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_permission("agents:create")],
)
async def create(
    payload: AgentCreate, db: DBSession
) -> AgentRead:
    try:
        agent = await create_agent(db, payload)
    except DuplicateAgent as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"agent with name '{exc}' already exists",
        ) from exc
    return AgentRead.model_validate(agent)


@router.get(
    "",
    response_model=list[AgentRead],
    dependencies=[require_permission("agents:read")],
)
async def index(
    db: DBSession,
    include: list[str] = Query(default_factory=list),
) -> list[AgentRead]:
    """List every agent. ``?include=system_tools`` mirrors the detail
    endpoint and eagerly attaches each agent's system-tool grants so a
    single round trip can populate UI surfaces (e.g. the flow editor's
    agent catalog) that branch on grants."""
    with_system_tools = "system_tools" in include
    agents = await list_agents(db, with_system_tool_grants=with_system_tools)
    reads: list[AgentRead] = []
    for agent in agents:
        read = AgentRead.model_validate(agent)
        if with_system_tools:
            read.system_tools = [
                AgentSystemToolGrantRead.model_validate(g)
                for g in agent.agent_system_tool_grants
            ]
        reads.append(read)
    return reads


@router.get(
    "/{agent_id}",
    response_model=AgentRead,
    dependencies=[require_permission("agents:read")],
)
async def detail(
    agent_id: uuid.UUID,
    db: DBSession,
    include: list[str] = Query(default_factory=list),
) -> AgentRead:
    with_operations = "operations" in include
    with_system_tools = "system_tools" in include
    try:
        agent = await get_agent(
            db,
            agent_id,
            with_operations=with_operations,
            with_system_tool_grants=with_system_tools,
        )
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    read = AgentRead.model_validate(agent)
    if with_operations:
        read.operations = [
            AgentOperationRead.model_validate(a) for a in agent.agent_operations
        ]
    if with_system_tools:
        read.system_tools = [
            AgentSystemToolGrantRead.model_validate(g)
            for g in agent.agent_system_tool_grants
        ]
    return read


@router.put(
    "/{agent_id}",
    response_model=AgentRead,
    dependencies=[require_permission("agents:update")],
)
async def update(
    agent_id: uuid.UUID,
    payload: AgentUpdate,
    db: DBSession,
) -> AgentRead:
    try:
        agent = await update_agent(db, agent_id, payload)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    return AgentRead.model_validate(agent)


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("agents:delete")],
)
async def destroy(agent_id: uuid.UUID, db: DBSession) -> Response:
    try:
        await delete_agent(db, agent_id)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{agent_id}/operations",
    response_model=list[AgentOperationRead],
    dependencies=[require_permission("agents:update")],
)
async def assign(
    agent_id: uuid.UUID,
    payload: AgentOperationAssign,
    db: DBSession,
) -> list[AgentOperationRead]:
    try:
        assignments = await assign_operations(db, agent_id, payload.operation_ids)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return [AgentOperationRead.model_validate(a) for a in assignments]


@router.get(
    "/{agent_id}/operations",
    response_model=list[AgentOperationRead],
    dependencies=[require_permission("agents:read")],
)
async def list_assigned(
    agent_id: uuid.UUID, db: DBSession
) -> list[AgentOperationRead]:
    try:
        assignments = await list_assigned_operations(db, agent_id)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    return [AgentOperationRead.model_validate(a) for a in assignments]


@router.post(
    "/{agent_id}/system-tools",
    response_model=list[AgentSystemToolGrantRead],
    dependencies=[require_permission("agents:update")],
)
async def assign_system_tools_endpoint(
    agent_id: uuid.UUID,
    payload: AgentSystemToolGrantAssign,
    db: DBSession,
) -> list[AgentSystemToolGrantRead]:
    """Bulk-replace the agent's system-tool grants.

    Empty ``tool_names`` clears every existing grant. Names must match
    catalog tools currently registered in the in-process registry; any
    unknown name aborts the whole request with 400 so the user knows
    immediately which grant failed instead of getting a silently-empty
    assignment.
    """
    try:
        grants = await assign_system_tools(db, agent_id, payload.tool_names)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    return [AgentSystemToolGrantRead.model_validate(g) for g in grants]


@router.get(
    "/{agent_id}/system-tools",
    response_model=list[AgentSystemToolGrantRead],
    dependencies=[require_permission("agents:read")],
)
async def list_assigned_system_tools_endpoint(
    agent_id: uuid.UUID, db: DBSession
) -> list[AgentSystemToolGrantRead]:
    try:
        grants = await list_assigned_system_tools(db, agent_id)
    except AgentNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="agent not found"
        ) from exc
    return [AgentSystemToolGrantRead.model_validate(g) for g in grants]
