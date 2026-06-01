from __future__ import annotations

import uuid
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from luna_core.engine.template_paths import resolve_path
from luna_core.mcp.system_tools import get_default_registry
from luna_core.models.agent import Agent, AgentOperation, AgentSystemToolGrant
from luna_core.models.connector import Operation
from luna_core.models.user import User
from luna_core.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    InstructionsPreviewOut,
    InstructionsPreviewSourceDiag,
)
from luna_core.services.context_sources import (
    SourceLoadContext,
    UnknownSourceError,
    extract_context_sources,
    get_context_source,
)


class AgentNotFound(LookupError):
    pass


class DuplicateAgent(ValueError):
    pass


async def create_agent(db: AsyncSession, payload: AgentCreate) -> Agent:
    agent = Agent(
        name=payload.name,
        role=payload.role,
        instructions=payload.instructions,
        llm_provider_id=payload.llm_provider_id,
        model=payload.model,
        temperature=payload.temperature,
        output_schema=payload.output_schema,
        required_sources=extract_context_sources(payload.instructions),
    )
    db.add(agent)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateAgent(payload.name) from exc
    await db.refresh(agent)
    return agent


async def list_agents(
    db: AsyncSession,
    *,
    with_system_tool_grants: bool = False,
) -> list[Agent]:
    """Return every agent in creation order.

    ``with_system_tool_grants`` eagerly loads each agent's
    ``AgentSystemToolGrant`` rows so callers can decide downstream
    UX based on whether a tool is granted (e.g. the flow editor only
    surfaces the stash editor for agents that actually have
    ``stash_records`` granted). Opt-in to avoid paying the join cost
    on the hot list path that doesn't need it.
    """
    stmt = select(Agent).order_by(Agent.created_at.desc())
    if with_system_tool_grants:
        stmt = stmt.options(selectinload(Agent.agent_system_tool_grants))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_agent(
    db: AsyncSession,
    agent_id: uuid.UUID,
    *,
    with_operations: bool = False,
    with_system_tool_grants: bool = False,
) -> Agent:
    # Compose selectinload options based on what the caller asked for;
    # nothing is eagerly fetched by default so the cheap detail path
    # stays a single PK lookup.
    if with_operations or with_system_tool_grants:
        load_options: list[Any] = []
        if with_operations:
            load_options.append(
                selectinload(Agent.agent_operations).selectinload(
                    AgentOperation.operation
                )
            )
        if with_system_tool_grants:
            load_options.append(selectinload(Agent.agent_system_tool_grants))
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id).options(*load_options)
        )
        agent = result.scalar_one_or_none()
    else:
        agent = await db.get(Agent, agent_id)
    if agent is None:
        raise AgentNotFound(str(agent_id))
    return agent


async def delete_agent(db: AsyncSession, agent_id: uuid.UUID) -> None:
    agent = await get_agent(db, agent_id)
    await db.delete(agent)
    await db.commit()


async def update_agent(
    db: AsyncSession, agent_id: uuid.UUID, payload: AgentUpdate
) -> Agent:
    agent = await get_agent(db, agent_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(agent, field, value)
    # Always recompute required_sources from the (possibly updated) instructions
    # so the persisted list stays in sync with what the template references.
    agent.required_sources = extract_context_sources(agent.instructions)
    await db.commit()
    await db.refresh(agent)
    return agent


async def assign_operations(
    db: AsyncSession, agent_id: uuid.UUID, operation_ids: list[uuid.UUID]
) -> list[AgentOperation]:
    await get_agent(db, agent_id)

    # validate operation IDs exist
    existing = await db.execute(
        select(Operation.id).where(Operation.id.in_(operation_ids))
    )
    existing_ids = {row[0] for row in existing.all()}
    missing = set(operation_ids) - existing_ids
    if missing:
        raise ValueError(f"operations not found: {sorted(str(m) for m in missing)}")

    # wipe and reinsert (idempotent assignment)
    await db.execute(
        delete(AgentOperation).where(AgentOperation.agent_id == agent_id)
    )
    assignments = [
        AgentOperation(agent_id=agent_id, operation_id=op_id)
        for op_id in operation_ids
    ]
    db.add_all(assignments)
    await db.commit()
    return await list_assigned_operations(db, agent_id)


async def list_assigned_operations(
    db: AsyncSession, agent_id: uuid.UUID
) -> list[AgentOperation]:
    await get_agent(db, agent_id)
    result = await db.execute(
        select(AgentOperation)
        .where(AgentOperation.agent_id == agent_id)
        .options(selectinload(AgentOperation.operation))
    )
    return list(result.scalars().all())


async def assign_system_tools(
    db: AsyncSession, agent_id: uuid.UUID, tool_names: list[str]
) -> list[AgentSystemToolGrant]:
    """Replace the agent's system-tool grants with ``tool_names``.

    Mirrors ``assign_operations``: idempotent bulk replace (wipe then
    insert). Validates that every requested name corresponds to a
    catalog tool currently registered in the in-process system tool
    registry — granting an unknown tool would silently expand the
    agent's tool list to nothing useful, so we reject loudly instead.

    Passing an empty list clears all existing grants — the user-facing
    "unassign everything" path.
    """
    await get_agent(db, agent_id)

    registry = get_default_registry()
    known_names = {t.name for t in registry.list_catalog()}
    requested = list(tool_names)
    missing = [n for n in requested if n not in known_names]
    if missing:
        raise ValueError(f"system tools not in catalog: {sorted(set(missing))}")
    # Deduplicate while preserving the caller's order so the grants
    # come out in a stable, debuggable sequence on subsequent listings.
    seen: set[str] = set()
    deduped: list[str] = []
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    await db.execute(
        delete(AgentSystemToolGrant).where(
            AgentSystemToolGrant.agent_id == agent_id
        )
    )
    grants = [
        AgentSystemToolGrant(agent_id=agent_id, tool_name=name)
        for name in deduped
    ]
    db.add_all(grants)
    await db.commit()
    return await list_assigned_system_tools(db, agent_id)


async def list_assigned_system_tools(
    db: AsyncSession, agent_id: uuid.UUID
) -> list[AgentSystemToolGrant]:
    """Return the agent's current system-tool grants.

    Dangling grants (names not in the current catalog — possible if a
    tool was removed from the code after being granted) are returned as
    is; the runner's name-based filter will naturally drop them at
    dispatch time. The UI should highlight them so the user can clean up.
    """
    await get_agent(db, agent_id)
    result = await db.execute(
        select(AgentSystemToolGrant).where(
            AgentSystemToolGrant.agent_id == agent_id
        )
    )
    return list(result.scalars().all())


async def preview_instructions(
    *,
    db: AsyncSession,
    redis: Redis,
    user: User,
    instructions: str,
    source_bindings: dict[str, str] | None = None,
) -> InstructionsPreviewOut:
    """Render an agent's instruction template using live context loaders.

    Resolution mirrors the engine's `_format_template` so the preview matches
    what the agent would see at run time. Each referenced source is loaded
    against a synthetic state that stamps `state.trigger.user_id` from the
    authenticated user — matching how the trigger router seeds runs in
    production.

    Sources that fail to load (loader exception, missing binding, unknown
    source) are reported in `diagnostics` and their references are replaced
    with `[unresolved <name>]` markers so the rest of the prompt still
    renders. We never raise from a loader failure: the preview's job is to
    show the user what's working and what isn't.
    """
    bindings = source_bindings or {}
    required = extract_context_sources(instructions)
    diagnostics: list[InstructionsPreviewSourceDiag] = []

    # Synthetic state mirrors the engine's runtime: implicit-id sources
    # find their target via state.trigger.user_id; explicit-id sources
    # use the binding-id passed in.
    state: dict[str, Any] = {
        "trigger": {"user_id": str(user.id)},
        "inputs": {},
        "outputs": {},
    }
    load_ctx = SourceLoadContext(db=db, redis=redis, state=state)

    loaded_context: dict[str, Any] = {}
    for name in required:
        try:
            source = get_context_source(name)
        except UnknownSourceError:
            diagnostics.append(
                InstructionsPreviewSourceDiag(
                    name=name,
                    status="unknown-source",
                    detail=f"no source registered under {name!r}",
                )
            )
            continue

        if source.id_implicit:
            source_id: str | None = None
        else:
            source_id = bindings.get(name)
            if not source_id:
                diagnostics.append(
                    InstructionsPreviewSourceDiag(
                        name=name,
                        status="missing-binding",
                        detail=(
                            f"source {name!r} needs an explicit id; pass it in "
                            "`source_bindings`"
                        ),
                    )
                )
                continue

        try:
            data = await source.loader(load_ctx, source_id)
        except Exception as exc:  # noqa: BLE001 — loader failures are domain errors
            diagnostics.append(
                InstructionsPreviewSourceDiag(
                    name=name,
                    status="loader-error",
                    detail=str(exc),
                )
            )
            continue

        if not isinstance(data, dict):
            diagnostics.append(
                InstructionsPreviewSourceDiag(
                    name=name,
                    status="loader-error",
                    detail=f"loader returned {type(data).__name__}, expected dict",
                )
            )
            continue

        loaded_context[name] = data
        diagnostics.append(
            InstructionsPreviewSourceDiag(name=name, status="ok")
        )

    state["context"] = loaded_context

    resolved = _render_template(instructions, state, missing=loaded_context)
    return InstructionsPreviewOut(
        resolved=resolved,
        required_sources=required,
        diagnostics=diagnostics,
    )


def _render_template(
    text: str, state: dict[str, Any], *, missing: dict[str, Any]
) -> str:
    """Mirror of `engine.nodes._format_template`, but emits an explicit
    `[unresolved <source>]` marker for `${context.<x>...}` refs whose
    source didn't load — so the preview surfaces the gap instead of
    silently producing an empty string the way the engine does.

    The engine's behavior at run time is "missing → empty string"; here
    we trade a little fidelity for a much clearer preview.
    """
    result = text
    cursor = 0
    while True:
        start = result.find("${", cursor)
        if start == -1:
            return result
        end = result.find("}", start)
        if end == -1:
            return result
        path = result[start + 2 : end]
        replacement: str
        if path.startswith("context."):
            # `context` then either `<source>` or `<source>.<rest>` — split
            # on the first dot after the prefix to find the source name.
            tail = path[len("context.") :]
            source_name = tail.split(".", 1)[0] if "." in tail else tail
            if source_name not in missing:
                replacement = f"[unresolved {source_name}]"
            else:
                replacement = _stringify(_dot_get(state, path))
        else:
            replacement = _stringify(_dot_get(state, path))
        result = result[:start] + replacement + result[end + 1 :]
        cursor = start + len(replacement)


def _dot_get(state: dict[str, Any], path: str) -> Any:
    # Walks dotted paths with optional ``[*]`` segments — mirrors what the
    # engine runtime does so preview and runtime agree on shape.
    return resolve_path(state, path)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
