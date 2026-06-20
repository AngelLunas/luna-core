"""Conversation router — generic AI-chat surface for host apps.

CRUD over the conversation primitive + a ``send_message`` turn driven by
``ChatRunner`` + a ``/stream`` WebSocket. The CRUD endpoints are self-contained;
the turn endpoints need chat infrastructure the **host wires into ``app.state``**,
so luna-core stays free of any host's agent/tool/metering choices:

    app.state.chat_runner: ChatRunner                      # required for turns
    app.state.chat_agent_resolver:                         # required for turns
        Callable[[AsyncSession, Conversation], Awaitable[Agent]]
    app.state.chat_metering_hook:                          # optional
        Callable[[MeteringContext], Awaitable[None]] | None

The resolver picks which agent answers a given conversation (a host with one
orchestrator just returns it); the metering hook, if set, runs after each turn.
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, status
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.dependencies import (
    CurrentUser,
    DBSession,
    RedisClient,
    get_redis_client,
)
from luna_core.engine.agent import SuspendedForApproval
from luna_core.engine.chat import ChatRunner
from luna_core.engine.websocket import WebSocketManager
from luna_core.models.agent import Agent
from luna_core.models.conversation import Conversation
from luna_core.models.tool_approval import ToolApprovalStatus
from luna_core.schemas.conversation import (
    ConversationCreate,
    ConversationMessageRead,
    ConversationRead,
    ConversationUpdate,
    SendMessageRequest,
    SendMessageResponse,
)
from luna_core.schemas.tool_approval import ToolApprovalDecision, ToolApprovalRead
from luna_core.services.conversation import (
    ConversationNotFound,
    create_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    update_conversation_title,
)
from luna_core.services.tool_approval import (
    ToolApprovalNotFound,
    ToolApprovalNotPending,
    count_pending,
    decide,
    get_approval,
    list_approvals,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

AgentResolver = Callable[[AsyncSession, Conversation], Awaitable[Agent]]

# Multi-agent handoff: after a turn, the resolver is consulted again; if a tool
# changed which agent is active (e.g. a terminal ``route_to_*`` tool), the same
# request continues with the new agent on the existing history. Capped so a
# misbehaving pair of agents can't ping-pong forever. A single-agent host never
# trips this — the resolver returns the same agent and the loop exits at once.
MAX_HANDOFF_HOPS = 4


@dataclass
class MeteringContext:
    """Passed to the optional ``chat_metering_hook`` after each turn. ``usage``
    carries the turn's token counts once luna-core exposes them (§5.2); until
    then it is ``None``."""

    user_id: uuid.UUID | None
    conversation_id: uuid.UUID
    agent: Agent
    output: Any
    usage: dict | None = None


# --- CRUD ---
@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create(
    payload: ConversationCreate, db: DBSession, user: CurrentUser
) -> ConversationRead:
    conversation = await create_conversation(db, user_id=user.id, title=payload.title)
    return ConversationRead.model_validate(conversation)


@router.get("", response_model=list[ConversationRead])
async def index(db: DBSession, user: CurrentUser) -> list[ConversationRead]:
    rows = await list_conversations(db, user_id=user.id)
    return [ConversationRead.model_validate(c) for c in rows]


@router.get("/{conversation_id}", response_model=ConversationRead)
async def detail(
    conversation_id: uuid.UUID, db: DBSession, user: CurrentUser
) -> ConversationRead:
    conversation = await _get_owned(db, conversation_id, user.id)
    return ConversationRead.model_validate(conversation)


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update(
    conversation_id: uuid.UUID,
    payload: ConversationUpdate,
    db: DBSession,
    user: CurrentUser,
) -> ConversationRead:
    try:
        conversation = await update_conversation_title(
            db, conversation_id, title=payload.title, user_id=user.id
        )
    except ConversationNotFound as exc:
        raise _not_found(conversation_id) from exc
    return ConversationRead.model_validate(conversation)


@router.get(
    "/{conversation_id}/messages", response_model=list[ConversationMessageRead]
)
async def messages(
    conversation_id: uuid.UUID, db: DBSession, user: CurrentUser
) -> list[ConversationMessageRead]:
    try:
        rows = await list_messages(db, conversation_id, user_id=user.id)
    except ConversationNotFound as exc:
        raise _not_found(conversation_id) from exc
    return [ConversationMessageRead.model_validate(m) for m in rows]


# --- turn (one assistant message) ---
@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send(
    conversation_id: uuid.UUID,
    payload: SendMessageRequest,
    request: Request,
    db: DBSession,
    redis: RedisClient,
    user: CurrentUser,
) -> SendMessageResponse:
    """Run one assistant turn over the conversation. Deltas + lifecycle events
    stream on the ``/stream`` WebSocket; this call returns the final output, or
    ``awaiting_approval`` when the turn paused for human tool approval."""
    conversation = await _get_owned(db, conversation_id, user.id)
    if await count_pending(db, conversation_id) > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="conversation has pending tool approvals; resolve them first",
        )
    runner = _chat_runner(request)
    resolver = _agent_resolver(request)
    agent = await resolver(db, conversation)

    attachments = [
        {"type": "image", "media_id": str(mid)} for mid in payload.media_ids
    ] or None
    result = await runner.send(
        agent=agent,
        conversation_id=conversation_id,
        new_message=payload.new_message,
        db=db,
        redis=redis,
        system_prompt=await _augment_system_prompt(
            request, db, conversation, agent, payload.new_message
        ),
        extra_call_context={"user_id": str(user.id)},
        attachments=attachments,
    )
    agent, result = await _follow_handoffs(
        db, request, conversation, redis, user.id, agent, result,
        query=payload.new_message,
    )
    return await _turn_response(db, request, conversation_id, agent, user.id, result)


@router.get(
    "/{conversation_id}/tool-approvals",
    response_model=list[ToolApprovalRead],
)
async def tool_approvals(
    conversation_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
    status_filter: str | None = None,
) -> list[ToolApprovalRead]:
    """List this conversation's tool approvals (default: all). The frontend
    fetches ``status=pending`` on load so the approve/reject buttons survive a
    page reload, independent of the live WebSocket."""
    await _get_owned(db, conversation_id, user.id)
    rows = await list_approvals(db, conversation_id, status=status_filter)
    return [ToolApprovalRead.model_validate(r) for r in rows]


@router.post(
    "/{conversation_id}/tool-approvals/{approval_id}/decision",
    response_model=SendMessageResponse,
)
async def decide_tool_approval(
    conversation_id: uuid.UUID,
    approval_id: uuid.UUID,
    payload: ToolApprovalDecision,
    request: Request,
    db: DBSession,
    redis: RedisClient,
    user: CurrentUser,
) -> SendMessageResponse:
    """Approve or reject a pending tool call. When the last pending approval of
    the turn is resolved, the turn resumes in-request: the decided tools run and
    the LLM is re-invoked (unless the whole turn was rejected with no reason)."""
    conversation = await _get_owned(db, conversation_id, user.id)
    approval = await get_approval(db, approval_id)
    if approval.conversation_id != conversation_id:
        raise _approval_not_found(approval_id)
    try:
        await decide(
            db,
            approval_id,
            decision=payload.decision,
            reason=payload.reason,
            resolved_by=user.id,
        )
    except ToolApprovalNotFound as exc:
        raise _approval_not_found(approval_id) from exc
    except ToolApprovalNotPending as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="approval already resolved",
        ) from exc

    # More approvals of this turn still pending → keep waiting.
    if await count_pending(db, conversation_id) > 0:
        pending = await list_approvals(
            db, conversation_id, status=ToolApprovalStatus.pending.value
        )
        return SendMessageResponse(
            conversation_id=conversation_id,
            status="awaiting_approval",
            pending=[ToolApprovalRead.model_validate(p) for p in pending],
        )

    # All resolved → resume the turn.
    resolver = _agent_resolver(request)
    runner = _chat_runner(request)
    agent = await resolver(db, conversation)
    result = await runner.resume(
        agent=agent,
        conversation_id=conversation_id,
        db=db,
        redis=redis,
        system_prompt=agent.instructions or None,
        extra_call_context={"user_id": str(user.id)},
    )
    agent, result = await _follow_handoffs(
        db, request, conversation, redis, user.id, agent, result
    )
    return await _turn_response(db, request, conversation_id, agent, user.id, result)


# --- live stream ---
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """Lazily-built manager keyed by ``conversation_id``. ConversationIO
    publishes turn events on ``run_event_channel(conversation_id)`` (the default
    ``channel_fn``); a chat timeline rehydrates from its messages + the live
    delta stream, so no snapshot function is needed."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager(get_redis_client())
    return _ws_manager


@router.websocket("/{conversation_id}/stream")
async def stream(websocket: WebSocket, conversation_id: uuid.UUID) -> None:
    await get_ws_manager().connect(conversation_id, websocket)


# --- helpers ---
async def _augment_system_prompt(
    request: Request,
    db: AsyncSession,
    conversation: Conversation,
    agent: Agent,
    query: str | None,
) -> str | None:
    """Build the turn's system prompt, optionally enriched by a host RAG hook.

    If the host registered ``app.state.chat_context_provider`` — an async
    ``(db, conversation, agent, query) -> str | None`` — its returned text is
    appended to the agent's instructions for this turn, so retrieved context
    (e.g. the user's relevant past cases) is injected automatically rather than
    relying on the model to call a search tool. Hosts without the hook are
    unaffected; failures never break the turn."""
    base = agent.instructions or None
    provider = getattr(request.app.state, "chat_context_provider", None)
    if provider is None or not query:
        return base
    try:
        extra = await provider(db, conversation, agent, query)
    except Exception:  # noqa: BLE001 — context augmentation must never break a turn
        extra = None
    if not extra:
        return base
    return f"{base}\n\n{extra}" if base else extra


async def _follow_handoffs(
    db: AsyncSession,
    request: Request,
    conversation: Conversation,
    redis: Any,
    user_id: uuid.UUID,
    agent: Agent,
    result: Any,
    query: str | None = None,
) -> tuple[Agent, Any]:
    """Continue the request under a new agent while the resolver keeps changing.

    A turn can hand the conversation off (a terminal ``route_to_*`` tool flips the
    host's routing); we re-consult the resolver and, if the active agent changed,
    run it on the existing history with no new user message so the now-active
    agent answers in the same request. Stops when the agent stabilises, the turn
    suspends for approval, or the hop cap is hit."""
    runner = _chat_runner(request)
    resolver = _agent_resolver(request)
    hops = 0
    while not isinstance(result, SuspendedForApproval) and hops < MAX_HANDOFF_HOPS:
        next_agent = await resolver(db, conversation)
        if next_agent.id == agent.id:
            break
        agent = next_agent
        result = await runner.send(
            agent=agent,
            conversation_id=conversation.id,
            new_message=None,
            db=db,
            redis=redis,
            system_prompt=await _augment_system_prompt(
                request, db, conversation, agent, query
            ),
            extra_call_context={"user_id": str(user_id)},
        )
        hops += 1
    return agent, result


async def _turn_response(
    db: AsyncSession,
    request: Request,
    conversation_id: uuid.UUID,
    agent: Agent,
    user_id: uuid.UUID,
    result: Any,
) -> SendMessageResponse:
    """Map a runner result to the API response. A suspended turn becomes
    ``awaiting_approval`` with the pending calls; a finished turn becomes
    ``completed`` and fires the optional metering hook."""
    if isinstance(result, SuspendedForApproval):
        pending = await list_approvals(
            db, conversation_id, status=ToolApprovalStatus.pending.value
        )
        return SendMessageResponse(
            conversation_id=conversation_id,
            status="awaiting_approval",
            pending=[ToolApprovalRead.model_validate(p) for p in pending],
        )

    hook = getattr(request.app.state, "chat_metering_hook", None)
    if hook is not None:
        await hook(
            MeteringContext(
                user_id=user_id,
                conversation_id=conversation_id,
                agent=agent,
                output=result,
            )
        )
    return SendMessageResponse(
        conversation_id=conversation_id, status="completed", output=result
    )


def _approval_not_found(approval_id: uuid.UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"tool approval {approval_id} not found",
    )


async def _get_owned(
    db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID
) -> Conversation:
    try:
        return await get_conversation(db, conversation_id, user_id=user_id)
    except ConversationNotFound as exc:
        raise _not_found(conversation_id) from exc


def _not_found(conversation_id: uuid.UUID) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"conversation {conversation_id} not found",
    )


def _chat_runner(request: Request) -> ChatRunner:
    runner = getattr(request.app.state, "chat_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="chat is not configured on this host (app.state.chat_runner)",
        )
    return runner


def _agent_resolver(request: Request) -> AgentResolver:
    resolver = getattr(request.app.state, "chat_agent_resolver", None)
    if resolver is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="chat is not configured (app.state.chat_agent_resolver)",
        )
    return resolver
