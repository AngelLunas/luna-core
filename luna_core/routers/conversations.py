"""Conversation router — generic AI-chat surface for host apps.

CRUD over the conversation primitive + a ``send_message`` turn driven by
``ChatRunner`` + a ``/stream`` WebSocket. The CRUD endpoints are self-contained;
the turn endpoints need chat infrastructure the **host wires into ``app.state``**,
so luna-core stays free of any host's agent/tool/metering choices:

    app.state.chat_runner: ChatRunner                      # required for turns
    app.state.chat_agent_resolver: AgentResolver           # required for turns
    app.state.chat_metering_hook:                          # optional
        Callable[[MeteringContext], Awaitable[None]] | None

The resolver picks which agent answers a given conversation (a host with one
orchestrator just returns it; see ``AgentResolver`` for the routing-aware
shape a multi-agent host can opt into); the metering hook, if set, runs after
each turn.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, status
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.db import AsyncSessionLocal
from luna_core.core.dependencies import (
    CurrentUser,
    DBSession,
    RedisClient,
    get_redis_client,
)
from luna_core.core.config import settings
from luna_core.core.security import decode_access_token
from luna_core.engine.agent import SuspendedForApproval
from luna_core.engine.chat import ChatRunner, ConversationIO
from luna_core.llm.base import AbortSignalError, abort_key
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
from luna_core.models.event import RunEventType
from luna_core.services.agent_routing import ROUTING_NODE, handoff_preamble
from luna_core.services.auto_title import maybe_title_conversation
from luna_core.services.conversation import (
    ConversationNotFound,
    create_conversation,
    delete_conversation,
    finalize_partial_messages,
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
    reject_pending_with_reason,
    should_cascade_rejection,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

logger = logging.getLogger(__name__)

# The resolver picks which agent answers a conversation's next turn. Two
# accepted shapes:
#   (db, conversation)                       — the original contract
#   (db, conversation, *, query, io)         — routing-aware: also receives the
#       turn's new user text (None on continuations/resumes) and a
#       ConversationIO to emit routing lifecycle events (routing_started /
#       routing_decided) on the conversation's live channel. A host that
#       classifies cold-start turns (see services.agent_routing) uses both.
# ``_resolve_agent`` inspects the callable once per call and passes only what
# it accepts, so existing hosts keep working unchanged.
AgentResolver = Callable[..., Awaitable[Agent]]

# Multi-agent handoff: after a turn, the resolver is consulted again; if a tool
# changed which agent is active (e.g. a terminal ``route_to_*`` tool), the same
# request continues with the new agent on the existing history. Capped so a
# misbehaving pair of agents can't ping-pong forever. A single-agent host never
# trips this — the resolver returns the same agent and the loop exits at once.
MAX_HANDOFF_HOPS = 4

# Fire-and-forget auto-title tasks. Kept in a module-level set so the event loop
# doesn't garbage-collect a task mid-flight (asyncio holds only a weak ref); each
# discards itself on completion.
_auto_title_tasks: set[asyncio.Task] = set()


def _maybe_spawn_auto_title(
    request: Request,
    conversation_id: uuid.UUID,
    owner_id: uuid.UUID | None,
    agent: Agent,
    response: SendMessageResponse,
    needs_title: bool,
) -> None:
    """After a completed turn, kick off titling OFF the request path.

    Opt-in: only runs when the host set ``app.state.chat_auto_title_llm_router``
    (an ``LLMRouter``) — hosts that don't stay unaffected. Skips owner-less
    conversations (domain-owned) and any turn that didn't finish cleanly.
    ``needs_title`` is the pre-turn snapshot of ``conversation.title is None`` so
    we don't even spawn on later turns; the task itself re-checks idempotently."""
    if not needs_title or owner_id is None or response.status != "completed":
        return
    llm_router = getattr(request.app.state, "chat_auto_title_llm_router", None)
    if llm_router is None:
        return
    task = asyncio.create_task(
        maybe_title_conversation(
            conversation_id=conversation_id,
            user_id=owner_id,
            llm_router=llm_router,
            redis=get_redis_client(),
            provider_id=agent.llm_provider_id,
            model=agent.model,
        )
    )
    _auto_title_tasks.add(task)
    task.add_done_callback(_auto_title_tasks.discard)


def _resolver_takes_routing_kwargs(resolver: AgentResolver) -> bool:
    """Whether the host resolver accepts the routing-aware kwargs (see
    ``AgentResolver``)."""
    try:
        params = inspect.signature(resolver).parameters
    except (TypeError, ValueError):
        return False
    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return True
    return "query" in params and "io" in params


async def _resolve_agent(
    resolver: AgentResolver,
    db: AsyncSession,
    conversation: Conversation,
    *,
    query: str | None = None,
    io: ConversationIO | None = None,
) -> Agent:
    """Call the host resolver with as much context as it accepts."""
    if _resolver_takes_routing_kwargs(resolver):
        return await resolver(db, conversation, query=query, io=io)
    return await resolver(db, conversation)


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


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    conversation_id: uuid.UUID,
    request: Request,
    db: DBSession,
    redis: RedisClient,
    user: CurrentUser,
) -> Response:
    """Delete a conversation and all its history (cascades). Owner-only.

    If the host registered ``app.state.on_conversation_delete``, it runs first —
    while the messages still exist — so the host can clean up side artifacts (e.g.
    images the chat uploaded) before the cascade removes the rows that point to them.
    """
    try:
        await _get_owned(db, conversation_id, user.id)
        hook = getattr(request.app.state, "on_conversation_delete", None)
        if hook is not None:
            await hook(db, conversation_id, user.id)
        await delete_conversation(db, conversation_id, user_id=user.id)
    except ConversationNotFound as exc:
        raise _not_found(conversation_id) from exc
    # Drop any lingering abort flag for this id so it can't leak to a reused key.
    await redis.delete(abort_key(conversation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    # Snapshot now (pre-turn), while the row is fresh: whether this conversation
    # still needs an auto-title. Used after the turn to titling off the hot path.
    needs_title = conversation.title is None
    if await count_pending(db, conversation_id) > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="conversation has pending tool approvals; resolve them first",
        )
    runner = _chat_runner(request)
    resolver = _agent_resolver(request)
    # The turn's live channel, up before the agent is even chosen — a
    # routing-aware resolver emits routing_started/routing_decided on it so
    # the client can show the decision phase ahead of the first token.
    io = ConversationIO(db, redis, conversation_id)
    agent = await _resolve_agent(
        resolver, db, conversation, query=payload.new_message, io=io
    )

    # Clear any stale abort flag from a prior turn (the key lingers for its TTL
    # and is never auto-cleared) so it can't kill this fresh turn.
    await redis.delete(abort_key(conversation_id))

    attachments = payload.attachment_blocks() or None
    try:
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
            image_resolver=await _image_resolver(
                request, db, conversation, agent, user.id,
                media_ids=payload.attachment_media_ids(),
            ),
        )
        agent, result = await _follow_handoffs(
            db, request, conversation, redis, user.id, agent, result,
            query=payload.new_message,
        )
    except AbortSignalError:
        # The user stopped the turn. The provider persisted the streamed-so-far
        # text as a PARTIAL message; promote it to final so it stays in the thread
        # (list_messages hides partials). Clear the flag for the next turn.
        await redis.delete(abort_key(conversation_id))
        await finalize_partial_messages(db, conversation_id)
        return SendMessageResponse(conversation_id=conversation_id, status="aborted")
    response = await _turn_response(db, request, conversation_id, agent, user.id, result)
    _maybe_spawn_auto_title(
        request, conversation_id, conversation.user_id, agent, response, needs_title
    )
    return response


@router.post("/{conversation_id}/abort", status_code=status.HTTP_202_ACCEPTED)
async def abort_turn(
    conversation_id: uuid.UUID,
    db: DBSession,
    redis: RedisClient,
    user: CurrentUser,
) -> dict[str, str]:
    """Stop the conversation's in-flight turn. Sets the abort flag the streaming
    provider checks before each chunk: it stops reading the LLM stream (no more
    tokens burned), persists whatever streamed as a partial message, and the
    blocking ``/messages`` call returns ``status="aborted"``. A no-op if nothing
    is running."""
    await _get_owned(db, conversation_id, user.id)
    await redis.set(
        abort_key(conversation_id), "1", ex=settings.run_abort_key_ttl_seconds
    )
    return {"status": "aborting"}


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
    the LLM is re-invoked (unless the whole turn was rejected with no reason).

    A rejection carrying a reason also rejects the turn's remaining gated calls
    with that same reason (see ``reject_pending_with_reason``), so correcting one
    call of a multi-call plan resumes the turn instead of stalling on siblings."""
    conversation = await _get_owned(db, conversation_id, user.id)
    needs_title = conversation.title is None  # pre-resume snapshot (see send())
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

    # A correction is about the plan, not about the single call it was typed
    # under: reject the turn's remaining gated calls with the same reason so the
    # turn resumes now and the LLM re-proposes with the correction in hand.
    if should_cascade_rejection(payload.decision, payload.reason):
        await reject_pending_with_reason(
            db,
            conversation_id,
            reason=payload.reason or "",
            resolved_by=user.id,
        )

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

    # All resolved → resume the turn (no new user text, so a routing-aware
    # resolver just returns the sticky agent).
    resolver = _agent_resolver(request)
    runner = _chat_runner(request)
    agent = await _resolve_agent(
        resolver, db, conversation, io=ConversationIO(db, redis, conversation_id)
    )
    result = await runner.resume(
        agent=agent,
        conversation_id=conversation_id,
        db=db,
        redis=redis,
        system_prompt=agent.instructions or None,
        extra_call_context={"user_id": str(user.id)},
        image_resolver=await _image_resolver(
            request, db, conversation, agent, user.id
        ),
    )
    agent, result = await _follow_handoffs(
        db, request, conversation, redis, user.id, agent, result
    )
    response = await _turn_response(db, request, conversation_id, agent, user.id, result)
    _maybe_spawn_auto_title(
        request, conversation_id, conversation.user_id, agent, response, needs_title
    )
    return response


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


def _user_from_token(token: str | None) -> uuid.UUID | None:
    """Decode an access token (passed as a WS query param, since browsers can't
    set WebSocket headers) → the user id, or None if missing/invalid/expired."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:  # covers expired (ExpiredSignatureError subclass)
        return None
    sub = payload.get("sub")
    try:
        return uuid.UUID(sub) if sub else None
    except (ValueError, TypeError):
        return None


@router.websocket("/{conversation_id}/stream")
async def stream(
    websocket: WebSocket,
    conversation_id: uuid.UUID,
    token: str | None = None,
) -> None:
    # Authenticate the socket: a valid access token whose user owns this
    # conversation. Reject before accepting otherwise (the stream is per-user).
    user_id = _user_from_token(token)
    if user_id is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    async with AsyncSessionLocal() as db:
        try:
            await get_conversation(db, conversation_id, user_id=user_id)
        except ConversationNotFound:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
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
    ``(db, conversation, agent, query) -> str | None`` — it is consulted on every
    turn (``query`` is ``None`` when the turn has no new user text, e.g. a handoff
    or image-only turn) and its returned text is appended to the agent's
    instructions, so context (the current date/time, the user's relevant past
    cases, …) is injected automatically rather than relying on the model to call a
    tool. Hosts without the hook are unaffected; failures never break the turn."""
    base = agent.instructions or None
    provider = getattr(request.app.state, "chat_context_provider", None)
    if provider is None:
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

    A turn can hand the conversation off (the terminal transfer tool flips the
    host's routing); we re-consult the resolver and, if the active agent changed,
    run it on the existing history with no new user message so the now-active
    agent answers in the same request. Stops when the agent stabilises, the turn
    suspends for approval, or the hop cap is hit.

    Each continuation turn gets the handoff preamble appended to its system
    prompt: the receiving agent's transcript ends with the *previous* agent's
    transfer call, and without an explicit "you are the new agent now — carry
    on with the pending request" models routinely keep narrating the old
    agent's voice ("I'll pass you to the doctor") instead of answering."""
    runner = _chat_runner(request)
    resolver = _agent_resolver(request)
    io = ConversationIO(db, redis, conversation.id) if redis is not None else None
    hops = 0
    while not isinstance(result, SuspendedForApproval) and hops < MAX_HANDOFF_HOPS:
        next_agent = await _resolve_agent(
            resolver, db, conversation, query=query, io=io
        )
        if next_agent.id == agent.id:
            break
        await _emit_routing_decided(io, from_agent=agent, to_agent=next_agent)
        agent = next_agent
        system_prompt = await _augment_system_prompt(
            request, db, conversation, agent, query
        )
        preamble = handoff_preamble(agent)
        system_prompt = (
            f"{system_prompt}\n\n{preamble}" if system_prompt else preamble
        )
        result = await runner.send(
            agent=agent,
            conversation_id=conversation.id,
            new_message=None,
            db=db,
            redis=redis,
            system_prompt=system_prompt,
            extra_call_context={"user_id": str(user_id)},
            image_resolver=await _image_resolver(
                request, db, conversation, agent, user_id
            ),
        )
        hops += 1
    return agent, result


async def _emit_routing_decided(
    io: ConversationIO | None, *, from_agent: Agent, to_agent: Agent
) -> None:
    """Broadcast a handoff on the conversation's live channel — the client's
    cue to switch the speaking identity before the new agent's first token.
    Best-effort: the event is UI sugar and must never cost the user the turn."""
    if io is None:
        return
    try:
        await io.emit(
            RunEventType.routing_decided,
            node_id=ROUTING_NODE,
            payload={
                "agent_name": to_agent.name,
                "from_agent": from_agent.name,
                "trigger": "handoff",
            },
        )
    except Exception:  # noqa: BLE001 — see docstring
        logger.exception("routing_decided emit failed for %s", io.scope_id)


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


async def _image_resolver(
    request: Request,
    db: AsyncSession,
    conversation: Conversation,
    agent: Agent,
    user_id: uuid.UUID,
    media_ids: list[uuid.UUID] | None = None,
) -> Any | None:
    """Per-agent image resolver from the optional host hook
    ``app.state.chat_image_resolver_factory``. The host decides whether THIS agent
    should see attached photos inline (vision-capable models) and how to scope them
    (e.g. only the latest turn's); a text-only agent gets ``None`` and keeps reading
    the ``[image attached: …]`` notes. No hook ⇒ ``None`` ⇒ unchanged behaviour.

    ``media_ids`` are the photos on the message being sent in THIS request. The
    resolver is built before the runner persists that message, so a host that
    scopes by "the latest stored user turn" would otherwise miss them (a fresh
    conversation: none; a later turn: the previous photos). Hosts whose factory
    accepts a ``media_ids`` keyword get them; older 4-argument factories are
    called as before."""
    factory = getattr(request.app.state, "chat_image_resolver_factory", None)
    if factory is None:
        return None
    if media_ids and _accepts_media_ids(factory):
        return await factory(db, user_id, agent, conversation, media_ids=media_ids)
    return await factory(db, user_id, agent, conversation)


def _accepts_media_ids(factory: Any) -> bool:
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    return "media_ids" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
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
