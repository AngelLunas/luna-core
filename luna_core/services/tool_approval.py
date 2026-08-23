"""Tool-approval service — the durable intent-to-execute a gated tool.

The runner creates pending rows when a turn suspends; the API lists them (so
the buttons survive reload) and resolves them. ``decide`` transitions the row
atomically (``UPDATE ... WHERE status='pending'``) so a double-click or a racing
request can't resolve the same approval twice.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.tool_approval import ToolApproval, ToolApprovalStatus


class ToolApprovalNotFound(LookupError):
    pass


class ToolApprovalNotPending(ValueError):
    """The approval was already resolved (lost the race / double submit)."""


async def create_pending_approvals(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    tool_uses: list[dict[str, Any]],
    agent_name: str | None = None,
) -> list[ToolApproval]:
    """Persist one pending row per gated ``tool_use`` block of the assistant
    message that just suspended the turn.

    ``agent_name`` records who is asking, so a client can name the proposer
    instead of assuming one. It comes from the runner, never from the tool's
    arguments — those are written by the model."""
    rows = [
        ToolApproval(
            conversation_id=conversation_id,
            tool_use_id=str(tu.get("id", "")),
            tool_name=str(tu.get("name", "")),
            tool_input=tu.get("input") or {},
            agent_name=agent_name,
            status=ToolApprovalStatus.pending.value,
        )
        for tu in tool_uses
    ]
    db.add_all(rows)
    await db.commit()
    for row in rows:
        await db.refresh(row)
    return rows


async def list_approvals(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    status: str | None = None,
) -> list[ToolApproval]:
    stmt = select(ToolApproval).where(
        ToolApproval.conversation_id == conversation_id
    )
    if status is not None:
        stmt = stmt.where(ToolApproval.status == status)
    stmt = stmt.order_by(ToolApproval.created_at)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_pending(db: AsyncSession, conversation_id: uuid.UUID) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(ToolApproval)
        .where(
            ToolApproval.conversation_id == conversation_id,
            ToolApproval.status == ToolApprovalStatus.pending.value,
        )
    )
    return int(result.scalar() or 0)


async def get_approval(
    db: AsyncSession, approval_id: uuid.UUID
) -> ToolApproval:
    obj = await db.get(ToolApproval, approval_id)
    if obj is None:
        raise ToolApprovalNotFound(str(approval_id))
    return obj


async def decide(
    db: AsyncSession,
    approval_id: uuid.UUID,
    *,
    decision: str,
    reason: str | None = None,
    resolved_by: uuid.UUID | None = None,
) -> ToolApproval:
    """Atomically resolve a pending approval. Raises ``ToolApprovalNotPending``
    if it was already resolved (so only the winning request resumes the turn)."""
    stmt = (
        update(ToolApproval)
        .where(
            ToolApproval.id == approval_id,
            ToolApproval.status == ToolApprovalStatus.pending.value,
        )
        .values(
            status=decision,
            reason=reason,
            resolved_at=datetime.now(timezone.utc),
            resolved_by=resolved_by,
        )
        .returning(ToolApproval.id)
    )
    result = await db.execute(stmt)
    if result.first() is None:
        await db.rollback()
        # Disambiguate not-found vs already-resolved for the caller.
        if await db.get(ToolApproval, approval_id) is None:
            raise ToolApprovalNotFound(str(approval_id))
        raise ToolApprovalNotPending(str(approval_id))
    await db.commit()
    return await get_approval(db, approval_id)


def should_cascade_rejection(decision: str, reason: str | None) -> bool:
    """Whether resolving this approval should also reject the turn's remaining
    gated calls.

    Only a rejection carrying a correction cascades. A plain discard is a
    per-call "not this one" and leaves its siblings for the user to decide.
    """
    return decision == ToolApprovalStatus.rejected.value and bool(
        (reason or "").strip()
    )


async def reject_pending_with_reason(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    reason: str,
    resolved_by: uuid.UUID | None = None,
) -> list[ToolApproval]:
    """Reject every still-pending approval of the suspended turn, carrying the
    correction the user wrote on one of them.

    A gated plan reaches the user as several tool calls, but a correction ("the
    light runs 6pm-12") is about the plan, not about the one call it was typed
    under. Without this the turn stalls on siblings the user believes they
    already answered, and the LLM never sees the correction. Rejecting them with
    the same reason lets the turn resume now and hands the whole plan back for
    the LLM to re-propose.

    Safe to scope by conversation: ``send`` refuses to start a turn while any
    approval is pending, so every pending row belongs to the one suspended turn.
    """
    stmt = (
        update(ToolApproval)
        .where(
            ToolApproval.conversation_id == conversation_id,
            ToolApproval.status == ToolApprovalStatus.pending.value,
        )
        .values(
            status=ToolApprovalStatus.rejected.value,
            reason=reason,
            resolved_at=datetime.now(timezone.utc),
            resolved_by=resolved_by,
        )
        .returning(ToolApproval.id)
    )
    ids = [row[0] for row in (await db.execute(stmt)).all()]
    await db.commit()
    if not ids:
        return []
    rows = await db.execute(select(ToolApproval).where(ToolApproval.id.in_(ids)))
    return list(rows.scalars().all())


async def decisions_by_tool_use(
    db: AsyncSession, conversation_id: uuid.UUID
) -> dict[str, ToolApproval]:
    """All resolved approvals of a conversation, keyed by ``tool_use_id`` — the
    runner consults this on resume to know which tool_uses were approved vs
    rejected (and the rejection reason)."""
    rows = await list_approvals(db, conversation_id)
    return {
        row.tool_use_id: row
        for row in rows
        if row.status != ToolApprovalStatus.pending.value
    }
