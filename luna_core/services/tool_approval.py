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
) -> list[ToolApproval]:
    """Persist one pending row per gated ``tool_use`` block of the assistant
    message that just suspended the turn."""
    rows = [
        ToolApproval(
            conversation_id=conversation_id,
            tool_use_id=str(tu.get("id", "")),
            tool_name=str(tu.get("name", "")),
            tool_input=tu.get("input") or {},
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
