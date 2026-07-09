"""Conversation surface service — CRUD over the persistent chat primitive.

Generic: any host app that exposes AI conversations over an API uses these.
Ownership is optional (``user_id``); when a caller passes ``user_id`` the
lookups enforce it, so a host with auth can scope conversations per user while a
domain-owned conversation (no owner) still works.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.conversation import Conversation, ConversationMessage


class ConversationNotFound(LookupError):
    pass


async def create_conversation(
    db: AsyncSession,
    *,
    user_id: uuid.UUID | None = None,
    title: str | None = None,
) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def list_conversations(
    db: AsyncSession, *, user_id: uuid.UUID
) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    user_id: uuid.UUID | None = None,
) -> Conversation:
    """Fetch a conversation. When ``user_id`` is given, enforce ownership (a
    mismatched or null owner is treated as not-found — no cross-tenant leak)."""
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise ConversationNotFound(str(conversation_id))
    if user_id is not None and conversation.user_id != user_id:
        raise ConversationNotFound(str(conversation_id))
    return conversation


async def update_conversation_title(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    title: str | None,
    user_id: uuid.UUID | None = None,
) -> Conversation:
    conversation = await get_conversation(db, conversation_id, user_id=user_id)
    conversation.title = title
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def delete_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    user_id: uuid.UUID | None = None,
) -> None:
    """Delete a conversation (ownership-checked). Its messages, tool approvals and
    routing rows cascade away via their FKs."""
    conversation = await get_conversation(db, conversation_id, user_id=user_id)
    await db.delete(conversation)
    await db.commit()


async def list_messages(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    user_id: uuid.UUID | None = None,
    include_partial: bool = False,
) -> list[ConversationMessage]:
    """Messages of a conversation in sequence order. Ownership is checked first
    when ``user_id`` is given. Partial rows (an interrupted assistant turn) are
    excluded by default."""
    await get_conversation(db, conversation_id, user_id=user_id)
    stmt = select(ConversationMessage).where(
        ConversationMessage.conversation_id == conversation_id
    )
    if not include_partial:
        stmt = stmt.where(ConversationMessage.is_partial.is_(False))
    stmt = stmt.order_by(ConversationMessage.sequence)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def finalize_partial_messages(
    db: AsyncSession, conversation_id: uuid.UUID
) -> int:
    """Promote a conversation's partial assistant rows to final (is_partial=False).

    Used when a turn is aborted: the text streamed-so-far was persisted as a
    partial, but the user chose to stop there, so that text IS the turn's final
    content and must show in the thread (``list_messages`` hides partials).
    Returns how many rows were promoted.
    """
    result = await db.execute(
        update(ConversationMessage)
        .where(
            ConversationMessage.conversation_id == conversation_id,
            ConversationMessage.is_partial.is_(True),
        )
        .values(is_partial=False)
    )
    await db.commit()
    return result.rowcount or 0
