"""Conversation surface: ownership enforcement + schema role coercion.

These cover the genuinely new logic added with the conversation API surface
without standing up Postgres — mirroring the fake-DB style of
``test_conversation_io.py``.
"""
from __future__ import annotations

import uuid

import pytest

from luna_core.models.conversation import (
    Conversation,
    ConversationMessageRole,
)
from luna_core.schemas.conversation import ConversationMessageRead
from luna_core.services.conversation import (
    ConversationNotFound,
    get_conversation,
)


class _FakeDB:
    """Minimal ``AsyncSession`` stand-in: only ``get`` is exercised here."""

    def __init__(self, row: Conversation | None) -> None:
        self._row = row

    async def get(self, _model, _pk):
        return self._row


def _conversation(user_id: uuid.UUID | None) -> Conversation:
    return Conversation(id=uuid.uuid4(), user_id=user_id, title=None)


@pytest.mark.asyncio
async def test_get_conversation_missing_raises():
    with pytest.raises(ConversationNotFound):
        await get_conversation(_FakeDB(None), uuid.uuid4())


@pytest.mark.asyncio
async def test_get_conversation_enforces_owner():
    owner = uuid.uuid4()
    other = uuid.uuid4()
    db = _FakeDB(_conversation(owner))
    # owner sees it
    assert (await get_conversation(db, uuid.uuid4(), user_id=owner)).user_id == owner
    # a different user gets not-found (no cross-tenant leak)
    with pytest.raises(ConversationNotFound):
        await get_conversation(db, uuid.uuid4(), user_id=other)


@pytest.mark.asyncio
async def test_get_conversation_no_owner_filter_returns_row():
    db = _FakeDB(_conversation(None))
    # without a user_id filter, a domain-owned (ownerless) conversation resolves
    assert await get_conversation(db, uuid.uuid4()) is not None


def test_message_read_coerces_role_enum_to_value():
    read = ConversationMessageRead(
        id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        sequence=1,
        role=ConversationMessageRole.assistant,  # enum in → plain value out
        content=[],
        is_partial=False,
        created_at="2026-06-15T00:00:00Z",
    )
    assert read.role == "assistant"
