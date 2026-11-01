"""Conversation list search: the title filter the API adds with ``q``.

No Postgres: a fake session captures the statement and the test reads the SQL
it would send, in the fake-DB style of ``test_conversation_service.py``.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql

from luna_core.services.conversation import list_conversations, title_pattern


class _CaptureDB:
    """Minimal ``AsyncSession`` stand-in: records the statement, answers no rows."""

    def __init__(self) -> None:
        self.stmt = None

    async def execute(self, stmt):
        self.stmt = stmt
        return _NoRows()


class _NoRows:
    def scalars(self):
        return self

    def all(self):
        return []


def _compiled(db: _CaptureDB):
    return db.stmt.compile(dialect=postgresql.dialect())


def _sql(db: _CaptureDB) -> str:
    return str(_compiled(db))


def test_title_pattern_matches_anywhere_and_escapes_wildcards():
    assert title_pattern("riego") == "%riego%"
    assert title_pattern("  riego  ") == "%riego%"
    # LIKE's own wildcards are literal in what the grower typed.
    assert title_pattern("50%") == "%50\\%%"
    assert title_pattern("6T_01") == "%6T\\_01%"
    assert title_pattern("a\\b") == "%a\\\\b%"


@pytest.mark.asyncio
async def test_list_conversations_filters_titles_case_insensitively():
    db = _CaptureDB()
    await list_conversations(db, user_id=uuid.uuid4(), limit=20, offset=0, q="Riego")
    sql = _sql(db)
    assert "ILIKE" in sql and "ESCAPE" in sql
    assert "%Riego%" in _compiled(db).params.values()
    assert "ORDER BY" in sql and "updated_at DESC" in sql
    assert "LIMIT" in sql


@pytest.mark.asyncio
async def test_list_conversations_blank_query_adds_no_filter():
    db = _CaptureDB()
    await list_conversations(db, user_id=uuid.uuid4(), limit=20, q="   ")
    assert "ILIKE" not in _sql(db)
    db = _CaptureDB()
    await list_conversations(db, user_id=uuid.uuid4(), limit=20)
    assert "ILIKE" not in _sql(db)
