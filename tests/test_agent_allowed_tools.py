"""Unit tests for AgentRunner._allowed_tool_names.

The unioning of connector-op assignments and system-tool grants is the
gate that decides which tools the LLM actually sees. Tested against a
hand-rolled async session fake so we exercise the SQL composition
without spinning up Postgres.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import select

from luna_core.engine.agent import AgentRunner
from luna_core.models.agent import AgentOperation, AgentSystemToolGrant
from luna_core.models.connector import Operation


@dataclass
class _Row:
    """Mimics what ``row[0]`` returns from a single-column SELECT."""

    value: Any

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError(idx)
        return self.value


class _FakeResult:
    def __init__(self, rows: list[_Row]):
        self._rows = rows

    def all(self) -> list[_Row]:
        return list(self._rows)


class _FakeSession:
    """Routes ``execute(select(X)...)`` by the selected column to a
    canned list of rows. Just enough surface to test the two queries
    ``_allowed_tool_names`` runs.
    """

    def __init__(
        self,
        *,
        operation_names: list[str] | None = None,
        grant_names: list[str] | None = None,
    ):
        self._operation_names = operation_names or []
        self._grant_names = grant_names or []

    async def execute(self, stmt) -> _FakeResult:
        # Detect which select we're handling by sniffing the columns.
        # Both queries select exactly one column.
        columns = [str(c).split(".")[-1] for c in stmt.selected_columns]
        if len(columns) != 1:
            raise AssertionError(f"unexpected multi-column select: {columns}")
        col = columns[0]
        if col == "name":
            return _FakeResult([_Row(n) for n in self._operation_names])
        if col == "tool_name":
            return _FakeResult([_Row(n) for n in self._grant_names])
        raise AssertionError(f"unexpected column in select: {col}")


@pytest.fixture
def runner() -> AgentRunner:
    # Collaborators don't matter for _allowed_tool_names — pass None-like
    # placeholders. The method only touches the db session.
    return AgentRunner(llm_router=None, mcp_client=None)  # type: ignore[arg-type]


@pytest.fixture
def agent_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-0000000000a1")


@pytest.mark.asyncio
async def test_no_assignments_returns_none(runner, agent_id):
    db = _FakeSession()
    assert await runner._allowed_tool_names(db, agent_id) is None


@pytest.mark.asyncio
async def test_only_operation_assignments(runner, agent_id):
    db = _FakeSession(operation_names=["list_jobs", "create_job"])
    assert await runner._allowed_tool_names(db, agent_id) == {
        "list_jobs",
        "create_job",
    }


@pytest.mark.asyncio
async def test_only_system_tool_grants(runner, agent_id):
    db = _FakeSession(grant_names=["stash_records"])
    assert await runner._allowed_tool_names(db, agent_id) == {"stash_records"}


@pytest.mark.asyncio
async def test_union_of_both(runner, agent_id):
    db = _FakeSession(
        operation_names=["list_jobs"],
        grant_names=["stash_records", "future_tool"],
    )
    assert await runner._allowed_tool_names(db, agent_id) == {
        "list_jobs",
        "stash_records",
        "future_tool",
    }


@pytest.mark.asyncio
async def test_duplicate_name_across_sources_dedupes_in_union(runner, agent_id):
    # If a connector op and a system tool happened to share a name, the
    # union just dedupes — runtime dispatch still resolves correctly
    # because system tools win in AgentRunner.run.
    db = _FakeSession(
        operation_names=["overlapping"],
        grant_names=["overlapping", "unique"],
    )
    assert await runner._allowed_tool_names(db, agent_id) == {
        "overlapping",
        "unique",
    }


# ---- Smoke check that the two query shapes the production method emits
# match what _FakeSession routes by (column name). This protects against
# silent breakage if someone renames the underlying columns. -----------


def test_grant_query_selects_tool_name():
    stmt = select(AgentSystemToolGrant.tool_name)
    [col] = list(stmt.selected_columns)
    assert str(col).endswith(".tool_name")


def test_operation_query_selects_name():
    stmt = select(Operation.name)
    [col] = list(stmt.selected_columns)
    assert str(col).endswith(".name")


def test_agent_operation_relationship_still_exists():
    # If someone refactors AgentOperation away, _allowed_tool_names
    # silently stops filtering by connector ops. Catch that here.
    assert hasattr(AgentOperation, "agent_id")
    assert hasattr(AgentOperation, "operation_id")


# ---- list_agents with system tool include --------------------------------


@pytest.mark.asyncio
async def test_list_agents_accepts_with_system_tool_grants_flag():
    # Smoke-test the new parameter exists and doesn't crash with an
    # empty DB. The eager-load behavior itself is an implementation
    # detail that would need a real DB to verify meaningfully —
    # router-level coverage is the right place for that.
    from unittest.mock import AsyncMock, MagicMock

    from luna_core.services import agent as agent_service

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    agents_with = await agent_service.list_agents(db, with_system_tool_grants=True)
    agents_without = await agent_service.list_agents(db)
    assert agents_with == []
    assert agents_without == []
