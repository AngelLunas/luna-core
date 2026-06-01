"""Tests for assign_system_tools service.

Service is the thin wrapper around (a) registry validation, (b) wipe +
reinsert, (c) list-back. The DB pieces are exercised against an
``AsyncMock`` session — what we actually care about is the validation
and dedup logic that runs *before* any writes hit the DB.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna_core.mcp.system_tools import SystemToolRegistry
from luna_core.mcp.system_tools.registry import SystemTool


def _build_registry(*names: str) -> SystemToolRegistry:
    reg = SystemToolRegistry()
    for name in names:
        reg.register(
            SystemTool(
                name=name,
                description=name,
                input_schema={"type": "object"},
                handler=lambda _args, *, call_context: {},  # noqa: ARG005
                scope="catalog",
            )
        )
    return reg


@pytest.fixture
def agent_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-0000000000aa")


@pytest.mark.asyncio
async def test_unknown_name_raises_before_any_write(agent_id):
    from luna_core.services import agent as agent_service

    db = AsyncMock()
    # add_all is sync on a real SQLAlchemy AsyncSession; overriding the
    # AsyncMock default avoids a RuntimeWarning about the coroutine that
    # the production code never awaits (correctly).
    db.add_all = MagicMock()
    fake_registry = _build_registry("stash_records")
    with (
        patch.object(agent_service, "get_default_registry", return_value=fake_registry),
        patch.object(agent_service, "get_agent", new=AsyncMock(return_value=object())),
    ):
        with pytest.raises(ValueError, match="not in catalog"):
            await agent_service.assign_system_tools(
                db, agent_id, ["stash_records", "ghost_tool"]
            )
    # No execute / add_all / commit should have been called — validation
    # short-circuits before touching the DB.
    db.execute.assert_not_called()
    db.add_all.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_known_names_dedupes_and_writes(agent_id):
    from luna_core.services import agent as agent_service

    db = AsyncMock()
    # add_all is sync on a real SQLAlchemy AsyncSession; overriding the
    # AsyncMock default avoids a RuntimeWarning about the coroutine that
    # the production code never awaits (correctly).
    db.add_all = MagicMock()
    fake_registry = _build_registry("stash_records", "future_tool")
    with (
        patch.object(agent_service, "get_default_registry", return_value=fake_registry),
        patch.object(agent_service, "get_agent", new=AsyncMock(return_value=object())),
        patch.object(
            agent_service,
            "list_assigned_system_tools",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await agent_service.assign_system_tools(
            db, agent_id, ["stash_records", "stash_records", "future_tool"]
        )

    # add_all called once with the deduplicated set in input order.
    args, _kwargs = db.add_all.call_args
    grants = args[0]
    assert [g.tool_name for g in grants] == ["stash_records", "future_tool"]
    assert all(g.agent_id == agent_id for g in grants)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_list_clears_existing(agent_id):
    from luna_core.services import agent as agent_service

    db = AsyncMock()
    # add_all is sync on a real SQLAlchemy AsyncSession; overriding the
    # AsyncMock default avoids a RuntimeWarning about the coroutine that
    # the production code never awaits (correctly).
    db.add_all = MagicMock()
    fake_registry = _build_registry("stash_records")
    with (
        patch.object(agent_service, "get_default_registry", return_value=fake_registry),
        patch.object(agent_service, "get_agent", new=AsyncMock(return_value=object())),
        patch.object(
            agent_service,
            "list_assigned_system_tools",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await agent_service.assign_system_tools(db, agent_id, [])

    # delete still runs (idempotent wipe), then add_all is called with []
    # — clear-all is the explicit semantics.
    args, _kwargs = db.add_all.call_args
    assert args[0] == []
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_names_sorted_in_error_message(agent_id):
    from luna_core.services import agent as agent_service

    db = AsyncMock()
    # add_all is sync on a real SQLAlchemy AsyncSession; overriding the
    # AsyncMock default avoids a RuntimeWarning about the coroutine that
    # the production code never awaits (correctly).
    db.add_all = MagicMock()
    fake_registry = _build_registry("stash_records")
    with (
        patch.object(agent_service, "get_default_registry", return_value=fake_registry),
        patch.object(agent_service, "get_agent", new=AsyncMock(return_value=object())),
    ):
        with pytest.raises(ValueError) as exc:
            await agent_service.assign_system_tools(
                db, agent_id, ["zebra", "alpha", "zebra"]
            )
    # Sorted + deduplicated — keeps the error message stable across runs.
    assert "['alpha', 'zebra']" in str(exc.value)
