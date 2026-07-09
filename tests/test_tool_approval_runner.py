"""Human-in-the-loop tool approval — runner semantics.

Covers the decision logic without a DB, mirroring the fake-driven style of the
other engine tests:
  - ``_should_reinvoke``: re-call the LLM unless the whole turn was rejected with
    no reasons.
  - ``_execute_tool_uses`` with decisions: a rejected tool_use produces a
    rejection tool_result and does NOT run the handler; an approved (or
    never-gated) one runs it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from luna_core.engine.agent import AgentRunner, _should_reinvoke
from luna_core.mcp.system_tools.registry import SystemTool
from luna_core.models.tool_approval import ToolApproval, ToolApprovalStatus


def _tool_use(tid: str, name: str = "log_care_event", inp: dict | None = None) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": inp or {}}


def _approval(tid: str, status: ToolApprovalStatus, reason: str | None = None):
    return ToolApproval(
        tool_use_id=tid,
        tool_name="log_care_event",
        status=status.value,
        reason=reason,
    )


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event_type, node_id=None, payload=None):
        self.events.append((event_type, payload))
        return SimpleNamespace(sequence=1)


def test_should_reinvoke_cases():
    a, b = _tool_use("a"), _tool_use("b")
    # approve → re-invoke
    assert _should_reinvoke([a], {"a": _approval("a", ToolApprovalStatus.approved)})
    # reject plain → do NOT re-invoke
    assert not _should_reinvoke(
        [a], {"a": _approval("a", ToolApprovalStatus.rejected)}
    )
    # reject WITH reason → re-invoke (LLM must acknowledge)
    assert _should_reinvoke(
        [a], {"a": _approval("a", ToolApprovalStatus.rejected, reason="do X")}
    )
    # mixed (one approved, one rejected-plain) → re-invoke
    assert _should_reinvoke(
        [a, b],
        {
            "a": _approval("a", ToolApprovalStatus.approved),
            "b": _approval("b", ToolApprovalStatus.rejected),
        },
    )
    # never-gated auto tool (no decision) counts as executed → re-invoke
    assert _should_reinvoke([a], {})


def _runner() -> AgentRunner:
    return AgentRunner(llm_router=SimpleNamespace(), mcp_client=SimpleNamespace())


def _tool(handler) -> SystemTool:
    return SystemTool(
        name="log_care_event",
        description="",
        input_schema={},
        handler=handler,
        scope="catalog",
        terminal=False,
    )


@pytest.mark.asyncio
async def test_rejected_tool_use_skips_handler():
    called: list = []

    async def handler(args, *, call_context):
        called.append(args)
        return {"ok": True}

    blocks, terminal, _tv = await _runner()._execute_tool_uses(
        [_tool_use("a")],
        decisions={"a": _approval("a", ToolApprovalStatus.rejected, reason="no")},
        system_by_name={"log_care_event": _tool(handler)},
        call_context={},
        emitter=_FakeEmitter(),
        node_id="chat",
    )

    assert called == []  # handler never ran
    assert terminal is False
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "a"
    assert blocks[0]["is_error"] is True
    assert "rejected" in blocks[0]["content"].lower()
    assert "no" in blocks[0]["content"]  # the reason is surfaced to the LLM


@pytest.mark.asyncio
async def test_approved_tool_use_runs_handler():
    called: list = []

    async def handler(args, *, call_context):
        called.append(args)
        return {"saved": True}

    blocks, _terminal, _tv = await _runner()._execute_tool_uses(
        [_tool_use("a", inp={"x": 1})],
        decisions={"a": _approval("a", ToolApprovalStatus.approved)},
        system_by_name={"log_care_event": _tool(handler)},
        call_context={},
        emitter=_FakeEmitter(),
        node_id="chat",
    )

    assert called == [{"x": 1}]  # handler ran with the tool input
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "a"
    assert "is_error" not in blocks[0]
