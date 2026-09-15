"""The runner hands the caller's timezone (``extra_call_context["timezone"]``)
to every model call, so a provider that has anything tied to local time — the
Claude CLI's own date note, a web search's location — uses the user's, never
the server's."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from luna_core.engine.agent import AgentRunner


class _Emitter:
    def for_session(self, _db):
        return self

    async def emit(self, *_a, **_k):
        return SimpleNamespace(sequence=1)

    async def save_message(self, **_k):
        return SimpleNamespace(id=uuid.uuid4())


class _RecordingRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return [{"type": "text", "text": "ok"}]


async def _run(extra_call_context):
    router = _RecordingRouter()
    runner = AgentRunner(llm_router=router, mcp_client=SimpleNamespace())  # type: ignore[arg-type]

    async def _no_tools(_db, _agent_id, _names):
        return [], {}

    runner._resolve_tools = _no_tools  # type: ignore[method-assign]
    agent = SimpleNamespace(
        id=uuid.uuid4(), name="a", model="m", temperature=0.0,
        llm_provider_id=uuid.uuid4(), output_schema=None, builtin_tools=[],
    )
    await runner.run(
        agent,  # type: ignore[arg-type]
        history=[], new_message="hola", scope_id=uuid.uuid4(), node_id="chat",
        emitter=_Emitter(), db=None, redis=None,  # type: ignore[arg-type]
        system_prompt="s", extra_call_context=extra_call_context,
    )
    return router.calls


@pytest.mark.asyncio
async def test_the_callers_timezone_reaches_the_model_call():
    calls = await _run({"user_id": "u", "timezone": "America/Bogota"})
    assert [c["timezone"] for c in calls] == ["America/Bogota"]


@pytest.mark.asyncio
async def test_no_timezone_is_passed_as_none():
    calls = await _run({"user_id": "u"})
    assert [c["timezone"] for c in calls] == [None]
