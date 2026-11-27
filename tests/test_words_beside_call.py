"""A call the user sees never reaches them alone: a tool with a ``words_cue``
called with no text of its own makes the model write, once, the message that
goes with it — above the call, in the same assistant message — and then runs.
A call that came with words, or a tool with no cue, costs no extra request.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.engine.agent import AgentRunner
from luna_core.llm.base import ToolDefinition
from luna_core.mcp.system_tools.registry import SystemTool

CUE = "Say what sets each option apart and which you recommend."


class _Emitter:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.saved: list = []

    async def emit(self, event_type, node_id=None, payload=None):
        return SimpleNamespace(sequence=1)

    async def save_message(self, node_id=None, role=None, content=None):
        self.saved.append((role, content))

    async def add_text_to_last_assistant(self, text: str):
        self.added.append(text)

    def live_only(self):
        return SimpleNamespace(for_session=lambda db: None)

    def for_session(self, db):
        return self


class _ScriptedLLM:
    def __init__(self, replies: list[list[dict]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _ask(call_id: str = "q1") -> dict:
    return {"type": "tool_use", "id": call_id, "name": "ask_choice", "input": {"question": "¿Cuándo?"}}


async def _run(monkeypatch, llm: _ScriptedLLM, emitter: _Emitter, *, cue: str | None = CUE):
    ran: list = []

    async def ask_choice(args, *, call_context):
        ran.append(args)
        return {"shown": True}

    tool = SystemTool(
        name="ask_choice", description="", input_schema={"type": "object"},
        handler=ask_choice, scope="catalog", terminal=True, words_cue=cue,
    )
    runner = AgentRunner(llm_router=llm, mcp_client=SimpleNamespace())

    async def resolve_tools(db, agent_id, context_tool_names):
        return [ToolDefinition(name="ask_choice", input_schema={"type": "object"})], {"ask_choice": tool}

    async def approval_names(db, agent_id):
        return set()

    monkeypatch.setattr(runner, "_resolve_tools", resolve_tools)
    monkeypatch.setattr(runner, "_approval_required_names", approval_names)
    agent = SimpleNamespace(
        id=uuid.uuid4(), name="savia-mentor", llm_provider_id=None, temperature=0.2,
        model="m", reasoning_effort=None, output_schema=None, builtin_tools=None,
    )
    result = await runner.run(
        agent=agent, history=[], new_message="Quiero un plan", scope_id=uuid.uuid4(),
        node_id="chat", emitter=emitter, db=None, redis=None, system_prompt="s",
        approval_enabled=True,
    )
    return result, ran


@pytest.mark.asyncio
async def test_a_card_with_no_words_gets_them_once_above_it(monkeypatch):
    llm = _ScriptedLLM([[_ask()], [{"type": "text", "text": "Julio es lo menos lluvioso; te lo recomiendo."}]])
    emitter = _Emitter()
    result, ran = await _run(monkeypatch, llm, emitter)

    assert ran == [{"question": "¿Cuándo?"}]  # the card still goes on screen
    assert result == {"question": "¿Cuándo?"}  # and it ends the turn
    assert emitter.added == ["Julio es lo menos lluvioso; te lo recomiendo."]
    assert len(llm.calls) == 2
    words_call = llm.calls[1]["messages"]
    answered = words_call[-1]["content"]
    assert answered[0]["type"] == "tool_result" and answered[0]["tool_use_id"] == "q1"
    assert "not shown yet" in answered[0]["content"].lower()
    assert CUE in answered[-1]["text"]


@pytest.mark.asyncio
async def test_a_card_that_came_with_words_needs_no_second_request(monkeypatch):
    llm = _ScriptedLLM([[{"type": "text", "text": "Enero cae en lluvias; julio no."}, _ask()]])
    emitter = _Emitter()
    _, ran = await _run(monkeypatch, llm, emitter)
    assert len(llm.calls) == 1 and emitter.added == [] and len(ran) == 1


@pytest.mark.asyncio
async def test_a_tool_with_no_cue_runs_without_words(monkeypatch):
    llm = _ScriptedLLM([[_ask()]])
    emitter = _Emitter()
    _, ran = await _run(monkeypatch, llm, emitter, cue=None)
    assert len(llm.calls) == 1 and emitter.added == [] and len(ran) == 1
