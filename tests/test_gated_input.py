"""A gated call is read against its schema before it is put to the user.

- ``read_tool_input`` decodes JSON text where the schema wants a list or an
  object, and names what cannot be read (with line and column) or is missing.
- A proposal that cannot be read never becomes an approval: the model gets the
  problem as a tool_result and the loop goes on.
- On resume, the stored call is read the same way, so a decoded proposal is
  not mistaken for a correction by the user.
- The words beside an approval go out as the turn's own call did, so the
  provider's prompt cache matches.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

from luna_core.engine import agent as agent_mod
from luna_core.engine.agent import AgentRunner, SuspendedForApproval
from luna_core.engine.tool_input import read_tool_input
from luna_core.llm.base import ToolDefinition
from luna_core.mcp.system_tools.registry import SystemTool
from luna_core.models.tool_approval import ToolApproval, ToolApprovalStatus

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "crop": {"type": "string"},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entries": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    },
    "required": ["crop", "phases"],
}


# --- read_tool_input -----------------------------------------------------------


def test_json_text_is_decoded_where_a_list_is_expected():
    phases = [{"name": "Plántula", "entries": [{"title": "Regar", "type": "routine"}]}]
    read, problems = read_tool_input(
        {"crop": "Huerto", "phases": json.dumps(phases)}, PLAN_SCHEMA
    )
    assert problems == []
    assert read == {"crop": "Huerto", "phases": phases}


def test_unreadable_text_is_named_with_line_and_column():
    broken = '[{"name":"Plántula","entries":[{"title":"Abrigar","type":checkpoint}]}]'
    read, problems = read_tool_input({"crop": "Huerto", "phases": broken}, PLAN_SCHEMA)
    assert read["phases"] == broken  # left as sent
    assert len(problems) == 1
    assert problems[0].startswith("`phases` must be a JSON array")
    assert "line 1, column" in problems[0]


def test_nested_text_is_decoded_inside_items():
    entries = [{"title": "Regar"}]
    read, problems = read_tool_input(
        {"crop": "Huerto", "phases": [{"name": "Plántula", "entries": json.dumps(entries)}]},
        PLAN_SCHEMA,
    )
    assert problems == []
    assert read["phases"][0]["entries"] == entries


def test_nested_unreadable_text_names_its_path():
    _read, problems = read_tool_input(
        {"crop": "Huerto", "phases": [{"name": "Plántula", "entries": "[{oops}]"}]},
        PLAN_SCHEMA,
    )
    assert problems and problems[0].startswith("`phases[0].entries` must be a JSON array")


def test_a_required_field_left_out_is_a_problem():
    _read, problems = read_tool_input({"phases": []}, PLAN_SCHEMA)
    assert problems == ["`crop` is required"]


def test_text_that_parses_to_the_wrong_kind_is_a_problem():
    _read, problems = read_tool_input({"crop": "Huerto", "phases": '{"a": 1}'}, PLAN_SCHEMA)
    assert problems == ["`phases` must be a JSON array"]


def test_plain_strings_and_unknown_fields_are_left_alone():
    read, problems = read_tool_input(
        {"crop": "[not a list]", "phases": [], "extra": "{x}"}, PLAN_SCHEMA
    )
    assert problems == []
    assert read == {"crop": "[not a list]", "phases": [], "extra": "{x}"}


# --- the run loop --------------------------------------------------------------


class _Emitter:
    def __init__(self) -> None:
        self.events: list = []
        self.saved: list = []

    async def emit(self, event_type, node_id=None, payload=None):
        self.events.append((event_type, payload))
        return SimpleNamespace(sequence=1)

    async def save_message(self, node_id=None, role=None, content=None):
        self.saved.append((role, content))

    def for_session(self, db):
        return self


class _ScriptedLLM:
    def __init__(self, replies: list[list[dict]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _plan_call(call_id: str, phases) -> dict:
    return {
        "type": "tool_use",
        "id": call_id,
        "name": "create_crop_plan",
        "input": {"crop": "Huerto", "phases": phases},
    }


def _note_call(call_id: str) -> dict:
    return {"type": "tool_use", "id": call_id, "name": "remember", "input": {"text": "x"}}


async def _run(monkeypatch, llm: _ScriptedLLM, emitter: _Emitter, ran: list):
    created: list = []

    async def fake_create(db, *, conversation_id, tool_uses, agent_name=None):
        rows = [
            ToolApproval(
                tool_use_id=tu["id"], tool_name=tu["name"], tool_input=tu["input"],
                status=ToolApprovalStatus.pending.value,
            )
            for tu in tool_uses
        ]
        created.extend(rows)
        return rows

    async def remember(args, *, call_context):
        ran.append(args)
        return {"ok": True}

    note_tool = SystemTool(
        name="remember", description="", input_schema={"type": "object"},
        handler=remember, scope="catalog",
    )
    runner = AgentRunner(llm_router=llm, mcp_client=SimpleNamespace())

    async def resolve_tools(db, agent_id, context_tool_names):
        return (
            [
                ToolDefinition(name="create_crop_plan", input_schema=PLAN_SCHEMA),
                ToolDefinition(name="remember", input_schema={"type": "object"}),
            ],
            {"remember": note_tool},
        )

    async def approval_names(db, agent_id):
        return {"create_crop_plan"}

    monkeypatch.setattr(runner, "_resolve_tools", resolve_tools)
    monkeypatch.setattr(runner, "_approval_required_names", approval_names)
    monkeypatch.setattr(agent_mod, "create_pending_approvals", fake_create)
    agent = SimpleNamespace(
        id=uuid.uuid4(), name="savia-mentor", llm_provider_id=None, temperature=0.2,
        model="m", reasoning_effort=None, output_schema=None, builtin_tools=None,
    )
    result = await runner.run(
        agent=agent, history=[], new_message="Quiero crear el plan", scope_id=uuid.uuid4(),
        node_id="chat", emitter=emitter, db=None, redis=None, system_prompt="s",
        approval_enabled=True,
    )
    return result, created


@pytest.mark.asyncio
async def test_an_unreadable_proposal_goes_back_to_the_model(monkeypatch):
    good = [{"name": "Plántula", "entries": []}]
    llm = _ScriptedLLM([
        [
            {"type": "text", "text": "Te dejo el plan."},
            _note_call("n1"),
            _plan_call("p1", '[{"name":"Plántula","entries":[{"type":checkpoint}]}]'),
        ],
        [{"type": "text", "text": "Te dejo el plan."}, _plan_call("p2", good)],
    ])
    emitter, ran = _Emitter(), []
    result, created = await _run(monkeypatch, llm, emitter, ran)

    assert isinstance(result, SuspendedForApproval)
    assert [a.tool_use_id for a in created] == ["p2"]  # only the readable one was put to the user
    assert created[0].tool_input["phases"] == good
    assert ran == [{"text": "x"}]  # the call that needs no approval ran in the refused turn

    answers = [
        m["content"] for m in llm.calls[1]["messages"]
        if m["role"] == "user" and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(answers) == 1
    by_id = {b["tool_use_id"]: b for b in answers[0]}
    assert list(by_id) == ["n1", "p1"]  # every call of the turn answered, in order
    assert "is_error" not in by_id["n1"]
    assert by_id["p1"]["is_error"] is True
    assert "`phases` must be a JSON array" in by_id["p1"]["content"]


@pytest.mark.asyncio
async def test_json_text_that_reads_becomes_a_proposal_with_the_list(monkeypatch):
    phases = [{"name": "Plántula", "entries": []}]
    llm = _ScriptedLLM([[{"type": "text", "text": "Listo."}, _plan_call("p1", json.dumps(phases))]])
    result, created = await _run(monkeypatch, llm, _Emitter(), [])

    assert isinstance(result, SuspendedForApproval)
    assert created[0].tool_input["phases"] == phases
    assert len(llm.calls) == 1  # no extra round trip for text that reads


@pytest.mark.asyncio
async def test_a_decoded_proposal_is_not_taken_for_a_user_correction():
    phases = [{"name": "Plántula", "entries": []}]
    seen: list = []

    async def handler(args, *, call_context):
        seen.append(args)
        return {"saved": True}

    tool = SystemTool(
        name="create_crop_plan", description="", input_schema=PLAN_SCHEMA,
        handler=handler, scope="catalog",
    )
    approval = ToolApproval(
        tool_use_id="p1", tool_name="create_crop_plan",
        status=ToolApprovalStatus.approved.value,
        tool_input={"crop": "Huerto", "phases": phases},  # as read before approval
    )
    blocks, _t, _v = await AgentRunner(
        llm_router=SimpleNamespace(), mcp_client=SimpleNamespace()
    )._execute_tool_uses(
        [_plan_call("p1", json.dumps(phases))],  # as the stored message holds it
        decisions={"p1": approval},
        system_by_name={"create_crop_plan": tool},
        call_context={},
        emitter=_Emitter(),
        node_id="chat",
    )
    assert seen == [{"crop": "Huerto", "phases": phases}]
    assert "edited_by_user" not in json.loads(blocks[0]["content"])


# --- words beside an approval ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_words_call_goes_out_as_the_turn_did():
    class _LLM:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def complete(self, **kwargs):
            self.kwargs = kwargs
            return [{"type": "text", "text": "Hola"}]

    class _Conv(_Emitter):
        async def add_text_to_last_assistant(self, text):
            pass

        def live_only(self):
            return SimpleNamespace(for_session=lambda db: None)

    llm = _LLM()
    policy, resolver = object(), object()
    agent = SimpleNamespace(
        name="a", llm_provider_id=None, temperature=0.2, model="m",
        reasoning_effort=None, builtin_tools=["web_search"],
    )
    await AgentRunner(llm_router=llm, mcp_client=SimpleNamespace())._words_beside(
        result="r", cue="c", agent=agent, messages=[], tool_uses=[], system="s", tools=[], scope_id=None,
        node_id="chat", emitter=_Conv(), timezone=None,
        image_resolver=resolver, context_policy=policy,
    )
    assert llm.kwargs["builtin_tools"] == ["web_search"]
    assert llm.kwargs["context_policy"] is policy
    assert llm.kwargs["image_resolver"] is resolver
