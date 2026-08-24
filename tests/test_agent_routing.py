"""The generic multi-agent routing pieces: the transfer tool built from host
cards, the cold-start classifier's tolerant parsing, and the handoff preamble
the conversation router appends to continuation turns."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.services.agent_routing import (
    AgentCard,
    build_directory,
    classify_route,
    handoff_preamble,
    make_transfer_tool,
)

ORCH = AgentCard(
    name="orch", display_name="Assistant", purpose="Default agent."
)
DOC = AgentCard(
    name="doc", display_name="Doctor", purpose="Health consults."
)
LOCKED = AgentCard(
    name="locked",
    display_name="Locked",
    purpose="May only hand back to orch.",
    targets=("orch",),
)
CARDS = (ORCH, DOC, LOCKED)


def _tool(calls: list[tuple[uuid.UUID, str]]):
    async def set_active(conversation_id: uuid.UUID, agent_name: str) -> None:
        calls.append((conversation_id, agent_name))

    return make_transfer_tool(cards=CARDS, set_active=set_active)


def _ctx(caller: str, scope: uuid.UUID | None = None) -> dict:
    return {"agent_name": caller, "scope_id": str(scope or uuid.uuid4())}


# --- directory + tool shape -------------------------------------------------

def test_directory_lists_every_card_and_can_exclude():
    directory = build_directory(CARDS)
    assert "orch — Assistant: Default agent." in directory
    assert "doc — Doctor" in directory
    assert "doc" not in build_directory(CARDS, exclude="doc").split("\n")[0]


def test_tool_is_terminal_and_enumerates_cards():
    tool = _tool([])
    assert tool.terminal
    assert sorted(tool.input_schema["properties"]["agent"]["enum"]) == [
        "doc", "locked", "orch",
    ]
    # The directory rides in the description so every agent holding the tool
    # knows who exists without prompt changes.
    assert "doc — Doctor: Health consults." in tool.description


# --- handler behaviour ------------------------------------------------------

@pytest.mark.asyncio
async def test_transfer_flips_routing_and_addresses_the_next_agent():
    calls: list[tuple[uuid.UUID, str]] = []
    scope = uuid.uuid4()
    result = await _tool(calls).handler(
        {"agent": "doc", "reason": "pest report"},
        call_context=_ctx("orch", scope),
    )
    assert calls == [(scope, "doc")]
    assert result["transferred_to"] == "doc"
    # The note is read by the RECEIVING agent in the transcript — it must name
    # it as the speaker and forbid the announce/greet anti-patterns.
    note = result["handoff_note"]
    assert "Doctor" in note and "never announce a transfer" in note


@pytest.mark.asyncio
async def test_transfer_soft_fails_on_unknown_self_and_disallowed_targets():
    calls: list[tuple[uuid.UUID, str]] = []
    tool = _tool(calls)
    assert "error" in await tool.handler(
        {"agent": "nope"}, call_context=_ctx("orch")
    )
    assert "error" in await tool.handler(
        {"agent": "orch"}, call_context=_ctx("orch")
    )
    # ``locked`` may only reach orch; doc is refused.
    assert "error" in await tool.handler(
        {"agent": "doc"}, call_context=_ctx("locked")
    )
    assert calls == []  # nothing flipped on any refusal


@pytest.mark.asyncio
async def test_transfer_without_scope_raises_programming_error():
    with pytest.raises(RuntimeError):
        await _tool([]).handler({"agent": "doc"}, call_context={"agent_name": "orch"})


# --- cold-start classifier --------------------------------------------------

def _llm(reply: str):
    async def complete(**_kw):
        return [{"type": "text", "text": reply}]

    return SimpleNamespace(complete=complete)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        '{"agent": "doc", "reason": "symptoms"}',
        '```json\n{"agent": "doc", "reason": "symptoms"}\n```',
        'Sure! Here is my pick: {"agent": "doc", "reason": "symptoms"}',
    ],
)
async def test_classifier_parses_clean_fenced_and_embedded_json(reply):
    decision = await classify_route(
        llm_router=_llm(reply),
        provider_id=uuid.uuid4(),
        model="cheap",
        cards=CARDS,
        message="my plant has spots",
        default="orch",
        scope_id=uuid.uuid4(),
    )
    assert decision.agent == "doc"
    assert decision.reason == "symptoms"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["gibberish", '{"agent": "intruder"}', ""])
async def test_classifier_degrades_to_default_on_unusable_output(reply):
    decision = await classify_route(
        llm_router=_llm(reply),
        provider_id=uuid.uuid4(),
        model="cheap",
        cards=CARDS,
        message="hola",
        default="orch",
        scope_id=uuid.uuid4(),
    )
    assert decision.agent == "orch"


@pytest.mark.asyncio
async def test_classifier_rejects_a_default_outside_the_cards():
    with pytest.raises(ValueError):
        await classify_route(
            llm_router=_llm("{}"),
            provider_id=uuid.uuid4(),
            model="cheap",
            cards=CARDS,
            message="hola",
            default="ghost",
            scope_id=uuid.uuid4(),
        )


# --- handoff preamble -------------------------------------------------------

def test_preamble_names_the_receiver_and_forbids_the_anti_patterns():
    agent = SimpleNamespace(name="doc", role="Plant doctor")
    text = handoff_preamble(agent)
    assert "doc (Plant doctor)" in text
    assert "do not announce any transfer" in text
    assert "not by you" in text
