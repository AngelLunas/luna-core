"""The conversation router re-dispatches a turn when a handoff changes the active
agent (a terminal ``route_to_*`` tool flips the host's routing), so a second agent
answers in the same request — and a single-agent host is never affected."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.routers.conversations import (
    MAX_HANDOFF_HOPS,
    _augment_system_prompt,
    _follow_handoffs,
)


class _Agent:
    def __init__(self, name: str) -> None:
        self.id = uuid.uuid4()
        self.name = name
        self.instructions = "x"


def _request(runner, resolver) -> SimpleNamespace:
    state = SimpleNamespace(chat_runner=runner, chat_agent_resolver=resolver)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _runner(calls: list[str]):
    async def send(**kw):
        calls.append(kw["agent"].name)
        assert kw["new_message"] is None  # continuation, no new user turn
        return f"answer from {kw['agent'].name}"

    return SimpleNamespace(send=send)


@pytest.mark.asyncio
async def test_single_agent_host_never_redispatches():
    orch = _Agent("orch")
    calls: list[str] = []

    async def resolver(_db, _convo):
        return orch

    request = _request(_runner(calls), resolver)
    convo = SimpleNamespace(id=uuid.uuid4())
    agent, result = await _follow_handoffs(
        None, request, convo, None, uuid.uuid4(), orch, "original"
    )
    assert agent is orch
    assert result == "original"  # untouched
    assert calls == []  # runner never re-invoked


@pytest.mark.asyncio
async def test_one_handoff_redispatches_to_new_agent():
    orch, doc = _Agent("orch"), _Agent("doc")
    seq = [doc, doc]  # handoff to doc, then stable
    calls: list[str] = []

    async def resolver(_db, _convo):
        return seq.pop(0)

    request = _request(_runner(calls), resolver)
    convo = SimpleNamespace(id=uuid.uuid4())
    agent, result = await _follow_handoffs(
        None, request, convo, None, uuid.uuid4(), orch, "answer from orch"
    )
    assert agent.name == "doc"
    assert result == "answer from doc"
    assert calls == ["doc"]  # exactly one re-dispatch


@pytest.mark.asyncio
async def test_pingpong_is_capped():
    orch, doc = _Agent("orch"), _Agent("doc")
    flip = [doc, orch] * MAX_HANDOFF_HOPS * 2  # never stabilises
    calls: list[str] = []

    async def resolver(_db, _convo):
        return flip.pop(0)

    request = _request(_runner(calls), resolver)
    convo = SimpleNamespace(id=uuid.uuid4())
    _agent, _result = await _follow_handoffs(
        None, request, convo, None, uuid.uuid4(), orch, "x"
    )
    assert len(calls) == MAX_HANDOFF_HOPS  # bounded, no infinite loop


# --- optional RAG prompt augmentation -------------------------------------

def _req_with_provider(provider) -> SimpleNamespace:
    state = SimpleNamespace()
    if provider is not None:
        state.chat_context_provider = provider
    return SimpleNamespace(app=SimpleNamespace(state=state))


class _Ag:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.name = "doctor"


@pytest.mark.asyncio
async def test_augment_noop_without_provider():
    # No host hook → instructions are returned untouched (the feature is opt-in).
    request = _req_with_provider(None)
    out = await _augment_system_prompt(request, None, object(), _Ag("BASE"), "q")
    assert out == "BASE"


@pytest.mark.asyncio
async def test_augment_appends_provider_context():
    async def provider(_db, _conv, _agent, _query):
        return "PAST CASES"

    out = await _augment_system_prompt(
        _req_with_provider(provider), None, object(), _Ag("BASE"), "q"
    )
    assert out == "BASE\n\nPAST CASES"


@pytest.mark.asyncio
async def test_augment_skips_when_no_query_or_empty_context():
    async def provider(_db, _conv, _agent, _query):
        return None

    # empty query → provider not even consulted
    assert await _augment_system_prompt(
        _req_with_provider(provider), None, object(), _Ag("BASE"), ""
    ) == "BASE"
    # provider returns nothing → base unchanged
    assert await _augment_system_prompt(
        _req_with_provider(provider), None, object(), _Ag("BASE"), "q"
    ) == "BASE"


@pytest.mark.asyncio
async def test_augment_never_raises_on_provider_error():
    async def provider(_db, _conv, _agent, _query):
        raise RuntimeError("retrieval down")

    out = await _augment_system_prompt(
        _req_with_provider(provider), None, object(), _Ag("BASE"), "q"
    )
    assert out == "BASE"  # failures degrade to the base prompt
