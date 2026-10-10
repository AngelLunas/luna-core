"""Host context riding a user turn (``{"type": "context"}``): every provider
shows it to the model as text, and the send route builds it from the host
hook — or leaves it out when there is no hook, no text, or the hook fails."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from luna_core.llm.providers.anthropic import _canonical_to_anthropic_messages
from luna_core.llm.providers.generic import (
    _canonical_to_openai_messages,
    _canonical_to_responses_input,
)
from luna_core.routers.conversations import _turn_context_block

_TURN = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "¿riego hoy?"},
            {"type": "context", "context": "NOW is 14:30"},
        ],
    }
]


def test_every_provider_renders_the_context_as_text():
    assert _canonical_to_anthropic_messages(_TURN, None, tools_declared=False) == [
        {"role": "user", "content": [{"type": "text", "text": "¿riego hoy?\nNOW is 14:30"}]}
    ]
    assert _canonical_to_openai_messages(_TURN, "") == [
        {"role": "user", "content": "¿riego hoy?\nNOW is 14:30"}
    ]
    assert _canonical_to_responses_input(_TURN) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "¿riego hoy?"},
                {"type": "input_text", "text": "NOW is 14:30"},
            ],
        }
    ]


def _request(hook):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chat_turn_context_provider=hook)))


@pytest.mark.asyncio
async def test_the_hook_text_becomes_a_context_block():
    async def hook(_db, _conversation, _agent):
        return "NOW is 14:30"

    block = await _turn_context_block(_request(hook), None, None, None)
    assert block == {"type": "context", "context": "NOW is 14:30"}


@pytest.mark.asyncio
async def test_no_hook_no_text_or_a_failing_hook_add_nothing():
    async def empty(*_a):
        return None

    async def boom(*_a):
        raise RuntimeError("down")

    no_hook = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert await _turn_context_block(no_hook, None, None, None) is None
    assert await _turn_context_block(_request(empty), None, None, None) is None
    assert await _turn_context_block(_request(boom), None, None, None) is None
