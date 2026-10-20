"""Guards around a chat turn: what it may carry, one at a time per conversation,
and the side call that titles it.

- A turn over the caps is refused BEFORE anything runs or is stored, with a
  stable code (and the figures) the client words for the user — who gets their
  message back to fix.
- Two turns over the same conversation both paid a full model call and
  interleaved their messages; the second is now refused while the first runs.
- The title came from the answering agent's own model (a large one, for some
  agents) and its usage was ledgered under a random scope nobody could
  attribute; it now runs on the provider's fast model, under the conversation.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from luna_core.core.config import settings
from luna_core.routers.conversations import _check_turn_size, _turn_in_progress
from luna_core.services import auto_title
from luna_core.services.turn_lock import conversation_turn, turn_lock_key


def test_a_message_over_the_cap_is_refused_with_its_figures():
    _check_turn_size("x" * settings.chat_max_message_chars, settings.chat_max_attachments)
    with pytest.raises(HTTPException) as exc:
        _check_turn_size("x" * (settings.chat_max_message_chars + 1), 0)
    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "message_too_long"
    assert exc.value.detail["limit"] == settings.chat_max_message_chars
    assert exc.value.detail["count"] == settings.chat_max_message_chars + 1
    assert exc.value.detail["message"]


def test_too_many_attachments_are_refused():
    with pytest.raises(HTTPException) as exc:
        _check_turn_size("hola", settings.chat_max_attachments + 1)
    assert exc.value.detail["code"] == "too_many_attachments"
    assert exc.value.detail["limit"] == settings.chat_max_attachments


def test_a_busy_conversation_refuses_with_a_code():
    exc = _turn_in_progress()
    assert exc.status_code == 409 and exc.detail["code"] == "turn_in_progress"


class _Redis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_one_turn_at_a_time_per_conversation():
    redis = _Redis()
    conversation, other = uuid.uuid4(), uuid.uuid4()
    async with conversation_turn(redis, conversation, ttl_seconds=60) as first:
        assert first
        async with conversation_turn(redis, conversation, ttl_seconds=60) as second:
            assert not second
        # Another conversation is not held up.
        async with conversation_turn(redis, other, ttl_seconds=60) as elsewhere:
            assert elsewhere
        # The refused caller did not release the holder's lock.
        assert turn_lock_key(conversation) in redis.store
    assert turn_lock_key(conversation) not in redis.store
    async with conversation_turn(redis, conversation, ttl_seconds=60) as again:
        assert again


@pytest.mark.asyncio
async def test_a_lock_taken_over_after_expiry_is_not_released_by_the_old_holder():
    redis = _Redis()
    conversation = uuid.uuid4()
    async with conversation_turn(redis, conversation, ttl_seconds=60) as held:
        assert held
        redis.store[turn_lock_key(conversation)] = "someone-else"  # expired, retaken
    assert redis.store[turn_lock_key(conversation)] == "someone-else"


class _Router:
    def __init__(self, fast):
        self.fast = fast
        self.calls: list[dict] = []

    async def fast_model(self, _provider_id):
        if isinstance(self.fast, Exception):
            raise self.fast
        return self.fast

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return [{"type": "text", "text": "Riego de la semana"}]


@pytest.mark.asyncio
async def test_the_title_runs_on_the_fast_model_and_counts_against_the_conversation():
    router = _Router("small-model")
    model = await auto_title._title_model(router, uuid.uuid4(), "large-model")
    assert model == "small-model"
    conversation = uuid.uuid4()
    text = await auto_title._generate_title(
        router, uuid.uuid4(), model, "¿riego hoy?", scope_id=conversation
    )
    assert text == "Riego de la semana"
    call = router.calls[0]
    assert call["model"] == "small-model"
    assert call["make_io"](None).scope_id == conversation
    assert call["run_id"] != conversation
    # The message is data to name, not a question to the titler.
    assert call["messages"][0]["content"][0]["text"] == "<message>\n¿riego hoy?\n</message>"
    assert call["output_schema"]["required"] == ["title"]


@pytest.mark.asyncio
async def test_a_structured_answer_gives_the_title_alone():
    class Structured(_Router):
        async def complete(self, **kwargs):
            return [{"type": "text", "text": '{"title": "Riego con CalMag"}'}]

    text = await auto_title._generate_title(
        Structured("small"), uuid.uuid4(), "small", "¿cuánto CalMag?", scope_id=uuid.uuid4()
    )
    assert text == "Riego con CalMag"


@pytest.mark.asyncio
async def test_without_a_fast_model_the_title_keeps_the_callers_model():
    assert await auto_title._title_model(_Router(None), uuid.uuid4(), "large") == "large"
    assert await auto_title._title_model(_Router(RuntimeError("down")), uuid.uuid4(), "large") == "large"
