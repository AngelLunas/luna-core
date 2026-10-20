"""Every ledger row says who a call served and what it was doing.

The ledger had no owner: a host mapped scopes back to users kind by kind, and
whatever ran under a scope nobody recorded (a title, a look at photos, an
embedding) was spent by nobody. The entry point now sets ``usage_context`` and
every call inside inherits it, nested: ``chat`` → ``chat.title``.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from luna_core.services import usage
from luna_core.services.turn_lock import active_turns, conversation_turn, user_turns_key


def test_activities_nest_and_never_repeat():
    assert usage._nest(None, "chat") == "chat"
    assert usage._nest("chat", "title") == "chat.title"
    assert usage._nest("chat", None) == "chat"
    assert usage._nest("chat", "chat") == "chat"  # a handoff's turn inside a turn
    assert usage._nest("ronda.answers", "ronda") == "ronda.answers"  # a job reusing the review


def test_the_context_nests_inward_and_is_reset_on_exit():
    user = uuid.uuid4()
    with usage.usage_context(user_id=user, purpose="chat"):
        with usage.usage_context(purpose="title", agent_name="titler") as inner:
            assert (inner.user_id, inner.purpose, inner.agent_name) == (user, "chat.title", "titler")
        assert usage.current_attribution().purpose == "chat"
    assert usage.current_attribution() == usage.UsageAttribution()


class _DB:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)


@pytest.mark.asyncio
async def test_a_row_carries_the_current_attribution():
    db, user = _DB(), uuid.uuid4()

    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 100, 10, 110

    with usage.usage_context(user_id=user, purpose="chat"), usage.usage_context(purpose="image", agent_name="vision"):
        await usage.record_usage(db, scope_id=uuid.uuid4(), message_id=None, model="m", usage=Usage())
    row = db.rows[0]
    assert (row.user_id, row.purpose, row.agent_name) == (user, "chat.image", "vision")


@pytest.mark.asyncio
async def test_an_explicit_user_wins_and_no_context_leaves_the_row_unowned():
    db, voice_user = _DB(), uuid.uuid4()
    await usage.record_usage_counts(
        db, scope_id=voice_user, message_id=None, model="stt", input_tokens=5,
        output_tokens=1, total_tokens=6, user_id=voice_user,
    )
    await usage.record_usage_counts(
        db, scope_id=uuid.uuid4(), message_id=None, model="m", input_tokens=1,
        output_tokens=1, total_tokens=2,
    )
    assert db.rows[0].user_id == voice_user
    assert db.rows[1].user_id is None and db.rows[1].purpose is None


@pytest.mark.asyncio
async def test_the_request_user_is_bound_for_that_request_only():
    user = uuid.uuid4()

    async def request():
        usage.bind_request_user(user)
        return usage.current_attribution().user_id

    assert await asyncio.create_task(request()) == user
    assert usage.current_attribution().user_id is None  # the next request starts clean


def test_the_report_refuses_an_unknown_grouping():
    with pytest.raises(ValueError):
        asyncio.run(usage.usage_report(None, since=None, group_by=("colour",)))


class _Redis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.sets: dict[str, set] = {}

    async def set(self, key, value, *, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)

    async def exists(self, key):
        return key in self.values

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def srem(self, key, member):
        self.sets.get(key, set()).discard(member)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def expire(self, key, seconds):
        return True


@pytest.mark.asyncio
async def test_a_users_running_turns_are_counted_across_conversations():
    redis, user = _Redis(), uuid.uuid4()
    async with conversation_turn(redis, uuid.uuid4(), ttl_seconds=60, owner_id=user):
        async with conversation_turn(redis, uuid.uuid4(), ttl_seconds=60, owner_id=user):
            assert await active_turns(redis, user) == 2
        assert await active_turns(redis, user) == 1
    assert await active_turns(redis, user) == 0
    # A lock whose process died (its key expired) is not counted, and is dropped.
    redis.sets[user_turns_key(user)] = {"conversation_turn_lock:gone"}
    assert await active_turns(redis, user) == 0
    assert not redis.sets[user_turns_key(user)]
