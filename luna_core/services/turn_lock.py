"""One turn at a time per conversation, and how many a user has running.

Two turns running over the same conversation both load the same history, both
pay for a full model call, and interleave their messages in the transcript.
``conversation_turn`` holds a Redis lock for the length of a turn (a send, an
approval decision and the resume it triggers); a caller that does not get it
refuses the request instead of waiting. The TTL only matters when the process
holding the lock dies mid-turn: the lock is otherwise released on exit.

With an ``owner_id``, the lock is also listed under its user, so a host can
count the turns one account has running across all its conversations
(``active_turns``) — the lock per conversation does not stop ten conversations
running at once. A listed lock whose key has expired (its process died) is not
counted, and is dropped when next counted.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis


def turn_lock_key(conversation_id: uuid.UUID) -> str:
    return f"conversation_turn_lock:{conversation_id}"


def user_turns_key(owner_id: uuid.UUID) -> str:
    return f"user_turns:{owner_id}"


def _text(value: object) -> str | None:
    return value.decode() if isinstance(value, bytes) else value  # type: ignore[return-value]


@asynccontextmanager
async def conversation_turn(
    redis: Redis,
    conversation_id: uuid.UUID,
    *,
    ttl_seconds: int,
    owner_id: uuid.UUID | None = None,
) -> AsyncIterator[bool]:
    """Yield whether this caller holds the conversation's turn lock. Released
    on exit only by its holder (a lock that expired and was taken by another
    turn is left alone)."""
    key = turn_lock_key(conversation_id)
    token = uuid.uuid4().hex
    acquired = bool(await redis.set(key, token, nx=True, ex=ttl_seconds))
    listed = acquired and owner_id is not None
    if listed:
        await redis.sadd(user_turns_key(owner_id), key)
        await redis.expire(user_turns_key(owner_id), ttl_seconds)
    try:
        yield acquired
    finally:
        if listed:
            await redis.srem(user_turns_key(owner_id), key)
        if acquired and _text(await redis.get(key)) == token:
            await redis.delete(key)


async def active_turns(redis: Redis, owner_id: uuid.UUID) -> int:
    """How many turns ``owner_id`` has running right now."""
    listing = user_turns_key(owner_id)
    count = 0
    for member in await redis.smembers(listing):
        key = _text(member)
        if await redis.exists(key):
            count += 1
        else:
            await redis.srem(listing, key)
    return count


__all__ = ["active_turns", "conversation_turn", "turn_lock_key", "user_turns_key"]
