"""One turn at a time per conversation.

Two turns running over the same conversation both load the same history, both
pay for a full model call, and interleave their messages in the transcript.
``conversation_turn`` holds a Redis lock for the length of a turn (a send, an
approval decision and the resume it triggers); a caller that does not get it
refuses the request instead of waiting. The TTL only matters when the process
holding the lock dies mid-turn: the lock is otherwise released on exit.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis


def turn_lock_key(conversation_id: uuid.UUID) -> str:
    return f"conversation_turn_lock:{conversation_id}"


@asynccontextmanager
async def conversation_turn(
    redis: Redis, conversation_id: uuid.UUID, *, ttl_seconds: int
) -> AsyncIterator[bool]:
    """Yield whether this caller holds the conversation's turn lock. Released
    on exit only by its holder (a lock that expired and was taken by another
    turn is left alone)."""
    key = turn_lock_key(conversation_id)
    token = uuid.uuid4().hex
    acquired = bool(await redis.set(key, token, nx=True, ex=ttl_seconds))
    try:
        yield acquired
    finally:
        if acquired:
            held = await redis.get(key)
            if isinstance(held, bytes):
                held = held.decode()
            if held == token:
                await redis.delete(key)


__all__ = ["conversation_turn", "turn_lock_key"]
