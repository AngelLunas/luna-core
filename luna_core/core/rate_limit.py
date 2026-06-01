from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: int


async def check_rate_limit(
    redis: Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    """Fixed-window counter using INCR + EXPIRE.

    The first request in a window sets the TTL; subsequent requests in the
    same window inherit it. Returns RateLimitResult.allowed=False once the
    counter exceeds `limit`.
    """
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.ttl(key)
    count, ttl = await pipe.execute()

    if ttl == -1:
        await redis.expire(key, window_seconds)
        ttl = window_seconds

    remaining = max(0, limit - int(count))
    retry_after = int(ttl) if ttl and ttl > 0 else window_seconds

    return RateLimitResult(
        allowed=int(count) <= limit,
        remaining=remaining,
        retry_after=retry_after,
    )


async def reset_rate_limit(redis: Redis, key: str) -> None:
    await redis.delete(key)
