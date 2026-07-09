"""LLM token-usage ledger service.

``record_usage`` is the single write point: the streaming provider calls it once
per completed turn with the provider's ``usage`` object (OpenAI-shaped). Token
fields are read defensively — providers vary in what they report, and cached
tokens live under ``prompt_tokens_details.cached_tokens`` when present.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.usage import LLMUsage


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _cached_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return _int(cached) if cached is not None else None


async def record_usage(
    db: AsyncSession,
    *,
    scope_id: uuid.UUID,
    message_id: uuid.UUID | None,
    model: str,
    usage: Any,
) -> LLMUsage:
    """Persist one usage row from a provider ``usage`` object. Does NOT commit —
    the caller (provider) commits the turn's session as one unit."""
    row = LLMUsage(
        scope_id=scope_id,
        message_id=message_id,
        model=model,
        input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
        output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        cached_input_tokens=_cached_tokens(usage),
        total_tokens=_int(getattr(usage, "total_tokens", 0)),
    )
    db.add(row)
    return row


async def usage_totals_for_scope(
    db: AsyncSession, scope_id: uuid.UUID
) -> dict[str, int]:
    """Summed token counts for one execution scope — what that flow run /
    conversation has cost so far."""
    result = await db.execute(
        select(
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0),
        ).where(LLMUsage.scope_id == scope_id)
    )
    input_tokens, output_tokens, total_tokens = result.one()
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
