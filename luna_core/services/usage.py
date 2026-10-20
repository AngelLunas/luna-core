"""LLM token-usage ledger service.

``record_usage`` is the single write point: the streaming provider calls it once
per completed turn with the provider's ``usage`` object (OpenAI-shaped). Token
fields are read defensively — providers vary in what they report, and cached
tokens live under ``prompt_tokens_details.cached_tokens`` when present.

Attribution. Every row says who the call served and what it was doing, from
``usage_context`` — set at each entry point (a chat turn, a background job, a
side call) and nested inward, so a title generated after a chat turn records
``chat.title`` for the same user without the titling code knowing either. The
context is a ``ContextVar``: it follows the request's awaits and the tasks
spawned from it, and is reset when its block exits.

Cost. Prices (``LLMPrice``) are applied when the ledger is read: a row's cost is
its uncached input, cache reads, cache writes, audio and output tokens at their
own prices. A model with no price row is charged at the dearest price known
for each kind of token, so a forgotten price never makes calls free; with no
price at all, cost is unknown (``None``) rather than zero.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Numeric, and_, case, cast, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.pricing import LLMPrice
from luna_core.models.usage import LLMUsage

logger = logging.getLogger(__name__)


# --- attribution --------------------------------------------------------------
@dataclass(frozen=True)
class UsageAttribution:
    """Who a model call serves and what it is doing. ``scope_id`` is the ledger
    key for calls that have no IO of their own (an embedding)."""

    user_id: uuid.UUID | None = None
    purpose: str | None = None
    agent_name: str | None = None
    scope_id: uuid.UUID | None = None


_ATTRIBUTION: ContextVar[UsageAttribution] = ContextVar(
    "luna_usage_attribution", default=UsageAttribution()
)


def current_attribution() -> UsageAttribution:
    return _ATTRIBUTION.get()


def _nest(outer: str | None, inner: str | None) -> str | None:
    """``chat`` + ``title`` → ``chat.title``. An activity already on the path
    is not added again: a handoff runs another chat turn inside the first, a
    job that reuses a review runs ``ronda`` inside ``ronda.answers``."""
    if not inner:
        return outer
    if not outer:
        return inner
    if inner in outer.split("."):
        return outer
    return f"{outer}.{inner}"


@contextmanager
def usage_context(
    *,
    user_id: uuid.UUID | None = None,
    purpose: str | None = None,
    agent_name: str | None = None,
    scope_id: uuid.UUID | None = None,
) -> Iterator[UsageAttribution]:
    """Attribute every model call inside the block. ``purpose`` nests under the
    enclosing one; the other fields replace it when given."""
    outer = _ATTRIBUTION.get()
    inner = replace(
        outer,
        user_id=user_id or outer.user_id,
        purpose=_nest(outer.purpose, purpose),
        agent_name=agent_name or outer.agent_name,
        scope_id=scope_id or outer.scope_id,
    )
    token = _ATTRIBUTION.set(inner)
    try:
        yield inner
    finally:
        _ATTRIBUTION.reset(token)


def bind_request_user(user_id: uuid.UUID) -> None:
    """Attribute the rest of this request to ``user_id`` (the authenticated
    principal): a model call any endpoint makes on the way — an embedding of a
    record being saved — counts for the account that asked. Set once per
    request task, never reset: a request runs in a task (and context) of its
    own, so nothing leaks into the next one."""
    if _ATTRIBUTION.get().user_id is None:
        _ATTRIBUTION.set(replace(_ATTRIBUTION.get(), user_id=user_id))


# --- writes -------------------------------------------------------------------
def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _cached_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    return _int(cached) if cached is not None else None


def _cache_write_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    written = getattr(details, "cache_write_tokens", None) if details is not None else None
    return _int(written) if written is not None else None


def _attributed(*, user_id: uuid.UUID | None = None, **fields: Any) -> LLMUsage:
    """A ledger row carrying the current attribution; an explicit ``user_id``
    (voice knows its user without a context) wins over the context's."""
    who = _ATTRIBUTION.get()
    return LLMUsage(
        user_id=user_id or who.user_id,
        purpose=who.purpose,
        agent_name=who.agent_name,
        **fields,
    )


async def record_usage(
    db: AsyncSession,
    *,
    scope_id: uuid.UUID,
    message_id: uuid.UUID | None,
    model: str,
    usage: Any,
) -> LLMUsage:
    """Persist one usage row from a provider ``usage`` object, attributed from
    the current ``usage_context``. Does NOT commit — the caller (provider)
    commits the turn's session as one unit."""
    row = _attributed(
        scope_id=scope_id,
        message_id=message_id,
        model=model,
        input_tokens=_int(getattr(usage, "prompt_tokens", 0)),
        output_tokens=_int(getattr(usage, "completion_tokens", 0)),
        cached_input_tokens=_cached_tokens(usage),
        cache_write_input_tokens=_cache_write_tokens(usage),
        total_tokens=_int(getattr(usage, "total_tokens", 0)),
    )
    db.add(row)
    return row


def usage_scope(emitter: Any, fallback: uuid.UUID) -> uuid.UUID:
    """The ledger key for a call made through ``emitter``: its
    ``usage_scope_id`` when it has one — a sub-agent streams on a scope of its
    own but its usage belongs to the run or conversation that called it —
    else its ``scope_id``, else ``fallback`` (the call's run id)."""
    return (
        getattr(emitter, "usage_scope_id", None)
        or getattr(emitter, "scope_id", None)
        or fallback
    )


async def record_usage_counts(
    db: AsyncSession,
    *,
    scope_id: uuid.UUID,
    message_id: uuid.UUID | None,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_input_tokens: int | None = None,
    audio_input_tokens: int | None = None,
    user_id: uuid.UUID | None = None,
) -> LLMUsage:
    """Persist one usage row from explicit counts — for sources whose usage
    payload is not OpenAI-chat-shaped (e.g. the Realtime transcription API).
    Voice STT sessions have no run/conversation, so their ``scope_id`` is the
    authenticated user id. Does NOT commit — the caller batches the session's
    rows and commits once."""
    row = _attributed(
        scope_id=scope_id,
        message_id=message_id,
        model=model,
        input_tokens=_int(input_tokens),
        output_tokens=_int(output_tokens),
        cached_input_tokens=cached_input_tokens,
        audio_input_tokens=audio_input_tokens,
        total_tokens=_int(total_tokens),
        user_id=user_id,
    )
    db.add(row)
    return row


async def record_embedding_usage(model: str, input_tokens: int) -> None:
    """Ledger one embedding call on its own session (an embedding has no turn
    to commit with), keyed by the context's scope, else its user. Best-effort:
    a failed write is logged, never raised into the caller's work."""
    from luna_core.core.db import AsyncSessionLocal

    who = _ATTRIBUTION.get()
    scope = who.scope_id or who.user_id or uuid.UUID(int=0)
    try:
        with usage_context(purpose="embedding"):
            async with AsyncSessionLocal() as db:
                await record_usage_counts(
                    db,
                    scope_id=scope,
                    message_id=None,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    total_tokens=input_tokens,
                )
                await db.commit()
    except Exception:  # noqa: BLE001 — bookkeeping never breaks an embedding
        logger.exception("embedding usage not recorded")


# --- reads --------------------------------------------------------------------
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


def purpose_filter(prefix: str):
    """Rows whose purpose is ``prefix`` or nested under it."""
    return or_(LLMUsage.purpose == prefix, LLMUsage.purpose.like(f"{prefix}.%"))


def _outside(prefix: str):
    """Rows NOT under ``prefix`` — including rows with no purpose at all."""
    purpose = func.coalesce(LLMUsage.purpose, "")
    return ~or_(purpose == prefix, purpose.like(f"{prefix}.%"))


async def _dearest(db: AsyncSession) -> dict[str, Decimal] | None:
    """The highest price of each kind of token across all rows; ``None`` when
    no price has been entered."""
    row = (
        await db.execute(
            select(
                func.max(LLMPrice.input_per_mtok),
                func.max(LLMPrice.output_per_mtok),
                func.max(func.coalesce(LLMPrice.cache_read_per_mtok, LLMPrice.input_per_mtok)),
                func.max(func.coalesce(LLMPrice.cache_write_per_mtok, LLMPrice.input_per_mtok)),
                func.max(func.coalesce(LLMPrice.audio_input_per_mtok, LLMPrice.input_per_mtok)),
            )
        )
    ).one()
    if row[0] is None:
        return None
    return dict(zip(("input", "output", "cache_read", "cache_write", "audio"), row))


def _cost_expr(dearest: dict[str, Decimal]):
    """USD of one ledger row, joined (outer) to its model's price."""

    def price(column, fallback: Decimal):
        return func.coalesce(column, literal(fallback, Numeric(12, 4)))

    p_in = price(LLMPrice.input_per_mtok, dearest["input"])
    p_out = price(LLMPrice.output_per_mtok, dearest["output"])
    p_read = func.coalesce(LLMPrice.cache_read_per_mtok, LLMPrice.input_per_mtok, literal(dearest["cache_read"], Numeric(12, 4)))
    p_write = func.coalesce(LLMPrice.cache_write_per_mtok, LLMPrice.input_per_mtok, literal(dearest["cache_write"], Numeric(12, 4)))
    p_audio = func.coalesce(LLMPrice.audio_input_per_mtok, LLMPrice.input_per_mtok, literal(dearest["audio"], Numeric(12, 4)))
    read = func.coalesce(LLMUsage.cached_input_tokens, 0)
    write = func.coalesce(LLMUsage.cache_write_input_tokens, 0)
    audio = func.coalesce(LLMUsage.audio_input_tokens, 0)
    plain = func.greatest(LLMUsage.input_tokens - read - write - audio, 0)
    total = plain * p_in + read * p_read + write * p_write + audio * p_audio + LLMUsage.output_tokens * p_out
    # Priced by prompt length: past the threshold, the whole call at the
    # multiple (the prompt is every input token, cache reads and writes too).
    tier = case(
        (
            and_(
                LLMPrice.long_context_threshold.is_not(None),
                LLMUsage.input_tokens > LLMPrice.long_context_threshold,
            ),
            func.coalesce(LLMPrice.long_context_multiplier, 1),
        ),
        else_=1,
    )
    return cast(total * tier, Numeric(20, 8)) / 1_000_000


async def spend_usd(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    since: datetime,
    purpose: str | None = None,
    exclude: str | None = None,
) -> Decimal | None:
    """What ``user_id`` spent since ``since``, in USD — only on ``purpose``
    (and what nests under it), or on everything but ``exclude``; ``None``
    when no price is known."""
    dearest = await _dearest(db)
    if dearest is None:
        return None
    stmt = (
        select(func.coalesce(func.sum(_cost_expr(dearest)), 0))
        .select_from(LLMUsage)
        .outerjoin(LLMPrice, LLMPrice.model == LLMUsage.model)
        .where(LLMUsage.user_id == user_id, LLMUsage.created_at >= since)
    )
    if purpose:
        stmt = stmt.where(purpose_filter(purpose))
    if exclude:
        stmt = stmt.where(_outside(exclude))
    return Decimal((await db.execute(stmt)).scalar_one())


REPORT_GROUPS = ("user_id", "purpose", "model", "agent_name", "day")


async def usage_report(
    db: AsyncSession,
    *,
    since: datetime,
    until: datetime | None = None,
    group_by: tuple[str, ...] = ("user_id", "purpose"),
    user_id: uuid.UUID | None = None,
) -> list[dict[str, Any]]:
    """Calls, tokens and cost grouped by any of ``REPORT_GROUPS``, dearest
    first. ``cost_usd`` is ``None`` when no price is known."""
    unknown = [g for g in group_by if g not in REPORT_GROUPS]
    if unknown:
        raise ValueError(f"cannot group usage by {unknown}")
    columns = {
        "user_id": LLMUsage.user_id,
        "purpose": LLMUsage.purpose,
        "model": LLMUsage.model,
        "agent_name": LLMUsage.agent_name,
        "day": func.date_trunc("day", LLMUsage.created_at),
    }
    keys = [columns[g].label(g) for g in group_by]
    dearest = await _dearest(db)
    cost = func.sum(_cost_expr(dearest)) if dearest else literal(None)
    stmt = (
        select(
            *keys,
            func.count().label("calls"),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0).label("input_tokens"),
            func.coalesce(func.sum(LLMUsage.cached_input_tokens), 0).label("cache_read_tokens"),
            func.coalesce(func.sum(LLMUsage.cache_write_input_tokens), 0).label("cache_write_tokens"),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0).label("output_tokens"),
            cost.label("cost_usd"),
            # A model in the group with no price row: charged at the dearest.
            func.bool_or(LLMPrice.model.is_(None)).label("unpriced"),
        )
        .select_from(LLMUsage)
        .outerjoin(LLMPrice, LLMPrice.model == LLMUsage.model)
        .where(LLMUsage.created_at >= since)
        .group_by(*keys)
    )
    if until is not None:
        stmt = stmt.where(LLMUsage.created_at < until)
    if user_id is not None:
        stmt = stmt.where(LLMUsage.user_id == user_id)
    stmt = stmt.order_by(cost.desc().nulls_last() if dearest else func.sum(LLMUsage.input_tokens).desc())
    return [dict(r._mapping) for r in (await db.execute(stmt)).all()]


__all__ = [
    "REPORT_GROUPS",
    "UsageAttribution",
    "bind_request_user",
    "current_attribution",
    "purpose_filter",
    "record_embedding_usage",
    "record_usage",
    "record_usage_counts",
    "spend_usd",
    "usage_context",
    "usage_report",
    "usage_scope",
]
