"""Model prices from the provider's own published list.

No API returns token prices: the Models API lists ids, names and capabilities,
and the organization's cost report gives what was billed, not per-token rates.
What Anthropic does publish in a machine-readable form is its pricing page as
markdown (``settings.anthropic_pricing_url``). ``sync_anthropic_prices`` reads
its model table, maps each row's model NAME to the ids that carry it (the
Models API's ``display_name``, for the ids it lists and the ones the ledger
holds), and upserts ``LLMPrice`` rows marked ``anthropic-docs``.

It never deletes a price and never writes a partial one: a page it cannot read
(layout changed, network down) raises ``PricingParseError`` and leaves every
price as it was — a stale price is visible (``updated_at``), a zero price is a
budget that never trips.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.models.pricing import LLMPrice
from luna_core.models.usage import LLMUsage

logger = logging.getLogger(__name__)

SOURCE_ANTHROPIC = "anthropic-docs"
SOURCE_MANUAL = "manual"


class PricingParseError(RuntimeError):
    """The published price list could not be read as expected."""


@dataclass(frozen=True)
class PublishedPrice:
    """One model's prices per million tokens, as published."""

    input: Decimal
    cache_write: Decimal  # the 5-minute write: the cache the providers ask for
    cache_read: Decimal
    output: Decimal
    long_context_threshold: int | None = None
    long_context_multiplier: Decimal | None = None


_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TAGS = re.compile(r"<[^>]+>")
_MONEY = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")
_TIER = re.compile(r"\(for prompts (up to|over) ([0-9][0-9,]*) tokens\)", re.IGNORECASE)


def _cell(text: str) -> str:
    return _TAGS.sub("", _LINK.sub(r"\1", text)).strip()


def _money(text: str) -> Decimal:
    found = _MONEY.search(_TAGS.sub("", text))
    if not found:
        raise PricingParseError(f"no price in {text!r}")
    try:
        return Decimal(found.group(1))
    except InvalidOperation as exc:
        raise PricingParseError(f"bad price {text!r}") from exc


def _model_table(markdown: str) -> list[list[str]]:
    """The rows of the first table whose header names base input, cache hits
    and output — the per-model price table."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        header = [_cell(c).lower() for c in line.strip().strip("|").split("|")]
        if not ("base input tokens" in header and "output tokens" in header and any("cache hits" in h for h in header)):
            continue
        rows = []
        for row in lines[i + 2:]:
            if not row.strip().startswith("|"):
                break
            rows.append([c for c in row.strip().strip("|").split("|")])
        return [header, *rows]
    raise PricingParseError("no model price table found")


def parse_anthropic_pricing(markdown: str) -> dict[str, PublishedPrice]:
    """Model name ("Claude Opus 5.5") → its published prices. A model priced
    by prompt length comes as two rows ("up to N" / "over N"): the first gives
    the prices, the second the multiple past N."""
    header, *rows = _model_table(markdown)

    def col(name: str) -> int:
        for i, h in enumerate(header):
            if h.startswith(name):
                return i
        raise PricingParseError(f"no {name!r} column")

    c_in, c_write, c_read, c_out = col("base input"), col("5m cache writes"), col("cache hits"), col("output")
    base: dict[str, PublishedPrice] = {}
    over: dict[str, tuple[int, Decimal]] = {}
    for row in rows:
        if len(row) <= max(c_in, c_write, c_read, c_out):
            continue
        raw_name = _LINK.sub(r"\1", row[0])
        tier = _TIER.search(raw_name)
        name = re.sub(r"\s*\([^)]*\)", "", _cell(raw_name)).strip()
        price = PublishedPrice(
            input=_money(row[c_in]),
            cache_write=_money(row[c_write]),
            cache_read=_money(row[c_read]),
            output=_money(row[c_out]),
        )
        if tier and tier.group(1).lower() == "over":
            over[name] = (int(tier.group(2).replace(",", "")), price.input)
        else:
            base.setdefault(name, price)
    for name, (threshold, long_input) in over.items():
        if name in base and base[name].input:
            p = base[name]
            base[name] = PublishedPrice(
                p.input, p.cache_write, p.cache_read, p.output,
                long_context_threshold=threshold,
                long_context_multiplier=long_input / p.input,
            )
    if len(base) < 3:
        raise PricingParseError(f"only {len(base)} models read from the price list")
    return base


async def fetch_anthropic_pricing(url: str | None = None) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url or settings.anthropic_pricing_url)
        response.raise_for_status()
        return response.text


async def set_price(
    db: AsyncSession, model: str, price: PublishedPrice, *, source: str
) -> LLMPrice:
    """Upsert one model's price. Does not commit."""
    row = await db.get(LLMPrice, model)
    if row is None:
        row = LLMPrice(model=model)
        db.add(row)
    row.input_per_mtok = price.input
    row.output_per_mtok = price.output
    row.cache_read_per_mtok = price.cache_read
    row.cache_write_per_mtok = price.cache_write
    row.long_context_threshold = price.long_context_threshold
    row.long_context_multiplier = price.long_context_multiplier
    row.source = source
    return row


@dataclass
class SyncResult:
    updated: dict[str, str]  # model id → the published name it was priced as
    unpriced: list[str]  # ids the provider knows whose name the list lacks


async def sync_anthropic_prices(db: AsyncSession, provider: Any, *, markdown: str | None = None) -> SyncResult:
    """Price every Anthropic model id the provider lists or the ledger holds,
    from the published list. Commits once, only after the whole list read."""
    published = parse_anthropic_pricing(markdown or await fetch_anthropic_pricing())
    ledger_models = (
        await db.execute(select(LLMUsage.model).distinct().where(LLMUsage.model.like("claude-%")))
    ).scalars().all()
    names = await provider.display_names(ledger_models)
    result = SyncResult(updated={}, unpriced=[])
    for model, name in sorted(names.items()):
        price = published.get(name)
        if price is None:
            result.unpriced.append(model)
            continue
        await set_price(db, model, price, source=SOURCE_ANTHROPIC)
        result.updated[model] = name
    await db.commit()
    if result.unpriced:
        logger.info("price list names no price for %s", result.unpriced)
    return result


__all__ = [
    "PricingParseError",
    "PublishedPrice",
    "SOURCE_ANTHROPIC",
    "SOURCE_MANUAL",
    "SyncResult",
    "fetch_anthropic_pricing",
    "parse_anthropic_pricing",
    "set_price",
    "sync_anthropic_prices",
]
