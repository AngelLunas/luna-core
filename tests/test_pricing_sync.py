"""Prices come from the provider's published list, never from code.

No API returns token prices, so the sync reads Anthropic's pricing page (the
markdown its docs publish; ``fixtures/anthropic_pricing.md`` is a copy) and maps
each row's model name to the ids carrying it. A model priced by prompt length
keeps its threshold and multiple; a page that cannot be read raises and leaves
every price untouched.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from luna_core.services import pricing

PAGE = (Path(__file__).parent / "fixtures" / "anthropic_pricing.md").read_text()


def test_the_model_table_is_read_with_every_kind_of_token():
    prices = pricing.parse_anthropic_pricing(PAGE)
    opus = prices["Claude Opus 5.5"]
    assert (opus.input, opus.cache_write, opus.cache_read, opus.output) == (
        Decimal("4"), Decimal("5"), Decimal("0.20"), Decimal("20"),
    )
    assert opus.long_context_threshold is None
    # Links and footnote marks around a name or a figure do not leak in.
    assert "Claude Mythos 5.1" in prices
    assert prices["Claude Fable 5.1"].cache_read == Decimal("0.25")


def test_a_model_priced_by_prompt_length_keeps_its_threshold_and_multiple():
    haiku = pricing.parse_anthropic_pricing(PAGE)["Claude Haiku 5.5"]
    assert haiku.input == Decimal("0.10")
    assert haiku.long_context_threshold == 100_000
    assert haiku.long_context_multiplier == Decimal("5")


def test_a_page_without_the_table_refuses_to_price():
    with pytest.raises(pricing.PricingParseError):
        pricing.parse_anthropic_pricing("# Pricing\n\nCall us.")


class _DB:
    def __init__(self, ledger_models):
        self.ledger_models = ledger_models
        self.rows = {}
        self.committed = False

    async def execute(self, _stmt):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self.ledger_models))

    async def get(self, _model, key):
        return self.rows.get(key)

    def add(self, row):
        self.rows[row.model] = row

    async def commit(self):
        self.committed = True


class _Provider:
    async def display_names(self, models):
        names = {"claude-opus-5-5": "Claude Opus 5.5", "claude-haiku-5-5": "Claude Haiku 5.5",
                 "claude-new-thing": "Claude Unpublished"}
        names.update({m: "Claude Haiku 4.5" for m in models if m.startswith("claude-haiku-4-5")})
        return names


@pytest.mark.asyncio
async def test_every_id_is_priced_by_its_name_including_the_ledgers_dated_ones():
    db = _DB(["claude-haiku-4-5-20251001"])
    result = await pricing.sync_anthropic_prices(db, _Provider(), markdown=PAGE)
    assert set(result.updated) == {"claude-opus-5-5", "claude-haiku-5-5", "claude-haiku-4-5-20251001"}
    assert result.unpriced == ["claude-new-thing"]
    haiku = db.rows["claude-haiku-5-5"]
    assert (haiku.source, haiku.long_context_threshold, haiku.long_context_multiplier) == (
        "anthropic-docs", 100_000, Decimal("5"),
    )
    assert db.rows["claude-haiku-4-5-20251001"].input_per_mtok == Decimal("1")
    assert db.committed


@pytest.mark.asyncio
async def test_an_unreadable_page_writes_nothing():
    db = _DB([])
    with pytest.raises(pricing.PricingParseError):
        await pricing.sync_anthropic_prices(db, _Provider(), markdown="nothing here")
    assert not db.rows and not db.committed
