"""Prompt-cache writes reach the ledger apart from reads and plain input."""
from __future__ import annotations

import uuid

import pytest

from luna_core.llm.providers._turn import _anthropic_usage_shim
from luna_core.models.usage import LLMUsage
from luna_core.services.usage import record_usage


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)


@pytest.mark.asyncio
async def test_a_cache_write_is_recorded_apart_from_the_read():
    usage = _anthropic_usage_shim(
        {
            "input_tokens": 4,
            "cache_read_input_tokens": 60000,
            "cache_creation_input_tokens": 1500,
            "output_tokens": 300,
        }
    )
    db = _FakeSession()
    row = await record_usage(db, scope_id=uuid.uuid4(), message_id=None, model="m", usage=usage)

    assert isinstance(row, LLMUsage)
    assert row.input_tokens == 61504  # the prompt total, as before
    assert row.cached_input_tokens == 60000
    assert row.cache_write_input_tokens == 1500
    assert row.output_tokens == 300


@pytest.mark.asyncio
async def test_a_provider_that_reports_no_writes_leaves_them_unknown():
    from types import SimpleNamespace

    usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
    )
    row = await record_usage(_FakeSession(), scope_id=uuid.uuid4(), message_id=None, model="m", usage=usage)
    assert row.cache_write_input_tokens is None
