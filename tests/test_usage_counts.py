"""record_usage_counts builds the ledger row from explicit counts."""
from __future__ import annotations

import uuid

import pytest

from luna_core.services.usage import record_usage_counts


class _StubSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, row: object) -> None:
        self.added.append(row)


@pytest.mark.asyncio
async def test_record_usage_counts_row() -> None:
    db = _StubSession()
    scope = uuid.uuid4()

    row = await record_usage_counts(
        db,  # type: ignore[arg-type]
        scope_id=scope,
        message_id=None,
        model="gpt-4o-mini-transcribe",
        input_tokens=40,
        output_tokens=3,
        total_tokens=43,
        audio_input_tokens=35,
    )

    assert db.added == [row]
    assert row.scope_id == scope
    assert row.message_id is None
    assert row.model == "gpt-4o-mini-transcribe"
    assert row.input_tokens == 40
    assert row.output_tokens == 3
    assert row.total_tokens == 43
    assert row.audio_input_tokens == 35
    assert row.cached_input_tokens is None
