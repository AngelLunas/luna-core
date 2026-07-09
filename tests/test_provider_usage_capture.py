"""The streaming provider records real token usage once per completed turn.

When the provider returns a final usage-only chunk (``stream_options=
{"include_usage": true}``), ``complete()`` must persist one ``LLMUsage`` row on
the turn's session, keyed by the execution scope + message, with the reported
token counts. When no usage chunk arrives, nothing is recorded — the no-usage
path (and the existing provider tests) stay untouched.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from luna_core.llm.providers.generic import GenericProvider
from luna_core.models.usage import LLMUsage


def _chunk(content: str | None = None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _usage_chunk(usage: Any):
    # Final usage-only chunk: empty choices, carries the usage object.
    return SimpleNamespace(choices=[], usage=usage)


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def create(self, **_kwargs):
        return _FakeStream(self._chunks)


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.lists: dict[str, list] = {}

    async def exists(self, *_keys):
        return 0

    async def rpush(self, key, *vals):
        self.lists.setdefault(key, []).extend(vals)
        return len(self.lists[key])

    async def expire(self, *_a, **_k):
        return True

    async def incr(self, key):
        n = int(self.strings.get(key, 0)) + 1
        self.strings[key] = n
        return n

    async def set(self, key, value, ex=None):
        self.strings[key] = value

    async def get(self, key):
        return self.strings.get(key)

    async def publish(self, *_a, **_k):
        return None

    async def delete(self, *keys):
        for k in keys:
            self.strings.pop(k, None)
            self.lists.pop(k, None)


class _CapturingIO:
    def __init__(self, scope_id: uuid.UUID) -> None:
        self._scope_id = scope_id
        self._seq = 0

    @property
    def scope_id(self) -> uuid.UUID:
        return self._scope_id

    def for_session(self, _db):
        return self

    async def emit(self, *_a, **_k):
        self._seq += 1
        return SimpleNamespace(sequence=self._seq)

    async def save_message(self, *_a, **_k):
        return SimpleNamespace(id=uuid.uuid4())


class _CapturingSession:
    """Fake session that records ``add``ed rows (the usage path needs ``add``)."""

    def __init__(self, added: list[Any]) -> None:
        self._added = added

    def add(self, row):
        self._added.append(row)

    async def commit(self):
        return None

    async def refresh(self, _row):
        return None


async def _run(chunks: list[Any]) -> list[Any]:
    added: list[Any] = []

    @asynccontextmanager
    async def _session_factory():
        yield _CapturingSession(added)

    scope_id = uuid.uuid4()
    provider = GenericProvider(
        api_key="x", base_url="http://localhost", session_factory=_session_factory
    )
    provider._chat_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(chunks))
    )
    await provider.complete(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        system="",
        tools=[],
        temperature=0.0,
        model="fake-model",
        output_schema=None,
        run_id=scope_id,
        node_id="chat",
        redis=_FakeRedis(),
        make_io=lambda _db: _CapturingIO(scope_id),
    )
    return added, scope_id


@pytest.mark.asyncio
async def test_usage_chunk_records_one_ledger_row():
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2),
    )
    added, scope_id = await _run([_chunk("hi"), _usage_chunk(usage)])

    rows = [r for r in added if isinstance(r, LLMUsage)]
    assert len(rows) == 1
    row = rows[0]
    assert row.scope_id == scope_id
    assert row.model == "fake-model"
    assert row.input_tokens == 10
    assert row.output_tokens == 5
    assert row.total_tokens == 15
    assert row.cached_input_tokens == 2


@pytest.mark.asyncio
async def test_no_usage_chunk_records_nothing():
    added, _ = await _run([_chunk("hi")])
    assert [r for r in added if isinstance(r, LLMUsage)] == []
