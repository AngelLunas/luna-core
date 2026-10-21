"""Conversation compaction and context editing.

What must hold: a compacted history starts at its marker and keeps the turns
after it; the provider replays its own signed summary on the model that wrote
it and plain text on any other; attachment labels keep counting past the
summarized photos; a turn compacts only when the last call reached the
policy's size, and a failed compaction leaves the history whole; the client
never sees the summary.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from luna_core.engine.chat import ChatRunner
from luna_core.llm.base import ToolDefinition
from luna_core.llm.context import ContextPolicy
from luna_core.llm.providers.anthropic import AnthropicProvider, _canonical_to_anthropic_messages
from luna_core.models.conversation import ConversationMessageRole
from luna_core.models.event import RunEventType
from luna_core.schemas.conversation import ConversationMessageRead

from tests.test_anthropic_provider import (
    _CapturingIO,
    _FakeClient,
    _FakeRedis,
    _FakeStream,
    _complete,
    _final,
    _provider,
    _text_block,
)

NATIVE = {"type": "compaction", "content": "the summary", "signature": "sig"}


def _marker(model: str = "model-a", media: dict | None = None, through: int = 3) -> dict:
    return {
        "role": "system",
        "content": [{
            "type": "compaction", "native": NATIVE, "model": model,
            "media": media or {"image": 0, "video": 0},
            "compacted_through_sequence": through, "at": "2026-10-21T00:00:00+00:00",
        }],
    }


# ----------------------------------------------------------------- provider


def test_the_signed_summary_goes_first_on_the_model_that_wrote_it():
    history = [_marker(), {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    wire = _canonical_to_anthropic_messages(history, None, tools_declared=False, model="model-a")
    assert wire[0] == {"role": "assistant", "content": [NATIVE]}


def test_another_model_reads_the_summary_as_text():
    wire = _canonical_to_anthropic_messages([_marker()], None, tools_declared=False, model="model-b")
    assert wire[0]["role"] == "user"
    assert "the summary" in wire[0]["content"][0]["text"]


def test_labels_keep_counting_past_the_summarized_photos():
    history = [
        _marker(media={"image": 2, "video": 1}),
        {"role": "user", "content": [{"type": "text", "text": "and this"}, {"type": "image", "media_id": "m9"}]},
    ]
    wire = _canonical_to_anthropic_messages(history, None, tools_declared=False, model="model-a")
    assert "img-3" in wire[1]["content"][0]["text"]


@pytest.mark.asyncio
async def test_a_policy_clears_old_tool_results_server_side():
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))])
    tool = ToolDefinition(name="t", description="d", input_schema={"type": "object"})
    await _complete(
        _provider(client), _FakeRedis(), _CapturingIO(), tools=[tool],
        context_policy=ContextPolicy(clear_tool_results_at=70_000, keep_tool_results=4),
    )
    request = client.requests[0]
    edit = request["context_management"]["edits"][0]
    assert edit["type"] == "clear_tool_uses_20250919"
    assert edit["trigger"] == {"type": "input_tokens", "value": 70_000}
    assert edit["keep"] == {"type": "tool_uses", "value": 4}
    assert "context-management-2025-06-27" in request["betas"]


@pytest.mark.asyncio
async def test_a_history_with_the_block_names_the_compaction_beta():
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))])
    await _complete(
        _provider(client), _FakeRedis(), _CapturingIO(),
        messages=[_marker(), {"role": "user", "content": [{"type": "text", "text": "sigue"}]}],
    )
    assert "compact-2026-09-04" in client.requests[0]["betas"]


@pytest.mark.asyncio
async def test_compact_returns_the_block_and_ledgers_its_iterations():
    client = _FakeClient([])
    sent: list[dict[str, Any]] = []

    async def create(**request):
        sent.append(request)
        return SimpleNamespace(
            stop_reason="compaction",
            model="model-a",
            content=[SimpleNamespace(model_dump=lambda exclude_none=True: dict(NATIVE))],
            usage=SimpleNamespace(iterations=[SimpleNamespace(input_tokens=900, output_tokens=80)]),
        )

    client.beta.messages.create = create
    rows: list[Any] = []
    block = await _provider(client, rows).compact(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hola"}]}],
        system="s", tools=[], model="model-a", run_id=uuid.uuid4(), instructions="keep facts",
    )
    assert block == NATIVE
    assert sent[0]["compaction"] == {"type": "summarize", "instructions": "keep facts"}
    assert "compact-2026-09-04" in sent[0]["betas"]
    assert "fallbacks" not in sent[0] and "context_management" not in sent[0]
    assert len(rows) == 1 and rows[0].purpose == "compaction" and rows[0].input_tokens == 900


@pytest.mark.asyncio
async def test_no_summary_is_none():
    client = _FakeClient([])

    async def create(**_request):
        return SimpleNamespace(stop_reason="max_tokens", model="model-a", content=[], usage=SimpleNamespace(iterations=[]))

    client.beta.messages.create = create
    assert await _provider(client).compact(
        messages=[], system="s", tools=[], model="model-a", run_id=uuid.uuid4()
    ) is None


# ----------------------------------------------------------------- chat runner


class _Row:
    def __init__(self, sequence: int, role: ConversationMessageRole, content: list, is_partial: bool = False):
        self.id = uuid.uuid4()
        self.sequence = sequence
        self.role = role
        self.content = content
        self.is_partial = is_partial


def _text(sequence: int, role: ConversationMessageRole, text: str) -> _Row:
    return _Row(sequence, role, [{"type": "text", "text": text}])


class _DB:
    """``execute`` answers the stored rows; ``scalar`` the last call's size."""

    def __init__(self, rows: list[_Row], last_call_tokens: int = 0) -> None:
        self._rows = rows
        self._tokens = last_call_tokens

    async def execute(self, _stmt):
        rows = self._rows
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

    async def scalar(self, _stmt):
        return self._tokens


class _IO:
    def __init__(self, next_sequence: int) -> None:
        self.events: list[tuple[RunEventType, dict]] = []
        self.saved: list[_Row] = []
        self._next = next_sequence

    async def emit(self, event_type, node_id=None, payload=None):
        self.events.append((event_type, payload or {}))

    async def save_message(self, node_id, role, content, **_kw):
        row = _Row(self._next, ConversationMessageRole(role.value), content)
        self.saved.append(row)
        return row


U, A = ConversationMessageRole.user, ConversationMessageRole.assistant


def _conversation() -> list[_Row]:
    return [
        _Row(1, U, [{"type": "text", "text": "look"}, {"type": "image", "media_id": "m1"}]),
        _text(2, A, "I see it"),
        _text(3, U, "and the other one?"),
        _text(4, A, "two of them"),
    ]


def _runner(native: dict | None) -> tuple[ChatRunner, list]:
    runner = ChatRunner(llm_router=None, mcp_client=None)
    calls: list = []

    async def compact(agent, history, scope_id, db, **kw):
        calls.append(history)
        return native

    runner._runner.compact = compact
    return runner, calls


AGENT = SimpleNamespace(name="a", model="model-a")


@pytest.mark.asyncio
async def test_a_turn_past_the_size_compacts_all_but_the_last_reply():
    rows = _conversation()
    runner, calls = _runner(NATIVE)
    io = _IO(next_sequence=5)
    kept = await runner._compact_if_due(
        AGENT, uuid.uuid4(), rows, io, _DB(rows, last_call_tokens=90_000),
        system_prompt=None, timezone=None, image_resolver=None,
        policy=ContextPolicy(compact_at=85_000),
    )
    assert len(calls[0]) == 3  # exactly the messages of the last call
    marker = io.saved[0]
    block = marker.content[0]
    assert block["native"] == NATIVE and block["model"] == "model-a"
    assert block["compacted_through_sequence"] == 3
    assert block["media"] == {"image": 1, "video": 0}
    assert kept == [marker, rows[3]]
    assert [e for e, _ in io.events] == [RunEventType.compaction_started, RunEventType.compaction_completed]
    completed = io.events[1][1]
    assert completed["compacted"] is True and "native" not in completed


@pytest.mark.asyncio
async def test_under_the_size_nothing_happens():
    rows = _conversation()
    runner, calls = _runner(NATIVE)
    io = _IO(5)
    kept = await runner._compact_if_due(
        AGENT, uuid.uuid4(), rows, io, _DB(rows, last_call_tokens=40_000),
        system_prompt=None, timezone=None, image_resolver=None,
        policy=ContextPolicy(compact_at=85_000),
    )
    assert kept == rows and not calls and not io.events


@pytest.mark.asyncio
async def test_a_failed_compaction_keeps_the_whole_history():
    rows = _conversation()
    runner, _ = _runner(None)
    io = _IO(5)
    kept = await runner._compact_if_due(
        AGENT, uuid.uuid4(), rows, io, _DB(rows, last_call_tokens=90_000),
        system_prompt=None, timezone=None, image_resolver=None,
        policy=ContextPolicy(compact_at=85_000),
    )
    assert kept == rows and not io.saved
    assert io.events[-1] == (RunEventType.compaction_completed, {"compacted": False})


@pytest.mark.asyncio
async def test_history_starts_at_the_marker_and_keeps_what_follows():
    rows = _conversation()
    marker = _Row(5, ConversationMessageRole.system, _marker(through=3)["content"])
    after = [_text(6, U, "and tomorrow?"), _text(7, A, "same again")]
    stored = [*rows, marker, *after]
    runner = ChatRunner(llm_router=None, mcp_client=None)
    history = await runner._history_rows(_DB(stored), uuid.uuid4())
    assert history == [marker, rows[3], *after]


@pytest.mark.asyncio
async def test_a_reply_kept_past_the_marker_does_not_trigger_again():
    rows = _conversation()
    marker = _Row(5, ConversationMessageRole.system, _marker(through=3)["content"])
    runner, calls = _runner(NATIVE)
    io = _IO(6)
    history = [marker, rows[3], _text(6, U, "another")]
    # The only assistant row is older than the marker: its big bill is the
    # old history's, so there is no call to measure yet.
    await runner._compact_if_due(
        AGENT, uuid.uuid4(), history, io, _DB(history, last_call_tokens=90_000),
        system_prompt=None, timezone=None, image_resolver=None,
        policy=ContextPolicy(compact_at=85_000),
    )
    assert not calls


# ----------------------------------------------------------------- client


def test_the_client_sees_where_it_happened_never_the_summary():
    read = ConversationMessageRead(
        id=uuid.uuid4(), conversation_id=uuid.uuid4(), sequence=5, role="system",
        content=_marker()["content"], is_partial=False, created_at="2026-10-21T00:00:00Z",
    )
    assert read.content == [{
        "type": "compaction", "compacted_through_sequence": 3, "at": "2026-10-21T00:00:00+00:00",
    }]
