"""A client tool call while the model writes it: announced when its block
starts, then what the host's reader makes of the input so far — only when it
changed and not more often than the throttle allows."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.llm import tool_progress
from luna_core.llm.tool_progress import ToolWriting, register_input_reader
from luna_core.models.event import RunEventType


@pytest.fixture(autouse=True)
def _no_readers():
    tool_progress.clear_input_readers()
    yield
    tool_progress.clear_input_readers()


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _phases(text: str):
    return {"phases": text.count('"stage"')} if '"stage"' in text else None


def test_the_start_is_announced_with_the_call():
    w = ToolWriting()
    assert w.start(1, "tu_1", "create_crop_plan") == {
        "tool_use_id": "tu_1", "name": "create_crop_plan", "status": "started",
    }


def test_without_a_reader_nothing_more_is_said():
    w = ToolWriting()
    w.start(0, "tu_1", "remember")
    assert w.feed(0, '{"text": "x"}') is None


def test_the_reader_speaks_when_its_answer_changes_and_the_throttle_allows():
    register_input_reader("create_crop_plan", _phases)
    clock = _Clock()
    w = ToolWriting(clock=clock)
    w.start(0, "tu_1", "create_crop_plan")

    assert w.feed(0, '{"phases": [{"name": "A", "stage": "seedling"') == {
        "tool_use_id": "tu_1", "name": "create_crop_plan", "status": "writing", "phases": 1,
    }
    clock.now += 0.1
    assert w.feed(0, '}, {"name": "B", "stage": "veg"') is None  # too soon
    clock.now += 1
    assert w.feed(0, ', "entries": []') == {
        "tool_use_id": "tu_1", "name": "create_crop_plan", "status": "writing", "phases": 2,
    }
    clock.now += 1
    assert w.feed(0, "}") is None  # same answer as before


def test_a_reader_that_fails_is_silent():
    def broken(text):
        raise ValueError("nope")

    register_input_reader("create_crop_plan", broken)
    w = ToolWriting()
    w.start(0, "tu_1", "create_crop_plan")
    assert w.feed(0, "{") is None


def test_a_delta_for_an_unknown_block_is_ignored():
    assert ToolWriting().feed(3, "{") is None


@pytest.mark.asyncio
async def test_the_anthropic_stream_publishes_tool_writing(monkeypatch):
    from luna_core.llm.providers import anthropic as provider_mod

    published: list = []

    async def fake_publish(redis, run_id, event_type, node_id, payload, seq, **kwargs):
        published.append((event_type, payload))

    monkeypatch.setattr(provider_mod, "publish_run_event", fake_publish)
    provider = object.__new__(provider_mod.AnthropicProvider)

    async def next_seq(redis, key, base):
        return 1

    monkeypatch.setattr(provider, "_next_delta_sequence", next_seq)
    register_input_reader("create_crop_plan", _phases)
    turn = provider_mod._Turn(uuid.uuid4())
    turn.started = True

    async def on(event):
        await provider._on_event(event, turn, uuid.uuid4(), "chat", None, None)

    await on(SimpleNamespace(
        type="content_block_start", index=2,
        content_block=SimpleNamespace(type="tool_use", id="tu_9", name="create_crop_plan"),
    ))
    await on(SimpleNamespace(
        type="content_block_delta", index=2,
        delta=SimpleNamespace(type="input_json_delta", partial_json='{"phases": [{"stage": "veg"'),
    ))

    kinds = [(e, p["status"]) for e, p in published]
    assert kinds == [(RunEventType.tool_writing, "started"), (RunEventType.tool_writing, "writing")]
    assert published[1][1]["phases"] == 1
    assert published[1][1]["message_id"] == str(turn.message_id)
