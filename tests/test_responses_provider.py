"""Unit tests for the Responses-API translation layer in the generic provider —
the pure converters that map canonical history/tools ↔ the Responses wire shape
and parse its output back to canonical blocks. No network, no DB."""
from __future__ import annotations

import json
from types import SimpleNamespace

from luna_core.llm.base import ToolDefinition
from luna_core.llm.providers.generic import (
    _ResponsesUsage,
    _canonical_to_responses_input,
    _responses_output_to_blocks,
    _tools_to_responses,
)


# --- _tools_to_responses ----------------------------------------------------

def test_tools_to_responses_builtin_first_then_functions():
    tools = [ToolDefinition(name="get_plant", description="d", input_schema={"type": "object"})]
    out = _tools_to_responses(tools, ["web_search"])
    assert out[0] == {"type": "web_search"}  # built-in first
    assert out[1] == {
        "type": "function",
        "name": "get_plant",
        "description": "d",
        "parameters": {"type": "object"},
    }


def test_tools_to_responses_no_builtin():
    out = _tools_to_responses(
        [ToolDefinition(name="f", description="", input_schema={})], None
    )
    assert len(out) == 1 and out[0]["type"] == "function"
    # empty schema is filled with a valid object schema
    assert out[0]["parameters"] == {"type": "object", "properties": {}}


# --- _canonical_to_responses_input ------------------------------------------

def test_input_user_text_becomes_input_text_part():
    items = _canonical_to_responses_input(
        [{"role": "user", "content": [{"type": "text", "text": "hola"}]}]
    )
    assert items == [{"role": "user", "content": [{"type": "input_text", "text": "hola"}]}]


def test_input_tool_use_and_result_become_function_call_pair_in_order():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "diagnose"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "inspect_image", "input": {"media_ids": ["m1"]}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "white powder"}],
        },
    ]
    items = _canonical_to_responses_input(messages)
    call = next(i for i in items if i.get("type") == "function_call")
    output = next(i for i in items if i.get("type") == "function_call_output")
    assert call["call_id"] == "c1" and call["name"] == "inspect_image"
    assert json.loads(call["arguments"]) == {"media_ids": ["m1"]}
    assert output["call_id"] == "c1" and output["output"] == "white powder"
    # the call must precede its output
    assert items.index(call) < items.index(output)


def test_input_assistant_text_and_nonstr_tool_result():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "let me look"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c9", "content": {"ok": True}}]},
    ]
    items = _canonical_to_responses_input(messages)
    assert {"role": "assistant", "content": "let me look"} in items
    out = next(i for i in items if i.get("type") == "function_call_output")
    assert json.loads(out["output"]) == {"ok": True}  # dict serialized to JSON


def test_input_image_resolved_vs_note():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "media_id": "seen"},
                {"type": "image", "media_id": "unseen"},
            ],
        }
    ]
    items = _canonical_to_responses_input(messages, {"seen": "data:image/png;base64,X"})
    parts = items[0]["content"]
    assert {"type": "input_image", "image_url": "data:image/png;base64,X"} in parts
    notes = [p["text"] for p in parts if p["type"] == "input_text"]
    assert "[image attached: img-1 (shown below)]" in notes  # the rendered one
    assert "[image attached: img-2]" in notes  # the unresolved one, label only


# --- _responses_output_to_blocks --------------------------------------------

def _part(text, urls=()):
    return SimpleNamespace(
        type="output_text",
        text=text,
        annotations=[SimpleNamespace(type="url_citation", url=u) for u in urls],
    )


def test_output_message_with_citations_appends_sources():
    resp = SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[_part("powdery mildew", ["http://a", "http://b"])])]
    )
    blocks, thinking = _responses_output_to_blocks(resp)
    assert thinking is None
    assert len(blocks) == 1 and blocks[0]["type"] == "text"
    assert blocks[0]["text"].startswith("powdery mildew")
    assert "Sources: http://a, http://b" in blocks[0]["text"]


def test_output_function_call_becomes_tool_use():
    resp = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", call_id="c1", name="save_diagnosis", arguments='{"verdict":"x"}')]
    )
    blocks, _ = _responses_output_to_blocks(resp)
    assert blocks == [
        {"type": "tool_use", "id": "c1", "name": "save_diagnosis", "input": {"verdict": "x"}}
    ]


def test_output_bad_function_args_fall_back_to_raw():
    resp = SimpleNamespace(
        output=[SimpleNamespace(type="function_call", call_id="c2", name="f", arguments="not json")]
    )
    blocks, _ = _responses_output_to_blocks(resp)
    assert blocks[0]["input"] == {"_raw": "not json"}


def test_output_reasoning_thinking_and_websearch_kept_in_order():
    resp = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="weighing options")]),
            SimpleNamespace(
                type="web_search_call",
                id="ws_1",
                action=SimpleNamespace(
                    query="powdery mildew cure",
                    queries=["powdery mildew cure", "neem oil dosage", ""],
                ),
            ),
            SimpleNamespace(type="message", content=[_part("answer")]),
        ]
    )
    blocks, thinking = _responses_output_to_blocks(resp)
    assert thinking == "weighing options"
    types = [b["type"] for b in blocks]
    # web_search_call is kept in place so it renders in the order it ran.
    assert types == ["thinking", "web_search_call", "text"]
    ws = blocks[1]
    assert ws["id"] == "ws_1"
    # query + queries deduped, blanks dropped.
    assert ws["queries"] == ["powdery mildew cure", "neem oil dosage"]


def test_output_websearch_without_action_yields_empty_queries():
    resp = SimpleNamespace(output=[SimpleNamespace(type="web_search_call", id="ws_2")])
    blocks, _ = _responses_output_to_blocks(resp)
    assert blocks == [{"type": "web_search_call", "id": "ws_2", "queries": []}]


# --- _ResponsesUsage adapter -------------------------------------------------

def test_responses_usage_adapter_maps_fields():
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    a = _ResponsesUsage(usage)
    assert a.prompt_tokens == 100
    assert a.completion_tokens == 20
    assert a.total_tokens == 120
    assert a.prompt_tokens_details.cached_tokens == 40


def test_responses_usage_adapter_handles_missing_details():
    a = _ResponsesUsage(SimpleNamespace(input_tokens=5, output_tokens=1, total_tokens=6))
    assert a.prompt_tokens == 5
    assert a.prompt_tokens_details.cached_tokens is None
