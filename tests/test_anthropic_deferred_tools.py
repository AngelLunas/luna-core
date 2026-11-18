"""Tools granted on demand: sent with defer_loading behind the search that
loads them, and named in a one-line index so the model knows they exist."""
from __future__ import annotations

from luna_core.llm.base import ToolDefinition
from luna_core.llm.providers.anthropic import _deferred_index, _tools_to_anthropic


def _tools():
    return [
        ToolDefinition(name="read_card", description="Read the card."),
        ToolDefinition(
            name="update_record",
            description="Update a record by its name — weights and dates. Everything else too.",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            deferred=True,
        ),
    ]


def test_deferred_tools_go_behind_the_search():
    wire = _tools_to_anthropic(_tools())
    assert wire[0] == {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    by_name = {t.get("name"): t for t in wire}
    assert by_name["update_record"]["defer_loading"] is True
    assert "defer_loading" not in by_name["read_card"]


def test_no_deferred_tool_no_search():
    wire = _tools_to_anthropic([ToolDefinition(name="read_card", description="x")])
    assert all(t.get("type") != "tool_search_tool_regex_20251119" for t in wire)


def test_the_index_names_each_deferred_tool_by_its_first_sentence():
    index = _deferred_index(_tools())
    assert "- update_record: Update a record by its name — weights and dates" in index
    assert "read_card" not in index
    assert _deferred_index([ToolDefinition(name="a", description="b")]) == ""


def test_a_dropped_connection_is_told_from_an_api_answer():
    import anthropic
    import httpx

    from luna_core.llm.providers.anthropic import _is_connection_drop

    req = httpx.Request("POST", "https://example.invalid")
    assert _is_connection_drop(anthropic.APIConnectionError(request=req))

    class ReadError(Exception):
        pass

    wrapped = RuntimeError("stream broke")
    wrapped.__cause__ = ReadError("connection reset")
    assert _is_connection_drop(wrapped)
    assert not _is_connection_drop(ValueError("bad request"))


def _searched_history():
    search = {
        "type": "native",
        "native": {
            "id": "srvtoolu_1",
            "name": "tool_search_tool_regex",
            "type": "server_tool_use",
            "input": {"pattern": "^update_record$"},
        },
    }
    result = {
        "type": "native",
        "native": {
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [
                    {"type": "tool_reference", "tool_name": "update_record"},
                    {"type": "tool_reference", "tool_name": "move_items"},
                ],
            },
        },
    }
    call = {"type": "tool_use", "id": "toolu_1", "name": "update_record", "input": {}}
    return [
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
        {"role": "assistant", "content": [search, result, call]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ]


def _assistant_blocks(wire):
    return next(m for m in wire if m["role"] == "assistant")["content"]


def test_a_replayed_search_keeps_only_the_tools_this_request_can_load():
    from luna_core.llm.providers.anthropic import _canonical_to_anthropic_messages

    wire = _canonical_to_anthropic_messages(
        _searched_history(), None, tools_declared=True, loadable=frozenset({"update_record"})
    )
    blocks = _assistant_blocks(wire)
    assert [b["type"] for b in blocks] == ["server_tool_use", "tool_search_tool_result", "tool_use"]
    assert blocks[1]["content"]["tool_references"] == [
        {"type": "tool_reference", "tool_name": "update_record"}
    ]


def test_a_replayed_search_is_left_out_when_this_request_has_no_search():
    from luna_core.llm.providers.anthropic import _canonical_to_anthropic_messages

    wire = _canonical_to_anthropic_messages(_searched_history(), None, tools_declared=True)
    assert [b["type"] for b in _assistant_blocks(wire)] == ["tool_use"]
