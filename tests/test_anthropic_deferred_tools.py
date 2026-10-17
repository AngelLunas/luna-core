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
