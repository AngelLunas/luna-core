"""Pure-function tests for the template + binding helpers in engine.nodes.

These exercise the substitution and binding-id resolution paths without
spinning up a real LangGraph / DB session.
"""
from __future__ import annotations

import pytest

from luna_core.engine.nodes import (
    NodeExecutionError,
    _format_template,
    _resolve_binding_id,
    _resolve_value,
)


def test_format_template_substitutes_context_path():
    state = {"context": {"profile": {"name": "Angel", "rate": 45}}}
    out = _format_template(
        "Hi ${context.profile.name}, rate ${context.profile.rate}", state
    )
    assert out == "Hi Angel, rate 45"


def test_format_template_substitutes_inputs_and_trigger():
    state = {
        "inputs": {"profile_id": "abc"},
        "trigger": {"source": "manual"},
    }
    out = _format_template(
        "id=${inputs.profile_id} via=${trigger.source}", state
    )
    assert out == "id=abc via=manual"


def test_format_template_missing_path_becomes_empty_string():
    state = {"context": {"profile": {"name": "Angel"}}}
    # missing drill-down inside a loaded dict: silent empty (matches the doc
    # — only the top-level binding is fail-hard, intra-dict misses are soft).
    out = _format_template("missing=${context.profile.unknown}", state)
    assert out == "missing="


def test_resolve_value_returns_dict_intact():
    # When the path resolves to a structured value, callers (like the system
    # prompt) get the dict and can json.dumps it themselves.
    state = {"context": {"profile": {"name": "Angel"}}}
    result = _resolve_value("${context.profile}", state)
    assert result == {"name": "Angel"}


def test_resolve_binding_id_from_inputs():
    state = {"inputs": {"profile_id": "uuid-1"}}
    assert _resolve_binding_id({"from": "inputs.profile_id"}, state) == "uuid-1"


def test_resolve_binding_id_static_id():
    assert (
        _resolve_binding_id({"static_id": "literal-uuid"}, {})
        == "literal-uuid"
    )


def test_resolve_binding_id_static_id_empty_returns_none():
    assert _resolve_binding_id({"static_id": ""}, {}) is None
    assert _resolve_binding_id({"static_id": None}, {}) is None


def test_resolve_binding_id_missing_input_returns_none():
    assert _resolve_binding_id({"from": "inputs.profile_id"}, {}) is None


def test_resolve_binding_id_malformed_binding_raises():
    with pytest.raises(NodeExecutionError):
        _resolve_binding_id({}, {})
    with pytest.raises(NodeExecutionError):
        _resolve_binding_id({"from": ""}, {})
