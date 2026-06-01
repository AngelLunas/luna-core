from __future__ import annotations

import pytest

from luna_core.schemas.flow import FlowInputDef
from luna_core.services.flow import (
    FlowInputValidationError,
    validate_flow_inputs,
)


def _spec(**kw) -> FlowInputDef:
    return FlowInputDef(**kw)


def test_required_missing_raises_with_field_error():
    inputs = [_spec(name="profile_id", type="string", required=True)]
    with pytest.raises(FlowInputValidationError) as exc:
        validate_flow_inputs(inputs, {})
    assert "profile_id" in exc.value.errors


def test_wrong_type_string():
    inputs = [_spec(name="name", type="string", required=True)]
    with pytest.raises(FlowInputValidationError) as exc:
        validate_flow_inputs(inputs, {"name": 42})
    assert "name" in exc.value.errors


def test_boolean_not_accepted_as_integer():
    inputs = [_spec(name="count", type="integer", required=True)]
    with pytest.raises(FlowInputValidationError) as exc:
        validate_flow_inputs(inputs, {"count": True})
    assert "count" in exc.value.errors


def test_integer_accepted_as_number():
    inputs = [_spec(name="rate", type="number", required=True)]
    assert validate_flow_inputs(inputs, {"rate": 5}) == {"rate": 5}


def test_default_applied_when_optional_missing():
    inputs = [_spec(name="limit", type="integer", required=False, default=10)]
    assert validate_flow_inputs(inputs, {}) == {"limit": 10}


def test_optional_without_default_omitted():
    inputs = [_spec(name="note", type="string", required=False)]
    assert validate_flow_inputs(inputs, {}) == {}


def test_unknown_field_rejected():
    inputs = [_spec(name="profile_id", type="string", required=True)]
    with pytest.raises(FlowInputValidationError) as exc:
        validate_flow_inputs(inputs, {"profile_id": "abc", "extra": "nope"})
    assert "extra" in exc.value.errors


def test_object_type_accepts_dict():
    inputs = [_spec(name="filters", type="object", required=True)]
    result = validate_flow_inputs(inputs, {"filters": {"score_gte": 0.7}})
    assert result == {"filters": {"score_gte": 0.7}}


def test_object_type_rejects_non_dict():
    inputs = [_spec(name="filters", type="object", required=True)]
    with pytest.raises(FlowInputValidationError):
        validate_flow_inputs(inputs, {"filters": ["not", "a", "dict"]})


def test_passes_through_clean_payload():
    inputs = [
        _spec(name="profile_id", type="string", required=True),
        _spec(name="max_jobs", type="integer", required=False, default=20),
    ]
    result = validate_flow_inputs(
        inputs, {"profile_id": "abc-123", "max_jobs": 5}
    )
    assert result == {"profile_id": "abc-123", "max_jobs": 5}
