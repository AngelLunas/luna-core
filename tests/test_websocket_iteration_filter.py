"""Tests for the per-client iteration subscription filter on the run
event WebSocket.

The contract pinned here:

  1. ``_extract_routing_keys`` pulls (iteration_id, event_type) out of a
     serialized run event frame so the fanout can route in O(1) per
     client. Frames without an iteration_id are treated as broadcast.
  2. The control protocol accepted on the WebSocket is exactly:

        {"type": "subscribe",   "scope": "iteration", "id": "<uuid>"}
        {"type": "unsubscribe", "scope": "iteration", "id": "<uuid>"}

     Anything else is silently dropped — including malformed JSON,
     unknown scopes, missing fields — so existing clients that send any
     other text keep working.
  3. ``_frame_allowed`` mirrors the fanout rule: lifecycle iteration
     events and untagged events pass for every client; tagged sub-events
     pass only when the iteration_id is in the client's subscriptions.

These are pure-function tests on the manager internals — the live WS /
Redis path is exercised by integration tests elsewhere.
"""
from __future__ import annotations

import json
import uuid

import pytest

from luna_core.engine.websocket import (
    ITERATION_LIFECYCLE_TYPES,
    WebSocketManager,
    _ClientState,
    _extract_routing_keys,
)


def _frame(event_type: str, payload: dict | None = None) -> str:
    return json.dumps(
        {
            "id": str(uuid.uuid4()),
            "flow_run_id": "00000000-0000-0000-0000-0000000000aa",
            "sequence": 1,
            "timestamp": "2026-05-25T00:00:00+00:00",
            "event_type": event_type,
            "node_id": "scorer",
            "payload": payload or {},
        }
    )


# ---- _extract_routing_keys ---------------------------------------------


def test_extract_keys_returns_iteration_id_and_event_type():
    frame = _frame("tool_called", {"iteration_id": "abc-123", "tool": "x"})
    assert _extract_routing_keys(frame) == ("abc-123", "tool_called")


def test_extract_keys_returns_none_iteration_when_untagged():
    frame = _frame("node_started", {"type": "ai_agent"})
    assert _extract_routing_keys(frame) == (None, "node_started")


def test_extract_keys_tolerates_malformed_json():
    assert _extract_routing_keys("{not json") == (None, None)


def test_extract_keys_tolerates_non_object_payload():
    frame = json.dumps({"event_type": "x", "payload": "not a dict"})
    assert _extract_routing_keys(frame) == (None, "x")


def test_extract_keys_iteration_id_must_be_string():
    frame = _frame("tool_called", {"iteration_id": 42})
    assert _extract_routing_keys(frame) == (None, "tool_called")


# ---- _frame_allowed ----------------------------------------------------


def test_untagged_frame_always_allowed():
    state = _ClientState()  # no subscriptions
    assert WebSocketManager._frame_allowed(_frame("node_started"), state)


@pytest.mark.parametrize("event_type", sorted(ITERATION_LIFECYCLE_TYPES))
def test_lifecycle_iteration_events_always_allowed(event_type):
    state = _ClientState()  # no subscriptions
    frame = _frame(event_type, {"iteration_id": "abc-123"})
    assert WebSocketManager._frame_allowed(frame, state)


def test_tagged_subevent_blocked_without_subscription():
    state = _ClientState()
    frame = _frame("tool_called", {"iteration_id": "abc-123"})
    assert not WebSocketManager._frame_allowed(frame, state)


def test_tagged_subevent_allowed_with_subscription():
    state = _ClientState()
    state.iteration_subscriptions.add("abc-123")
    frame = _frame("tool_called", {"iteration_id": "abc-123"})
    assert WebSocketManager._frame_allowed(frame, state)


def test_tagged_subevent_blocked_when_subscription_is_for_other_iteration():
    state = _ClientState()
    state.iteration_subscriptions.add("other-iteration")
    frame = _frame("tool_called", {"iteration_id": "abc-123"})
    assert not WebSocketManager._frame_allowed(frame, state)


# ---- _handle_control_message protocol ----------------------------------


def _manager() -> WebSocketManager:
    """Throwaway manager used to exercise the control-message handler.

    The handler doesn't touch Redis or the client dict, so passing
    ``None`` for redis is fine — it only ever stores it on the instance.
    """
    return WebSocketManager.__new__(WebSocketManager)


@pytest.mark.asyncio
async def test_subscribe_adds_iteration_id_to_set():
    manager = _manager()
    state = _ClientState()
    await manager._handle_control_message(
        state,
        json.dumps(
            {"type": "subscribe", "scope": "iteration", "id": "abc-123"}
        ),
    )
    assert state.iteration_subscriptions == {"abc-123"}


@pytest.mark.asyncio
async def test_unsubscribe_removes_iteration_id_from_set():
    manager = _manager()
    state = _ClientState()
    state.iteration_subscriptions.add("abc-123")
    state.iteration_subscriptions.add("def-456")
    await manager._handle_control_message(
        state,
        json.dumps(
            {"type": "unsubscribe", "scope": "iteration", "id": "abc-123"}
        ),
    )
    assert state.iteration_subscriptions == {"def-456"}


@pytest.mark.asyncio
async def test_unsubscribe_unknown_id_is_a_clean_noop():
    manager = _manager()
    state = _ClientState()
    state.iteration_subscriptions.add("abc-123")
    await manager._handle_control_message(
        state,
        json.dumps(
            {"type": "unsubscribe", "scope": "iteration", "id": "never-sub"}
        ),
    )
    # Existing subscriptions untouched.
    assert state.iteration_subscriptions == {"abc-123"}


@pytest.mark.asyncio
async def test_subscribe_is_idempotent():
    manager = _manager()
    state = _ClientState()
    for _ in range(3):
        await manager._handle_control_message(
            state,
            json.dumps(
                {"type": "subscribe", "scope": "iteration", "id": "abc-123"}
            ),
        )
    assert state.iteration_subscriptions == {"abc-123"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[]",  # JSON, but not an object
        json.dumps({"type": "subscribe"}),  # missing scope + id
        json.dumps({"type": "subscribe", "scope": "node", "id": "x"}),  # wrong scope
        json.dumps({"type": "subscribe", "scope": "iteration", "id": ""}),  # empty id
        json.dumps({"type": "weird", "scope": "iteration", "id": "x"}),  # unknown type
        json.dumps({"type": "subscribe", "scope": "iteration", "id": 42}),  # id not str
        "",
    ],
)
@pytest.mark.asyncio
async def test_malformed_control_messages_silently_dropped(raw):
    manager = _manager()
    state = _ClientState()
    await manager._handle_control_message(state, raw)
    # No mutation, no exception.
    assert state.iteration_subscriptions == set()
