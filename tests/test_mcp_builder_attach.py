"""Tests for MCPServerBuilder's connector-op handler attachment.

Focused on the error-surfacing contract: when the underlying connector
raises ``ConnectorExecutionError`` (HTTP 4xx/5xx, transport failure,
etc.), the attached MCP tool must return the message verbatim as a
normal tool result with an ``error`` key rather than re-raising. The
prior behavior re-raised the exception, letting FastMCP collapse it
into a content block whose payload didn't always make it back to the
agent — the LLM saw an empty error and had no way to react.

The tests don't spin up a real FastMCP server. Instead they capture the
handler the builder would attach (via a stub `_attach_tool`) and invoke
it directly, which is what FastMCP's tool manager would do internally.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from luna_core.connectors.registry import (
    ConnectorExecutionError,
    ConnectorRegistry,
)
from luna_core.mcp.builder import MCPServerBuilder
from luna_core.models.connector import HTTPMethod, Operation


def _make_operation() -> Operation:
    op = Operation(
        connector_id=uuid.uuid4(),
        name="list_jobs",
        description="",
        method=HTTPMethod.GET,
        path="/jobs",
        input_schema={"type": "object"},
        output_schema={},
        parameters=[],
        fixed_headers={},
        fixed_body=None,
        is_active=True,
    )
    op.id = uuid.uuid4()
    return op


def _build_attached_handler(registry: ConnectorRegistry, op: Operation):
    """Run ``attach_operation`` against a stub server and return the
    handler the builder produced."""
    builder = MCPServerBuilder(registry, server_name="test")
    captured: dict = {}

    def fake_attach(server, handler, name, description, input_schema):
        captured["handler"] = handler

    builder._attach_tool = fake_attach  # type: ignore[method-assign]
    builder.attach_operation(server=MagicMock(), operation=op)
    return captured["handler"]


@pytest.mark.asyncio
async def test_connector_handler_returns_error_dict_on_http_failure(monkeypatch):
    op = _make_operation()
    registry = ConnectorRegistry()

    async def fake_execute(operation_id, input_data):
        # Mirror what _execute_http raises on a real 4xx response —
        # exact wording matches connectors/registry.py so the agent
        # sees the same string regardless of whether we mock the
        # registry or hit a real upstream.
        raise ConnectorExecutionError(
            "connector upwork.list_jobs returned HTTP 404: <html>not found</html>"
        )

    monkeypatch.setattr(registry, "execute", fake_execute)

    handler = _build_attached_handler(registry, op)
    result = await handler(cursor="0", skills=["three-js"])

    # Must NOT raise — FastMCP's exception serialization was losing
    # the message somewhere in the chain. A dict-shaped error is
    # predictable: the agent treats it the same way it treats
    # save_recommended_job's `{"error": "..."}` responses.
    assert isinstance(result, dict)
    assert "error" in result
    assert "HTTP 404" in result["error"]
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_connector_handler_returns_payload_on_success(monkeypatch):
    op = _make_operation()
    registry = ConnectorRegistry()

    async def fake_execute(operation_id, input_data):
        return {"data": {"totalCount": 10, "edges": []}}

    monkeypatch.setattr(registry, "execute", fake_execute)

    handler = _build_attached_handler(registry, op)
    result = await handler(cursor="0", skills=["three-js"])

    # Success path is unchanged — full payload passes through.
    assert result == {"data": {"totalCount": 10, "edges": []}}


@pytest.mark.asyncio
async def test_connector_handler_reraises_non_connector_exceptions(monkeypatch):
    # Programmer errors (KeyError, RuntimeError from a misconfigured
    # registry, etc.) must still surface to the framework as
    # exceptions. The "swallow → return error dict" path is scoped
    # to ConnectorExecutionError specifically — broader catches would
    # mask real bugs behind a tool result the agent can't act on.
    op = _make_operation()
    registry = ConnectorRegistry()

    async def fake_execute(operation_id, input_data):
        raise RuntimeError("programmer mistake")

    monkeypatch.setattr(registry, "execute", fake_execute)

    handler = _build_attached_handler(registry, op)
    with pytest.raises(RuntimeError, match="programmer mistake"):
        await handler()
