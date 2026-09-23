"""A worker that boots beside the MCP server must wait for it to listen.

The gotcha this pins: process start order is not a contract. A scheduled job
firing seconds after a host boots asked for `tools/list` before the server
accepted connections and died with "All connection attempts failed" — 3
seconds before the port opened, with no model call made and nothing to retry
it. The retry covers ONLY failures that never established a connection; a
call that did reach the server is never repeated, or its writes would run
twice.
"""
from __future__ import annotations

import httpx
import pytest

from luna_core.mcp.client import MCPClient, MCPTransportError


class _Response:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _Transport:
    """Raises the queued exceptions in order, then answers."""

    def __init__(self, failures: list[BaseException], body: dict | None = None):
        self.failures = list(failures)
        self.body = body or {"jsonrpc": "2.0", "result": {"tools": []}}
        self.attempts = 0

    async def post(self, url, json):  # noqa: A002 — httpx's own keyword
        self.attempts += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Response(self.body)


def _client(transport: _Transport, *, attempts: int = 6) -> MCPClient:
    client = MCPClient(
        base_url="http://server:8765",
        connect_attempts=attempts,
        connect_backoff_seconds=0.0,
    )
    client._client = transport  # type: ignore[assignment]
    return client


@pytest.mark.asyncio
async def test_waits_out_a_server_that_is_not_listening_yet():
    transport = _Transport([
        httpx.ConnectError("All connection attempts failed"),
        httpx.ConnectTimeout("timed out"),
    ])
    tools = await _client(transport).list_tools()
    assert tools == []
    assert transport.attempts == 3


@pytest.mark.asyncio
async def test_gives_up_after_the_configured_attempts():
    transport = _Transport([httpx.ConnectError("refused")] * 10)
    with pytest.raises(MCPTransportError):
        await _client(transport, attempts=4).list_tools()
    assert transport.attempts == 4


@pytest.mark.asyncio
async def test_a_call_that_reached_the_server_is_never_repeated():
    """A read timeout means the tool may already be running: retrying it
    would run its writes a second time."""
    transport = _Transport([httpx.ReadTimeout("slow tool")] * 5)
    with pytest.raises(MCPTransportError):
        await _client(transport).call_tool("create_thing", {})
    assert transport.attempts == 1
