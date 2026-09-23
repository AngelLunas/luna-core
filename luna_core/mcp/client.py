"""HTTP client for a running MCP Server.

Uses MCP's JSON-RPC over HTTP transport. The client speaks the protocol
directly via httpx rather than dragging in a heavier MCP client SDK — the two
operations we need (`tools/list`, `tools/call`) are trivial requests.

The MCP server URL is configured via settings.mcp_server_url; the host app
overrides per call when needed (e.g. multi-tenant deployments).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from luna_core.core.config import settings
from luna_core.mcp.schemas import ToolCallResult, ToolDefinition

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 30.0,
        auth_headers: dict[str, str] | None = None,
        connect_attempts: int | None = None,
        connect_backoff_seconds: float | None = None,
    ):
        self._base_url = (base_url or settings.mcp_server_url).rstrip("/")
        self._connect_attempts = max(
            1,
            settings.mcp_connect_attempts
            if connect_attempts is None
            else connect_attempts,
        )
        self._connect_backoff = (
            settings.mcp_connect_backoff_seconds
            if connect_backoff_seconds is None
            else connect_backoff_seconds
        )
        # follow_redirects: fastmcp's streamable-http handler returns a 307
        # from /mcp to /mcp/ (Starlette trailing-slash normalization).
        # The Accept header is required by fastmcp 2.x — it rejects `*/*`
        # with 406, even when `json_response=True` makes the body plain JSON.
        merged_headers = {
            "Accept": "application/json, text/event-stream",
            **(auth_headers or {}),
        }
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=merged_headers,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_tools(self) -> list[ToolDefinition]:
        result = await self._rpc("tools/list", {})
        tools = result.get("tools", [])
        return [
            ToolDefinition(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema") or t.get("input_schema") or {},
            )
            for t in tools
        ]

    async def call_tool(self, name: str, input: dict[str, Any]) -> ToolCallResult:
        try:
            result = await self._rpc(
                "tools/call", {"name": name, "arguments": input}
            )
        except MCPRemoteError as exc:
            return ToolCallResult(
                name=name, output=None, is_error=True, error_message=str(exc)
            )

        is_error = bool(result.get("isError"))
        output: Any = result.get("structuredContent")
        if output is None:
            output = _flatten_content_blocks(result.get("content", []))
        return ToolCallResult(
            name=name,
            output=output,
            is_error=is_error,
            error_message=str(output) if is_error else None,
        )

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        envelope = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        try:
            response = await self._post_with_connect_retry(envelope)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPTransportError(f"MCP transport failed: {exc}") from exc
        body = response.json()
        if body.get("error"):
            error = body["error"]
            raise MCPRemoteError(
                f"MCP {method} failed: {error.get('message', 'unknown error')}"
            )
        return body.get("result", {})

    async def _post_with_connect_retry(self, envelope: dict[str, Any]) -> httpx.Response:
        """POST the envelope, waiting out a server that is not listening yet."""
        delay = self._connect_backoff
        for attempt in range(1, self._connect_attempts + 1):
            try:
                return await self._client.post("/mcp", json=envelope)
            except _CONNECT_FAILURES as exc:
                if attempt == self._connect_attempts:
                    raise
                logger.warning(
                    "MCP server unreachable at %s (attempt %d/%d): %s — retrying in %.1fs",
                    self._base_url,
                    attempt,
                    self._connect_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover


# Only failures that never established a connection: the server was not
# listening, so this request never reached it and cannot have taken effect.
# Anything that did reach the server (a timeout mid-call, a 5xx) is NOT
# retried here — a repeated `tools/call` would run the tool's writes twice.
_CONNECT_FAILURES = (httpx.ConnectError, httpx.ConnectTimeout)


def _flatten_content_blocks(blocks: list[dict[str, Any]]) -> Any:
    """Concatenate text blocks; pass-through structured ones."""
    text_parts: list[str] = []
    structured: list[Any] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        else:
            structured.append(block)
    if structured and not text_parts:
        return structured if len(structured) > 1 else structured[0]
    if structured and text_parts:
        return {"text": "\n".join(text_parts), "structured": structured}
    return "\n".join(text_parts)


class MCPTransportError(RuntimeError):
    pass


class MCPRemoteError(RuntimeError):
    pass


__all__ = ["MCPClient", "MCPRemoteError", "MCPTransportError"]
