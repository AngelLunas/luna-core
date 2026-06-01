"""Build a FastMCP server that exposes Operations + host-registered tools.

The builder is library-shaped: it returns a configured `FastMCP` instance and
does NOT start a transport. The host application is responsible for running
the server (HTTP/SSE/stdio) and choosing where it listens.

Two flavors of tools end up on the same FastMCP server:

  1. **Connector tools** — one per active `Operation` row in the DB. Their
     handler delegates to `ConnectorRegistry.execute(operation.id, …)`.

  2. **Internal tools** — registered in code by the host app via
     `register_tool(...)`. These live in the host's process (e.g. writing to
     sentinel-specific tables) and never round-trip through the connector
     registry.

Internal tools are useful when the LLM needs to invoke host-owned business
logic — saving a recommended job, posting an internal notification, updating
a domain object — without exposing that endpoint as a REST connector.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.connectors.registry import ConnectorExecutionError, ConnectorRegistry
from luna_core.models.connector import Operation

logger = logging.getLogger(__name__)


ToolHandler = Callable[..., Awaitable[Any]]


@dataclass
class InternalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class MCPServerBuilder:
    """Materializes DB operations + internal tools into a single MCP server."""

    def __init__(
        self,
        registry: ConnectorRegistry,
        server_name: str = "luna-core",
    ):
        self._registry = registry
        self._server_name = server_name
        self._internal_tools: list[InternalTool] = []

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
    ) -> "MCPServerBuilder":
        """Register an internal tool to be exposed on the MCP server.

        Returns self so calls can be chained. Registration order is preserved
        and duplicate names override earlier entries (last-writer-wins).
        """
        # Drop a previous registration of the same name so re-registration
        # behaves predictably (handy in dev / reloads).
        self._internal_tools = [
            t for t in self._internal_tools if t.name != name
        ]
        self._internal_tools.append(
            InternalTool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
            )
        )
        return self

    async def build(self, db: AsyncSession):
        try:
            from fastmcp import FastMCP
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "fastmcp is required to build the MCP server; "
                "install with `pip install fastmcp`"
            ) from exc

        # `stateless_http=True` lets the streamable-http transport accept
        # one-shot JSON-RPC requests without forcing each client to do the
        # MCP `initialize` handshake first — matches how MCPClient talks to
        # the server. `json_response=True` returns a plain JSON body instead
        # of an SSE stream for tools/list and tools/call, which is what our
        # client expects.
        server = FastMCP(
            self._server_name,
            stateless_http=True,
            json_response=True,
            # `replace` makes attach_operation/attach_internal idempotent — a
            # re-registration with the same name overwrites the entry
            # silently instead of warning. We rely on this for dynamic
            # connector updates and dev reloads.
            on_duplicate_tools="replace",
        )

        operations = await self._load_operations(db)
        for op in operations:
            self.attach_operation(server, op)

        for tool in self._internal_tools:
            self.attach_internal(server, tool)

        logger.info(
            "MCP server '%s' built with %d connector tool(s) and %d internal tool(s)",
            self._server_name,
            len(operations),
            len(self._internal_tools),
        )
        return server

    async def refresh_from_db(self, server: Any, db: AsyncSession) -> None:
        """Re-sync the MCP server's connector tools with the active Operations
        in the DB. Detaches tools whose operation is no longer active and
        attaches any new ones. Idempotent.

        Use this when something writes Operations directly (seed scripts,
        admin scripts) instead of going through the service layer.
        """
        operations = await self._load_operations(db)
        wanted = {op.name for op in operations}
        existing_names = {t.name for t in server._tool_manager.list_tools()}
        # Only touch tools we own (connector tools). Internal tools live in
        # `self._internal_tools` and are not refreshed here.
        internal_names = {t.name for t in self._internal_tools}
        for stale in existing_names - wanted - internal_names:
            self.detach(server, stale)
        for op in operations:
            # attach_operation is idempotent — on_duplicate_tools="replace"
            # makes a re-registration overwrite the previous entry.
            self.attach_operation(server, op)

    async def _load_operations(self, db: AsyncSession) -> list[Operation]:
        result = await db.execute(
            select(Operation).where(Operation.is_active.is_(True))
        )
        return list(result.scalars().all())

    def attach_operation(self, server: Any, operation: Operation) -> None:
        """Attach (or replace) a connector tool for `operation` on `server`.

        Safe to call at any time after `build()` — the MCP server picks up
        the new tool on the next `tools/list` request from any client.
        """
        op_id = operation.id
        name = operation.name
        description = operation.description or ""
        input_schema = operation.input_schema or {"type": "object", "properties": {}}
        registry = self._registry

        async def handler(**kwargs):
            try:
                return await registry.execute(op_id, kwargs)
            except ConnectorExecutionError as exc:
                # Surface the upstream HTTP status + body verbatim to the
                # agent as a normal tool result. Re-raising would let
                # FastMCP collapse the exception into a content block
                # whose payload doesn't reliably make it back to the
                # client (we've seen the agent receive an empty error
                # string while logs showed the real "HTTP 404: ..."
                # message). The agent contract for "this call failed"
                # is a dict with an "error" key — same shape system
                # tools already use, so the LLM treats both paths
                # identically.
                logger.warning(
                    "connector tool '%s' failed: %s (operation_id=%s)",
                    name,
                    exc,
                    op_id,
                )
                return {"error": str(exc)}
            except Exception:  # noqa: BLE001
                logger.exception(
                    "MCP tool '%s' failed (operation_id=%s)", name, op_id
                )
                raise

        handler.__name__ = _safe_tool_name(name)
        handler.__doc__ = description
        self._attach_tool(server, handler, name, description, input_schema)

    def attach_internal(self, server: Any, tool: InternalTool) -> None:
        """Attach (or replace) an internal (host-owned) tool on `server`."""
        name = tool.name
        description = tool.description
        input_schema = tool.input_schema or {"type": "object", "properties": {}}
        user_handler = tool.handler

        async def handler(**kwargs):
            try:
                return await user_handler(**kwargs)
            except Exception:  # noqa: BLE001
                logger.exception("MCP internal tool '%s' failed", name)
                raise

        handler.__name__ = _safe_tool_name(name)
        handler.__doc__ = description
        self._attach_tool(server, handler, name, description, input_schema)

    @staticmethod
    def detach(server: Any, name: str) -> bool:
        """Remove a tool from `server` by name. Returns True if removed."""
        manager = getattr(server, "_tool_manager", None)
        if manager is None:
            return False
        # fastmcp 2.x exposes `remove_tool` on the manager; fall back to
        # mutating its internal dict if the method moves in a future release.
        if hasattr(manager, "remove_tool"):
            try:
                manager.remove_tool(name)
                return True
            except KeyError:
                return False
        tools = getattr(manager, "_tools", None)
        if isinstance(tools, dict) and name in tools:
            del tools[name]
            return True
        return False

    def _attach_tool(
        self,
        server: Any,
        handler: ToolHandler,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        # FastMCP 2.x derives a tool's JSON schema from the handler signature
        # and rejects **kwargs. Our handlers are intentionally schema-less
        # at the code level — the schema comes from the Operation row — so we
        # build a Tool with an explicit `parameters` and inject it through the
        # tool manager, which bypasses signature inspection.
        try:
            from fastmcp.tools.tool import Tool

            tool = Tool(
                fn=handler,
                name=name,
                description=description,
                parameters=input_schema,
            )
            server._tool_manager.add_tool(tool)
            return
        except Exception:  # noqa: BLE001
            pass
        # Fallbacks for older / future fastmcp versions whose API differs.
        try:
            server.add_tool(
                handler,
                name=name,
                description=description,
                input_schema=input_schema,
            )
        except TypeError:
            decorated = server.tool(name=name, description=description)(handler)
            setattr(decorated, "_input_schema", input_schema)


def _safe_tool_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out) or f"tool_{uuid.uuid4().hex[:8]}"
    if s[0].isdigit():
        s = "_" + s
    return s


__all__ = ["InternalTool", "MCPServerBuilder"]
