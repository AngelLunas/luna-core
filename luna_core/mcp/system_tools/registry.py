"""System tool registry types and the default in-process registry.

System tools are first-class agent tools whose implementation lives in
process (a Python handler) instead of behind a remote MCP server. They
flow through the same dispatch path as connector-backed tools — the
agent doesn't distinguish them. What sets them apart is *where* they
come from and *how* they get into the agent's tool list:

  - **catalog scope**: registered for the lifetime of the process,
    visible (eventually) in the Agents view, toggleable per agent via
    the existing ``AgentOperation`` model. These are the reusable
    capabilities. Their input_schema is **generic** so the same tool
    serves every flow regardless of per-node configuration; the
    per-node config parameterizes the prompt and editor UX, not the
    catalog shape.

  - **context scope**: registered alongside catalog tools, but NOT
    seeded as Operation rows and NOT subject to ``AgentOperation``
    filtering. A runtime that owns a particular context (the iteration
    runtime, for ``yield_iteration``) injects them into the agent's
    tool list by name when its conditions apply.

The two scopes share one dispatch path: when the agent calls a tool
by name, the registry is consulted first; a hit short-circuits the
remote MCP call and invokes the local handler. The ``terminal`` flag
on a system tool tells the AgentRunner to exit its tool-calling loop
after the handler runs successfully.

Handler contract: an async callable taking the tool input dict and a
keyword-only ``call_context`` dict. The context carries per-call state
(flow_run_id, redis client, iteration_index, etc.) that handlers need
but isn't available at registration time. Handlers must not raise on
agent mistakes — return an error dict so the agent can self-correct
on the next turn — but they may raise for programming errors
(misconfigured call_context, missing collaborators, etc).

Each concrete tool lives in its own module under
``luna_core.mcp.system_tools.<name>``; ``install_builtins`` in the
package's ``__init__`` orchestrates registration. Do not bundle
multiple tools into one file — one tool, one file.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

SystemToolScope = Literal["catalog", "context"]

SystemToolHandler = Callable[..., Awaitable[Any]]
"""``async def handler(args: dict, *, call_context: dict) -> Any``.

Typed loosely because keyword-only signatures don't compose nicely
with ``Callable``; the dispatcher always invokes handlers with
``handler(args, call_context=...)`` so the keyword name is part of
the contract.
"""


@dataclass(frozen=True)
class SystemTool:
    """One registered system tool.

    Immutable on purpose: registrations should be set up once at module
    import time and remain stable. Re-registering the same name raises
    so accidental shadowing surfaces immediately.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: SystemToolHandler
    scope: SystemToolScope
    terminal: bool = False


class SystemToolRegistry:
    """In-process registry of system tools.

    Holds both scopes in one map keyed by tool name. ``list_catalog``
    returns the tools eligible for AgentOperation filtering;
    ``get`` returns any registered tool regardless of scope (used by
    the dispatcher, which doesn't care about scope). ``get_many`` is
    the convenience the iteration runtime uses to fetch context tools
    by name when wiring up an iterative agent run.
    """

    def __init__(self) -> None:
        self._tools: dict[str, SystemTool] = {}

    def register(self, tool: SystemTool) -> None:
        if tool.name in self._tools:
            existing = self._tools[tool.name]
            raise ValueError(
                f"system tool {tool.name!r} already registered "
                f"(existing scope={existing.scope!r}, new scope={tool.scope!r})"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> SystemTool | None:
        return self._tools.get(name)

    def get_many(self, names: list[str]) -> list[SystemTool]:
        """Fetch tools by name, ignoring any unknown entries.

        Unknown names are silently dropped rather than raising so a
        runtime can ask "give me whatever of these you have" without
        having to pre-check. If a name should always resolve, the
        caller is responsible for asserting.
        """
        results: list[SystemTool] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is not None:
                results.append(tool)
        return results

    def list_catalog(self) -> list[SystemTool]:
        return [t for t in self._tools.values() if t.scope == "catalog"]

    def list_all(self) -> list[SystemTool]:
        return list(self._tools.values())

    def unregister(self, name: str) -> None:
        """Remove a registration. Tests only."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Wipe every registration. Tests only — production code never
        calls this."""
        self._tools.clear()


# Module-level default registry. Production code uses this; tests can
# either monkeypatch it or construct their own ``SystemToolRegistry``
# instance and inject it where needed. Module-level mirrors how the
# existing ``context_sources`` registry already works in luna-core.
_default_registry = SystemToolRegistry()


def get_default_registry() -> SystemToolRegistry:
    return _default_registry


__all__ = [
    "SystemTool",
    "SystemToolHandler",
    "SystemToolRegistry",
    "SystemToolScope",
    "get_default_registry",
]
