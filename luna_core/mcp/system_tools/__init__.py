"""System tool package.

This file is the orchestrator: it re-exports the registry types from
``registry.py`` and provides ``install_builtins`` which composes the
per-tool ``register`` functions. It owns no tool logic. To add a new
built-in: create ``luna_core/mcp/system_tools/<name>.py`` with the
schema + handler + ``register(registry)`` function, then add a single
line to ``install_builtins`` calling its ``register``.
"""
from __future__ import annotations

from luna_core.mcp.system_tools import list_scratchpad, stash_records, yield_iteration
from luna_core.mcp.system_tools.registry import (
    SystemTool,
    SystemToolHandler,
    SystemToolRegistry,
    SystemToolScope,
    get_default_registry,
)


def install_builtins(registry: SystemToolRegistry | None = None) -> None:
    """Register every built-in system tool into ``registry``.

    Defaults to the module-level default registry. Idempotent within a
    single process is *not* guaranteed — the registry raises on
    re-registration so a duplicate call surfaces loudly. The host
    application calls this exactly once at startup; tests construct
    isolated registries and pass them in.
    """
    target = registry if registry is not None else get_default_registry()
    stash_records.register(target)
    list_scratchpad.register(target)
    yield_iteration.register(target)


__all__ = [
    "SystemTool",
    "SystemToolHandler",
    "SystemToolRegistry",
    "SystemToolScope",
    "get_default_registry",
    "install_builtins",
]
