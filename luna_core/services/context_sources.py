"""Runtime context source registry.

A context source represents an external bag of data that an agent's
template can reference via ``${context.<source_name>.<path>}``. Each
product registers its sources at startup, then flow nodes declare
``context_bindings`` mapping each required source to an id pulled from
``state.inputs.*``, ``state.trigger.*``, ``state.outputs.*`` or a static
literal. At node execution time the engine resolves the id, calls the
loader, and exposes the resulting dict on ``state.context[<source>]``.

The registry itself is process-local and intentionally tiny — there is
no DB-backed catalog and no per-source configuration. Sources are
contracts with the database written in code; the UI only references
them.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — import only for typing
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class SourceLoadContext:
    """Per-call context handed to a source loader.

    ``state`` carries the current flow state dict (inputs/trigger/outputs/
    context) so a loader for an ``id_implicit`` source can resolve its
    target from e.g. ``state.trigger.user_id``.
    """

    db: "AsyncSession"
    redis: "Redis"
    state: dict[str, Any]


SourceLoader = Callable[[SourceLoadContext, str | None], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ContextSource:
    name: str
    description: str
    loader: SourceLoader
    schema: dict[str, Any]
    id_implicit: bool


class DuplicateSourceError(ValueError):
    """A source with the same name has already been registered."""


class UnknownSourceError(LookupError):
    """No source registered under that name."""


_REGISTRY: dict[str, ContextSource] = {}


def register_context_source(
    *,
    name: str,
    description: str,
    loader: SourceLoader,
    schema: dict[str, Any] | None = None,
    id_implicit: bool = False,
) -> ContextSource:
    """Register a context source. Idempotent for the exact same loader.

    Re-registering the same name with a different loader raises so
    misconfiguration (two products colliding on a name) fails loudly at
    startup rather than producing surprising prompts later.
    """
    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing.loader is loader:
            return existing
        raise DuplicateSourceError(
            f"context source {name!r} already registered with a different loader"
        )
    source = ContextSource(
        name=name,
        description=description,
        loader=loader,
        schema=schema or {},
        id_implicit=id_implicit,
    )
    _REGISTRY[name] = source
    logger.info("registered context source %s", name)
    return source


def get_context_source(name: str) -> ContextSource:
    source = _REGISTRY.get(name)
    if source is None:
        raise UnknownSourceError(name)
    return source


def list_context_sources() -> list[ContextSource]:
    return sorted(_REGISTRY.values(), key=lambda s: s.name)


def clear_context_sources() -> None:
    """Test-only: wipe the registry. Production code never calls this."""
    _REGISTRY.clear()


# Matches ${context.<name>...} where <name> is the identifier right after
# "context." and before the next "." or "}". The trailing path is ignored
# because we only care which sources a template references.
_CONTEXT_REF_RE = re.compile(r"\$\{context\.([A-Za-z_][A-Za-z0-9_]*)")


def extract_context_sources(text: str | None) -> list[str]:
    """Return the deduplicated, sorted list of context source names that a
    template references. Returns an empty list for None / empty input.
    """
    if not text:
        return []
    return sorted(set(_CONTEXT_REF_RE.findall(text)))


__all__ = [
    "ContextSource",
    "DuplicateSourceError",
    "SourceLoadContext",
    "SourceLoader",
    "UnknownSourceError",
    "clear_context_sources",
    "extract_context_sources",
    "get_context_source",
    "list_context_sources",
    "register_context_source",
]
