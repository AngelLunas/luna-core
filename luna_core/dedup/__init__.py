"""Dedup-checker registry — abstract, populated by host apps at boot.

A "dedup checker" answers, for each record an agent wants to stash, the
question *"have I seen this before?"*. It is **content-defined**: the
checker knows what field projection to look at, where to look (a DB
table, an embedding index, a remote cache), and how to phrase the
verdict the LLM will read in the tool_result.

This package owns only the registry types. Concrete checkers live in
the host app that knows the domain — e.g. luna-sentinel registers a
``sentinel.jobs`` checker that queries the ``jobs`` table. luna-core
itself ships no checkers, so this module stays abstract per the core
boundary rule.

Wiring in three lines, mirroring ``SystemToolRegistry``:

  1. Host app's bootstrap calls ``get_default_registry().register(...)``
     once per checker at startup.
  2. The flow editor lists registered checkers via
     ``GET /dedup-checkers`` so a node author can pick one in the UI.
  3. At dispatch time, ``engine/nodes.py`` resolves the configured
     checker by name + field mapping into a callable injected as
     ``call_context["stash_dedup_checker"]``; ``stash_records`` then
     consults it for each record and rejects matches with a verdict
     the LLM can act on.
"""
from __future__ import annotations

from luna_core.dedup.node_config import (
    StashDedupBinding,
    StashDedupConfigError,
    build_call_context_checker,
    format_stash_dedup_addendum,
    resolve_stash_dedup_binding,
)
from luna_core.dedup.registry import (
    DedupChecker,
    DedupCheckerHandler,
    DedupCheckerRegistry,
    DedupFieldSpec,
    DedupVerdict,
    get_default_registry,
)

__all__ = [
    "DedupChecker",
    "DedupCheckerHandler",
    "DedupCheckerRegistry",
    "DedupFieldSpec",
    "DedupVerdict",
    "StashDedupBinding",
    "StashDedupConfigError",
    "build_call_context_checker",
    "format_stash_dedup_addendum",
    "get_default_registry",
    "resolve_stash_dedup_binding",
]
