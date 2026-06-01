"""Registry types for dedup checkers.

A ``DedupChecker`` is a small contract: given a list of projected
records (already cut down to the fields it cares about) plus the
per-call ``call_context`` the AgentRunner threads through every
system tool handler, return a verdict per record — ``None`` for
"new, go ahead and stash" or a ``DedupVerdict`` for "already exists,
do not stash, here's why".

Two kinds of "abstract" matter here:

  - **Field abstraction**: the checker declares ``required_fields``
    (name + type + description) so the UI can build a *field mapping*
    between the record shape a node produces and the fields the
    checker needs. The checker never sees raw record dicts; it sees
    the projected dict with its own canonical field names. That
    mapping lives in node config; the runtime applies it at dispatch.

  - **Implementation abstraction**: this file contains zero domain
    logic. Concrete checkers live in host apps that know the domain.
    Adding "dedup against contacts" is one file in sentinel; no
    luna-core change. That's the same boundary
    ``SystemToolRegistry`` enforces.

Re-registering the same name raises so accidental shadowing surfaces
immediately — same posture as ``SystemToolRegistry``.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


# Primitive types a dedup field can declare. Mirrors the carry/stash
# field primitives in ``luna_core/engine/iteration.py`` so the editor
# can reuse the same row-based widget. Kept narrow on purpose.
DEDUP_FIELD_PRIMITIVE_TYPES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "array"}
)


@dataclass(frozen=True)
class DedupFieldSpec:
    """One field a checker needs from each record to do its lookup.

    ``name`` is the checker's canonical field name (e.g. ``external_id``).
    The node config supplies a mapping from this name to a field on the
    record the agent actually produces (e.g. ``record['source_external_id']``).
    ``type`` and ``description`` are for the editor's UX — type filters
    which record fields show up as valid options in the mapping dropdown,
    description renders as the row hint.

    ``optional`` controls whether the node config *must* map this field.
    Required fields (the default) force the editor to wire them up,
    typically because the checker can't even run the cheap path (e.g.
    exact match) without them. Optional fields only sharpen a lookup —
    the canonical use is semantic fingerprint context that improves
    accuracy but isn't structurally required (skills, client_name).
    Skipping an optional field in the mapping means the projection
    omits it from the slim dict the handler receives, and the handler
    handles the missing key gracefully.
    """

    name: str
    type: str = "string"
    description: str = ""
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DedupFieldSpec.name must be a non-empty string")
        if self.type not in DEDUP_FIELD_PRIMITIVE_TYPES:
            raise ValueError(
                f"DedupFieldSpec.type must be one of "
                f"{sorted(DEDUP_FIELD_PRIMITIVE_TYPES)}; got {self.type!r}"
            )


@dataclass(frozen=True)
class DedupVerdict:
    """The "this record is a duplicate" answer the checker returns.

    The fields mirror what ``save_recommended_job`` already returns to
    agents so the LLM sees a consistent shape across dedup-aware tools:

      - ``match_kind``: free-form discriminator the checker defines
        (e.g. ``"exact"``, ``"semantic"``, ``"in_collection"``). The
        LLM uses it only as a hint; the runtime does not branch on it.
      - ``existing_id``: stable id of the matching entity in the
        checker's storage (``Job.id`` for sentinel.jobs). Lets the
        agent — or a downstream tool — refer back to the existing
        record if it needs to.
      - ``reason``: short human-readable string the LLM reads. Should
        explain *why* it's a duplicate at a level a tool retry can act
        on (e.g. "already saved with this external_id under this user").
    """

    match_kind: str
    existing_id: str | None = None
    reason: str = ""


# Handler signature: takes the *projected* records (already mapped to
# the checker's canonical field names) and the per-call ``call_context``,
# returns one verdict per record in the same order. ``None`` means "no
# duplicate found" — the runtime treats those records as eligible for
# stashing. The handler MUST return exactly ``len(records)`` entries
# (we assert this at dispatch so misbehaving checkers surface loudly
# instead of silently dropping records).
DedupCheckerHandler = Callable[..., Awaitable[list["DedupVerdict | None"]]]


@dataclass(frozen=True)
class DedupChecker:
    """One registered checker.

    Immutable on purpose: registrations should be set up once at module
    import time and remain stable for the process lifetime.
    """

    name: str
    description: str
    required_fields: list[DedupFieldSpec]
    handler: DedupCheckerHandler
    # Optional richer label for the UI (the dropdown shows ``label`` if
    # set, otherwise falls back to ``name``). Lets a host expose a
    # human name like "Jobs database" while keeping ``name`` machine-
    # friendly like ``sentinel.jobs``.
    label: str = ""

    def display_label(self) -> str:
        return self.label or self.name


class DedupCheckerRegistry:
    """In-process registry of dedup checkers.

    Mirrors ``SystemToolRegistry`` in shape (register / get / list)
    rather than inventing a new pattern; the AgentRunner-adjacent code
    already follows this convention.
    """

    def __init__(self) -> None:
        self._checkers: dict[str, DedupChecker] = {}

    def register(self, checker: DedupChecker) -> None:
        if checker.name in self._checkers:
            existing = self._checkers[checker.name]
            raise ValueError(
                f"dedup checker {checker.name!r} already registered "
                f"(existing label={existing.display_label()!r})"
            )
        self._checkers[checker.name] = checker

    def get(self, name: str) -> DedupChecker | None:
        return self._checkers.get(name)

    def list_all(self) -> list[DedupChecker]:
        return list(self._checkers.values())

    def unregister(self, name: str) -> None:
        """Remove a registration. Tests only."""
        self._checkers.pop(name, None)

    def clear(self) -> None:
        """Wipe every registration. Tests only — production never calls."""
        self._checkers.clear()


# Module-level default registry. Mirrors ``SystemToolRegistry``'s
# pattern so callers don't have to learn two different conventions.
_default_registry = DedupCheckerRegistry()


def get_default_registry() -> DedupCheckerRegistry:
    return _default_registry


__all__ = [
    "DEDUP_FIELD_PRIMITIVE_TYPES",
    "DedupChecker",
    "DedupCheckerHandler",
    "DedupCheckerRegistry",
    "DedupFieldSpec",
    "DedupVerdict",
    "get_default_registry",
]
