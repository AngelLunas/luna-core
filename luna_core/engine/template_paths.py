"""Path resolution for ``${context.<source>.<path>}`` markers.

The instructions editor inserts chips with paths like ``skills[*]`` (for
arrays of scalars) or ``work_experiences[*].title`` (project a field over
each item). Both the runtime template formatter and the preview endpoint
need to walk those paths the same way; this module is the single source
of truth.

Grammar (matches the frontend ``CONTEXT_REF_RE`` and the chip palette):

  segment := <name> | <name>[*] | [*]
  path    := segment ( '.' segment )*

Semantics:

  - ``a.b.c``            → walk dict/attr keys.
  - ``items[*]`` (last)  → return the whole list (identity over a list).
  - ``items[*].field``   → project ``field`` over each item, returning a list.
  - any None hit         → return None.
  - ``[*]`` on a non-list → return None (the chip is asking for iteration
    over something that isn't iterable).

Why ``[*]`` matters: the palette derives chips from each source's JSON
schema; for an array property it emits ``field[*]`` to mark "this is a
list" in the UI, even when the array is terminal. The earlier resolver
treated ``skills[*]`` as a literal dict key, which always missed and
produced an empty substitution.
"""
from __future__ import annotations

from typing import Any


def tokenize_path(path: str) -> list[str]:
    """Split ``a.b[*].c`` into ``['a', 'b', '[*]', 'c']``.

    Empty / pure-``[*]`` chunks are kept so the walker can react to them;
    a leading ``[*]`` (rare) becomes a standalone segment too.
    """
    segments: list[str] = []
    for chunk in path.split("."):
        cursor = chunk
        while cursor:
            if cursor.startswith("[*]"):
                segments.append("[*]")
                cursor = cursor[3:]
                continue
            bracket = cursor.find("[")
            if bracket == -1:
                segments.append(cursor)
                cursor = ""
            else:
                if bracket > 0:
                    segments.append(cursor[:bracket])
                cursor = cursor[bracket:]
    return segments


def resolve_path(state: Any, path: str) -> Any:
    """Walk ``path`` (dotted, with optional ``[*]`` segments) over ``state``.

    See module docstring for grammar and semantics. Always returns the
    raw value (or None) — stringification is the caller's job.
    """
    if not path:
        return state
    return _walk(state, tokenize_path(path))


def _walk(cursor: Any, segments: list[str]) -> Any:
    for index, seg in enumerate(segments):
        if cursor is None:
            return None
        if seg == "[*]":
            if not isinstance(cursor, list):
                return None
            rest = segments[index + 1 :]
            if not rest:
                # Terminal ``[*]`` — caller asked for the whole list.
                return cursor
            # Project the remaining path over each item.
            return [_walk(item, rest) for item in cursor]
        if isinstance(cursor, dict):
            cursor = cursor.get(seg)
        elif isinstance(cursor, list):
            # Bare numeric indexing isn't part of the grammar — bail out.
            return None
        else:
            cursor = getattr(cursor, seg, None)
    return cursor


__all__ = ["resolve_path", "tokenize_path"]
