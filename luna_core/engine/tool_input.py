"""A tool call's input read against its schema, before anything acts on it.

Models sometimes send a list or an object as JSON *text* — mostly when the
value is large and nested. Text that parses is decoded in place; text that
does not parse cannot run, and naming the field, the line and the column lets
the model fix it in one more call.
"""
from __future__ import annotations

import json
from typing import Any

_KINDS: dict[str, type] = {"array": list, "object": dict}


def _structured_kind(schema: dict[str, Any]) -> str | None:
    kind = (schema or {}).get("type")
    if isinstance(kind, list):
        found = [k for k in kind if k in _KINDS]
        return found[0] if len(found) == 1 else None
    return kind if kind in _KINDS else None


def _walk(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    kind = _structured_kind(schema)
    if kind is not None and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            where = (
                f" (line {exc.lineno}, column {exc.colno}: {exc.msg})"
                if isinstance(exc, json.JSONDecodeError)
                else ""
            )
            problems.append(
                f"`{path}` must be a JSON {kind}, and the text sent for it is not valid JSON{where}"
            )
            return value
        if not isinstance(decoded, _KINDS[kind]):
            problems.append(f"`{path}` must be a JSON {kind}")
            return value
        value = decoded
    if isinstance(value, dict):
        props = (schema or {}).get("properties") or {}
        if props:
            value = {
                key: _walk(item, props[key], f"{path}.{key}" if path else key, problems)
                if key in props
                else item
                for key, item in value.items()
            }
    elif isinstance(value, list):
        items = (schema or {}).get("items")
        if isinstance(items, dict):
            value = [_walk(item, items, f"{path}[{i}]", problems) for i, item in enumerate(value)]
    return value


def read_tool_input(
    tool_input: dict[str, Any], schema: dict[str, Any], *, check_required: bool = True
) -> tuple[dict[str, Any], list[str]]:
    """``(input with JSON text decoded where the schema wants a list or an
    object, problems)``. A problem is a field that still cannot be read, or a
    required one left out (unless the host names those itself); with any, the
    call must not run as sent."""
    problems: list[str] = []
    value = _walk(dict(tool_input or {}), {**(schema or {}), "type": "object"}, "", problems)
    if check_required:
        for name in (schema or {}).get("required") or []:
            if value.get(name) in (None, ""):
                problems.append(f"`{name}` is required")
    return value, problems
