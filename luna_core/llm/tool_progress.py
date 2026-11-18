"""A client tool call while the model is still writing it.

A large input (a whole crop plan) can take minutes to stream, and nothing
else happens on screen meanwhile. Providers announce the call when its block
starts and, as its JSON streams in, publish what a host-registered reader
makes of the text so far — for a plan, the phase being written and the
entries so far. Without a reader only the start is announced.

Readers get the partial JSON text (never complete until the block ends) and
return a small dict for the client, or None when there is nothing new to say.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

InputReader = Callable[[str], dict[str, Any] | None]

_READERS: dict[str, InputReader] = {}

# A reader's answer goes out when it changed, at most this often: the stream
# is a few hundred deltas a second and the client only needs the gist.
MIN_INTERVAL_S = 0.5


def register_input_reader(tool_name: str, reader: InputReader) -> None:
    """Make ``tool_name``'s streaming input readable as progress. A host calls
    it once at startup; registering a name again replaces its reader."""
    _READERS[tool_name] = reader


def clear_input_readers() -> None:
    _READERS.clear()


@dataclass
class _Call:
    tool_use_id: str
    name: str
    text: list[str] = field(default_factory=list)
    last: dict[str, Any] | None = None
    last_at: float = 0.0


class ToolWriting:
    """One turn's client tool calls being written, by content-block index."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._calls: dict[int, _Call] = {}
        self._clock = clock

    def start(self, index: int, tool_use_id: str, name: str) -> dict[str, Any]:
        self._calls[index] = _Call(tool_use_id=tool_use_id, name=name)
        return {"tool_use_id": tool_use_id, "name": name, "status": "started"}

    def feed(self, index: int, partial_json: str) -> dict[str, Any] | None:
        """The payload to publish for this delta, or None."""
        call = self._calls.get(index)
        if call is None or not partial_json:
            return None
        call.text.append(partial_json)
        reader = _READERS.get(call.name)
        if reader is None:
            return None
        now = self._clock()
        if now - call.last_at < MIN_INTERVAL_S:
            return None
        try:
            read = reader("".join(call.text))
        except Exception:  # noqa: BLE001 — progress is a courtesy, never a failure
            return None
        if not read or read == call.last:
            return None
        call.last, call.last_at = read, now
        return {"tool_use_id": call.tool_use_id, "name": call.name, "status": "writing", **read}
