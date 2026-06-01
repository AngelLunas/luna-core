"""Per-iteration context propagation for run events.

When an ai_agent node runs in ``scratchpad`` iteration mode, every event
emitted from inside the loop body — assistant message deltas, tool calls,
tool results, the completion event itself — needs to carry an
``iteration_id`` in its payload so the UI can group them back under the
right iteration block. With parallel execution this is essential:
several iterations are emitting interleaved events to the same per-run
channel, and without a routing key the timeline is unreadable.

We propagate the tag via a ``ContextVar`` rather than threading it
through every function signature. The reasons:

* ``asyncio.create_task`` (which ``asyncio.gather`` uses internally)
  copies the current ``contextvars.Context`` into the new task — so each
  parallel iteration's tasks see *their own* tag without polluting
  siblings. This is the property we need.
* The streaming LLM provider publishes ``agent_text_delta`` via the
  module-level ``publish_run_event`` helper, not through the emitter
  instance owned by the node executor. A ContextVar reaches both paths
  without rewiring the provider's call chain.
* The event emitter is the natural choke point — one ``_inject_iteration_tag``
  call inside ``EventEmitter.emit`` and ``publish_run_event`` covers
  every persisted event and every transient delta with no per-callsite
  bookkeeping.

The tag is the iteration's UUID. The full lifecycle info
(``iteration_index``, ``item_id``, ``collection``) is carried only on
the ``iteration_started`` event payload — sub-events stay light, the UI
joins on ``iteration_id``.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_current_iteration_id: ContextVar[uuid.UUID | None] = ContextVar(
    "luna_core_iteration_id", default=None
)


def get_current_iteration_id() -> uuid.UUID | None:
    """Return the iteration UUID active in the current task, or ``None``.

    Returns ``None`` when called outside any iteration body (the common
    case — node executors, lifecycle emitters, etc).
    """
    return _current_iteration_id.get()


@contextmanager
def iteration_scope(iteration_id: uuid.UUID) -> Iterator[None]:
    """Bind ``iteration_id`` to the current asyncio context.

    Use as a ``with`` block around the body of one iteration. Setting the
    value via ``ContextVar.set`` returns a Token that ``reset`` restores
    on exit, so nested scopes (which shouldn't happen in practice but
    are safe regardless) unwind cleanly.

    Asyncio runs each Task inside a copy of its parent's context, so when
    a parallel iteration calls ``asyncio.create_task`` / ``gather`` the
    tag does NOT propagate into those children unless they explicitly
    inherit it. That's fine for our case: the only thing running inside
    the scope is the agent run itself plus its tool calls, which all
    happen on the same task as the iteration body.
    """
    token = _current_iteration_id.set(iteration_id)
    try:
        yield
    finally:
        _current_iteration_id.reset(token)


def inject_iteration_tag(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``payload`` (or a fresh dict) with ``iteration_id`` set when active.

    Returns the *same dict* it was given when the tag is absent — avoids
    a needless copy on the common no-iteration path. When the tag is
    present and the payload already carries an ``iteration_id`` (caller
    set it explicitly, e.g. the ``iteration_*`` lifecycle events
    themselves), the caller's value wins.
    """
    iteration_id = _current_iteration_id.get()
    if iteration_id is None:
        return payload or {}
    if payload is None:
        return {"iteration_id": str(iteration_id)}
    if "iteration_id" in payload:
        return payload
    payload["iteration_id"] = str(iteration_id)
    return payload


__all__ = [
    "get_current_iteration_id",
    "inject_iteration_tag",
    "iteration_scope",
]
