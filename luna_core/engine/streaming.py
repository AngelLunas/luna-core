"""Transport abstractions for agent execution.

One ``AgentRunner`` turn produces two kinds of side effect: lifecycle
*events* (an assistant turn started, a token streamed, a tool was called)
and durable *transcript* messages (the user/assistant/tool blocks that
make up the conversation). Historically both were welded to the flow
engine — ``EventEmitter`` keyed everything off a ``flow_run_id`` and
persisted to ``RunEvent`` / ``AgentMessage``.

These protocols name the genuine shared concept so the same agent loop
can run in two contexts as co-equals, never importing one another:

  - inside a flow node → events become ``RunEvent`` rows on the run's
    channel; transcript becomes ``AgentMessage`` rows. The flow
    implementation is ``EventEmitter``.
  - inside a chat turn → events fan out on the conversation's channel;
    transcript becomes ``ConversationMessage`` rows. The chat
    implementation lives alongside the ``ChatRunner``.

``AgentRunner`` and the streaming provider depend only on these
protocols, never on a concrete implementation.

``node_id`` is an optional sub-scope label *within* an execution scope.
Flows use it to tell apart the nodes of one run; chat has no sub-scope
and passes ``None``. It keeps the flow column name (rather than a
generic "segment") because it is also the persisted field and the key
several system tools read from the call context.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.event import AgentMessageRole, RunEventType


class SupportsSequence(Protocol):
    """Anything carrying a monotonic per-scope ``sequence`` — a persisted
    ``RunEvent`` for flows, or a lightweight event record for chat. The
    streaming provider reads ``.sequence`` off an emitted lifecycle event
    to anchor the ordering of the transient deltas that follow it."""

    sequence: int


class EventSink(Protocol):
    """Where one agent execution's lifecycle events go."""

    @property
    def scope_id(self) -> uuid.UUID:
        """The streaming / abort / cache namespace for this execution: a
        ``flow_run_id`` for flows, a ``conversation_id`` for chat."""
        ...

    async def emit(
        self,
        event_type: RunEventType,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SupportsSequence:
        """Persist and/or broadcast a lifecycle event and return it."""
        ...


class TranscriptStore(Protocol):
    """Where one agent execution's durable transcript messages go."""

    async def save_message(
        self,
        node_id: str | None,
        role: AgentMessageRole,
        content: list[dict[str, Any]],
        is_partial: bool = False,
        thinking: str | None = None,
        message_id: uuid.UUID | None = None,
    ) -> Any:
        """Persist one transcript message and return the persisted row.

        ``thinking`` is already present as a block inside ``content``; the
        separate argument is a convenience for backends (like the flow
        ``AgentMessage``) that also keep it in a dedicated column.
        Backends without such a column (chat) ignore it."""
        ...


class AgentIO(EventSink, TranscriptStore, Protocol):
    """The full transport surface one ``AgentRunner`` turn needs: a place
    for lifecycle events and a place for the transcript. ``EventEmitter``
    (flow) and the chat emitter both satisfy it, so a single object can be
    passed where both capabilities are required."""

    def for_session(self, db: AsyncSession) -> AgentIO:
        """Mint a sibling bound to a different DB session, same scope and
        redis. The streaming provider opens its own short-lived sessions
        per persisted turn and uses this to write through them without
        knowing whether the execution is a flow run or a chat turn."""
        ...


# A scope-and-redis-bound factory: hand it a freshly opened session and it
# returns an ``AgentIO`` for that session. ``EventEmitter.for_session`` (and
# its chat counterpart) are exactly this — that's how the provider stays
# agnostic to the flow-vs-chat distinction.
IOFactory = Callable[[AsyncSession], AgentIO]


__all__ = [
    "AgentIO",
    "EventSink",
    "IOFactory",
    "SupportsSequence",
    "TranscriptStore",
]
