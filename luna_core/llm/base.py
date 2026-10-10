"""Provider-agnostic LLM interface and shared types.

All providers translate between their native wire format and the canonical
content-block list used throughout luna-core. The canonical format is:

  assistant: [
    {"type": "thinking", "thinking": "..."},
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "tc_1", "name": "...", "input": {...}},
  ]
  user (tool results): [
    {"type": "tool_result", "tool_use_id": "tc_1", "content": "..."}
  ]
  user (host context riding the turn — rendered to the model as text,
  persisted with the turn, never shown as something the user wrote):
    {"type": "context", "context": "..."}
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory


class ToolDefinition(BaseModel):
    """Provider-agnostic tool spec the LLM sees in a tool-calling turn."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AbortSignalError(RuntimeError):
    """Raised when an abort signal is observed mid-stream.

    Carries the run/node identifiers so the runner can correlate cleanup.
    Partial content is always persisted as `AgentMessage(is_partial=True)`
    before this exception bubbles up.
    """

    def __init__(self, run_id: uuid.UUID | str, node_id: str):
        super().__init__(f"run {run_id} aborted at node {node_id}")
        self.run_id = run_id
        self.node_id = node_id


class LLMRateLimitError(RuntimeError):
    """Provider returned 429 / rate-limited. Router may retry with backoff."""


class LLMRefusalError(RuntimeError):
    """The model declined the request on policy grounds (and any fallback the
    provider tried declined too). Not retried: the same request would be
    declined again. ``category`` is the provider's reason, when it gives one."""

    def __init__(self, message: str, category: str | None = None):
        super().__init__(message)
        self.category = category


# How hard a model should think before answering, from cheapest to deepest.
# Provider-agnostic: each provider maps a level onto what its model accepts
# (and ignores it where the model has no such control). ``None`` everywhere
# means "the model's own default".
REASONING_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")


class BaseLLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        temperature: float,
        model: str,
        output_schema: dict[str, Any] | None,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        make_io: IOFactory | None = None,
        image_resolver: Callable[[str], Awaitable[str | None]] | None = None,
        builtin_tools: list[str] | None = None,
        timezone: str | None = None,
        reasoning_effort: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return canonical assistant content blocks for one tool-calling turn.

        ``reasoning_effort`` is one of ``REASONING_EFFORTS`` (``None`` = the
        model's default); a provider maps it to its own control or ignores it.

        A provider may attach a ``native`` key to a block it returns: its own
        wire form of that block, which it replays verbatim when the block comes
        back in history (signed reasoning, server-side tool results). Other
        providers ignore the key.

        ``timezone`` is the caller's IANA timezone (``None`` = unknown). A provider
        with anything tied to local time — the Claude CLI's own date note, a web
        search's location — uses it, never the server's clock.

        ``make_io`` lets the caller inject where the assistant turn and its
        lifecycle events are persisted (a flow ``EventEmitter`` or a chat
        emitter), bound to sessions the provider opens itself. When omitted,
        the provider falls back to the flow ``EventEmitter`` for the given
        ``run_id`` — preserving the original behavior for direct callers.

        ``image_resolver`` maps an attached media ``media_id`` to a renderable
        URL (typically a ``data:`` base64 URL) so a vision model sees the pixels.
        It is asked for videos too — the host answers with the video's poster
        frame (a still image), or ``None``. Omitted → attached media render as
        text notes (the text-model path).
        """

    async def embed(self, text: str) -> list[float]:
        ...


def abort_key(run_id: uuid.UUID | str) -> str:
    return f"abort:{run_id}"


def stream_key(run_id: uuid.UUID | str, message_id: uuid.UUID | str) -> str:
    """Redis list holding the chunks of one in-flight assistant turn.

    Keyed by ``message_id`` (not ``node_id``) so parallel iterations of
    the same ai_agent node — each generating its own message_id per LLM
    call — don't share a stream cache. Without that isolation: chunks
    from sibling iterations interleave into the same list, the
    snapshot/synth path attributes them all to whichever iteration's
    meta survived the most recent overwrite, and the first iteration
    that completes (calling ``_save_partial``) DELETEs the cache,
    wiping the still-in-flight siblings' history.
    """
    return f"stream:msg:{run_id}:{message_id}"


def inflight_meta_key(
    run_id: uuid.UUID | str, message_id: uuid.UUID | str
) -> str:
    # Per-message sidecar to `stream_key`: holds the started-event
    # sequence (plus iteration_id when emitted from inside an iteration
    # scope) for the assistant turn currently writing chunks into the
    # stream list. Lets a fresh WebSocket subscriber reconstruct
    # synthetic delta frames covering everything published before it
    # connected. Keyed by message_id so parallel iterations of the same
    # node each get their own meta (see docstring on ``stream_key``).
    return f"stream_meta:msg:{run_id}:{message_id}"


def delta_event_id(
    message_id: uuid.UUID | str, kind: str, chunk_index: int
) -> uuid.UUID:
    # Deterministic id for a single streamed delta. Live publishes and
    # mid-stream snapshot rehydrations both derive the same id for the same
    # (message_id, kind, chunk_index) triple, so the client reducer dedupes
    # them by id — no separate dedup table, no double-counted text on
    # reconnect.
    return uuid.uuid5(uuid.NAMESPACE_OID, f"delta:{message_id}:{kind}:{chunk_index}")


def run_state_key(run_id: uuid.UUID | str) -> str:
    return f"run_state:{run_id}"


__all__ = [
    "AbortSignalError",
    "BaseLLMProvider",
    "LLMRateLimitError",
    "LLMRefusalError",
    "REASONING_EFFORTS",
    "ToolDefinition",
    "abort_key",
    "delta_event_id",
    "inflight_meta_key",
    "run_state_key",
    "stream_key",
]
