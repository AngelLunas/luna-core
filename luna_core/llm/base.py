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
"""
from __future__ import annotations

import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis


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
    ) -> list[dict[str, Any]]:
        """Return canonical assistant content blocks for one tool-calling turn."""

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
    "ToolDefinition",
    "abort_key",
    "delta_event_id",
    "inflight_meta_key",
    "run_state_key",
    "stream_key",
]
