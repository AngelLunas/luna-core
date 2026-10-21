"""How much of a conversation a model is sent: the context policy.

A conversation's history grows with every turn — mostly with tool results —
and every call sends all of it. Two levers keep it bounded, both decided per
agent by the host (its price tier, its budget) and carried out by the provider:

- ``clear_tool_results_at``: once a call's prompt passes this many tokens, the
  provider clears the oldest tool results before the model reads them (keeping
  the latest ``keep_tool_results`` and clearing at least ``clear_at_least``
  tokens each time, so the prompt cache is reset rarely and for a real gain).
  The stored history is never touched.
- ``compact_at``: when the last call's prompt reached this many tokens, the
  next turn first summarizes everything before it into one compaction block;
  from then on the history starts at that block. The summarized messages stay
  stored; the model no longer receives them. ``compaction_instructions``
  replaces the provider's own summarization prompt.

A provider that has neither lever ignores the policy.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPolicy:
    clear_tool_results_at: int | None = None
    keep_tool_results: int = 4
    clear_at_least: int = 15_000
    compact_at: int | None = None
    compaction_instructions: str | None = None


#: The canonical block a compaction leaves in the history: ``native`` is the
#: provider's signed summary (never sent to a client), ``model`` the model that
#: wrote it, ``media`` how many labeled attachments the summarized messages
#: held (labels keep counting past them), ``compacted_through_sequence`` the
#: last stored message it summarizes.
COMPACTION_BLOCK = "compaction"


def compaction_block(message: dict) -> dict | None:
    """The compaction block a canonical message carries, if any."""
    content = message.get("content") or []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == COMPACTION_BLOCK:
                return block
    return None


def public_compaction(block: dict) -> dict:
    """What a client sees of a compaction: where it happened, not the summary."""
    return {
        "type": COMPACTION_BLOCK,
        "compacted_through_sequence": block.get("compacted_through_sequence"),
        "at": block.get("at"),
    }


__all__ = ["COMPACTION_BLOCK", "ContextPolicy", "compaction_block", "public_compaction"]
