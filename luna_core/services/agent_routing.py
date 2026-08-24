"""Generic multi-agent routing — the pieces a host composes to route a chat.

A host that runs several chat agents needs three things luna-core can provide
without knowing the host's domain:

  - :class:`AgentCard` — how the host describes each routable agent (who it is,
    when it should answer, who it may hand off to). Cards are plain data the
    host authors next to its seeds; adding an agent to the routing system is
    adding a card, nothing else.
  - :func:`make_transfer_tool` — ONE terminal system tool, ``transfer_to_agent``,
    whose directory of destinations is generated from the cards. It replaces
    per-destination ``route_to_x`` tools: agents pass the conversation baton by
    naming a card, and the tool result carries an explicit takeover note the
    *next* agent reads in the transcript (so it never mistakes the handoff for
    its own narration).
  - :func:`classify_route` — a cold-start classifier: one cheap structured LLM
    call that picks the answering agent for a conversation that has no sticky
    routing yet. The host decides when to call it (typically only on the first
    message) and what policy text to inject.

:func:`handoff_preamble` is the fourth piece, used by the conversation router
itself: a system-prompt block appended to the continuation turn after a handoff
so the receiving agent knows the transfer already happened and that it must
carry on with the user's pending request in its own voice.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from luna_core.mcp.system_tools.registry import SystemTool

logger = logging.getLogger(__name__)

TRANSFER_TOOL_NAME = "transfer_to_agent"

# Event-stream segment label for routing lifecycle events (mirrors the "chat"
# node the turn events ride under).
ROUTING_NODE = "routing"


@dataclass(frozen=True, slots=True)
class AgentCard:
    """How the host describes one routable agent to the routing system.

    ``purpose`` is the "when should I answer" text — it feeds both the transfer
    tool's directory (what other agents read to decide a handoff) and the
    cold-start classifier's prompt, so write it as a routing criterion, not as
    marketing copy. ``targets`` restricts who this agent may hand off to
    (``None`` = any other card)."""

    name: str
    display_name: str
    purpose: str
    targets: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """What :func:`classify_route` returns: the chosen card name and the
    classifier's stated reason (may be empty when the model omitted it or the
    call fell back to the default)."""

    agent: str
    reason: str = ""


def build_directory(
    cards: Sequence[AgentCard], *, exclude: str | None = None
) -> str:
    """The agent directory as prompt-ready lines, one per card."""
    return "\n".join(
        f"- {c.name} — {c.display_name}: {c.purpose}"
        for c in cards
        if c.name != exclude
    )


def handoff_preamble(agent: Any) -> str:
    """System-prompt block for the continuation turn right after a handoff.

    The receiving agent sees a transcript whose last assistant message is the
    *previous* agent calling the transfer tool — without this block, models
    routinely continue that narration ("I'll pass you to X") instead of
    answering as themselves. ``agent`` is the receiving agent row (only
    ``name`` / ``role`` are read, so tests can pass any stand-in)."""
    role = getattr(agent, "role", None)
    who = f"{agent.name} ({role})" if role else str(agent.name)
    return (
        "[HANDOFF] This conversation was just transferred to you — you are "
        f"now {who}, and the next assistant message is yours. The transfer "
        "tool call you see at the end of the history was made by the PREVIOUS "
        "agent, not by you: do not repeat it, do not announce any transfer, "
        "and do not greet as if the chat were starting. Read the user's "
        "pending request above and continue it directly in your own voice — "
        "if it is clear, act on it now; if it is not, ask the questions you "
        "need."
    )


def make_transfer_tool(
    *,
    cards: Sequence[AgentCard],
    set_active: Callable[[uuid.UUID, str], Awaitable[None]],
    tool_name: str = TRANSFER_TOOL_NAME,
) -> SystemTool:
    """Build the single generic handoff tool from the host's cards.

    ``set_active`` persists the host's sticky routing (conversation → agent
    name); the conversation id comes from the call context (``scope_id``), the
    calling agent from ``agent_name`` — both injected by the engine, so the
    model can neither spoof the caller nor route a foreign conversation.

    The tool is **terminal**: calling it ends the caller's turn and the
    conversation router re-dispatches the same request to the new agent. Soft
    failures (unknown target, transferring to yourself, a target the caller's
    card doesn't allow) return ``{"error": ...}`` so the turn continues and the
    model can correct itself instead of half-transferring."""
    by_name = {c.name: c for c in cards}
    names = sorted(by_name)

    async def _transfer(
        args: dict[str, Any], *, call_context: dict[str, Any]
    ) -> dict[str, Any]:
        target = str(args.get("agent") or "")
        card = by_name.get(target)
        if card is None:
            return {"error": f"unknown agent {target!r}; valid agents: {names}"}
        caller = str(call_context.get("agent_name") or "")
        if caller == target:
            return {
                "error": f"you are already {target}; answer the user directly "
                "instead of transferring."
            }
        caller_card = by_name.get(caller)
        if (
            caller_card is not None
            and caller_card.targets is not None
            and target not in caller_card.targets
        ):
            allowed = sorted(caller_card.targets)
            return {"error": f"{caller} may only transfer to: {allowed}"}
        try:
            conversation_id = uuid.UUID(str(call_context["scope_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "transfer tool missing scope_id in call_context"
            ) from exc
        await set_active(conversation_id, target)
        # The NEXT agent reads this result in the transcript when it takes the
        # continuation turn — address the note to it, explicitly, so the model
        # can't mistake the handoff for narration it should carry on.
        return {
            "transferred_to": target,
            "handoff_note": (
                f"Handoff complete: the conversation now belongs to "
                f"{card.display_name} ({target}). The next assistant message "
                f"IS {card.display_name} speaking — it must address the "
                "user's pending request directly, never announce a transfer, "
                "and never greet as if the chat were new."
            ),
        }

    description = (
        "Transfer this conversation to another agent, who takes over and "
        "answers the user directly on the same history. Call it INSTEAD of "
        "answering, with no accompanying text — your turn ends immediately "
        "and the chosen agent writes the next message. Use it when the "
        "user's request is another agent's job, or when they explicitly ask "
        "for that agent.\n\nAgents:\n" + build_directory(cards)
    )
    return SystemTool(
        name=tool_name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": names,
                    "description": "Canonical name of the agent to hand the "
                    "conversation to.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the handoff (what the "
                    "user needs).",
                },
            },
            "required": ["agent"],
            "additionalProperties": False,
        },
        handler=_transfer,
        scope="catalog",
        terminal=True,
    )


async def classify_route(
    *,
    llm_router: Any,
    provider_id: uuid.UUID,
    model: str,
    cards: Sequence[AgentCard],
    message: str,
    default: str,
    scope_id: uuid.UUID,
    guidelines: str | None = None,
) -> RouteDecision:
    """Pick the agent that should answer ``message`` — one cheap LLM call.

    Generic on purpose: the host passes its cards, its policy text
    (``guidelines``) and which card is the safe ``default``; luna-core owns the
    prompt shape and the tolerant parsing. Any malformed output degrades to the
    default rather than raising — routing must never cost the user a turn (the
    caller is still expected to guard against transport errors)."""
    valid = {c.name for c in cards}
    if default not in valid:
        raise ValueError(f"default {default!r} is not one of the cards")
    system = (
        "You are a routing classifier for a multi-agent assistant. Read the "
        "user's message and decide which ONE agent should answer it.\n\n"
        "Agents:\n" + build_directory(cards) + "\n\n"
        + (f"Policy:\n{guidelines}\n\n" if guidelines else "")
        + 'Respond with JSON only: {"agent": "<name>", "reason": "<short '
        'reason>"}. The agent MUST be one of the names above.'
    )
    blocks = await llm_router.complete(
        provider_id=provider_id,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": message}]}
        ],
        system=system,
        tools=[],
        temperature=0.0,
        model=model,
        # Strict structured-output providers (OpenAI/Azure) require EVERY
        # property to be listed in ``required`` — so ``reason`` is mandatory
        # too (a short reason is cheap and feeds the routing log anyway).
        output_schema={
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": sorted(valid)},
                "reason": {"type": "string"},
            },
            "required": ["agent", "reason"],
            "additionalProperties": False,
        },
        run_id=scope_id,
        node_id=ROUTING_NODE,
        make_io=lambda _db: _SilentIO(scope_id),
    )
    text = "".join(
        b.get("text", "") for b in blocks if b.get("type") == "text"
    ).strip()
    parsed = _parse_json_object(text)
    agent = parsed.get("agent") if isinstance(parsed, dict) else None
    if agent not in valid:
        logger.warning(
            "route classifier returned unusable output %r — defaulting to %s",
            text[:200],
            default,
        )
        return RouteDecision(agent=default, reason="")
    return RouteDecision(agent=agent, reason=str(parsed.get("reason") or ""))


class _Sequenced:
    __slots__ = ("sequence",)

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence


class _SilentIO:
    """``AgentIO`` for the classifier call: no events, no transcript.

    Without an injected IO the streaming provider falls back to the flow
    ``EventEmitter`` and tries to persist run_events/agent_messages rows keyed
    by a flow run that doesn't exist. The classifier is a side call — its
    output is a routing decision, not conversation content — so nothing it
    streams belongs on any channel or in any transcript. (Token usage is still
    recorded by the provider's ledger, keyed by the scope id.)"""

    def __init__(self, scope_id: uuid.UUID) -> None:
        self._scope_id = scope_id

    @property
    def scope_id(self) -> uuid.UUID:
        return self._scope_id

    def for_session(self, _db: Any) -> "_SilentIO":
        return self

    async def emit(
        self,
        event_type: Any,
        node_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> _Sequenced:
        return _Sequenced(0)

    async def save_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant parse of a model's JSON reply: bare object, fenced block, or
    an object embedded in prose. ``None`` when nothing parses."""
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    embedded = re.search(r"\{.*\}", text, re.DOTALL)
    if embedded:
        candidates.append(embedded.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


__all__ = [
    "ROUTING_NODE",
    "TRANSFER_TOOL_NAME",
    "AgentCard",
    "RouteDecision",
    "build_directory",
    "classify_route",
    "handoff_preamble",
    "make_transfer_tool",
]
