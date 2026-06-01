"""End-to-end-ish test: state -> resolved system prompt.

Mirrors what nodes.py does for an ai_agent node, minus the actual MCP /
LLM call. Confirms ${context.*} and ${inputs.*} substitution and the
prompt-assembly contract together.
"""
from __future__ import annotations

import types

from luna_core.engine.agent import build_system_prompt
from luna_core.engine.nodes import _format_template


def _fake_agent(role: str = "", instructions: str = "", output_schema=None):
    """Minimal stand-in for the ORM Agent record. build_system_prompt only
    reads .role / .instructions / .output_schema so we don't need the real
    model."""
    return types.SimpleNamespace(
        role=role,
        instructions=instructions,
        output_schema=output_schema or {},
    )


def test_resolved_prompt_substitutes_context_and_inputs():
    agent = _fake_agent(
        role="Job matcher",
        instructions=(
            "Profile: ${context.profile.name} (${context.profile.title})\n"
            "Looking at profile_id=${inputs.profile_id}"
        ),
    )
    state = {
        "context": {"profile": {"name": "Angel", "title": "FSE"}},
        "inputs": {"profile_id": "uuid-1"},
    }

    resolved_role = _format_template(agent.role, state)
    resolved_instructions = _format_template(agent.instructions, state)
    prompt = build_system_prompt(
        agent, role=resolved_role, instructions=resolved_instructions
    )

    assert "Role: Job matcher" in prompt
    assert "Profile: Angel (FSE)" in prompt
    assert "profile_id=uuid-1" in prompt
    # No unresolved tokens left behind.
    assert "${" not in prompt


def test_build_system_prompt_falls_back_to_agent_fields_when_no_override():
    agent = _fake_agent(role="Analyst", instructions="Be careful.")
    prompt = build_system_prompt(agent)
    assert "Role: Analyst" in prompt
    assert "Be careful." in prompt


def test_build_system_prompt_includes_output_schema_instruction():
    agent = _fake_agent(
        role="Scorer",
        instructions="Score it.",
        output_schema={"type": "object"},
    )
    prompt = build_system_prompt(agent)
    assert "JSON object" in prompt
