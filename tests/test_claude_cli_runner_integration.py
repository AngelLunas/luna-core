"""AgentRunner + ClaudeCLIProvider integration — the propose-only loop.

Exercises the full turn mechanics with a real subprocess (tests/fake_claude):
invocation 1 proposes a tool_use over the MCP catalog path, the RUNNER (not
the CLI) executes the system tool, and invocation 2 receives the rendered
transcript — tool call and result included — and produces the final text.
This is the architectural contract of the claude-cli provider: the CLI only
proposes; execution, approvals and transcript stay in the runner.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from luna_core.engine.agent import AgentRunner
from luna_core.llm.base import ToolDefinition
from luna_core.llm.providers.claude_cli import ClaudeCLIProvider
from luna_core.mcp.system_tools.registry import SystemTool

FAKE_BINARY = str(Path(__file__).parent / "fake_claude")


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.lists: dict[str, list] = {}

    async def exists(self, *keys):
        return sum(1 for k in keys if k in self.strings)

    async def rpush(self, key, *vals):
        self.lists.setdefault(key, []).extend(vals)
        return len(self.lists[key])

    async def expire(self, *_a, **_k):
        return True

    async def incr(self, key):
        n = int(self.strings.get(key, 0)) + 1
        self.strings[key] = n
        return n

    async def set(self, key, value, ex=None):
        self.strings[key] = value

    async def get(self, key):
        return self.strings.get(key)

    async def publish(self, *_a, **_k):
        return None

    async def delete(self, *keys):
        for k in keys:
            self.strings.pop(k, None)
            self.lists.pop(k, None)


class _CapturingEmitter:
    """AgentIO stand-in: records messages/events, hands itself out per-session."""

    def __init__(self) -> None:
        self._seq = 0
        self.events: list[tuple[Any, dict[str, Any]]] = []
        self.messages: list[dict[str, Any]] = []

    def for_session(self, _db):
        return self

    async def emit(self, event_type, *, node_id, payload):
        self._seq += 1
        self.events.append((event_type, payload))
        return SimpleNamespace(sequence=self._seq)

    async def save_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(id=uuid.uuid4())


class _RouterShim:
    """LLMRouter stand-in: routes every complete() to one provider instance."""

    def __init__(self, provider: ClaudeCLIProvider, redis: _FakeRedis) -> None:
        self._provider = provider
        self._redis = redis

    async def complete(self, *, provider_id, **kwargs):
        return await self._provider.complete(redis=self._redis, **kwargs)


@pytest.mark.asyncio
async def test_propose_only_loop_executes_tool_in_runner(tmp_path):
    scenario = {"invocations": [
        # Invocation 1: the model proposes the tool (denied+stopped in the
        # real CLI; the fake just emits what the provider would see).
        [
            {"event": {
                "type": "assistant",
                "message": {"content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "mcp__luna__water_plant",
                    "input": {"plant_name": "Marta", "amount_ml": 500},
                }]},
            }},
            {"event": {
                "type": "result", "subtype": "error_max_turns",
                "is_error": True, "result": "", "usage": {},
                "permission_denials": [], "num_turns": 2,
            }},
        ],
        # Invocation 2: transcript now carries the executed tool result.
        [
            {"event": {
                "type": "assistant",
                "message": {"content": [
                    {"type": "text", "text": "Listo, riego registrado."}
                ]},
            }},
            {"event": {
                "type": "result", "subtype": "success", "is_error": False,
                "result": "Listo, riego registrado.", "usage": {},
                "permission_denials": [], "num_turns": 1,
            }},
        ],
    ]}
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    os.environ["FAKE_CLAUDE_DIR"] = str(tmp_path)

    @asynccontextmanager
    async def _session_factory():
        yield SimpleNamespace(
            add=lambda _row: None,
            commit=_noop,
            refresh=_noop,
        )

    provider = ClaudeCLIProvider(
        binary_path=FAKE_BINARY,
        timeout_seconds=10,
        max_concurrency=1,
        session_factory=_session_factory,
    )
    redis = _FakeRedis()
    runner = AgentRunner(
        llm_router=_RouterShim(provider, redis),  # type: ignore[arg-type]
        mcp_client=SimpleNamespace(),  # type: ignore[arg-type]
    )

    handler_calls: list[dict[str, Any]] = []

    async def _water_plant(args: dict[str, Any], *, call_context: dict[str, Any]):
        handler_calls.append(args)
        return {"ok": True, "watering_id": "w-1"}

    tool = SystemTool(
        name="water_plant",
        description="Registra un riego",
        input_schema={"type": "object", "properties": {}},
        handler=_water_plant,
        scope="catalog",
    )

    async def _resolve_tools(_db, _agent_id, _context_names):
        return (
            [ToolDefinition(
                name="water_plant",
                description="Registra un riego",
                input_schema={"type": "object", "properties": {}},
            )],
            {"water_plant": tool},
        )

    runner._resolve_tools = _resolve_tools  # type: ignore[method-assign]

    agent = SimpleNamespace(
        id=uuid.uuid4(),
        name="savia-orchestrator",
        model="haiku",
        temperature=0.2,
        llm_provider_id=uuid.uuid4(),
        output_schema=None,
        builtin_tools=[],
    )
    emitter = _CapturingEmitter()
    result = await runner.run(
        agent,  # type: ignore[arg-type]
        history=[],
        new_message="riega a Marta con 500 ml",
        scope_id=uuid.uuid4(),
        node_id="chat",
        emitter=emitter,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
        redis=redis,  # type: ignore[arg-type]
        system_prompt="Eres Savia.",
    )

    # The runner executed the system tool with the CLI-proposed input…
    assert handler_calls == [{"plant_name": "Marta", "amount_ml": 500}]
    # …and the turn finished on invocation 2's text.
    assert result == "Listo, riego registrado."

    # Invocation 2 received the full transcript: the user ask, the proposed
    # call, and the runner's tool result — the stateless-render contract.
    second = json.loads((tmp_path / "invocation-1.json").read_text())
    prompt = "".join(
        b["text"]
        for b in json.loads(second["prompt"])["message"]["content"]
        if b["type"] == "text"
    )
    assert "<user>riega a Marta con 500 ml</user>" in prompt
    assert '<tool_call id="toolu_1" name="water_plant">' in prompt
    # the fresh tool result is what invocation 2 must respond to
    assert '<latest>\n<tool_result id="toolu_1">' in prompt
    assert '"ok": true' in prompt

    # Both invocations advertised the catalog (tools present both times).
    first = json.loads((tmp_path / "invocation-0.json").read_text())
    assert first["tools_file"][0]["name"] == "water_plant"
    assert second["tools_file"][0]["name"] == "water_plant"


async def _noop(*_a, **_k):
    return None
