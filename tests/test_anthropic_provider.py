"""The Anthropic Messages API provider, driven with a fake SDK client.

What it must do, whatever the model: stream text and reasoning live, return
canonical blocks the AgentRunner understands, replay signed reasoning and
server-tool blocks verbatim, and decide every model-specific request field
(thinking, effort, output cap, web tools, fallbacks) from what the Models API
reports for that model — never from the model's name.
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import (
    BetaServerToolUseBlock,
    BetaTextBlock,
    BetaThinkingBlock,
    BetaToolUseBlock,
    BetaWebSearchResultBlock,
    BetaWebSearchToolResultBlock,
)

from luna_core.llm.base import LLMRateLimitError, LLMRefusalError, ToolDefinition
from luna_core.llm.providers.anthropic import (
    AnthropicProvider,
    _canonical_to_anthropic_messages,
)
from luna_core.models.event import RunEventType

# ----------------------------------------------------------------- fakes


def _support(ok: bool = True) -> SimpleNamespace:
    return SimpleNamespace(supported=ok)


def _model_info(
    model_id: str = "model-a",
    *,
    adaptive: bool = True,
    efforts: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max"),
    web_search: bool = True,
    code_execution: bool = True,
    images: bool = True,
    structured: bool = True,
    max_tokens: int | None = 128000,
    fallbacks: tuple[str, ...] = ("model-b",),
) -> SimpleNamespace:
    effort = SimpleNamespace(
        supported=bool(efforts),
        **{lvl: _support(lvl in efforts) for lvl in ("low", "medium", "high", "xhigh", "max")},
    )
    return SimpleNamespace(
        id=model_id,
        max_tokens=max_tokens,
        allowed_fallback_models=list(fallbacks),
        capabilities=SimpleNamespace(
            thinking=SimpleNamespace(
                supported=True,
                types=SimpleNamespace(
                    adaptive=_support(adaptive), enabled=_support(), disabled=_support()
                ),
            ),
            effort=effort,
            server_tools=SimpleNamespace(
                supported=web_search,
                web_search=_support(web_search),
                code_execution=_support(code_execution),
            ),
            image_input=_support(images),
            structured_outputs=_support(structured),
        ),
    )


def _usage(**counts: int) -> SimpleNamespace:
    data = {"input_tokens": 10, "output_tokens": 5, **counts}
    return SimpleNamespace(model_dump=lambda: dict(data))


def _final(content: list[Any], stop_reason: str = "end_turn", **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model="model-a",
        usage=_usage(),
        stop_details=extra.get("stop_details"),
    )


def _start(block: Any, index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_start", index=index, content_block=block)


def _stop(block: Any, index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index, content_block=block)


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _thinking_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
    )


class _FakeStream:
    def __init__(self, events: list[Any], final: Any, error: Exception | None = None):
        self._events = events
        self._final = final
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event
            if self._error is not None:
                raise self._error

        return gen()

    async def get_final_message(self):
        return self._final


class _FakeClient:
    """``beta.messages.stream`` answers one scripted round per call;
    ``beta.models.retrieve`` / ``models.list`` answer from ``infos``."""

    def __init__(self, rounds: list[_FakeStream], infos: list[Any] | None = None):
        self._rounds = list(rounds)
        self._infos = infos or [_model_info()]
        self.requests: list[dict[str, Any]] = []
        self.retrieves: list[tuple[str, Any]] = []
        client = self

        class _Messages:
            def stream(self, **request):
                client.requests.append(request)
                return client._rounds.pop(0)

        class _BetaModels:
            async def retrieve(self, model_id, betas=None):
                client.retrieves.append((model_id, betas))
                for info in client._infos:
                    if info.id == model_id:
                        return info
                raise KeyError(model_id)

        class _Models:
            def list(self):
                async def gen():
                    for info in client._infos:
                        yield info

                return gen()

        self.beta = SimpleNamespace(messages=_Messages(), models=_BetaModels())
        self.models = _Models()


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, Any] = {}
        self.lists: dict[str, list] = {}
        self.published: list[dict[str, Any]] = []
        self.aborted = False

    async def exists(self, *_keys):
        return 1 if self.aborted else 0

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

    async def publish(self, _channel, payload):
        self.published.append(json.loads(payload))

    async def delete(self, *keys):
        for k in keys:
            self.strings.pop(k, None)
            self.lists.pop(k, None)

    def events(self, event_type: RunEventType) -> list[dict[str, Any]]:
        return [m["payload"] for m in self.published if m["event_type"] == event_type.value]


class _CapturingIO:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []
        self.events: list[Any] = []
        self._seq = 0

    def for_session(self, _db):
        return self

    async def emit(self, event_type, node_id=None, payload=None):
        self._seq += 1
        self.events.append(event_type)
        return SimpleNamespace(sequence=self._seq)

    async def save_message(
        self, node_id, role, content, is_partial=False, thinking=None, message_id=None
    ):
        self.saved.append(
            {"content": content, "is_partial": is_partial, "thinking": thinking}
        )


class _Session:
    def __init__(self, sink: list[Any]) -> None:
        self._sink = sink

    def add(self, row):
        self._sink.append(row)

    async def commit(self):
        return None


def _provider(client: _FakeClient, usage_rows: list[Any] | None = None) -> AnthropicProvider:
    rows = usage_rows if usage_rows is not None else []

    @asynccontextmanager
    async def session():
        yield _Session(rows)

    return AnthropicProvider(api_key="k", client=client, session_factory=session)


async def _complete(provider: AnthropicProvider, redis: _FakeRedis, io: _CapturingIO, **kw):
    args: dict[str, Any] = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hola"}]}],
        "system": "be brief",
        "tools": [],
        "temperature": 0.7,
        "model": "model-a",
        "output_schema": None,
        "run_id": uuid.uuid4(),
        "node_id": "chat",
        "redis": redis,
        "make_io": io.for_session,
    }
    args.update(kw)
    return await provider.complete(**args)


def _thinking_block(text: str = "pensando", signature: str = "sig-1") -> BetaThinkingBlock:
    return BetaThinkingBlock.model_construct(type="thinking", thinking=text, signature=signature)


def _text_block(text: str) -> BetaTextBlock:
    return BetaTextBlock.model_construct(type="text", text=text, citations=None)


# ----------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_streams_reasoning_and_text_and_returns_signed_blocks():
    thinking, text = _thinking_block(), _text_block("Hola, ¿qué tal?")
    stream = _FakeStream(
        [
            _start(thinking),
            _thinking_delta("pensando"),
            _stop(thinking),
            _start(_text_block(""), 1),
            _text_delta("Hola, "),
            _text_delta("¿qué tal?"),
        ],
        _final([thinking, text]),
    )
    client = _FakeClient([stream])
    redis, io, rows = _FakeRedis(), _CapturingIO(), []

    blocks = await _complete(_provider(client, rows), redis, io)

    assert blocks == [
        {
            "type": "thinking",
            "thinking": "pensando",
            "native": {"type": "thinking", "thinking": "pensando", "signature": "sig-1"},
        },
        {"type": "text", "text": "Hola, ¿qué tal?"},
    ]
    assert [p["text"] for p in redis.events(RunEventType.agent_text_delta)] == [
        "Hola, ",
        "¿qué tal?",
    ]
    assert [p["text"] for p in redis.events(RunEventType.agent_thinking_delta)] == ["pensando"]
    assert io.saved == [{"content": blocks, "is_partial": False, "thinking": "pensando"}]
    assert RunEventType.agent_message_completed in io.events
    assert len(rows) == 1 and rows[0].model == "model-a"


@pytest.mark.asyncio
async def test_request_fields_come_from_the_models_api():
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))])
    await _complete(
        _provider(client), _FakeRedis(), _CapturingIO(), reasoning_effort="high"
    )

    request = client.requests[0]
    assert request["model"] == "model-a"
    assert request["max_tokens"] == 128000
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert request["output_config"] == {"effort": "high"}
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["system"] == "be brief"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    # The current models take no sampling controls: never sent.
    assert "temperature" not in request
    assert client.retrieves == [("model-a", ["server-side-fallback-2026-06-01"])]


@pytest.mark.asyncio
async def test_a_model_without_those_capabilities_gets_none_of_them():
    info = _model_info(adaptive=False, efforts=(), fallbacks=(), max_tokens=None)
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))], [info])
    await _complete(_provider(client), _FakeRedis(), _CapturingIO(), reasoning_effort="max")

    request = client.requests[0]
    assert "thinking" not in request
    assert "output_config" not in request
    assert "fallbacks" not in request and "betas" not in request
    assert request["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_effort_steps_down_to_the_deepest_level_the_model_takes():
    info = _model_info(efforts=("low", "medium", "high", "max"))
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))], [info])
    await _complete(_provider(client), _FakeRedis(), _CapturingIO(), reasoning_effort="xhigh")
    assert client.requests[0]["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_model_capabilities_are_cached_between_calls():
    final = _final([_text_block("ok")])
    client = _FakeClient([_FakeStream([], final), _FakeStream([], final)])
    provider = _provider(client)
    await _complete(provider, _FakeRedis(), _CapturingIO())
    await _complete(provider, _FakeRedis(), _CapturingIO())
    assert len(client.retrieves) == 1


@pytest.mark.asyncio
async def test_structured_output_rides_output_config_format():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client = _FakeClient([_FakeStream([], _final([_text_block('{"a": "x"}')]))])
    await _complete(_provider(client), _FakeRedis(), _CapturingIO(), output_schema=schema)
    assert client.requests[0]["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }


@pytest.mark.asyncio
async def test_tools_go_out_as_client_tools_and_calls_come_back_as_tool_use():
    call = BetaToolUseBlock.model_construct(
        type="tool_use", id="toolu_1", name="get_weather", input={"city": "Bogotá"}
    )
    client = _FakeClient([_FakeStream([_start(call)], _final([call], "tool_use"))])
    tool = ToolDefinition(name="get_weather", description="weather", input_schema={})

    blocks = await _complete(_provider(client), _FakeRedis(), _CapturingIO(), tools=[tool])

    assert client.requests[0]["tools"] == [
        {
            "name": "get_weather",
            "description": "weather",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert "tool_choice" not in client.requests[0]
    assert blocks == [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Bogotá"}}
    ]


def test_history_replays_native_blocks_and_drops_unsigned_reasoning():
    native_thinking = {"type": "thinking", "thinking": "t", "signature": "s"}
    history = [
        {"role": "user", "content": [{"type": "text", "text": "hola"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "native": native_thinking},
                {"type": "thinking", "thinking": "from another provider"},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": {"ok": True}},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "boom", "is_error": True},
            ],
        },
    ]
    wire = _canonical_to_anthropic_messages(history, None, tools_declared=True)
    assert wire == [
        {"role": "user", "content": [{"type": "text", "text": "hola"}]},
        {
            "role": "assistant",
            "content": [
                native_thinking,
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": '{"ok": true}'},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "boom", "is_error": True},
            ],
        },
    ]


def test_history_without_tools_declared_flattens_tool_blocks_to_text():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "hola"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "native": {"type": "thinking"}},
                {"type": "text", "text": "voy", "native": {"type": "text", "text": "voy"}},
                {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "x"}]},
    ]
    wire = _canonical_to_anthropic_messages(history, None, tools_declared=False)
    assert wire[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "voy"},
            {"type": "text", "text": "[called lookup with {}]"},
        ],
    }
    assert wire[2] == {"role": "user", "content": [{"type": "text", "text": "[tool result: x]"}]}


def test_attached_media_keep_their_labels_and_show_after_the_text():
    png = "data:image/png;base64,AAAA"
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "mira"},
                {"type": "image", "media_id": "m1"},
                {"type": "image", "media_id": "m2"},
                {"type": "video", "media_id": "v1"},
            ],
        }
    ]
    wire = _canonical_to_anthropic_messages(history, {"m1": png, "v1": png}, tools_declared=False)
    parts = wire[0]["content"]
    assert parts[0] == {
        "type": "text",
        "text": "mira\n[image attached: img-1 (shown below)]\n[image attached: img-2]\n"
        "[video attached: vid-1 (first frame shown below)]",
    }
    assert [p["type"] for p in parts[1:]] == ["image", "image"]
    assert parts[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}


@pytest.mark.asyncio
async def test_a_model_without_image_input_gets_notes_only():
    seen: list[str] = []

    async def resolver(media_id: str) -> str:
        seen.append(media_id)
        return "data:image/png;base64,AAAA"

    info = _model_info(images=False)
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))], [info])
    await _complete(
        _provider(client),
        _FakeRedis(),
        _CapturingIO(),
        messages=[{"role": "user", "content": [{"type": "image", "media_id": "m1"}]}],
        image_resolver=resolver,
    )
    assert seen == []
    assert client.requests[0]["messages"][0]["content"] == [
        {"type": "text", "text": "[image attached: img-1]"}
    ]


@pytest.mark.asyncio
async def test_web_search_runs_server_side_and_is_kept_for_replay():
    use = BetaServerToolUseBlock.model_construct(
        type="server_tool_use", id="srvtoolu_1", name="web_search", input={"query": "clima"}
    )
    hit = BetaWebSearchResultBlock.model_construct(
        type="web_search_result", title="Clima", url="https://x.test", encrypted_content="e"
    )
    result = BetaWebSearchToolResultBlock.model_construct(
        type="web_search_tool_result", tool_use_id="srvtoolu_1", content=[hit]
    )
    answer = _text_block("Hace sol.")
    stream = _FakeStream(
        [_start(use), _stop(use), _start(result, 1), _start(answer, 2), _text_delta("Hace sol.")],
        _final([use, result, answer]),
    )
    client = _FakeClient([stream])
    redis = _FakeRedis()

    blocks = await _complete(
        _provider(client), redis, _CapturingIO(),
        builtin_tools=["web_search", "web_fetch", "image_generation"],
        timezone="America/Bogota",
    )

    assert client.requests[0]["tools"] == [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "user_location": {"type": "approximate", "timezone": "America/Bogota"},
        },
        {"type": "web_fetch_20260209", "name": "web_fetch"},
    ]
    assert blocks[0]["type"] == "web_search_call"
    assert blocks[0]["queries"] == ["clima"]
    assert blocks[0]["native"]["id"] == "srvtoolu_1"
    assert blocks[1]["type"] == "native"
    assert blocks[1]["native"]["type"] == "web_search_tool_result"
    assert blocks[2] == {"type": "text", "text": "Hace sol."}
    statuses = [p["status"] for p in redis.events(RunEventType.builtin_tool_call)]
    assert statuses == ["searching", "completed", "result"]
    result_event = redis.events(RunEventType.builtin_tool_call)[-1]
    assert result_event["query"] == "clima"
    assert result_event["results"] == [{"title": "Clima", "url": "https://x.test"}]
    # Replayed in the same order, verbatim.
    wire = _canonical_to_anthropic_messages(
        [{"role": "assistant", "content": blocks}], None, tools_declared=True
    )
    assert [b["type"] for b in wire[0]["content"]] == [
        "server_tool_use",
        "web_search_tool_result",
        "text",
    ]


@pytest.mark.asyncio
async def test_basic_web_tools_when_the_model_cannot_run_code():
    info = _model_info(code_execution=False)
    client = _FakeClient([_FakeStream([], _final([_text_block("ok")]))], [info])
    await _complete(_provider(client), _FakeRedis(), _CapturingIO(), builtin_tools=["web_search"])
    assert client.requests[0]["tools"] == [{"type": "web_search_20250305", "name": "web_search"}]


@pytest.mark.asyncio
async def test_a_paused_turn_is_resumed_in_place():
    use = BetaServerToolUseBlock.model_construct(
        type="server_tool_use", id="srvtoolu_1", name="web_search", input={"query": "q"}
    )
    rounds = [
        _FakeStream([_start(use)], _final([use], "pause_turn")),
        _FakeStream([_start(_text_block("")), _text_delta("listo")], _final([_text_block("listo")])),
    ]
    client = _FakeClient(rounds)
    io = _CapturingIO()

    blocks = await _complete(_provider(client), _FakeRedis(), io, builtin_tools=["web_search"])

    assert len(client.requests) == 2
    assert client.requests[1]["messages"][-1] == {
        "role": "assistant",
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}}
        ],
    }
    assert [b["type"] for b in blocks] == ["web_search_call", "text"]
    assert len(io.saved) == 1


@pytest.mark.asyncio
async def test_a_refusal_raises_and_keeps_what_streamed():
    final = _final(
        [_text_block("Empie")], "refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )
    client = _FakeClient([_FakeStream([_start(_text_block("")), _text_delta("Empie")], final)])
    io = _CapturingIO()

    with pytest.raises(LLMRefusalError) as caught:
        await _complete(_provider(client), _FakeRedis(), io)

    assert caught.value.category == "cyber"
    assert io.saved == [
        {"content": [{"type": "text", "text": "Empie"}], "is_partial": True, "thinking": None}
    ]


@pytest.mark.asyncio
async def test_a_fallback_keeps_only_the_declined_models_text():
    declined_thinking = _thinking_block("x", "s-old")
    marker = SimpleNamespace(type="fallback", model_dump=lambda **_: {"type": "fallback"})
    final = _final([declined_thinking, _text_block("Hola "), marker, _text_block("mundo")])
    client = _FakeClient([_FakeStream([], final)])

    blocks = await _complete(_provider(client), _FakeRedis(), _CapturingIO())

    assert blocks == [{"type": "text", "text": "Hola "}, {"type": "text", "text": "mundo"}]


@pytest.mark.asyncio
async def test_rate_limit_becomes_the_engines_rate_limit_error():
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.test"))
    error = anthropic.RateLimitError("slow down", response=response, body=None)
    client = _FakeClient([_FakeStream([], None, error=error)])
    with pytest.raises(LLMRateLimitError):
        await _complete(_provider(client), _FakeRedis(), _CapturingIO())


@pytest.mark.asyncio
async def test_fast_model_is_the_newest_active_model_of_the_fast_line():
    def info(model_id, line, created, lifecycle="active"):
        return SimpleNamespace(
            id=model_id, line=line, lifecycle=lifecycle,
            created_at=datetime(2026, created, 1, tzinfo=timezone.utc),
        )

    client = _FakeClient(
        [],
        [
            info("h-old", "haiku", 1),
            info("h-new", "haiku", 5),
            info("h-retired", "haiku", 9, "retired"),
            info("s-newest", "sonnet", 10),
        ],
    )
    provider = _provider(client)
    assert await provider.fast_model() == "h-new"
    assert await provider.latest_model("sonnet") == "s-newest"
    assert await provider.latest_model("opus") is None
