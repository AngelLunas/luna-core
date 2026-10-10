"""Anthropic Messages API provider.

Drives ``POST /v1/messages`` (streaming) through the official SDK. The
canonical block format the engine uses is already Anthropic-shaped, so the
translation is thin; what this module adds is everything a model-specific
request needs, and none of it is assumed from a model's name: the Models API
says, per model, which thinking types and effort levels it accepts, whether
it runs server-side web search, takes images, constrains output to a schema,
how many output tokens it may write and which fallback models it has. That
answer is cached per model (``anthropic_capabilities_ttl_seconds``).

Tool calling is the runner's, as with every provider: our tools go out as
client tools and come back as ``tool_use`` blocks the AgentRunner executes.
Built-in tools (``web_search``, ``web_fetch``) run server-side inside the
turn; a turn the server pauses mid-search is resumed in place.

Blocks the engine cannot rebuild from canonical fields alone — signed
reasoning, cited text, server-tool calls and results — carry their wire form
under ``native`` and are replayed verbatim, in order, when the turn comes
back as history: a model rejects a later call whose earlier reasoning was
altered.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import anthropic
from redis.asyncio import Redis

from luna_core.core.config import settings
from luna_core.engine.emitter import EventEmitter, publish_run_event
from luna_core.llm.base import (
    REASONING_EFFORTS,
    AbortSignalError,
    LLMRateLimitError,
    LLMRefusalError,
    ToolDefinition,
    abort_key,
    delta_event_id,
    inflight_meta_key,
    stream_key,
)
from luna_core.llm.providers._turn import (
    _LABELED_KINDS,
    ImageResolver,
    SessionFactory,
    StreamingTurnProvider,
    _anthropic_usage_shim,
    _data_url_to_image_block,
    _MediaLabels,
    _media_note,
)
from luna_core.models.event import AgentMessageRole, RunEventType
from luna_core.services.usage import record_usage

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory

logger = logging.getLogger(__name__)

# Server-side fallback on a policy decline: the API re-runs the request on
# the model it recommends for the decline's category, inside the same call.
# Reading a model's fallback list needs the first header; asking for the
# "default" routing needs the second.
_FALLBACK_LIST_BETA = "server-side-fallback-2026-06-01"
_FALLBACK_DEFAULT_BETA = "server-side-fallback-2026-07-01"

# Built-in tool versions, by whether the model can run code server-side: the
# newer versions filter results with code execution under the hood.
_WEB_TOOLS: dict[str, tuple[str, str]] = {
    # name → (version with dynamic filtering, basic version)
    "web_search": ("web_search_20260209", "web_search_20250305"),
    "web_fetch": ("web_fetch_20260209", "web_fetch_20250910"),
}

# Output cap when the Models API does not report one for a model.
_DEFAULT_MAX_TOKENS = 16000


def _supported(capability: Any) -> bool:
    return bool(getattr(capability, "supported", False))


class _ModelSpec:
    """What one model accepts, as the Models API reports it."""

    def __init__(self, info: Any) -> None:
        caps = getattr(info, "capabilities", None)
        self.model_id: str = info.id
        self.max_tokens: int = getattr(info, "max_tokens", None) or _DEFAULT_MAX_TOKENS
        thinking_types = getattr(getattr(caps, "thinking", None), "types", None)
        self.adaptive_thinking = _supported(getattr(thinking_types, "adaptive", None))
        effort = getattr(caps, "effort", None)
        self.efforts: tuple[str, ...] = (
            tuple(lvl for lvl in REASONING_EFFORTS if _supported(getattr(effort, lvl, None)))
            if _supported(effort)
            else ()
        )
        server_tools = getattr(caps, "server_tools", None)
        self.web_search = _supported(getattr(server_tools, "web_search", None))
        self.code_execution = _supported(getattr(server_tools, "code_execution", None))
        self.images = _supported(getattr(caps, "image_input", None))
        self.structured_outputs = _supported(getattr(caps, "structured_outputs", None))
        self.has_fallbacks = bool(getattr(info, "allowed_fallback_models", None))

    def effort_for(self, requested: str | None) -> str | None:
        """The requested level if the model takes it, else the deepest level
        it takes below that one; None when it takes none."""
        if not requested or not self.efforts or requested not in REASONING_EFFORTS:
            return None
        for level in reversed(REASONING_EFFORTS[: REASONING_EFFORTS.index(requested) + 1]):
            if level in self.efforts:
                return level
        return None


def _tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        schema = dict(tool.input_schema or {})
        schema.setdefault("type", "object")
        if schema["type"] == "object":
            schema.setdefault("properties", {})
        out.append(
            {"name": tool.name, "description": tool.description, "input_schema": schema}
        )
    return out


def _server_tools(
    builtin_tools: list[str] | None, spec: _ModelSpec, timezone: str | None
) -> list[dict[str, Any]]:
    """The agent's built-in tools this model runs server-side. Names the
    provider has no equivalent for, or a model without server tools, are
    dropped (logged): an agent's builtin list is shared across providers."""
    out: list[dict[str, Any]] = []
    for name in builtin_tools or []:
        versions = _WEB_TOOLS.get(name)
        if versions is None or not spec.web_search:
            logger.warning(
                "anthropic provider: ignoring builtin tool %r for %s", name, spec.model_id
            )
            continue
        tool: dict[str, Any] = {
            "type": versions[0] if spec.code_execution else versions[1],
            "name": name,
        }
        # A search asked about "this week" answers for the user's week, not
        # the server's.
        if name == "web_search" and timezone:
            tool["user_location"] = {"type": "approximate", "timezone": timezone}
        if all(t["name"] != name for t in out):
            out.append(tool)
    return out


def _tool_result_content(payload: Any) -> str:
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


def _flatten_tool_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """A history tool block as plain text, for a call that declares no tools
    (the API refuses tool blocks then)."""
    if block.get("type") == "tool_use":
        return {
            "type": "text",
            "text": f"[called {block.get('name')} with "
            f"{json.dumps(block.get('input', {}), default=str)}]",
        }
    if block.get("type") == "tool_result":
        return {
            "type": "text",
            "text": f"[tool result: {_tool_result_content(block.get('content'))}]",
        }
    return None


def _canonical_to_anthropic_messages(
    messages: list[dict[str, Any]],
    image_urls: dict[str, str] | None,
    *,
    tools_declared: bool,
) -> list[dict[str, Any]]:
    """Canonical history → Messages API ``messages``.

    Assistant blocks carrying ``native`` are replayed as written; reasoning
    from another provider (no ``native``) is dropped — it has no signature
    the API would accept. Attached media get the same ``img-N`` / ``vid-N``
    labels as every provider; a resolved one is shown after the text, in
    label order, so the note ``(shown below)`` matches the picture."""
    result: list[dict[str, Any]] = []
    labels = _MediaLabels()
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", []) or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if role == "user":
            parts: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") != "tool_result":
                    continue
                if not tools_declared:
                    parts.append(_flatten_tool_block(b))
                    continue
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b.get("tool_use_id"),
                    "content": _tool_result_content(b.get("content")),
                }
                if b.get("is_error"):
                    item["is_error"] = True
                parts.append(item)
            texts = [b["text"] for b in content if b.get("type") == "text" and b.get("text")]
            notes: list[str] = []
            images: list[dict[str, Any]] = []
            for b in content:
                kind = b.get("type")
                if kind not in _LABELED_KINDS:
                    continue
                label = labels.next(kind)
                url = (image_urls or {}).get(str(b.get("media_id")))
                image = _data_url_to_image_block(url) if url else None
                notes.append(_media_note(kind, label, image is not None))
                if image is not None:
                    images.append(image)
            text = "\n".join([*texts, *notes])
            if text:
                parts.append({"type": "text", "text": text})
            parts.extend(images)
            if parts:
                result.append({"role": "user", "content": parts})
        elif role == "assistant":
            parts = []
            for b in content:
                btype = b.get("type")
                native = b.get("native")
                if not tools_declared and btype == "tool_use":
                    parts.append(_flatten_tool_block(b))
                elif not tools_declared and native is not None:
                    # Replayed server-tool or signed blocks need tools (or an
                    # untouched turn) to be valid; their text survives below.
                    if btype == "text" and b.get("text"):
                        parts.append({"type": "text", "text": b["text"]})
                elif native is not None:
                    parts.append(native)
                elif btype == "text" and b.get("text"):
                    parts.append({"type": "text", "text": b["text"]})
                elif btype == "tool_use":
                    parts.append(
                        {
                            "type": "tool_use",
                            "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input", {}) or {},
                        }
                    )
            if parts:
                result.append({"role": "assistant", "content": parts})
        elif role == "system":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            if text:
                result.append({"role": "user", "content": [{"type": "text", "text": text}]})
    return result


def _dump(block: Any) -> dict[str, Any]:
    return block.model_dump(mode="json", exclude_none=True)


def _after_fallback(content: list[Any]) -> list[Any]:
    """A declined attempt's blocks the fallback model continued from: only its
    text stands before the last ``fallback`` marker (reasoning and calls made
    by the declined model are not part of the turn); the marker goes too."""
    last = max(
        (i for i, b in enumerate(content) if getattr(b, "type", None) == "fallback"),
        default=-1,
    )
    if last == -1:
        return list(content)
    kept = [b for b in content[:last] if getattr(b, "type", None) == "text"]
    return kept + list(content[last + 1 :])


def _to_canonical(content: list[Any]) -> list[dict[str, Any]]:
    """Final message content → canonical blocks, order kept exactly."""
    blocks: list[dict[str, Any]] = []
    for block in _after_fallback(content):
        btype = getattr(block, "type", None)
        if btype == "thinking":
            blocks.append(
                {"type": "thinking", "thinking": block.thinking or "", "native": _dump(block)}
            )
        elif btype == "redacted_thinking":
            blocks.append({"type": "thinking", "thinking": "", "native": _dump(block)})
        elif btype == "text":
            item: dict[str, Any] = {"type": "text", "text": block.text}
            if getattr(block, "citations", None):
                item["native"] = _dump(block)
            blocks.append(item)
        elif btype == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input or {},
                }
            )
        elif btype == "server_tool_use" and block.name == "web_search":
            query = (block.input or {}).get("query")
            blocks.append(
                {
                    "type": "web_search_call",
                    "id": block.id,
                    "queries": [query] if query else [],
                    "native": _dump(block),
                }
            )
        elif btype == "server_tool_use" and block.name == "web_fetch":
            blocks.append(
                {
                    "type": "web_fetch_call",
                    "id": block.id,
                    "url": (block.input or {}).get("url"),
                    "native": _dump(block),
                }
            )
        elif btype is not None:
            # Server-tool results and anything else the model returns: no
            # canonical meaning, kept only to be replayed in place.
            blocks.append({"type": "native", "native": _dump(block)})
    return blocks


def _search_hits(result_block: Any) -> list[dict[str, str]]:
    content = getattr(result_block, "content", None)
    if not isinstance(content, list):  # an error object, not a hit list
        return []
    return [
        {"title": getattr(hit, "title", "") or "", "url": hit.url}
        for hit in content
        if getattr(hit, "url", None)
    ]


class _Turn:
    """One assistant turn's streaming state, across paused rounds."""

    def __init__(self, run_id: uuid.UUID) -> None:
        self.message_id = uuid.uuid4()
        self.s_key = stream_key(run_id, self.message_id)
        self.d_key = f"delta_seq:{run_id}:{self.message_id}"
        self.started = False
        self.delta_seq_base = 0
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.text_chunk_index = 0
        self.thinking_chunk_index = 0
        self.blocks: list[dict[str, Any]] = []
        self.usage: dict[str, int] = {}
        self.model: str | None = None
        # server_tool_use id → (tool name, query or url), for the live events.
        self.server_calls: dict[str, tuple[str, str | None]] = {}

    def add_usage(self, usage: Any) -> None:
        for key, value in (usage.model_dump() if usage is not None else {}).items():
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


class AnthropicProvider(StreamingTurnProvider):
    """Chat provider on the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        session_factory: SessionFactory | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(session_factory=session_factory)
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key or "missing", base_url=base_url or None
        )
        self._specs: dict[str, tuple[float, _ModelSpec]] = {}
        self._latest: dict[str, tuple[float, str | None]] = {}

    # ------------------------------------------------------------ models API
    async def _spec(self, model: str) -> _ModelSpec:
        cached = self._specs.get(model)
        now = time.monotonic()
        if cached is not None and now - cached[0] < settings.anthropic_capabilities_ttl_seconds:
            return cached[1]
        info = await self._client.beta.models.retrieve(model, betas=[_FALLBACK_LIST_BETA])
        spec = _ModelSpec(info)
        self._specs[model] = (now, spec)
        return spec

    async def latest_model(self, line: str) -> str | None:
        """The newest active model of a model line ("haiku", "sonnet", "opus",
        …) as the Models API lists it; None when the line lists none."""
        now = time.monotonic()
        cached = self._latest.get(line)
        if cached is not None and now - cached[0] < settings.anthropic_capabilities_ttl_seconds:
            return cached[1]
        best: Any = None
        async for info in self._client.models.list():
            if getattr(info, "line", None) != line:
                continue
            if getattr(info, "lifecycle", "active") != "active":
                continue
            if best is None or info.created_at > best.created_at:
                best = info
        model = best.id if best is not None else None
        self._latest[line] = (now, model)
        return model

    async def fast_model(self) -> str | None:
        """The fast line's newest model — for side calls that need an answer,
        not depth."""
        return await self.latest_model("haiku")

    # ------------------------------------------------------------------ chat
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        temperature: float,  # the current models take no sampling controls
        model: str | None,
        output_schema: dict[str, Any] | None,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        make_io: IOFactory | None = None,
        image_resolver: ImageResolver | None = None,
        builtin_tools: list[str] | None = None,
        timezone: str | None = None,
        reasoning_effort: str | None = None,
    ) -> list[dict[str, Any]]:
        if not model:
            raise ValueError("anthropic provider requires an explicit model")
        spec = await self._spec(model)
        build_io = make_io or (lambda session: EventEmitter(session, redis, run_id))

        image_urls = (
            await self._resolve_image_urls(messages, image_resolver) if spec.images else {}
        )
        wire_tools = [
            *_server_tools(builtin_tools, spec, timezone),
            *_tools_to_anthropic(tools),
        ]
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": spec.max_tokens,
            "messages": _canonical_to_anthropic_messages(
                messages, image_urls or None, tools_declared=bool(wire_tools)
            ),
            # Caches the longest stable prefix (tools, system, earlier turns):
            # every call of a tool loop re-reads it at a fraction of the cost.
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            request["system"] = system
        if wire_tools:
            request["tools"] = wire_tools
        if spec.adaptive_thinking:
            # "summarized" streams a readable account of the reasoning; without
            # it the thinking arrives empty and the user sees only a pause.
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
        output_config: dict[str, Any] = {}
        effort = spec.effort_for(reasoning_effort)
        if effort:
            output_config["effort"] = effort
        if output_schema and spec.structured_outputs:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        if output_config:
            request["output_config"] = output_config
        if spec.has_fallbacks:
            request["betas"] = [_FALLBACK_DEFAULT_BETA]
            request["fallbacks"] = "default"

        turn = _Turn(run_id)
        for _round in range(settings.anthropic_max_pause_continuations + 1):
            final = await self._stream_round(
                request, turn, run_id=run_id, node_id=node_id, redis=redis, build_io=build_io
            )
            turn.blocks.extend(_to_canonical(final.content))
            turn.add_usage(final.usage)
            turn.model = final.model
            if final.stop_reason != "pause_turn":
                break
            # The server paused a long server-tool turn: hand its content back
            # and the model continues the same turn where it stopped.
            request["messages"] = [
                *request["messages"],
                {"role": "assistant", "content": [_dump(b) for b in final.content]},
            ]

        if final.stop_reason == "refusal":
            await self._save_partial_blocks(
                turn.text_parts, turn.thinking_parts, run_id, node_id, redis,
                turn.message_id, build_io,
            )
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusalError(
                f"{model} declined the request (category={category})", category
            )
        if final.stop_reason == "max_tokens" and any(
            b["type"] == "tool_use" for b in turn.blocks
        ):
            # A tool call cut off mid-input would run with half its arguments.
            await self._save_partial_blocks(
                turn.text_parts, turn.thinking_parts, run_id, node_id, redis,
                turn.message_id, build_io,
            )
            raise RuntimeError(
                f"{model} hit max_tokens ({spec.max_tokens}) inside a tool call"
            )

        thinking = (
            "\n\n".join(b["thinking"] for b in turn.blocks if b["type"] == "thinking" and b["thinking"])
            or None
        )
        async with self._session_factory() as db:
            emitter = build_io(db)
            await emitter.save_message(
                node_id=node_id,
                role=AgentMessageRole.assistant,
                content=turn.blocks,
                is_partial=False,
                thinking=thinking,
                message_id=turn.message_id,
            )
            if turn.started:
                await emitter.emit(
                    RunEventType.agent_message_completed,
                    node_id=node_id,
                    payload={
                        "message_id": str(turn.message_id),
                        "text_chunks": turn.text_chunk_index,
                        "thinking_chunks": turn.thinking_chunk_index,
                    },
                )
            if turn.usage:
                await record_usage(
                    db,
                    scope_id=run_id,
                    message_id=turn.message_id,
                    model=turn.model or model,
                    usage=_anthropic_usage_shim(turn.usage),
                )
            await db.commit()
        await redis.delete(
            turn.s_key, turn.d_key, inflight_meta_key(run_id, turn.message_id)
        )
        return turn.blocks

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "the Anthropic API has no embeddings; the router uses its embedding provider"
        )

    # ------------------------------------------------------------- internals
    async def _stream_round(
        self,
        request: dict[str, Any],
        turn: _Turn,
        *,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
    ) -> Any:
        a_key = abort_key(run_id)
        try:
            async with self._client.beta.messages.stream(**request) as stream:
                async for event in stream:
                    if await redis.exists(a_key):
                        await self._save_partial_blocks(
                            turn.text_parts, turn.thinking_parts, run_id, node_id,
                            redis, turn.message_id, build_io,
                        )
                        raise AbortSignalError(run_id, node_id)
                    await self._on_event(event, turn, run_id, node_id, redis, build_io)
                return await stream.get_final_message()
        except AbortSignalError:
            raise
        except anthropic.RateLimitError as exc:
            await self._save_partial_blocks(
                turn.text_parts, turn.thinking_parts, run_id, node_id, redis,
                turn.message_id, build_io,
            )
            raise LLMRateLimitError(str(exc)) from exc
        except Exception:
            await self._save_partial_blocks(
                turn.text_parts, turn.thinking_parts, run_id, node_id, redis,
                turn.message_id, build_io,
            )
            raise

    async def _on_event(
        self,
        event: Any,
        turn: _Turn,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
    ) -> None:
        etype = getattr(event, "type", None)
        if etype == "content_block_start":
            await self._ensure_started(turn, run_id, node_id, redis, build_io)
            block = event.content_block
            btype = getattr(block, "type", None)
            if btype == "server_tool_use" and block.name == "web_search":
                await self._publish_builtin(
                    turn, run_id, node_id, redis, {"tool": "web_search", "status": "searching"}
                )
            elif btype == "web_search_tool_result":
                _name, query = turn.server_calls.get(block.tool_use_id, ("web_search", None))
                await self._publish_builtin(
                    turn, run_id, node_id, redis, {"tool": "web_search", "status": "completed"}
                )
                await self._publish_builtin(
                    turn, run_id, node_id, redis,
                    {
                        "tool": "web_search",
                        "status": "result",
                        "query": query,
                        "queries": [query] if query else [],
                        "results": _search_hits(block),
                    },
                )
            elif btype == "web_fetch_tool_result":
                _name, url = turn.server_calls.get(block.tool_use_id, ("web_fetch", None))
                result = getattr(block, "content", None)
                fetched = {"tool": "web_fetch", "url": getattr(result, "url", None) or url}
                await self._publish_builtin(
                    turn, run_id, node_id, redis, {**fetched, "status": "completed"}
                )
                await self._publish_builtin(
                    turn, run_id, node_id, redis, {**fetched, "status": "result"}
                )
        elif etype == "content_block_stop":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) != "server_tool_use":
                return
            args = block.input or {}
            if block.name == "web_search":
                turn.server_calls[block.id] = ("web_search", args.get("query"))
            elif block.name == "web_fetch":
                turn.server_calls[block.id] = ("web_fetch", args.get("url"))
                await self._publish_builtin(
                    turn, run_id, node_id, redis,
                    {"tool": "web_fetch", "status": "fetching", "url": args.get("url")},
                )
        elif etype == "content_block_delta":
            delta = event.delta
            dtype = getattr(delta, "type", None)
            if dtype == "text_delta" and delta.text:
                await self._publish_delta(turn, run_id, node_id, redis, "text", delta.text)
            elif dtype == "thinking_delta" and delta.thinking:
                await self._publish_delta(
                    turn, run_id, node_id, redis, "thinking", delta.thinking
                )

    async def _ensure_started(
        self,
        turn: _Turn,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
    ) -> None:
        if turn.started:
            return
        started_seq = await self._emit(
            build_io,
            run_id,
            RunEventType.agent_message_started,
            node_id,
            {"message_id": str(turn.message_id), "role": AgentMessageRole.assistant.value},
        )
        turn.started = True
        turn.delta_seq_base = started_seq or 0
        await self._write_inflight_meta(
            redis, run_id, node_id, turn.message_id, turn.delta_seq_base
        )

    async def _publish_delta(
        self,
        turn: _Turn,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        kind: str,
        text: str,
    ) -> None:
        if kind == "text":
            turn.text_parts.append(text)
            index = turn.text_chunk_index
            turn.text_chunk_index += 1
            event_type = RunEventType.agent_text_delta
        else:
            turn.thinking_parts.append(text)
            index = turn.thinking_chunk_index
            turn.thinking_chunk_index += 1
            event_type = RunEventType.agent_thinking_delta
        await self._push_stream(redis, turn.s_key, kind, text)
        seq = await self._next_delta_sequence(redis, turn.d_key, turn.delta_seq_base)
        await publish_run_event(
            redis,
            run_id,
            event_type,
            node_id,
            {"message_id": str(turn.message_id), "chunk_index": index, "text": text},
            seq,
            event_id=delta_event_id(turn.message_id, kind, index),
        )

    async def _publish_builtin(
        self,
        turn: _Turn,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        payload: dict[str, Any],
    ) -> None:
        """Live built-in tool activity — the same ``builtin_tool_call`` event
        every provider publishes (web_search: searching → completed → result
        with the hits; web_fetch: fetching → completed → result)."""
        seq = await self._next_delta_sequence(redis, turn.d_key, turn.delta_seq_base)
        await publish_run_event(
            redis,
            run_id,
            RunEventType.builtin_tool_call,
            node_id,
            {"message_id": str(turn.message_id), **payload},
            seq,
        )


__all__ = ["AnthropicProvider"]
