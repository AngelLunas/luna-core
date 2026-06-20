"""OpenAI-compatible streaming provider.

One implementation that drives any OpenAI-compatible chat-completions API
(OpenAI, Kimi/Moonshot, vLLM, LM Studio, Ollama's `/v1/chat/completions`).

Streaming flow:
  1. Open the SDK stream.
  2. For every chunk: check Redis `abort:{run_id}`. If set, persist whatever
     content accumulated as `AgentMessage(is_partial=True)` and raise
     AbortSignalError.
  3. Append the chunk's delta to Redis `stream:{run_id}:{node_id}` (LIST RPUSH
     for low-overhead resume) AND publish it on the run's pub/sub channel so
     WebSocket clients see tokens live.
  4. When the stream finishes, parse all accumulated chunks into canonical
     content blocks (`thinking`, `text`, `tool_use`), persist a single
     `AgentMessage(is_partial=False)` with the full content + extracted
     thinking string, and clean up the stream key.
  5. On any other exception during streaming, save whatever was accumulated as
     partial and re-raise.

Embeddings hit a separately-configured OpenAI-compatible `/v1/embeddings`
endpoint (defaults to a local text-embeddings-inference container serving
BAAI/bge-m3).
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, RateLimitError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal
from luna_core.engine.emitter import EventEmitter, publish_run_event
from luna_core.llm.base import (
    AbortSignalError,
    LLMRateLimitError,
    ToolDefinition,
    abort_key,
    delta_event_id,
    inflight_meta_key,
    stream_key,
)
from luna_core.models.event import AgentMessageRole, RunEventType
from luna_core.services.usage import record_usage

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory

logger = logging.getLogger(__name__)

# Resolves a media_id to a URL the provider can put on an ``image_url`` content
# part — in practice a ``data:`` base64 URL (works for both local-dev storage
# and object storage without exposing a public/signed URL). Host-injected
# (the host owns its media table + storage); ``None`` means "could not resolve",
# in which case the image falls back to a text note.
ImageResolver = Callable[[str], Awaitable[str | None]]


def _tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _canonical_to_openai_messages(
    messages: list[dict[str, Any]],
    system: str,
    image_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Translate canonical conversation history into OpenAI chat-completions
    `messages` array. Each canonical message is either:
      {"role": "user", "content": [<blocks>]}
      {"role": "assistant", "content": [<blocks>]}
      {"role": "system", "content": "..."} (rare — system is normally a param)

    ``image_urls`` maps ``media_id`` → a renderable URL (typically a ``data:``
    base64 URL). When supplied and an attached image resolves, that user turn is
    emitted as an OpenAI multimodal ``content`` parts list (text + ``image_url``)
    so a vision model sees the pixels; unresolved images (and the no-map case)
    fall back to the ``[image attached: media_id=…]`` text note a text model can
    still pass to a tool.
    """
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])
        if role == "user":
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            text_blocks = [b for b in content if b.get("type") == "text"]
            image_blocks = [b for b in content if b.get("type") == "image"]
            if tool_results:
                for block in tool_results:
                    payload = block.get("content")
                    if not isinstance(payload, str):
                        payload = json.dumps(payload)
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": payload,
                        }
                    )
            if text_blocks or image_blocks:
                resolved = [
                    (b, (image_urls or {}).get(str(b.get("media_id"))))
                    for b in image_blocks
                ]
                rendered = [(b, url) for b, url in resolved if url]
                joined_text = "\n".join(
                    b.get("text", "") for b in text_blocks if b.get("text")
                )
                if rendered:
                    # Vision path: emit a multimodal content parts list. Any
                    # image we could NOT resolve still rides as a text note.
                    note = "\n".join(
                        f"[image attached: media_id={b.get('media_id')}]"
                        for b, url in resolved
                        if not url
                    )
                    full_text = "\n".join(t for t in (joined_text, note) if t)
                    parts: list[dict[str, Any]] = []
                    if full_text:
                        parts.append({"type": "text", "text": full_text})
                    for _b, url in rendered:
                        parts.append(
                            {"type": "image_url", "image_url": {"url": url}}
                        )
                    result.append({"role": "user", "content": parts})
                else:
                    # Text path: render each attached media as a text note so a
                    # text model knows a media is present and can pass its id to
                    # a tool.
                    notes = [
                        f"[image attached: media_id={b.get('media_id')}]"
                        for b in image_blocks
                    ]
                    text = "\n".join(
                        p for p in [joined_text, *notes] if p
                    )
                    if text:
                        result.append({"role": "user", "content": text})
        elif role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                # `thinking` blocks are dropped from the wire payload.
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
        elif role == "system":
            text = ""
            if isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            if text:
                result.append({"role": "system", "content": text})
    return result


# --- Responses API (OpenAI) — used when an agent enables built-in tools ------
# The Responses API runs built-in tools (web_search, …) server-side within one
# request, while still letting the model call our own function tools. Its wire
# shape differs from chat-completions: `instructions` (system), `input` (a list of
# message + function_call/function_call_output items), flattened function tools,
# and an `output` list of typed items. We translate to/from the same canonical
# blocks the rest of the engine uses, so AgentRunner is unchanged.


def _tools_to_responses(
    tools: list[ToolDefinition], builtin_tools: list[str] | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"type": name} for name in (builtin_tools or [])]
    for tool in tools:
        out.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            }
        )
    return out


def _canonical_to_responses_input(
    messages: list[dict[str, Any]],
    image_urls: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Canonical history → Responses ``input`` items. tool_use/tool_result become
    top-level ``function_call`` / ``function_call_output`` items (the API tolerates
    a historical call for a tool not in the current set, so no special-casing)."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", []) or []
        if role == "user":
            for b in content:
                if b.get("type") == "tool_result":
                    payload = b.get("content")
                    if not isinstance(payload, str):
                        payload = json.dumps(payload, default=str)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": b.get("tool_use_id"),
                            "output": payload,
                        }
                    )
            parts: list[dict[str, Any]] = []
            for b in content:
                if b.get("type") == "text" and b.get("text"):
                    parts.append({"type": "input_text", "text": b["text"]})
                elif b.get("type") == "image":
                    url = (image_urls or {}).get(str(b.get("media_id")))
                    if url:
                        parts.append({"type": "input_image", "image_url": url})
                    else:
                        parts.append(
                            {
                                "type": "input_text",
                                "text": f"[image attached: media_id={b.get('media_id')}]",
                            }
                        )
            if parts:
                items.append({"role": "user", "content": parts})
        elif role == "assistant":
            text_parts = [
                b.get("text", "")
                for b in content
                if b.get("type") == "text" and b.get("text")
            ]
            if text_parts:
                items.append({"role": "assistant", "content": "\n".join(text_parts)})
            for b in content:
                if b.get("type") == "tool_use":
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": b.get("id"),
                            "name": b.get("name"),
                            "arguments": json.dumps(b.get("input", {}), default=str),
                        }
                    )
        elif role == "system":
            text = content if isinstance(content, str) else "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
            if text:
                items.append({"role": "user", "content": text})
    return items


def _responses_output_to_blocks(
    response: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Responses ``output`` items → canonical blocks (+ extracted thinking text).
    ``web_search_call`` and other server-side tool items are dropped — their
    results are already folded into the model's message (with url citations)."""
    blocks: list[dict[str, Any]] = []
    thinking_text: str | None = None
    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            summary = getattr(item, "summary", None) or []
            text = " ".join(getattr(s, "text", "") for s in summary).strip()
            if text:
                thinking_text = text
                blocks.append({"type": "thinking", "thinking": text})
        elif itype == "message":
            parts = getattr(item, "content", None) or []
            text = "".join(
                getattr(p, "text", "")
                for p in parts
                if getattr(p, "type", "") == "output_text"
            )
            urls: list[str] = []
            for p in parts:
                for ann in getattr(p, "annotations", None) or []:
                    if getattr(ann, "type", "") == "url_citation":
                        url = getattr(ann, "url", None)
                        if url and url not in urls:
                            urls.append(url)
            if text:
                if urls:
                    text = f"{text}\n\nSources: " + ", ".join(urls)
                blocks.append({"type": "text", "text": text})
        elif itype == "function_call":
            raw = getattr(item, "arguments", "") or "{}"
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"_raw": raw}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                    "name": getattr(item, "name", ""),
                    "input": parsed,
                }
            )
    return blocks, thinking_text


class _ResponsesUsage:
    """Adapt Responses usage (input/output_tokens) to the chat-shaped fields
    ``record_usage`` reads (prompt/completion_tokens + cached details)."""

    def __init__(self, usage: Any) -> None:
        self.prompt_tokens = getattr(usage, "input_tokens", 0)
        self.completion_tokens = getattr(usage, "output_tokens", 0)
        self.total_tokens = getattr(usage, "total_tokens", 0)
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        self.prompt_tokens_details = type(
            "_D", (), {"cached_tokens": cached}
        )()


class _StreamAccumulator:
    """Aggregates streaming chunks into canonical content blocks."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.thinking_parts: list[str] = []
        # tool_calls keyed by index → {"id", "name", "arguments_text"}
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add_text(self, text: str) -> None:
        if text:
            self.text_parts.append(text)

    def add_thinking(self, text: str) -> None:
        if text:
            self.thinking_parts.append(text)

    def add_tool_call_delta(self, delta_tool_calls: list[Any]) -> None:
        for tc in delta_tool_calls:
            index = getattr(tc, "index", 0)
            slot = self.tool_calls.setdefault(
                index, {"id": None, "name": "", "arguments_text": ""}
            )
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments_text"] += fn.arguments

    def to_canonical(self) -> tuple[list[dict[str, Any]], str | None]:
        blocks: list[dict[str, Any]] = []
        thinking_text = "".join(self.thinking_parts).strip() or None
        if thinking_text:
            blocks.append({"type": "thinking", "thinking": thinking_text})
        text = "".join(self.text_parts)
        if text:
            blocks.append({"type": "text", "text": text})
        for index in sorted(self.tool_calls):
            slot = self.tool_calls[index]
            args_raw = slot.get("arguments_text") or ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": slot.get("id") or f"tc_{index}",
                    "name": slot.get("name") or "",
                    "input": args,
                }
            )
        return blocks, thinking_text


def _extract_thinking_from_choice_delta(delta: Any) -> str | None:
    """Some OpenAI-compatible providers (notably Kimi/Moonshot) expose a
    `reasoning` or `reasoning_content` field on the delta. Return its text
    when present, otherwise None."""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if value:
            return value if isinstance(value, str) else str(value)
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = extra.get(key)
            if value:
                return value if isinstance(value, str) else str(value)
    return None


class GenericProvider:
    """OpenAI-compatible chat + embeddings provider.

    Constructed per LLMProvider row by the router (chat side) and once at
    process start by the host (embeddings side). Holds two AsyncOpenAI
    clients — one for chat (driven by the explicit api_key / base_url
    passed in) and one for embeddings (defaults to EMBEDDING_* env values).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        session_factory: Callable[
            [], AbstractAsyncContextManager[AsyncSession]
        ]
        | None = None,
    ):
        # `session_factory` opens a short-lived session for persisting agent
        # messages mid-stream. Defaults to luna-core's AsyncSessionLocal so
        # hosts that share that engine don't have to wire anything; hosts with
        # custom engines pass their own factory.
        self._session_factory = session_factory or AsyncSessionLocal
        self._chat_client = AsyncOpenAI(
            api_key=api_key or "missing",
            base_url=base_url,
        )
        self._embed_client = AsyncOpenAI(
            api_key=(
                embedding_api_key
                or settings.embedding_api_key
                or "missing"
            ),
            base_url=embedding_base_url or settings.embedding_base_url,
        )
        self._default_model = default_model
        self._embedding_model = embedding_model or settings.embedding_model

    # ------------------------------------------------------------------ chat
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        temperature: float,
        model: str | None,
        output_schema: dict[str, Any] | None,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        make_io: IOFactory | None = None,
        image_resolver: ImageResolver | None = None,
        builtin_tools: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Agents with built-in tools (e.g. web_search) run on the Responses API,
        # which executes those tools server-side while still calling our function
        # tools. Same canonical blocks out, so AgentRunner is unchanged.
        if builtin_tools:
            return await self._complete_via_responses(
                messages=messages,
                system=system,
                tools=tools,
                model=model,
                run_id=run_id,
                node_id=node_id,
                redis=redis,
                make_io=make_io,
                image_resolver=image_resolver,
                builtin_tools=builtin_tools,
            )
        accumulator = _StreamAccumulator()
        a_key = abort_key(run_id)

        # Where this assistant turn and its lifecycle events get persisted.
        # The caller injects a scope-bound factory (flow EventEmitter or chat
        # emitter); we call it with the short-lived sessions we open below, so
        # this loop never needs to know which context it's running in. Absent
        # an injection we default to the flow emitter for ``run_id`` — keeping
        # behavior identical for any direct caller.
        build_io = make_io or (
            lambda session: EventEmitter(session, redis, run_id)
        )

        # One assistant turn = one message_id. Generated here so that every
        # *_delta we publish AND the final AgentMessage row share the same
        # UUID — the frontend keys the rendered bubble off this id and can
        # match REST and WS frames byte-for-byte.
        message_id = uuid.uuid4()
        # Per-message stream cache key (NOT per-node). Parallel iterations
        # of the same ai_agent node each generate their own message_id
        # and therefore their own cache, so a sibling's _save_partial
        # never wipes our history out from under us mid-stream.
        s_key = stream_key(run_id, message_id)
        text_chunk_index = 0
        thinking_chunk_index = 0
        message_started = False
        # Sequence base for the live-only delta events. Captured from the
        # persisted agent_message_started so deltas sort right after it on
        # the client. Kept in a local because it's only meaningful within
        # this single stream — if the worker dies mid-stream the whole
        # turn is restarted with a fresh message_id, so there's nothing
        # to recover. The per-chunk increment lives in Redis (INCR on
        # delta_seq_key) so it survives any in-stream hiccups and stays
        # atomic across whatever process is driving the stream.
        delta_seq_base = 0
        delta_seq_redis_key = f"delta_seq:{run_id}:{message_id}"
        # Final-chunk token usage (when the provider honors stream_options).
        final_usage: Any = None

        # Pre-resolve attached-image media_ids to renderable URLs before the
        # (sync) message builder runs. Only when a resolver is injected — the
        # text-only path (no resolver) is byte-identical to before.
        image_urls = await self._resolve_image_urls(messages, image_resolver)

        request: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": _canonical_to_openai_messages(
                messages, system, image_urls or None
            ),
            "temperature": temperature,
            "stream": True,
            # Ask the provider for a final usage-only chunk (real token counts).
            # It arrives with empty ``choices`` and is captured below; the
            # streamed/persisted output is unchanged (that chunk carries no
            # content), so the flow path stays behavior-identical.
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = _tools_to_openai(tools)
            request["tool_choice"] = "auto"
        if output_schema:
            # Use JSON schema response_format where supported; fall back to
            # `json_object` mode (the LLM is still constrained by the prompt).
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": output_schema,
                    "strict": True,
                },
            }

        try:
            stream = await self._chat_client.chat.completions.create(**request)
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc

        try:
            async for chunk in stream:
                # Usage rides the final chunk (empty choices); capture before
                # the choices guard skips it.
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    final_usage = chunk_usage
                if not chunk.choices:
                    continue
                # abort check FIRST — short-circuit before doing any work
                if await redis.exists(a_key):
                    await self._save_partial(
                        accumulator, run_id, node_id, redis, message_id, build_io
                    )
                    raise AbortSignalError(run_id, node_id)

                choice = chunk.choices[0]
                delta = choice.delta
                if delta is None:
                    continue

                content_text = getattr(delta, "content", None)
                thinking_text = _extract_thinking_from_choice_delta(delta)
                tool_call_deltas = getattr(delta, "tool_calls", None)

                if content_text or thinking_text or tool_call_deltas:
                    if not message_started:
                        started_seq = await self._emit(
                            build_io,
                            run_id,
                            RunEventType.agent_message_started,
                            node_id,
                            {
                                "message_id": str(message_id),
                                "role": AgentMessageRole.assistant.value,
                            },
                        )
                        message_started = True
                        delta_seq_base = started_seq or 0
                        # Sidecar metadata for the WebSocket snapshot path:
                        # lets a reconnecting client reconstruct synthetic
                        # delta frames from the chunks RPUSH'd to s_key
                        # without having to query the DB for the started
                        # event's sequence.
                        await self._write_inflight_meta(
                            redis, run_id, node_id, message_id, delta_seq_base
                        )

                if content_text:
                    accumulator.add_text(content_text)
                    await self._push_stream(redis, s_key, "text", content_text)
                    chunk_seq = await self._next_delta_sequence(
                        redis, delta_seq_redis_key, delta_seq_base
                    )
                    await publish_run_event(
                        redis,
                        run_id,
                        RunEventType.agent_text_delta,
                        node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": text_chunk_index,
                            "text": content_text,
                        },
                        chunk_seq,
                        event_id=delta_event_id(
                            message_id, "text", text_chunk_index
                        ),
                    )
                    text_chunk_index += 1

                if thinking_text:
                    accumulator.add_thinking(thinking_text)
                    await self._push_stream(
                        redis, s_key, "thinking", thinking_text
                    )
                    chunk_seq = await self._next_delta_sequence(
                        redis, delta_seq_redis_key, delta_seq_base
                    )
                    await publish_run_event(
                        redis,
                        run_id,
                        RunEventType.agent_thinking_delta,
                        node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": thinking_chunk_index,
                            "text": thinking_text,
                        },
                        chunk_seq,
                        event_id=delta_event_id(
                            message_id, "thinking", thinking_chunk_index
                        ),
                    )
                    thinking_chunk_index += 1

                if tool_call_deltas:
                    accumulator.add_tool_call_delta(tool_call_deltas)
        except AbortSignalError:
            raise
        except RateLimitError as exc:
            await self._save_partial(
                accumulator, run_id, node_id, redis, message_id, build_io
            )
            raise LLMRateLimitError(str(exc)) from exc
        except Exception:
            await self._save_partial(
                accumulator, run_id, node_id, redis, message_id, build_io
            )
            raise

        blocks, thinking = accumulator.to_canonical()
        async with self._session_factory() as db:
            emitter = build_io(db)
            await emitter.save_message(
                node_id=node_id,
                role=AgentMessageRole.assistant,
                content=blocks,
                is_partial=False,
                thinking=thinking,
                message_id=message_id,
            )
            if message_started:
                await emitter.emit(
                    RunEventType.agent_message_completed,
                    node_id=node_id,
                    payload={
                        "message_id": str(message_id),
                        "text_chunks": text_chunk_index,
                        "thinking_chunks": thinking_chunk_index,
                    },
                )
            # Record the turn's real token cost on the same session/transaction
            # as its transcript. Only when the provider returned usage — keeps
            # the no-usage path (and its tests) untouched.
            if final_usage is not None:
                await record_usage(
                    db,
                    scope_id=run_id,
                    message_id=message_id,
                    model=request["model"],
                    usage=final_usage,
                )
            await db.commit()
        await redis.delete(
            s_key, delta_seq_redis_key, inflight_meta_key(run_id, message_id)
        )
        return blocks

    # -------------------------------------------------------------- responses
    async def _complete_via_responses(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        model: str | None,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        make_io: IOFactory | None,
        image_resolver: ImageResolver | None,
        builtin_tools: list[str],
    ) -> list[dict[str, Any]]:
        """One streaming turn via the OpenAI Responses API. Built-in tools (e.g.
        web_search) run server-side; our function tools come back as ``tool_use``
        blocks the AgentRunner executes and re-sends. Streams text/thinking deltas
        on the same channel + event ids as the chat-completions path, then persists
        the final turn (built from the authoritative ``response.completed`` output,
        so url citations and function calls are exact) + usage."""
        a_key = abort_key(run_id)
        build_io = make_io or (
            lambda session: EventEmitter(session, redis, run_id)
        )
        message_id = uuid.uuid4()
        s_key = stream_key(run_id, message_id)
        delta_seq_redis_key = f"delta_seq:{run_id}:{message_id}"
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        text_chunk_index = 0
        thinking_chunk_index = 0
        final_response: Any = None

        image_urls = await self._resolve_image_urls(messages, image_resolver)
        request: dict[str, Any] = {
            "model": model or self._default_model,
            "input": _canonical_to_responses_input(messages, image_urls or None),
            "tools": _tools_to_responses(tools, builtin_tools),
            "stream": True,
        }
        if system:
            request["instructions"] = system
        if request["tools"]:
            request["tool_choice"] = "auto"

        try:
            stream = await self._chat_client.responses.create(**request)
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc

        # Emit the lifecycle start up front so deltas sort right after it; the
        # turn always yields text or a tool call.
        started_seq = await self._emit(
            build_io,
            run_id,
            RunEventType.agent_message_started,
            node_id,
            {"message_id": str(message_id), "role": AgentMessageRole.assistant.value},
        )
        delta_seq_base = started_seq or 0
        await self._write_inflight_meta(redis, run_id, node_id, message_id, delta_seq_base)

        try:
            async for ev in stream:
                if await redis.exists(a_key):
                    await self._save_partial_blocks(
                        text_parts, thinking_parts, run_id, node_id, redis,
                        message_id, build_io,
                    )
                    raise AbortSignalError(run_id, node_id)
                etype = getattr(ev, "type", "")
                if etype == "response.completed":
                    final_response = ev.response
                elif etype == "response.output_text.delta":
                    delta = getattr(ev, "delta", "") or ""
                    if not delta:
                        continue
                    text_parts.append(delta)
                    await self._push_stream(redis, s_key, "text", delta)
                    seq = await self._next_delta_sequence(
                        redis, delta_seq_redis_key, delta_seq_base
                    )
                    await publish_run_event(
                        redis, run_id, RunEventType.agent_text_delta, node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": text_chunk_index,
                            "text": delta,
                        },
                        seq,
                        event_id=delta_event_id(message_id, "text", text_chunk_index),
                    )
                    text_chunk_index += 1
                elif etype == "response.reasoning_summary_text.delta":
                    delta = getattr(ev, "delta", "") or ""
                    if not delta:
                        continue
                    thinking_parts.append(delta)
                    await self._push_stream(redis, s_key, "thinking", delta)
                    seq = await self._next_delta_sequence(
                        redis, delta_seq_redis_key, delta_seq_base
                    )
                    await publish_run_event(
                        redis, run_id, RunEventType.agent_thinking_delta, node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": thinking_chunk_index,
                            "text": delta,
                        },
                        seq,
                        event_id=delta_event_id(
                            message_id, "thinking", thinking_chunk_index
                        ),
                    )
                    thinking_chunk_index += 1
                elif etype.startswith("response.web_search_call."):
                    # Live "searching the web…" signal (in_progress/searching/
                    # completed). Pub/sub-only, sequenced like the deltas.
                    seq = await self._next_delta_sequence(
                        redis, delta_seq_redis_key, delta_seq_base
                    )
                    await publish_run_event(
                        redis, run_id, RunEventType.builtin_tool_call, node_id,
                        {
                            "message_id": str(message_id),
                            "tool": "web_search",
                            "status": etype.rsplit(".", 1)[-1],
                        },
                        seq,
                    )
                elif etype == "response.output_item.done":
                    item = getattr(ev, "item", None)
                    if getattr(item, "type", "") == "web_search_call":
                        action = getattr(item, "action", None)
                        seq = await self._next_delta_sequence(
                            redis, delta_seq_redis_key, delta_seq_base
                        )
                        await publish_run_event(
                            redis, run_id, RunEventType.builtin_tool_call, node_id,
                            {
                                "message_id": str(message_id),
                                "tool": "web_search",
                                "status": "result",
                                # The main query + every query the model actually ran.
                                "query": getattr(action, "query", None),
                                "queries": list(getattr(action, "queries", None) or []),
                            },
                            seq,
                        )
        except AbortSignalError:
            raise
        except RateLimitError as exc:
            await self._save_partial_blocks(
                text_parts, thinking_parts, run_id, node_id, redis, message_id, build_io
            )
            raise LLMRateLimitError(str(exc)) from exc
        except Exception:
            await self._save_partial_blocks(
                text_parts, thinking_parts, run_id, node_id, redis, message_id, build_io
            )
            raise

        # Final blocks from the authoritative completed response (exact citations
        # + function calls); fall back to streamed text if it's somehow absent.
        if final_response is not None:
            blocks, thinking = _responses_output_to_blocks(final_response)
        else:
            blocks = []
            joined_thinking = "".join(thinking_parts).strip() or None
            if joined_thinking:
                blocks.append({"type": "thinking", "thinking": joined_thinking})
            joined_text = "".join(text_parts)
            if joined_text:
                blocks.append({"type": "text", "text": joined_text})
            thinking = joined_thinking

        async with self._session_factory() as db:
            emitter = build_io(db)
            await emitter.save_message(
                node_id=node_id,
                role=AgentMessageRole.assistant,
                content=blocks,
                is_partial=False,
                thinking=thinking,
                message_id=message_id,
            )
            await emitter.emit(
                RunEventType.agent_message_completed,
                node_id=node_id,
                payload={
                    "message_id": str(message_id),
                    "text_chunks": text_chunk_index,
                    "thinking_chunks": thinking_chunk_index,
                },
            )
            usage = getattr(final_response, "usage", None)
            if usage is not None:
                await record_usage(
                    db,
                    scope_id=run_id,
                    message_id=message_id,
                    model=request["model"],
                    usage=_ResponsesUsage(usage),
                )
            await db.commit()
        await redis.delete(
            s_key, delta_seq_redis_key, inflight_meta_key(run_id, message_id)
        )
        return blocks

    async def _save_partial_blocks(
        self,
        text_parts: list[str],
        thinking_parts: list[str],
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        message_id: uuid.UUID,
        build_io: IOFactory,
    ) -> None:
        """Persist whatever streamed before an abort/error as a partial turn
        (mirrors the chat-completions ``_save_partial``)."""
        s_key = stream_key(run_id, message_id)
        d_key = f"delta_seq:{run_id}:{message_id}"
        m_key = inflight_meta_key(run_id, message_id)
        thinking = "".join(thinking_parts).strip() or None
        blocks: list[dict[str, Any]] = []
        if thinking:
            blocks.append({"type": "thinking", "thinking": thinking})
        text = "".join(text_parts)
        if text:
            blocks.append({"type": "text", "text": text})
        if not blocks:
            await redis.delete(s_key, d_key, m_key)
            return
        try:
            async with self._session_factory() as db:
                emitter = build_io(db)
                await emitter.save_message(
                    node_id=node_id,
                    role=AgentMessageRole.assistant,
                    content=blocks,
                    is_partial=True,
                    thinking=thinking,
                    message_id=message_id,
                )
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist partial Responses turn for run %s node %s",
                run_id,
                node_id,
            )
        await redis.delete(s_key, d_key, m_key)

    # ------------------------------------------------------------------ embed
    async def embed(self, text: str) -> list[float]:
        response = await self._embed_client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return list(response.data[0].embedding)

    # ----------------------------------------------------------------- helpers
    async def _resolve_image_urls(
        self,
        messages: list[dict[str, Any]],
        image_resolver: ImageResolver | None,
    ) -> dict[str, str]:
        """Resolve every distinct attached-image ``media_id`` to a renderable
        URL via the injected resolver. No resolver → empty map (text-note path).
        Each id is resolved once even if it recurs across turns."""
        if image_resolver is None:
            return {}
        urls: dict[str, str] = {}
        for msg in messages:
            for block in msg.get("content", []) or []:
                if not isinstance(block, dict) or block.get("type") != "image":
                    continue
                media_id = block.get("media_id")
                if media_id is None:
                    continue
                key = str(media_id)
                if key in urls:
                    continue
                url = await image_resolver(key)
                if url:
                    urls[key] = url
        return urls

    async def _push_stream(
        self,
        redis: Redis,
        s_key: str,
        kind: str,
        text: str,
    ) -> None:
        # Append to a Redis LIST cache used both for crash-mid-stream recovery
        # and for the WebSocket snapshot path: when a client reconnects while
        # a turn is still streaming, the snapshot reader rehydrates a
        # synthetic delta from these chunks. The canonical broadcast for live
        # clients is still the pub/sub event emitted alongside each push.
        chunk = json.dumps({"kind": kind, "text": text})
        await redis.rpush(s_key, chunk)
        await redis.expire(s_key, settings.run_stream_key_ttl_seconds)

    async def _write_inflight_meta(
        self,
        redis: Redis,
        run_id: uuid.UUID,
        node_id: str,
        message_id: uuid.UUID,
        started_seq: int,
    ) -> None:
        # Capture the iteration tag of the *task that owns this turn* so
        # the WebSocket snapshot path can route mid-flight synthesized
        # delta frames to the right iteration block on the dashboard.
        # ``get_current_iteration_id`` returns None outside an iteration
        # scope, and we omit the key in that case so the wire shape for
        # non-iterative runs stays unchanged.
        from luna_core.engine.iteration_context import get_current_iteration_id

        # The key is per-message_id (not per-node) so parallel iterations
        # of the same ai_agent node each write their own meta — see
        # docstring on ``stream_key``/``inflight_meta_key``. We carry
        # ``node_id`` in the payload because the snapshot scanner no
        # longer parses it out of the key.
        meta: dict[str, Any] = {
            "message_id": str(message_id),
            "node_id": node_id,
            "started_seq": started_seq,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        iteration_id = get_current_iteration_id()
        if iteration_id is not None:
            meta["iteration_id"] = str(iteration_id)
        await redis.set(
            inflight_meta_key(run_id, message_id),
            json.dumps(meta),
            ex=settings.run_stream_key_ttl_seconds,
        )

    async def _emit(
        self,
        build_io: IOFactory,
        run_id: uuid.UUID,
        event_type: RunEventType,
        node_id: str,
        payload: dict[str, Any],
    ) -> int | None:
        """Persist + broadcast an event. Returns the assigned sequence so
        the caller can base downstream synthetic-sequence math on it
        (e.g. transient delta events that ride above the same baseline).

        ``run_id`` is retained only for the failure log; persistence goes
        through the injected ``build_io`` factory."""
        # Opens a short-lived session per event so the streaming loop never
        # holds a transaction open while awaiting the next LLM chunk.
        async with self._session_factory() as db:
            emitter = build_io(db)
            try:
                event = await emitter.emit(
                    event_type, node_id=node_id, payload=payload
                )
                return event.sequence
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to persist run event %s for run %s node %s",
                    event_type.value,
                    run_id,
                    node_id,
                )
                return None

    async def _next_delta_sequence(
        self, redis: Redis, key: str, base: int
    ) -> int:
        """Atomically allocate the next per-stream delta sequence.

        Uses INCR on a Redis key keyed by (run_id, message_id) so the
        counter survives any in-stream coordination hiccups and stays
        consistent if the same stream were ever driven by multiple
        coroutines. The returned sequence is ``base + INCR_value`` so
        deltas always sort right after the persisted agent_message_started
        event (whose sequence is ``base``). The key inherits the same TTL
        as the stream cache and is deleted at end-of-stream.
        """
        offset = await redis.incr(key)
        if offset == 1:
            await redis.expire(key, settings.run_stream_key_ttl_seconds)
        return base + int(offset)

    async def _save_partial(
        self,
        accumulator: _StreamAccumulator,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        message_id: uuid.UUID,
        build_io: IOFactory,
    ) -> None:
        # All three keys are per-message: deleting them on partial save
        # only affects this turn's cache, never a sibling iteration's
        # in-flight history.
        s_key = stream_key(run_id, message_id)
        d_key = f"delta_seq:{run_id}:{message_id}"
        m_key = inflight_meta_key(run_id, message_id)
        blocks, thinking = accumulator.to_canonical()
        if not blocks:
            await redis.delete(s_key, d_key, m_key)
            return
        try:
            async with self._session_factory() as db:
                emitter = build_io(db)
                await emitter.save_message(
                    node_id=node_id,
                    role=AgentMessageRole.assistant,
                    content=blocks,
                    is_partial=True,
                    thinking=thinking,
                    message_id=message_id,
                )
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "failed to persist partial AgentMessage for run %s node %s",
                run_id,
                node_id,
            )
        await redis.delete(s_key, d_key, m_key)


__all__ = ["GenericProvider"]
