"""Claude Code CLI provider — chat turns through a local `claude -p` binary.

Runs each ``complete()`` as one non-interactive CLI invocation authenticated by
the machine's Claude subscription login (no API key). Designed for personal /
test installs where the operator pays via subscription.

Propose-only tool calling: the turn's ``ToolDefinition`` list is advertised to
the CLI as native MCP tools via a catalog-only stdio server (see
``mcp_catalog``), but never allowlisted — the CLI auto-denies execution, and
``--max-turns 1`` stops the run right after the model's first turn. The
``tool_use`` blocks stream out *before* any execution or denial, so this
provider returns them as canonical blocks and the AgentRunner keeps owning
tool execution, approvals, and handoffs — exactly as with the HTTP provider.

Structured output (``output_schema``) rides the CLI's ``--json-schema`` flag;
the validated object comes back in the result event's ``structured_output``.

Stateless by design: history is rendered into the prompt each call
(``--no-session-persistence``); CLI sessions are never reused.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.llm.base import (
    AbortSignalError,
    LLMRateLimitError,
    ToolDefinition,
    abort_key,
    delta_event_id,
    inflight_meta_key,
    stream_key,
)
from luna_core.engine.emitter import publish_run_event
from luna_core.llm.providers.generic import (
    GenericProvider,
    ImageResolver,
    _canonical_to_openai_messages,
)
from luna_core.llm.providers.mcp_catalog import (
    TOOL_NAME_PREFIX,
    build_mcp_config,
)
from luna_core.models.event import AgentMessageRole, RunEventType
from luna_core.services.usage import record_usage

if TYPE_CHECKING:
    from luna_core.engine.streaming import IOFactory

logger = logging.getLogger(__name__)

# Curated model list served to the UI for kind=claude_cli providers. Aliases
# are stable — the CLI resolves each to the newest model of its family — so
# there is nothing to sync from upstream. Full model names also work (pin a
# version by typing one); the `claude_cli`-marked test probes every entry here
# so a stale list breaks in CI, not in production.
CLAUDE_CLI_MODELS: tuple[dict[str, str], ...] = (
    {"id": "haiku", "label": "Claude Haiku (latest)"},
    {"id": "sonnet", "label": "Claude Sonnet (latest)"},
    {"id": "opus", "label": "Claude Opus (latest)"},
    {"id": "fable", "label": "Claude Fable (latest)"},
)

# The CLI treats stdin as ONE user message, so prior turns are replayed as a
# tagged transcript. The protocol lives in the SYSTEM prompt (followed far more
# reliably than user-message framing): a small model given a bracketed
# transcript in the user turn happily continues writing the transcript —
# fabricating tool results and even the next user turn. XML tags plus the
# explicit "your reply is only the next message" rule, and the leak guard
# below, close that hole.
_HISTORY_PROTOCOL = """

# Conversation history protocol
The user message may contain a <conversation_history> block (earlier turns, \
tool calls you made and their real results) followed by a <latest> block: what \
you must respond to now (new user text, or results of tools you just called). \
Treat both as context only. Your reply is ONLY your next assistant message: \
natural prose for the user and/or real tool calls. Never write history tags, \
never narrate tool calls or invent tool results as text — call the tool."""

# Any of these at the start of a line means the model began "writing the
# transcript" instead of replying; everything from there on is discarded.
_LEAK_MARKERS = (
    "<conversation_history",
    "</conversation_history",
    "<latest>",
    "</latest>",
    "<user>",
    "<assistant>",
    "<tool_call",
    "<tool_result",
)


# Provider-agnostic builtin tool names (Agent.builtin_tools) → the CLI's own
# tools that implement them inside the turn. Names not listed are ignored
# (logged): an agent's builtin list is shared across providers, and each
# provider offers what it can — the HTTP provider does the same.
_BUILTIN_TOOL_MAP: dict[str, tuple[str, ...]] = {
    "web_search": ("WebSearch",),
    "web_fetch": ("WebFetch",),
}
_CLI_WEB_SEARCH_TOOL = "WebSearch"
_CLI_WEB_FETCH_TOOL = "WebFetch"


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def _user_text_and_images(content: Any) -> tuple[str, list[str]]:
    """Split an OpenAI-shaped user ``content`` (a string, or the multimodal
    parts list ``_canonical_to_openai_messages`` builds when images resolved)
    into its text and the image data URLs, in img-N order."""
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[str] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                images.append(url)
    return "\n".join(t for t in texts if t), images


def _render_entry(msg: dict[str, Any]) -> list[str]:
    role = msg.get("role")
    content = msg.get("content")
    if role == "user":
        text, _images = _user_text_and_images(content)
        return [f"<user>{_xml_escape(text)}</user>"]
    if role == "assistant":
        out: list[str] = []
        if content:
            out.append(f"<assistant>{_xml_escape(content)}</assistant>")
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            out.append(
                f'<tool_call id="{tc.get("id")}" name="{fn.get("name")}">'
                f"{_xml_escape(fn.get('arguments') or '{}')}</tool_call>"
            )
        return out
    if role == "tool":
        return [
            f'<tool_result id="{msg.get("tool_call_id")}">'
            f"{_xml_escape(str(content))}</tool_result>"
        ]
    return []


def _render_transcript(openai_msgs: list[dict[str, Any]]) -> str:
    """Render OpenAI-shaped history (from ``_canonical_to_openai_messages``,
    which owns the img-N labeling) into one prompt string for `claude -p`.

    A lone user message is sent raw. Otherwise everything up to the last
    assistant turn goes in <conversation_history> and the trailing run (new
    user text and/or fresh tool results) in <latest>. Images are not part of
    the text: ``_collect_images`` gathers them for the stream-json message."""
    if len(openai_msgs) == 1 and openai_msgs[0].get("role") == "user":
        text, _images = _user_text_and_images(openai_msgs[0].get("content"))
        return text
    last_assistant = max(
        (i for i, m in enumerate(openai_msgs) if m.get("role") == "assistant"),
        default=-1,
    )
    history = [
        line for m in openai_msgs[: last_assistant + 1] for line in _render_entry(m)
    ]
    latest = [
        line for m in openai_msgs[last_assistant + 1 :] for line in _render_entry(m)
    ]
    parts: list[str] = []
    if history:
        parts.append(
            "<conversation_history>\n" + "\n".join(history) + "\n</conversation_history>"
        )
    parts.append("<latest>\n" + "\n".join(latest) + "\n</latest>")
    return "\n\n".join(parts)


def _collect_images(openai_msgs: list[dict[str, Any]]) -> list[str]:
    """Every resolved image data URL across the history, in img-N order —
    the same order the text notes ``[image attached: img-N (shown below)]``
    use, so the model can match label to picture."""
    images: list[str] = []
    for msg in openai_msgs:
        if msg.get("role") == "user":
            images.extend(_user_text_and_images(msg.get("content"))[1])
    return images


def _data_url_to_image_block(url: str) -> dict[str, Any] | None:
    """``data:<media_type>;base64,<data>`` → an Anthropic image content block.
    Non-data URLs are skipped (the CLI has no way to fetch them for us)."""
    if not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep or ";base64" not in header:
        return None
    media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _stream_json_user_message(text: str, images: list[str]) -> str:
    """The one stdin line for ``--input-format stream-json``: a user message
    whose content is the prompt text plus any attached images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in images:
        block = _data_url_to_image_block(url)
        if block is not None:
            content.append(block)
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}}
    ) + "\n"


def _strip_leaked_transcript(text: str) -> str:
    """Cut an assistant text at the first line that starts a history tag."""
    cut = len(text)
    for marker in _LEAK_MARKERS:
        idx = text.find(marker)
        while idx != -1 and idx < cut:
            if idx == 0 or text[idx - 1] == "\n":
                cut = idx
                break
            idx = text.find(marker, idx + 1)
    return text[:cut].rstrip() if cut < len(text) else text


def _usage_shim(usage: dict[str, Any]) -> Any:
    """Adapt the CLI result event's Anthropic-shaped usage dict to the
    OpenAI-attribute shape ``record_usage`` reads."""
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_creation
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
    )


def _resolved_model(result_event: dict[str, Any], requested: str) -> str:
    """Prefer the canonical model the CLI reports (aliases resolve to a real
    model id); the busiest ``modelUsage`` entry is the main model — the CLI
    also runs small auxiliary calls on a lighter model."""
    model_usage = result_event.get("modelUsage") or {}
    best, best_tokens = None, -1
    for name, stats in model_usage.items():
        tokens = int(stats.get("inputTokens") or 0) + int(
            stats.get("outputTokens") or 0
        )
        if tokens > best_tokens:
            best, best_tokens = name, tokens
    return best or requested


class ClaudeCLIProvider(GenericProvider):
    """Chat provider backed by the local Claude Code CLI binary.

    Subclasses ``GenericProvider`` purely to reuse its emit/stream/partial-save
    plumbing (``_emit``, ``_push_stream``, ``_write_inflight_meta``,
    ``_next_delta_sequence``, ``_save_partial_blocks``) — the HTTP clients the
    parent builds are never used for chat, and ``embed()`` stays on the parent
    (the router routes embeddings to its dedicated env provider anyway).
    """

    def __init__(
        self,
        *,
        binary_path: str | None = None,
        timeout_seconds: int | None = None,
        max_concurrency: int | None = None,
        session_factory: Callable[
            [], AbstractAsyncContextManager[AsyncSession]
        ]
        | None = None,
    ):
        super().__init__(session_factory=session_factory)
        self._binary = binary_path or settings.claude_cli_binary
        self._timeout = timeout_seconds or settings.claude_cli_timeout_seconds
        self._sema = asyncio.Semaphore(
            max_concurrency or settings.claude_cli_max_concurrency
        )

    # ------------------------------------------------------------------ chat
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[ToolDefinition],
        temperature: float,  # CLI has no temperature control; accepted, unused
        model: str | None,
        output_schema: dict[str, Any] | None,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        make_io: IOFactory | None = None,
        image_resolver: ImageResolver | None = None,
        builtin_tools: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        cli_tools: list[str] = []
        for name in builtin_tools or []:
            mapped = _BUILTIN_TOOL_MAP.get(name)
            if mapped is None:
                logger.warning(
                    "claude-cli: ignoring builtin tool %r (no CLI equivalent; "
                    "supported: %s)", name, sorted(_BUILTIN_TOOL_MAP),
                )
                continue
            cli_tools.extend(t for t in mapped if t not in cli_tools)
        if not model:
            raise ValueError("claude-cli provider requires an explicit model")
        # Vision: resolved images (and video posters) become base64 blocks on
        # the stream-json user message, in the same img-N / vid-N order as their
        # text notes. No resolver → text notes only, like the HTTP provider's
        # text-model path.
        image_urls = await self._resolve_image_urls(messages, image_resolver)
        openai_msgs = _canonical_to_openai_messages(
            messages, system="", image_urls=image_urls or None
        )
        prompt = _render_transcript(openai_msgs)
        if prompt.startswith("<"):
            system = (system or "") + _HISTORY_PROTOCOL
        stdin_payload = _stream_json_user_message(
            prompt, _collect_images(openai_msgs)
        )

        from luna_core.engine.emitter import EventEmitter

        build_io = make_io or (
            lambda session: EventEmitter(session, redis, run_id)
        )
        message_id = uuid.uuid4()
        s_key = stream_key(run_id, message_id)
        d_key = f"delta_seq:{run_id}:{message_id}"

        async with self._sema:
            work_dir = tempfile.mkdtemp(prefix="claude-cli-")
            try:
                argv = self._build_argv(
                    model=model,
                    system=system,
                    tools=tools,
                    cli_tools=cli_tools,
                    output_schema=output_schema,
                    work_dir=work_dir,
                )
                return await self._run_turn(
                    argv=argv,
                    stdin_payload=stdin_payload,
                    stop_on_proposal=bool(cli_tools),
                    work_dir=work_dir,
                    output_schema=output_schema,
                    model=model,
                    run_id=run_id,
                    node_id=node_id,
                    redis=redis,
                    build_io=build_io,
                    message_id=message_id,
                    s_key=s_key,
                    d_key=d_key,
                )
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------- internals
    def _build_argv(
        self,
        *,
        model: str,
        system: str,
        tools: list[ToolDefinition],
        cli_tools: list[str],
        output_schema: dict[str, Any] | None,
        work_dir: str,
    ) -> list[str]:
        argv = [
            self._binary,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--no-session-persistence",
            "--strict-mcp-config",
        ]
        if cli_tools:
            # Builtin tools (web search) execute INSIDE the CLI and each one
            # spends a turn, so the cap is raised; a proposal of one of OUR
            # tools ends the run early instead (see _consume_stream).
            argv += [
                "--tools", ",".join(cli_tools),
                "--allowedTools", ",".join(cli_tools),
                "--max-turns", str(settings.claude_cli_builtin_max_turns),
            ]
        else:
            argv += ["--tools", "", "--max-turns", "1"]
        argv += [
            "--model",
            model,
            "--system-prompt",
            system or "",
            "--exclude-dynamic-system-prompt-sections",
        ]
        if tools:
            tools_file = os.path.join(work_dir, "tools.json")
            with open(tools_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.input_schema,
                        }
                        for t in tools
                    ],
                    f,
                )
            config_file = os.path.join(work_dir, "mcp-config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(build_mcp_config(tools_file), f)
            argv += ["--mcp-config", config_file]
        if output_schema:
            argv += ["--json-schema", json.dumps(output_schema)]
        return argv

    async def _run_turn(
        self,
        *,
        argv: list[str],
        stdin_payload: str,
        stop_on_proposal: bool,
        work_dir: str,
        output_schema: dict[str, Any] | None,
        model: str,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
        message_id: uuid.UUID,
        s_key: str,
        d_key: str,
    ) -> list[dict[str, Any]]:
        # The CLI must bill the subscription login, never an API key that
        # happens to be in the backend's environment.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,  # neutral cwd: no project CLAUDE.md/settings pickup
            env=env,
        )
        assert proc.stdin is not None and proc.stdout is not None

        state = _TurnState(stop_on_proposal=stop_on_proposal)
        aborted = asyncio.Event()

        async def _watch_abort() -> None:
            a_key = abort_key(run_id)
            while True:
                if await redis.exists(a_key):
                    aborted.set()
                    proc.kill()
                    return
                await asyncio.sleep(0.3)

        watcher = asyncio.create_task(_watch_abort())
        try:
            proc.stdin.write(stdin_payload.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            await asyncio.wait_for(
                self._consume_stream(
                    proc,
                    state,
                    run_id=run_id,
                    node_id=node_id,
                    redis=redis,
                    build_io=build_io,
                    message_id=message_id,
                    s_key=s_key,
                    d_key=d_key,
                ),
                timeout=self._timeout,
            )
            if state.proposal_complete:
                # One of our tools was proposed while builtin tools kept the
                # turn cap high: the runner takes over from here, so the CLI
                # (which would deny the call and let the model ramble) is cut.
                proc.kill()
            await proc.wait()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            await self._save_partial_blocks(
                state.delta_text, state.delta_thinking,
                run_id, node_id, redis, message_id, build_io,
            )
            raise RuntimeError(
                f"claude-cli timed out after {self._timeout}s (run {run_id})"
            ) from None
        finally:
            watcher.cancel()

        if aborted.is_set():
            await self._save_partial_blocks(
                state.delta_text, state.delta_thinking,
                run_id, node_id, redis, message_id, build_io,
            )
            raise AbortSignalError(run_id, node_id)

        result = state.result_event
        if result is None and state.proposal_complete:
            # Cut on purpose — no result event, hence no usage for this call.
            result = {"subtype": "proposal", "is_error": False}
        if result is None:
            stderr_tail = b""
            if proc.stderr is not None:
                stderr_tail = await proc.stderr.read()
            raise RuntimeError(
                "claude-cli exited without a result event "
                f"(code={proc.returncode}): {stderr_tail[-500:].decode(errors='replace')}"
            )
        blocks = state.to_canonical_blocks(result, output_schema)
        # error_max_turns is our expected stop when the model proposed tools
        # (--max-turns 1 halts before a second model turn).
        subtype = result.get("subtype")
        if result.get("is_error") and subtype not in (
            "success",
            "error_max_turns",
        ):
            message = str(result.get("result") or subtype)[:500]
            if "rate limit" in message.lower() or "overloaded" in message.lower():
                raise LLMRateLimitError(message)
            raise RuntimeError(f"claude-cli error ({subtype}): {message}")
        if not blocks:
            raise RuntimeError(
                f"claude-cli returned no content (subtype={subtype})"
            )

        thinking = "".join(state.delta_thinking).strip() or None
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
            if state.message_started:
                await emitter.emit(
                    RunEventType.agent_message_completed,
                    node_id=node_id,
                    payload={
                        "message_id": str(message_id),
                        "text_chunks": state.text_chunk_index,
                        "thinking_chunks": state.thinking_chunk_index,
                    },
                )
            usage = result.get("usage")
            if usage:
                await record_usage(
                    db,
                    scope_id=run_id,
                    message_id=message_id,
                    model=_resolved_model(result, model),
                    usage=_usage_shim(usage),
                )
            await db.commit()
        await redis.delete(s_key, d_key, inflight_meta_key(run_id, message_id))
        return blocks

    async def _consume_stream(
        self,
        proc: asyncio.subprocess.Process,
        state: _TurnState,
        *,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
        message_id: uuid.UUID,
        s_key: str,
        d_key: str,
    ) -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if state.proposal_seen and etype not in ("assistant", "stream_event"):
                # The proposing message is complete (the CLI moved on to
                # deny/execute it): every sibling tool_use is in hand.
                state.proposal_complete = True
                return
            if etype == "stream_event":
                inner = event.get("event", {})
                if inner.get("type") == "content_block_start":
                    block = inner.get("content_block") or {}
                    if (
                        block.get("type") == "tool_use"
                        and block.get("name") == _CLI_WEB_SEARCH_TOOL
                    ):
                        await self._ensure_started(
                            state, run_id, node_id, redis, build_io, message_id
                        )
                        await self._publish_builtin(
                            redis, run_id, node_id, message_id, d_key, state,
                            {"tool": "web_search", "status": "searching"},
                        )
                    continue
                if inner.get("type") != "content_block_delta":
                    continue
                delta = inner.get("delta", {})
                kind = delta.get("type")
                if kind == "text_delta":
                    text = delta.get("text", "")
                    if not text:
                        continue
                    await self._ensure_started(
                        state, run_id, node_id, redis, build_io, message_id
                    )
                    state.delta_text.append(text)
                    await self._push_stream(redis, s_key, "text", text)
                    seq = await self._next_delta_sequence(
                        redis, d_key, state.delta_seq_base
                    )
                    await publish_run_event(
                        redis,
                        run_id,
                        RunEventType.agent_text_delta,
                        node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": state.text_chunk_index,
                            "text": text,
                        },
                        seq,
                        event_id=delta_event_id(
                            message_id, "text", state.text_chunk_index
                        ),
                    )
                    state.text_chunk_index += 1
                elif kind == "thinking_delta":
                    text = delta.get("thinking", "")
                    if not text:
                        continue
                    await self._ensure_started(
                        state, run_id, node_id, redis, build_io, message_id
                    )
                    state.delta_thinking.append(text)
                    await self._push_stream(redis, s_key, "thinking", text)
                    seq = await self._next_delta_sequence(
                        redis, d_key, state.delta_seq_base
                    )
                    await publish_run_event(
                        redis,
                        run_id,
                        RunEventType.agent_thinking_delta,
                        node_id,
                        {
                            "message_id": str(message_id),
                            "chunk_index": state.thinking_chunk_index,
                            "text": text,
                        },
                        seq,
                        event_id=delta_event_id(
                            message_id, "thinking", state.thinking_chunk_index
                        ),
                    )
                    state.thinking_chunk_index += 1
            elif etype == "assistant":
                blocks = event.get("message", {}).get("content") or []
                if isinstance(blocks, str):
                    blocks = [{"type": "text", "text": blocks}]
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype not in ("text", "thinking", "tool_use"):
                        continue
                    await self._ensure_started(
                        state, run_id, node_id, redis, build_io, message_id
                    )
                    if btype == "tool_use":
                        if block.get("name") == _CLI_WEB_SEARCH_TOOL:
                            # Runs inside the CLI; kept in the transcript as
                            # the same web_search_call block the HTTP
                            # provider persists, not as a tool_use.
                            query = (block.get("input") or {}).get("query")
                            state.web_searches[str(block.get("id"))] = query
                            state.assistant_blocks.append(
                                {
                                    "type": "web_search_call",
                                    "id": block.get("id"),
                                    "queries": [query] if query else [],
                                }
                            )
                            continue
                        if block.get("name") == _CLI_WEB_FETCH_TOOL:
                            # Same idea for page reads: one block per fetch
                            # (url known here, unlike at content_block_start)
                            # and a live "fetching" signal for the UI.
                            url = (block.get("input") or {}).get("url")
                            state.web_fetches[str(block.get("id"))] = url
                            state.assistant_blocks.append(
                                {
                                    "type": "web_fetch_call",
                                    "id": block.get("id"),
                                    "url": url,
                                }
                            )
                            await self._publish_builtin(
                                redis, run_id, node_id, message_id, d_key, state,
                                {"tool": "web_fetch", "status": "fetching", "url": url},
                            )
                            continue
                        if state.stop_on_proposal:
                            state.proposal_seen = True
                    state.assistant_blocks.append(block)
            elif etype == "user":
                # Builtin tool results come back as user tool_result events;
                # the sidecar ``tool_use_result`` carries the structured hits.
                for block in event.get("message", {}).get("content", []) or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    call_id = block.get("tool_use_id")
                    if call_id in state.web_fetches:
                        sidecar = event.get("tool_use_result")
                        meta = sidecar if isinstance(sidecar, dict) else {}
                        fetched = {
                            "tool": "web_fetch",
                            "url": meta.get("url") or state.web_fetches[call_id],
                            "code": meta.get("code"),
                            "bytes": meta.get("bytes"),
                        }
                        await self._publish_builtin(
                            redis, run_id, node_id, message_id, d_key, state,
                            {**fetched, "status": "completed"},
                        )
                        await self._publish_builtin(
                            redis, run_id, node_id, message_id, d_key, state,
                            {**fetched, "status": "result"},
                        )
                        continue
                    search_id = call_id
                    if search_id not in state.web_searches:
                        continue
                    query = state.web_searches[search_id]
                    hits: list[dict[str, str]] = []
                    sidecar = event.get("tool_use_result")
                    # ``results`` mixes the search's hit groups (dicts with a
                    # ``content`` list of {title, url}) with plain strings
                    # (the CLI's own summaries); only the dicts carry links.
                    results = sidecar.get("results") if isinstance(sidecar, dict) else None
                    for res in results or []:
                        if not isinstance(res, dict):
                            continue
                        for hit in res.get("content") or []:
                            if isinstance(hit, dict) and hit.get("url"):
                                hits.append(
                                    {"title": hit.get("title") or "", "url": hit["url"]}
                                )
                    # Same status sequence the HTTP provider emits for the
                    # Responses API (…searching → completed → result), so a
                    # client that flips its chip on "completed" sees it.
                    await self._publish_builtin(
                        redis, run_id, node_id, message_id, d_key, state,
                        {"tool": "web_search", "status": "completed"},
                    )
                    await self._publish_builtin(
                        redis, run_id, node_id, message_id, d_key, state,
                        {
                            "tool": "web_search",
                            "status": "result",
                            "query": query,
                            "queries": [query] if query else [],
                            "results": hits,
                        },
                    )
            elif etype == "result":
                state.result_event = event

    async def _publish_builtin(
        self,
        redis: Redis,
        run_id: uuid.UUID,
        node_id: str,
        message_id: uuid.UUID,
        d_key: str,
        state: _TurnState,
        payload: dict[str, Any],
    ) -> None:
        """Live builtin-tool activity on the run channel — the same
        ``builtin_tool_call`` event the HTTP provider publishes for the
        Responses API (tool=web_search, status=searching|completed|result),
        so the chat UI renders both identically. The CLI adds what it can
        see: the search hits (``results``) and page reads (tool=web_fetch:
        fetching → completed → result with url/code/bytes)."""
        seq = await self._next_delta_sequence(redis, d_key, state.delta_seq_base)
        await publish_run_event(
            redis,
            run_id,
            RunEventType.builtin_tool_call,
            node_id,
            {"message_id": str(message_id), **payload},
            seq,
        )

    async def _ensure_started(
        self,
        state: _TurnState,
        run_id: uuid.UUID,
        node_id: str,
        redis: Redis,
        build_io: IOFactory,
        message_id: uuid.UUID,
    ) -> None:
        if state.message_started:
            return
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
        state.message_started = True
        state.delta_seq_base = started_seq or 0
        await self._write_inflight_meta(
            redis, run_id, node_id, message_id, state.delta_seq_base
        )


class _TurnState:
    """Accumulates one CLI invocation's stream into canonical output."""

    def __init__(self, *, stop_on_proposal: bool = False) -> None:
        self.assistant_blocks: list[dict[str, Any]] = []
        self.delta_text: list[str] = []
        self.delta_thinking: list[str] = []
        self.result_event: dict[str, Any] | None = None
        # Builtin-tools mode: end the call as soon as the model proposes one
        # of OUR tools (the CLI would otherwise deny it and keep going).
        self.stop_on_proposal = stop_on_proposal
        self.proposal_seen = False
        self.proposal_complete = False
        # In-flight CLI web searches / page reads: tool_use id → query / url.
        self.web_searches: dict[str, str | None] = {}
        self.web_fetches: dict[str, str | None] = {}
        self.message_started = False
        self.delta_seq_base = 0
        self.text_chunk_index = 0
        self.thinking_chunk_index = 0

    def to_canonical_blocks(
        self,
        result: dict[str, Any],
        output_schema: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in self.assistant_blocks:
            btype = block.get("type")
            if btype == "thinking":
                thinking = block.get("thinking") or ""
                if thinking:
                    blocks.append({"type": "thinking", "thinking": thinking})
            elif btype == "text":
                text = block.get("text") or ""
                stripped = _strip_leaked_transcript(text)
                if stripped != text:
                    logger.warning(
                        "claude-cli: model leaked history tags; truncated "
                        "%d chars", len(text) - len(stripped),
                    )
                if stripped:
                    blocks.append({"type": "text", "text": stripped})
            elif btype == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": _strip_tool_prefix(block.get("name", "")),
                        "input": block.get("input", {}) or {},
                    }
                )
            elif btype in ("web_search_call", "web_fetch_call"):
                blocks.append(block)
        has_tool_use = any(b["type"] == "tool_use" for b in blocks)
        if not has_tool_use:
            # Backup capture: a proposed call always lands in
            # permission_denials even if its assistant event was missed.
            for denial in result.get("permission_denials") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": denial.get("tool_use_id"),
                        "name": _strip_tool_prefix(denial.get("tool_name", "")),
                        "input": denial.get("tool_input", {}) or {},
                    }
                )
                has_tool_use = True
        structured = result.get("structured_output")
        if output_schema and structured is not None:
            # The schema-validated object is authoritative; drop free text so
            # callers parse exactly one JSON payload.
            blocks = [b for b in blocks if b["type"] == "thinking"]
            blocks.append(
                {
                    "type": "text",
                    "text": json.dumps(structured, ensure_ascii=False),
                }
            )
        elif not blocks and isinstance(result.get("result"), str):
            text = result["result"]
            if text:
                blocks.append({"type": "text", "text": text})
        return blocks


def _strip_tool_prefix(name: str) -> str:
    return name[len(TOOL_NAME_PREFIX):] if name.startswith(TOOL_NAME_PREFIX) else name


__all__ = ["CLAUDE_CLI_MODELS", "ClaudeCLIProvider"]
