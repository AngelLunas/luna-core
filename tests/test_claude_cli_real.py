"""Contract tests against the REAL Claude Code binary (subscription auth).

These pin the CLI behaviors the claude-cli provider is built on — the spikes
that validated the design, made permanent. They are skipped unless
CLAUDE_CLI_TEST_BIN points at a logged-in binary:

    CLAUDE_CLI_TEST_BIN=/path/to/claude pytest -m claude_cli tests/test_claude_cli_real.py

Each test costs one small model call on the subscription. Deliberately free of
luna_core imports so they can run on a bare host python (where the binary and
its login live) with just pytest installed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

BINARY = os.environ.get("CLAUDE_CLI_TEST_BIN")

pytestmark = [
    pytest.mark.claude_cli,
    pytest.mark.skipif(
        not BINARY, reason="set CLAUDE_CLI_TEST_BIN to run real-CLI contract tests"
    ),
]

BASE_ARGS = [
    "-p",
    "--tools", "",
    "--no-session-persistence",
    "--strict-mcp-config",
    "--exclude-dynamic-system-prompt-sections",
]

# Must mirror CLAUDE_CLI_MODELS in luna_core/llm/providers/claude_cli.py —
# duplicated here on purpose so this file needs no luna_core import.
CURATED_MODELS = ["haiku", "sonnet", "opus", "fable"]


def _run(
    extra_args: list[str],
    prompt: str,
    timeout: int = 120,
    *,
    allow_nonzero_exit: bool = False,
) -> str:
    proc = subprocess.run(
        [BINARY, *BASE_ARGS, *extra_args],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not allow_nonzero_exit:
        assert proc.returncode == 0, proc.stderr[-500:]
    return proc.stdout


def _catalog_config(tmp_path) -> str:
    """A one-tool MCP catalog served by a tiny inline stdio script — the same
    protocol subset luna's mcp_catalog implements."""
    server = tmp_path / "catalog.py"
    server.write_text(
        """
import json, sys
TOOL = {"name": "water_plant", "description": "Registra un riego de una planta",
        "inputSchema": {"type": "object", "properties": {
            "plant_name": {"type": "string"}, "amount_ml": {"type": "integer"}},
            "required": ["plant_name", "amount_ml"]}}
for line in sys.stdin:
    try: msg = json.loads(line)
    except Exception: continue
    mid = msg.get("id")
    m = msg.get("method", "")
    if m == "initialize":
        out = {"protocolVersion": msg["params"]["protocolVersion"],
               "capabilities": {"tools": {}},
               "serverInfo": {"name": "luna", "version": "0"}}
    elif m == "tools/list":
        out = {"tools": [TOOL]}
    elif mid is not None:
        out = {}
    else:
        continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": out}) + "\\n")
    sys.stdout.flush()
"""
    )
    config = tmp_path / "mcp-config.json"
    config.write_text(json.dumps({
        "mcpServers": {"luna": {"command": sys.executable, "args": [str(server)]}}
    }))
    return str(config)


def test_simple_call_returns_result_and_usage():
    out = json.loads(_run(
        ["--output-format", "json", "--model", "haiku", "--system-prompt", "Responde en una palabra."],
        "Di exactamente: OK",
    ))
    assert out["is_error"] is False
    assert "OK" in out["result"]
    assert out["usage"]["output_tokens"] > 0
    assert out["modelUsage"]  # canonical resolved models reported


def test_json_schema_yields_structured_output():
    schema = {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["diagnosis", "severity", "recommendations"],
    }
    out = json.loads(_run(
        ["--output-format", "json", "--model", "haiku", "--json-schema", json.dumps(schema),
         "--system-prompt", "Eres un agrónomo. Responde solo el JSON pedido."],
        "Hojas amarillas con venas verdes en la parte baja de mi tomatera.",
    ))
    structured = out["structured_output"]
    assert structured["severity"] in ("low", "medium", "high")
    assert structured["recommendations"]


def test_tool_use_streams_before_any_execution(tmp_path):
    """The load-bearing contract: with MCP tools advertised but not
    allowlisted and --max-turns 1, the assistant tool_use block arrives in
    stream-json, the tool is never executed, and the run stops itself.

    Hitting the turn cap makes the CLI exit non-zero even though the stream
    is complete — the provider decides by the result event, never the exit
    code, and this test pins that the stream is all there."""
    config = _catalog_config(tmp_path)
    stdout = _run(
        ["--output-format", "stream-json", "--verbose", "--model", "haiku",
         "--max-turns", "1",
         "--mcp-config", config,
         "--system-prompt",
         "Eres Savia. Usa las tools disponibles cuando pidan registrar algo."],
        "Riega la planta Marta con 500 ml usando la tool water_plant.",
        allow_nonzero_exit=True,
    )
    events = [json.loads(line) for line in stdout.splitlines() if line]
    tool_uses = [
        b
        for e in events
        if e.get("type") == "assistant"
        for b in e["message"].get("content", [])
        if b.get("type") == "tool_use"
    ]
    assert tool_uses, "model did not propose the advertised MCP tool"
    call = tool_uses[0]
    assert call["name"] == "mcp__luna__water_plant"
    assert call["input"]["plant_name"] == "Marta"
    assert call["input"]["amount_ml"] == 500

    result = next(e for e in events if e.get("type") == "result")
    # stopped by the turn cap, not by executing the tool
    assert result["subtype"] == "error_max_turns"
    denials = result.get("permission_denials") or []
    assert any(d["tool_name"] == "mcp__luna__water_plant" for d in denials)


def test_unrecognized_model_fails_fast_and_free():
    proc = subprocess.run(
        [BINARY, "-p", "--output-format", "json", "--model", "modelo-falso-123",
         "--tools", "", "--no-session-persistence"],
        input="di OK",
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = proc.stdout + proc.stderr
    assert "unrecognized_model" in combined
    # the run never reached the API: zero cost
    for line in proc.stdout.splitlines():
        try:
            out = json.loads(line)
        except json.JSONDecodeError:
            continue
        if out.get("type") == "result":
            assert out["is_error"] is True
            assert out.get("total_cost_usd", 0) == 0


@pytest.mark.parametrize("model", CURATED_MODELS)
def test_curated_model_list_is_current(model):
    """Every alias the UI offers must still be accepted by the CLI. Probes
    with a minimal call; a removed alias fails here instead of in prod."""
    out = json.loads(_run(
        ["--output-format", "json", "--model", model,
         "--system-prompt", "Responde en una palabra."],
        "Di exactamente: OK",
        timeout=180,
    ))
    assert out["is_error"] is False, f"alias '{model}' rejected: {out.get('result')}"


def _tiny_png_base64() -> str:
    """64x64 PNG, left half red / right half blue — stdlib only."""
    import base64
    import struct
    import zlib

    w = h = 64
    row = b"".join(b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff" for x in range(w))
    raw = b"".join(b"\x00" + row for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def test_stream_json_input_carries_images_to_the_model():
    """Vision contract: a base64 image block on the stream-json user message
    is seen by the model (stream-json in requires stream-json out)."""
    message = {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": "¿Qué colores tiene la imagen y cómo están "
                                 "distribuidos? Responde en una frase."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": _tiny_png_base64()}},
    ]}}
    stdout = _run(
        ["--input-format", "stream-json", "--output-format", "stream-json",
         "--verbose", "--model", "haiku", "--max-turns", "1",
         "--system-prompt", "Eres un analista de imágenes."],
        json.dumps(message) + "\n",
    )
    result = next(
        json.loads(l) for l in stdout.splitlines() if l and '"type":"result"' in l.replace(" ", "")
    )
    text = str(result.get("result", "")).lower()
    assert result["subtype"] == "success"
    assert ("rojo" in text or "red" in text) and ("azul" in text or "blue" in text)


def test_web_search_streams_observable_events():
    """Web-search contract: the CLI's own WebSearch runs headless and its
    activity is visible in stream-json — the tool_use with the query, and a
    tool_result whose ``tool_use_result`` sidecar lists title+url hits. The
    provider turns these into builtin_tool_call events for the UI."""
    stdout = _run(
        ["--output-format", "stream-json", "--verbose", "--model", "haiku",
         "--tools", "WebSearch", "--allowedTools", "WebSearch", "--max-turns", "6",
         "--system-prompt", "Usa WebSearch cuando necesites datos actuales."],
        "Busca en la web cuál es la versión estable más reciente de Python y "
        "responde con el número y la fuente.",
        timeout=240,
        # the model may search right up to the cap → exit 1 with a full stream
        allow_nonzero_exit=True,
    )
    events = [json.loads(l) for l in stdout.splitlines() if l]
    searches = [
        b for e in events if e.get("type") == "assistant"
        for b in e["message"].get("content", [])
        if b.get("type") == "tool_use" and b.get("name") == "WebSearch"
    ]
    assert searches, "model did not use WebSearch"
    assert searches[0]["input"].get("query")
    ids = {s["id"] for s in searches}
    sidecars = [
        e.get("tool_use_result") for e in events if e.get("type") == "user"
        and any(
            isinstance(b, dict) and b.get("tool_use_id") in ids
            for b in e.get("message", {}).get("content", [])
        )
    ]
    assert sidecars and isinstance(sidecars[0], dict)
    # `results` mixes hit groups (dicts with a content list) with plain
    # string summaries; only the dicts carry links.
    hits = [
        h
        for r in sidecars[0].get("results", [])
        if isinstance(r, dict)
        for h in r.get("content", [])
        if isinstance(h, dict)
    ]
    assert hits and all("url" in h for h in hits)
