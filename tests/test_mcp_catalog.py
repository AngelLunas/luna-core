"""mcp_catalog stdio server tests — real subprocess, real JSON-RPC."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from luna_core.llm.providers.mcp_catalog import build_mcp_config


def _speak(tools_file: str, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        [sys.executable, "-m", "luna_core.llm.providers.mcp_catalog", tools_file],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line]


def test_catalog_serves_tools_and_refuses_calls(tmp_path):
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps([
        {
            "name": "water_plant",
            "description": "Registra un riego",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]))
    replies = _speak(str(tools_file), [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "water_plant", "arguments": {}}},
    ])
    # notification got no reply: 3 replies for 3 requests
    assert [r["id"] for r in replies] == [0, 1, 2]
    init = replies[0]["result"]
    assert init["protocolVersion"] == "2025-11-25"
    assert init["serverInfo"]["name"] == "luna"
    listed = replies[1]["result"]["tools"]
    assert listed == [{
        "name": "water_plant",
        "description": "Registra un riego",
        "inputSchema": {"type": "object", "properties": {}},
    }]
    # execution is refused: this server is a catalog, never a runner
    call = replies[2]["result"]
    assert call["isError"] is True


def test_build_mcp_config_spawns_this_module(tmp_path):
    config = build_mcp_config("/tmp/tools.json")
    server = config["mcpServers"]["luna"]
    assert server["command"] == sys.executable
    assert server["args"] == [
        "-m", "luna_core.llm.providers.mcp_catalog", "/tmp/tools.json",
    ]
