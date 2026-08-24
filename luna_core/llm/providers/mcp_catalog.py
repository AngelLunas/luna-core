"""Catalog-only stdio MCP server for the Claude CLI provider.

The claude-cli provider advertises the turn's ``ToolDefinition`` list to the
Claude Code CLI as native MCP tools so the model proposes calls through the
mechanism it was trained on — but execution stays in the AgentRunner. This
module is that advertisement: a minimal JSON-RPC-over-stdio MCP server that
answers ``initialize`` and ``tools/list`` from a JSON file and never executes
anything. A ``tools/call`` should be unreachable (the provider never allowlists
these tools, so the CLI auto-denies before calling); if one arrives anyway it
returns an error result rather than pretending to run the tool.

Run as a module (the CLI spawns it per ``--mcp-config``):

    python -m luna_core.llm.providers.mcp_catalog /path/to/tools.json

The file holds ``[{"name", "description", "input_schema"}, ...]`` — the wire
shape of ``ToolDefinition``. Only stdlib is used: the subprocess must start
fast and never depend on the host app's collaborators.
"""
from __future__ import annotations

import json
import sys
from typing import Any

# Server name in the CLI's --mcp-config; tools surface as mcp__luna__<name>.
MCP_SERVER_NAME = "luna"
TOOL_NAME_PREFIX = f"mcp__{MCP_SERVER_NAME}__"


def build_mcp_config(tools_file: str) -> dict[str, Any]:
    """The ``--mcp-config`` payload that spawns this module as a stdio server."""
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", __name__, tools_file],
            }
        }
    }


def _load_tools(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        definitions = json.load(f)
    return [
        {
            "name": d["name"],
            "description": d.get("description", ""),
            "inputSchema": d.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        for d in definitions
    ]


def _reply(msg_id: Any, result: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n"
    )
    sys.stdout.flush()


def serve(tools_path: str) -> None:
    tools = _load_tools(tools_path)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method == "initialize":
            _reply(
                msg_id,
                {
                    "protocolVersion": msg.get("params", {}).get(
                        "protocolVersion", "2024-11-05"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": MCP_SERVER_NAME, "version": "1.0.0"},
                },
            )
        elif method == "tools/list":
            _reply(msg_id, {"tools": tools})
        elif method == "tools/call":
            # Unreachable by design (never allowlisted). Refuse loudly.
            _reply(
                msg_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "catalog-only MCP server: tools are executed "
                                "by the host runner, not the CLI"
                            ),
                        }
                    ],
                    "isError": True,
                },
            )
        elif msg_id is not None:
            # Politely answer anything else request-shaped; ignore notifications.
            _reply(msg_id, {})


if __name__ == "__main__":
    serve(sys.argv[1])
