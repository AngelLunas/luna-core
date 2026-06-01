"""System-tool catalog endpoint.

Lists the catalog-scope tools currently registered in the in-process
system-tool registry. Used by the agent editor to populate the
"System tools" picker — the user toggles names here, and the agent's
grants get persisted via the per-agent assign endpoint in the agents
router.

Catalog-only: context-scope tools (``yield_iteration``, future
``accept_item``, etc) are auto-injected by runtimes and not
user-assignable, so they don't surface in this endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter

from luna_core.core.dependencies import require_permission
from luna_core.mcp.system_tools import get_default_registry
from luna_core.schemas.system_tool import SystemToolRead

router = APIRouter(prefix="/system-tools", tags=["system-tools"])


@router.get(
    "",
    response_model=list[SystemToolRead],
    dependencies=[require_permission("agents:read")],
)
async def index() -> list[SystemToolRead]:
    registry = get_default_registry()
    return [
        SystemToolRead(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
        )
        for tool in registry.list_catalog()
    ]
