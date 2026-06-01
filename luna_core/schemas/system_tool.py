"""HTTP schemas for the system-tool catalog endpoint.

The catalog lives in process (``luna_core.mcp.system_tools``), not in
the DB — so this schema isn't backed by a SQLAlchemy model. The router
serializes ``SystemTool`` registry entries directly into this shape
for the agent editor's "System tools" picker.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SystemToolRead(BaseModel):
    """One catalog system tool as advertised to the agent editor.

    Only catalog-scope tools are returned (context-scope tools like
    ``yield_iteration`` are intrinsic to a particular runtime context
    and aren't user-assignable, so they don't appear in the picker).
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


__all__ = ["SystemToolRead"]
