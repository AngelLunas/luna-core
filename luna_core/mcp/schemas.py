from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Same shape as luna_core.llm.base.ToolDefinition — re-exported here to
    keep the MCP surface independent of the LLM module."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    name: str
    output: Any
    is_error: bool = False
    error_message: str | None = None


__all__ = ["ToolCallResult", "ToolDefinition"]
