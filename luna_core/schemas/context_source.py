from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ContextSourceRead(BaseModel):
    name: str
    description: str
    id_implicit: bool = False
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")

    model_config = {"populate_by_name": True}
