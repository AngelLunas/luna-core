"""Read-only listing of registered context sources.

Exposes the process-global registry built up by `register_context_source`
so the UI can render a chip palette (with each source's JSON Schema) for
the agent instructions editor. There is no create/update/delete — sources
are code contracts registered at startup, not user-managed records.
"""
from __future__ import annotations

from fastapi import APIRouter

from luna_core.core.dependencies import require_permission
from luna_core.schemas.context_source import ContextSourceRead
from luna_core.services.context_sources import list_context_sources

router = APIRouter(prefix="/context-sources", tags=["context-sources"])


@router.get(
    "",
    response_model=list[ContextSourceRead],
    dependencies=[require_permission("agents:read")],
)
async def index() -> list[ContextSourceRead]:
    return [
        ContextSourceRead(
            name=source.name,
            description=source.description,
            id_implicit=source.id_implicit,
            schema=source.schema,
        )
        for source in list_context_sources()
    ]
