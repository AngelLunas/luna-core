"""Dedup-checker catalog endpoint.

Lists the dedup checkers currently registered in the in-process
``DedupCheckerRegistry``. The flow editor uses this to populate the
"Dedup against" dropdown in the stash config inspector — picking an
entry here pins ``node.config.stash.dedup.checker`` to the entry's
``name`` and reveals the field-mapping editor for that checker's
``required_fields``.

Read-only; checkers are registered programmatically at process startup
by the host application (e.g. luna-sentinel registers ``sentinel.jobs``),
not via API.
"""
from __future__ import annotations

from fastapi import APIRouter

from luna_core.core.dependencies import require_permission
from luna_core.dedup import get_default_registry
from luna_core.schemas.dedup_checker import DedupCheckerRead, DedupFieldRead

router = APIRouter(prefix="/dedup-checkers", tags=["dedup-checkers"])


@router.get(
    "",
    response_model=list[DedupCheckerRead],
    dependencies=[require_permission("agents:read")],
)
async def index() -> list[DedupCheckerRead]:
    registry = get_default_registry()
    return [
        DedupCheckerRead(
            name=checker.name,
            label=checker.display_label(),
            description=checker.description,
            required_fields=[
                DedupFieldRead(
                    name=f.name,
                    type=f.type,
                    description=f.description,
                    optional=f.optional,
                )
                for f in checker.required_fields
            ],
        )
        for checker in registry.list_all()
    ]
