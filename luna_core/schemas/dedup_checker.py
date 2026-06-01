"""HTTP schemas for the dedup-checker catalog endpoint.

Lives next to ``system_tool.py`` because the surface is the same shape
(a small read-only catalog the flow editor lists to populate a picker).
Like that one, the registry is in-process (``luna_core.dedup``), not
backed by a SQLAlchemy model — the router serializes registry entries
directly into these schemas.
"""
from __future__ import annotations

from pydantic import BaseModel


class DedupFieldRead(BaseModel):
    """One field the checker needs from each record to do its lookup.

    The flow editor renders these as rows in the field-mapping table:
    ``name`` is the checker's canonical name (left column), ``type``
    constrains which record fields are eligible in the right-column
    dropdown, ``description`` shows as a row hint. ``optional`` flags
    a field the user can leave unmapped (sharpens the lookup but
    isn't structurally required).
    """

    name: str
    type: str
    description: str = ""
    optional: bool = False


class DedupCheckerRead(BaseModel):
    """One catalog dedup checker as advertised to the flow editor.

    ``required_fields`` drives the field-mapping UI: the editor builds
    one row per entry, asking the user to point each canonical field at
    a field on the records the agent will produce. ``label`` is the
    human-readable display name (falls back to ``name`` server-side).
    """

    name: str
    label: str
    description: str = ""
    required_fields: list[DedupFieldRead]


__all__ = ["DedupCheckerRead", "DedupFieldRead"]
