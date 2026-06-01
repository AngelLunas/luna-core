"""Idempotent seed of luna-core's own role/permission grants.

Host applications (e.g. luna-sentinel) call `seed_core_permissions` in
their startup lifespan to install the baseline `core` app permissions.
They also seed their own permissions for any app-specific roles.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.services.permission import seed_permissions

# Permission checks have no role hierarchy — `has_permission` matches the
# user's role exactly, so admin must carry the full list (including the
# `:read` permissions that `user` also has).
_USER_READS: list[str] = [
    "flows:read",
    "agents:read",
    "connectors:read",
    "llm_providers:read",
    "runs:read",
]
_ADMIN_WRITES: list[str] = [
    "flows:create",
    "flows:delete",
    "agents:create",
    "agents:update",
    "agents:delete",
    "connectors:create",
    "connectors:update",
    "connectors:delete",
    "connectors:test",
    "llm_providers:create",
    "llm_providers:update",
    "llm_providers:delete",
    "users:manage",
]

CORE_PERMISSIONS: list[tuple[str, str, str]] = [
    ("core", "user", p) for p in _USER_READS
] + [
    ("core", "admin", p) for p in (*_USER_READS, *_ADMIN_WRITES)
]


async def seed_core_permissions(db: AsyncSession) -> None:
    await seed_permissions(CORE_PERMISSIONS, db)
