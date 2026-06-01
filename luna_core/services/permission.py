from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.permission import Permission
from luna_core.models.user import User


class UserNotFound(LookupError):
    pass


async def has_permission(
    user: User, permission: str, db: AsyncSession
) -> bool:
    """Check if the user's role grants the given permission in any app."""
    result = await db.execute(
        select(Permission.id)
        .where(Permission.role == user.role, Permission.permission == permission)
        .limit(1)
    )
    return result.first() is not None


async def assign_role(
    user_id: uuid.UUID, role: str, db: AsyncSession
) -> User:
    """Update a user's role and return the updated row."""
    user = await db.get(User, user_id)
    if user is None:
        raise UserNotFound(str(user_id))
    user.role = role
    await db.commit()
    await db.refresh(user)
    return user


async def seed_permissions(
    permissions: list[tuple[str, str, str]], db: AsyncSession
) -> None:
    """Idempotent seed of (app, role, permission) tuples."""
    if not permissions:
        return

    rows = [
        {"app": app, "role": role, "permission": permission}
        for app, role, permission in permissions
    ]
    stmt = insert(Permission).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["app", "role", "permission"]
    )
    await db.execute(stmt)
    await db.commit()
