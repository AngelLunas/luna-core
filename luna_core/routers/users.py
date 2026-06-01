from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from luna_core.core.dependencies import DBSession, require_permission
from luna_core.models.user import User
from luna_core.schemas.user import RoleAssign, UserRead
from luna_core.services.permission import UserNotFound, assign_role

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "",
    response_model=list[UserRead],
    dependencies=[require_permission("users:manage")],
)
async def index(db: DBSession) -> list[UserRead]:
    result = await db.execute(select(User).order_by(User.created_at))
    return [UserRead.model_validate(u) for u in result.scalars().all()]


@router.put(
    "/{user_id}/role",
    response_model=UserRead,
    dependencies=[require_permission("users:manage")],
)
async def update_role(
    user_id: uuid.UUID, payload: RoleAssign, db: DBSession
) -> UserRead:
    try:
        user = await assign_role(user_id, payload.role, db)
    except UserNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        ) from exc
    return UserRead.model_validate(user)
