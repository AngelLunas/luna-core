import uuid
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal, get_db
from luna_core.core.redis import get_redis
from luna_core.core.security import decode_access_token
from luna_core.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def get_redis_client() -> Redis:
    return get_redis()


async def get_current_user_allow_unverified(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Authenticate the bearer token and load the user — WITHOUT the email-
    verification gate. Use for the handful of endpoints an unverified user must
    still reach (read own profile, re-send verification, log out)."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from exc

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    sub = payload.get("sub")
    try:
        user_id = uuid.UUID(sub)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        ) from exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user


async def get_current_user(
    user: Annotated[User, Depends(get_current_user_allow_unverified)],
) -> User:
    """The default protected-route principal: authenticated AND (when the host
    app requires it) email-verified. Unverified users get 403 ``email_not_verified``
    so the client can route them to the "confirm your email" screen."""
    if settings.email_verification_required and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="email_not_verified",
        )
    return user


async def authenticate_websocket(
    websocket: WebSocket, token: str | None, *, permission: str | None = None
) -> User | None:
    """The user behind a WebSocket, under the same rules as ``CurrentUser``:
    an access token, an active user and, when the host requires it, a
    verified email; with ``permission``, also what ``require_permission``
    checks on the matching REST reads. The token comes as ``?token=`` (browsers can't set
    headers on a WebSocket) or an ``Authorization: Bearer`` header (any other
    client). None on any failure: the caller closes the socket with 1008
    before accepting it, which the client sees as a 403 on the handshake.
    """
    if not token:
        scheme, _, value = (websocket.headers.get("authorization") or "").partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else None
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        return None
    if payload.get("type") != "access":
        return None
    try:
        user_id = uuid.UUID(payload.get("sub"))
    except (TypeError, ValueError):
        return None
    from luna_core.services.permission import has_permission

    async with AsyncSessionLocal() as db:
        user = await db.get(User, user_id)
        if user is None or not user.is_active:
            return None
        if settings.email_verification_required and not user.is_verified:
            return None
        if permission is not None and not await has_permission(user, permission, db):
            return None
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentUserAllowUnverified = Annotated[
    User, Depends(get_current_user_allow_unverified)
]
DBSession = Annotated[AsyncSession, Depends(get_db)]
RedisClient = Annotated[Redis, Depends(get_redis_client)]


def require_permission(permission: str):
    """FastAPI dependency factory that gates an endpoint on a permission."""
    from luna_core.services.permission import has_permission

    async def dependency(
        user: Annotated[User, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> User:
        if not await has_permission(user, permission, db):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return Depends(dependency)
