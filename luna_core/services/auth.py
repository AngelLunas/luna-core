from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from luna_core.models.refresh_token import RefreshToken
from luna_core.models.user import User


@dataclass(slots=True)
class IssuedTokens:
    access_token: str
    access_expires_in: int
    refresh_token: str
    refresh_expires_at: datetime
    user: User


async def register_user(db: AsyncSession, email: str, password: str) -> User:
    normalized_email = email.lower().strip()

    existing = await db.scalar(select(User).where(User.email == normalized_email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=normalized_email,
        password_hash=hash_password(password),
        is_active=True,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from exc

    await db.refresh(user)
    return user


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> User | None:
    normalized_email = email.lower().strip()
    user = await db.scalar(select(User).where(User.email == normalized_email))
    if user is None:
        # Run bcrypt anyway to avoid trivial timing oracle on user existence
        verify_password(password, "$2b$12$" + "x" * 53)
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def issue_tokens(db: AsyncSession, user: User) -> IssuedTokens:
    now = datetime.now(timezone.utc)
    refresh_token = generate_refresh_token()
    refresh_expires_at = now + timedelta(days=settings.refresh_token_expire_days)

    db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_refresh_token(refresh_token),
            expires_at=refresh_expires_at,
            revoked=False,
        )
    )
    await db.commit()

    access_token = create_access_token(subject=user.id)
    return IssuedTokens(
        access_token=access_token,
        access_expires_in=settings.access_token_expire_minutes * 60,
        refresh_token=refresh_token,
        refresh_expires_at=refresh_expires_at,
        user=user,
    )


async def rotate_refresh_token(
    db: AsyncSession, raw_refresh_token: str
) -> IssuedTokens:
    token_hash = hash_refresh_token(raw_refresh_token)

    stored = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    now = datetime.now(timezone.utc)
    expires_at = stored.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if stored.revoked or expires_at <= now:
        # Possible replay — revoke entire family for this user
        await _revoke_all_user_tokens(db, stored.user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired",
        )

    user = await db.get(User, stored.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    stored.revoked = True
    await db.flush()

    return await issue_tokens(db, user)


async def revoke_refresh_token(db: AsyncSession, raw_refresh_token: str) -> None:
    token_hash = hash_refresh_token(raw_refresh_token)
    stored = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if stored is not None and not stored.revoked:
        stored.revoked = True
        await db.commit()


async def _revoke_all_user_tokens(db: AsyncSession, user_id) -> None:
    tokens = await db.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False)
        )
    )
    for token in tokens:
        token.revoked = True
    await db.commit()
