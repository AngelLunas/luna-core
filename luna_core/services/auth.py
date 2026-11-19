import uuid
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
    if not user.is_active or user.password_hash is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def issue_tokens(
    db: AsyncSession, user: User, *, family_id: uuid.UUID | None = None
) -> IssuedTokens:
    """A new access/refresh pair. Without ``family_id`` it opens a new session
    (a sign-in); a rotation passes its own family to stay in it."""
    now = datetime.now(timezone.utc)
    refresh_token = generate_refresh_token()
    refresh_expires_at = now + timedelta(days=settings.refresh_token_expire_days)

    token_id = uuid.uuid4()
    db.add(
        RefreshToken(
            id=token_id,
            user_id=user.id,
            family_id=family_id or token_id,
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


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _within_reuse_grace(stored: RefreshToken, now: datetime) -> bool:
    if stored.rotated_at is None:
        return False
    grace = timedelta(seconds=settings.refresh_reuse_grace_seconds)
    return now - _aware(stored.rotated_at) <= grace


async def rotate_refresh_token(
    db: AsyncSession, raw_refresh_token: str
) -> IssuedTokens:
    stored = await _find_refresh_token(db, hash_refresh_token(raw_refresh_token))
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    now = datetime.now(timezone.utc)
    if _aware(stored.expires_at) <= now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired",
        )

    if stored.revoked and not _within_reuse_grace(stored, now):
        # Possible replay: sign out the session it belongs to.
        await _revoke_family(db, stored.family_id)
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

    if stored.revoked:
        # Reused inside the grace window: the client never saved the token
        # its rotation returned. It gets a sibling in the same family, and
        # whichever of the two it keeps revokes the other on its next
        # rotation. A session already closed (logout, a replay, the account
        # being deleted) has no live token left and stays closed.
        if not await _family_is_live(db, stored.family_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token revoked or expired",
            )
        return await issue_tokens(db, user, family_id=stored.family_id)

    stored.revoked = True
    stored.rotated_at = now
    await _revoke_family(db, stored.family_id, keep=stored.id, commit=False)
    await db.flush()

    return await issue_tokens(db, user, family_id=stored.family_id)


async def revoke_refresh_token(db: AsyncSession, raw_refresh_token: str) -> None:
    token_hash = hash_refresh_token(raw_refresh_token)
    stored = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    if stored is not None and not stored.revoked:
        stored.revoked = True
        await db.commit()


async def _find_refresh_token(db: AsyncSession, token_hash: str) -> RefreshToken | None:
    return await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )


async def _family_is_live(db: AsyncSession, family_id: uuid.UUID) -> bool:
    live = await db.scalar(
        select(RefreshToken.id)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked.is_(False))
        .limit(1)
    )
    return live is not None


async def _revoke_family(
    db: AsyncSession,
    family_id: uuid.UUID,
    *,
    keep: uuid.UUID | None = None,
    commit: bool = True,
) -> None:
    """Revoke the family's live tokens, except ``keep`` (which the caller
    revokes itself, with its rotation time)."""
    query = select(RefreshToken).where(
        RefreshToken.family_id == family_id, RefreshToken.revoked.is_(False)
    )
    if keep is not None:
        query = query.where(RefreshToken.id != keep)
    for token in await db.scalars(query):
        token.revoked = True
    if commit:
        await db.commit()
