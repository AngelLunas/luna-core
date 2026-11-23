"""Forgetting and changing a password.

Signed out, the user asks for a code by email and types it back with the new
password (``reset_password``); signed in, the current password authorises the
change (``change_password``). Either way every session the account had is
closed and the caller issues a fresh one, so a stolen password stops working
everywhere at once.

Like verification, luna-core owns the mechanism and the host app the email:
``app.state.send_password_email`` renders and sends each ``PasswordEmail``.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.security import hash_password, verify_password
from luna_core.models.email_verification_code import CodePurpose
from luna_core.models.refresh_token import RefreshToken
from luna_core.models.user import User
from luna_core.services.verification import CodeCheck, check_code, mint_code


class PasswordEmail(StrEnum):
    """The emails a host app sends for this flow."""

    # The code that resets the password.
    RESET_CODE = "reset_code"
    # Asked for a code, but the account signs in with Google or Apple only.
    NO_PASSWORD = "no_password"
    # The password changed: in case it was not them.
    CHANGED = "changed"


class PasswordRefusal(StrEnum):
    """Stable codes the client words."""

    WRONG_PASSWORD = "wrong_password"
    PASSWORD_NOT_SET = "password_not_set"
    TOO_MANY_REQUESTS = "too_many_requests"


async def user_by_email(db: AsyncSession, email: str) -> User | None:
    return await db.scalar(select(User).where(User.email == email.lower().strip()))


async def create_reset_code(db: AsyncSession, user: User) -> str:
    return await mint_code(
        db, user, CodePurpose.PASSWORD_RESET, settings.password_reset_code_ttl_minutes
    )


async def _revoke_sessions(db: AsyncSession, user_id: uuid.UUID) -> None:
    for token in await db.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False)
        )
    ):
        token.revoked = True


async def _set_password(db: AsyncSession, user: User, new_password: str) -> None:
    user.password_hash = hash_password(new_password)
    await _revoke_sessions(db, user.id)
    await db.commit()


async def reset_password(
    db: AsyncSession, user: User, raw_code: str, new_password: str
) -> CodeCheck:
    """Set ``new_password`` when ``raw_code`` is the user's live reset code.
    The code came to their inbox, so it also confirms the address."""
    check = await check_code(db, user, CodePurpose.PASSWORD_RESET, raw_code)
    if check is not CodeCheck.MATCHED:
        return check
    if not user.is_verified:
        user.is_verified = True
        user.verified_at = datetime.now(timezone.utc)
    await _set_password(db, user, new_password)
    return CodeCheck.MATCHED


async def change_password(
    db: AsyncSession, user: User, current_password: str, new_password: str
) -> PasswordRefusal | None:
    """Set ``new_password`` when ``current_password`` is right; the refusal
    otherwise."""
    if user.password_hash is None:
        return PasswordRefusal.PASSWORD_NOT_SET
    if not verify_password(current_password, user.password_hash):
        return PasswordRefusal.WRONG_PASSWORD
    await _set_password(db, user, new_password)
    return None
