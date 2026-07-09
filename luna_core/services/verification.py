"""Email-verification code lifecycle.

A registered user starts unverified. ``create_verification_code`` mints a fresh
single-use numeric code (invalidating any prior unused one) whose raw value is
e-mailed; ``verify_code`` checks the code the user typed into the app and, on a
match, flips the account to verified.

Brute force is bounded by a per-code attempt cap (the row is burned at the cap)
plus a short expiry. luna-core owns the *mechanism*; the host app owns delivery
(rendering + sending the branded e-mail), injected as a hook on ``app.state``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.security import generate_numeric_code, hash_token
from luna_core.models.email_verification_code import EmailVerificationCode
from luna_core.models.user import User


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    # One of: "verified", "already_verified", "invalid", "expired",
    # "too_many_attempts", "no_code".
    reason: str


async def create_verification_code(db: AsyncSession, user: User) -> str:
    """Mint a new code for ``user`` and return its raw value (only shown here).

    Any of the user's still-unused codes are burned so a re-send always leaves
    exactly one live code.
    """
    now = datetime.now(timezone.utc)
    await db.execute(
        update(EmailVerificationCode)
        .where(
            EmailVerificationCode.user_id == user.id,
            EmailVerificationCode.used_at.is_(None),
        )
        .values(used_at=now)
    )

    raw_code = generate_numeric_code()
    db.add(
        EmailVerificationCode(
            user_id=user.id,
            code_hash=hash_token(raw_code),
            expires_at=now
            + timedelta(minutes=settings.email_verification_code_ttl_minutes),
        )
    )
    await db.commit()
    return raw_code


async def verify_code(db: AsyncSession, user: User, raw_code: str) -> VerificationResult:
    """Check ``raw_code`` for ``user``'s live code and verify on a match."""
    if user.is_verified:
        return VerificationResult(ok=True, reason="already_verified")

    now = datetime.now(timezone.utc)
    code = await db.scalar(
        select(EmailVerificationCode)
        .where(
            EmailVerificationCode.user_id == user.id,
            EmailVerificationCode.used_at.is_(None),
        )
        .order_by(EmailVerificationCode.created_at.desc())
        .limit(1)
    )
    if code is None:
        return VerificationResult(ok=False, reason="no_code")

    expires_at = code.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        return VerificationResult(ok=False, reason="expired")

    code.attempts += 1
    if code.attempts > settings.email_verification_max_attempts:
        code.used_at = now  # burn it — the user must request a fresh code
        await db.commit()
        return VerificationResult(ok=False, reason="too_many_attempts")

    if hash_token(raw_code.strip()) != code.code_hash:
        await db.commit()  # persist the attempt bump
        return VerificationResult(ok=False, reason="invalid")

    code.used_at = now
    user.is_verified = True
    user.verified_at = now
    await db.commit()
    return VerificationResult(ok=True, reason="verified")
