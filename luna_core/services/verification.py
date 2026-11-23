"""Email-verification code lifecycle.

A registered user starts unverified. ``create_verification_code`` mints a fresh
single-use numeric code (invalidating any prior unused one) whose raw value is
e-mailed; ``verify_code`` checks the code the user typed into the app and, on a
match, flips the account to verified.

The same mechanism mints the codes that reset a password (see
``services.password``): ``mint_code`` and ``check_code`` take the purpose, and
each purpose keeps its own single live code.

Brute force is bounded by a per-code attempt cap (the row is burned at the cap)
plus a short expiry. luna-core owns the *mechanism*; the host app owns delivery
(rendering + sending the branded e-mail), injected as a hook on ``app.state``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.security import generate_numeric_code, hash_token
from luna_core.models.email_verification_code import CodePurpose, EmailVerificationCode
from luna_core.models.user import User


class CodeCheck(StrEnum):
    """Why a typed code did or did not pass. The refusals are the stable codes
    the client words."""

    VERIFIED = "verified"
    ALREADY_VERIFIED = "already_verified"
    MATCHED = "matched"
    INVALID = "invalid"
    EXPIRED = "expired"
    TOO_MANY_ATTEMPTS = "too_many_attempts"
    NO_CODE = "no_code"


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    reason: str


async def mint_code(
    db: AsyncSession, user: User, purpose: CodePurpose, ttl_minutes: int
) -> str:
    """A new code for ``user`` and ``purpose``; its raw value is only shown
    here. The user's other unused codes of that purpose are burned, so a
    re-send always leaves exactly one live code."""
    now = datetime.now(timezone.utc)
    await db.execute(
        update(EmailVerificationCode)
        .where(
            EmailVerificationCode.user_id == user.id,
            EmailVerificationCode.purpose == purpose,
            EmailVerificationCode.used_at.is_(None),
        )
        .values(used_at=now)
    )

    raw_code = generate_numeric_code()
    db.add(
        EmailVerificationCode(
            user_id=user.id,
            purpose=purpose,
            code_hash=hash_token(raw_code),
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
    )
    await db.commit()
    return raw_code


async def check_code(
    db: AsyncSession, user: User, purpose: CodePurpose, raw_code: str
) -> CodeCheck:
    """Check ``raw_code`` against the user's live code for ``purpose``. A
    match spends the code (the caller commits what it proves); a miss counts
    as an attempt."""
    now = datetime.now(timezone.utc)
    code = await db.scalar(
        select(EmailVerificationCode)
        .where(
            EmailVerificationCode.user_id == user.id,
            EmailVerificationCode.purpose == purpose,
            EmailVerificationCode.used_at.is_(None),
        )
        .order_by(EmailVerificationCode.created_at.desc())
        .limit(1)
    )
    if code is None:
        return CodeCheck.NO_CODE

    expires_at = code.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        return CodeCheck.EXPIRED

    code.attempts += 1
    if code.attempts > settings.email_verification_max_attempts:
        code.used_at = now  # burn it — the user must request a fresh code
        await db.commit()
        return CodeCheck.TOO_MANY_ATTEMPTS

    if hash_token(raw_code.strip()) != code.code_hash:
        await db.commit()  # persist the attempt bump
        return CodeCheck.INVALID

    code.used_at = now
    return CodeCheck.MATCHED


async def create_verification_code(db: AsyncSession, user: User) -> str:
    """Mint a new email verification code for ``user``."""
    return await mint_code(
        db,
        user,
        CodePurpose.VERIFY_EMAIL,
        settings.email_verification_code_ttl_minutes,
    )


async def verify_code(db: AsyncSession, user: User, raw_code: str) -> VerificationResult:
    """Check ``raw_code`` for ``user``'s live code and verify on a match."""
    if user.is_verified:
        return VerificationResult(ok=True, reason=CodeCheck.ALREADY_VERIFIED)

    check = await check_code(db, user, CodePurpose.VERIFY_EMAIL, raw_code)
    if check is not CodeCheck.MATCHED:
        return VerificationResult(ok=False, reason=check)

    user.is_verified = True
    user.verified_at = datetime.now(timezone.utc)
    await db.commit()
    return VerificationResult(ok=True, reason=CodeCheck.VERIFIED)
