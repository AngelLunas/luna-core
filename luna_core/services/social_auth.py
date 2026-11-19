"""Signing in with Google or Apple.

The app gets an ID token from the provider's native sheet and posts it here.
The token is checked against the provider's published keys, its audience
(our client or bundle IDs) and its issuer; then the identity finds its
account: by the identity itself, else by the verified email (linking it to an
account that already exists), else a new account, already verified.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, status
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.config import settings
from luna_core.core.crypto import decrypt_json, encrypt_json
from luna_core.models.refresh_token import RefreshToken
from luna_core.models.user import User
from luna_core.models.user_identity import UserIdentity

logger = logging.getLogger(__name__)


class Provider(StrEnum):
    GOOGLE = "google"
    APPLE = "apple"


GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"

# A token is accepted once: the same one posted again is a replay. Both
# providers' ID tokens live at most an hour.
_USED_TOKEN_TTL_SECONDS = 3600


@dataclass(slots=True)
class SocialClaims:
    provider: Provider
    subject: str
    email: str | None
    email_verified: bool
    audience: str


def _split(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _refused(detail: str = "Sign-in could not be verified") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


@lru_cache(maxsize=2)
def _jwks(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def _as_bool(value: Any) -> bool:
    # Apple sends "true"/"false" strings; Google sends booleans.
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


async def _decode(token: str, *, jwks_url: str, audiences: list[str], issuers) -> dict:
    try:
        key = await asyncio.to_thread(_jwks(jwks_url).get_signing_key_from_jwt, token)
        return jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            audience=audiences,
            issuer=issuers,
            options={"require": ["exp", "iat", "sub", "aud", "iss"]},
        )
    except (jwt.PyJWTError, httpx.HTTPError) as exc:
        logger.info("social token rejected: %s", exc)
        raise _refused() from exc


async def verify_google(id_token: str) -> SocialClaims:
    audiences = _split(settings.google_client_ids)
    if not audiences:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Google sign-in is not enabled")
    claims = await _decode(
        id_token, jwks_url=GOOGLE_JWKS_URL, audiences=audiences, issuers=GOOGLE_ISSUERS
    )
    return SocialClaims(
        provider=Provider.GOOGLE,
        subject=str(claims["sub"]),
        email=(claims.get("email") or None),
        email_verified=_as_bool(claims.get("email_verified")),
        audience=str(claims["aud"]),
    )


async def verify_apple(identity_token: str, nonce: str | None) -> SocialClaims:
    audiences = _split(settings.apple_bundle_ids)
    if not audiences:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Apple sign-in is not enabled")
    claims = await _decode(
        identity_token, jwks_url=APPLE_JWKS_URL, audiences=audiences, issuers=APPLE_ISSUER
    )
    # The app sends the nonce it asked Apple to sign; a token minted for
    # another request does not carry it.
    if nonce is not None and claims.get("nonce") != nonce:
        raise _refused()
    return SocialClaims(
        provider=Provider.APPLE,
        subject=str(claims["sub"]),
        email=(claims.get("email") or None),
        # Apple only hands out addresses it has verified (relay ones too).
        email_verified=_as_bool(claims.get("email_verified", True)),
        audience=str(claims["aud"]),
    )


async def claim_token_once(redis: Redis, token: str) -> None:
    """Refuse a token already used to sign in."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    first = await redis.set(f"social:used:{digest}", "1", nx=True, ex=_USED_TOKEN_TTL_SECONDS)
    if not first:
        raise _refused()


async def _identity(db: AsyncSession, provider: Provider, subject: str) -> UserIdentity | None:
    return await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == provider.value, UserIdentity.subject == subject
        )
    )


async def _user_by_email(db: AsyncSession, email: str) -> User | None:
    return await db.scalar(select(User).where(User.email == email))


async def _revoke_sessions(db: AsyncSession, user_id: uuid.UUID) -> None:
    for token in await db.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False)
        )
    ):
        token.revoked = True


async def sign_in(db: AsyncSession, claims: SocialClaims) -> User:
    """The account this identity signs into, creating it when there is none."""
    identity = await _identity(db, claims.provider, claims.subject)
    if identity is not None:
        user = await db.get(User, identity.user_id)
        if user is None or not user.is_active:
            raise _refused("User not found or inactive")
        return user

    email = claims.email.lower().strip() if claims.email else None
    if not email:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "The provider did not share an email address"
        )

    user = await _user_by_email(db, email)
    if user is not None:
        if not claims.email_verified:
            # Linking on an address the provider has not confirmed would hand
            # the account to whoever typed it there.
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
        if not user.is_active:
            raise _refused("User not found or inactive")
        if not user.is_verified:
            # Whoever registered this address never proved it; the provider
            # just did. The password they set stops working, so an account
            # opened in someone else's name cannot be entered with it.
            user.password_hash = None
            user.is_verified = True
            user.verified_at = datetime.now(UTC)
    else:
        user = User(
            email=email,
            password_hash=None,
            is_active=True,
            is_verified=claims.email_verified,
            verified_at=datetime.now(UTC) if claims.email_verified else None,
        )
        db.add(user)
        await db.flush()

    db.add(
        UserIdentity(
            user_id=user.id,
            provider=claims.provider.value,
            subject=claims.subject,
            email=email,
        )
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        # The same identity signing in twice at once: the other request won.
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Try signing in again") from exc
    await db.refresh(user)
    return user


async def providers_of(db: AsyncSession, user_id: uuid.UUID) -> list[str]:
    rows = await db.scalars(
        select(UserIdentity.provider).where(UserIdentity.user_id == user_id)
    )
    return sorted(set(rows))


# --- Apple: the refresh token Apple requires us to revoke ---------------------


def _apple_key_ready() -> bool:
    return bool(settings.apple_team_id and settings.apple_key_id and settings.apple_private_key)


def _apple_client_secret(client_id: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "iss": settings.apple_team_id,
            "iat": now,
            "exp": now + 300,
            "aud": APPLE_ISSUER,
            "sub": client_id,
        },
        # Railway variables keep a pasted key's line breaks as "\n".
        (settings.apple_private_key or "").replace("\\n", "\n"),
        algorithm="ES256",
        headers={"kid": settings.apple_key_id},
    )


async def store_apple_refresh_token(
    db: AsyncSession, claims: SocialClaims, authorization_code: str | None
) -> None:
    """Exchange the sign-in's authorization code for the refresh token that
    revokes our access later. Best effort: signing in never fails over it."""
    if not authorization_code or not _apple_key_ready():
        return
    identity = await _identity(db, Provider.APPLE, claims.subject)
    if identity is None or identity.provider_refresh_token is not None:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(
                APPLE_TOKEN_URL,
                data={
                    "client_id": claims.audience,
                    "client_secret": _apple_client_secret(claims.audience),
                    "code": authorization_code,
                    "grant_type": "authorization_code",
                },
            )
        res.raise_for_status()
        refresh = res.json().get("refresh_token")
    except Exception:  # noqa: BLE001 — never fail a sign-in over this
        logger.warning("apple code exchange failed", exc_info=True)
        return
    if refresh:
        identity.provider_refresh_token = encrypt_json(
            {"refresh_token": refresh, "client_id": claims.audience}
        )
        await db.commit()


async def _revoke_apple(identity: UserIdentity) -> None:
    stored = decrypt_json(identity.provider_refresh_token) if identity.provider_refresh_token else None
    if not stored or not _apple_key_ready():
        return
    client_id = stored.get("client_id") or _split(settings.apple_bundle_ids)[0]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(
                APPLE_REVOKE_URL,
                data={
                    "client_id": client_id,
                    "client_secret": _apple_client_secret(client_id),
                    "token": stored["refresh_token"],
                    "token_type_hint": "refresh_token",
                },
            )
        res.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.warning("apple revoke failed for identity %s", identity.id, exc_info=True)


async def forget_identities(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Revoke what the providers granted and drop the account's identities.
    For deleting an account; the caller commits."""
    identities = list(
        await db.scalars(select(UserIdentity).where(UserIdentity.user_id == user_id))
    )
    for identity in identities:
        if identity.provider == Provider.APPLE.value:
            await _revoke_apple(identity)
        await db.delete(identity)


# --- Apple: server-to-server notifications -----------------------------------


class AppleEvent(StrEnum):
    EMAIL_DISABLED = "email-disabled"
    EMAIL_ENABLED = "email-enabled"
    CONSENT_REVOKED = "consent-revoked"
    ACCOUNT_DELETE = "account-delete"


async def handle_apple_notification(db: AsyncSession, payload: str) -> None:
    """Apple tells us a person stopped using Sign in with Apple for this app
    (or deleted their Apple Account): that identity goes, and so do the
    sessions it opened. Mail forwarding changes need nothing from us."""
    audiences = _split(settings.apple_bundle_ids)
    if not audiences:
        return
    claims = await _decode(
        payload, jwks_url=APPLE_JWKS_URL, audiences=audiences, issuers=APPLE_ISSUER
    )
    raw = claims.get("events")
    events = json.loads(raw) if isinstance(raw, str) else (raw or {})
    kind = events.get("type")
    subject = events.get("sub")
    if kind not in (AppleEvent.CONSENT_REVOKED, AppleEvent.ACCOUNT_DELETE) or not subject:
        return
    identity = await _identity(db, Provider.APPLE, str(subject))
    if identity is None:
        return
    await _revoke_sessions(db, identity.user_id)
    await db.delete(identity)
    await db.commit()
