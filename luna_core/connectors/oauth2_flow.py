"""OAuth2 authorization_code helpers — handshake start + code exchange.

Used by the connector router to drive the popup-based OAuth2 flow:

  1. /oauth2/start  → `build_authorize_url(...)`  (returns the IdP URL).
  2. (user authorizes in the popup, IdP redirects to our callback URL.)
  3. /oauth2/callback → `exchange_code_for_tokens(...)`  (swaps code → tokens).

The `state` parameter is a short-lived JWT signed with the same key as
access tokens — that way the callback request is stateless. The JWT carries
the `connector_id` so the callback knows which conector to write tokens to.
PKCE is deliberately omitted: confidential clients (with a server-side
`client_secret`) get equivalent protection from the secret itself, and
adding PKCE would force us to persist the verifier server-side, defeating
the stateless `state` design. If we ever need to support public clients,
this is the place to wire it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from luna_core.connectors.auth import ConnectorAuthError
from luna_core.core.config import settings


_STATE_SCOPE = "oauth2_state"


class OAuth2StateError(ValueError):
    """Raised when an OAuth2 `state` token is missing / expired / invalid."""


def make_state(
    connector_id: uuid.UUID, *, ttl_seconds: int | None = None
) -> str:
    """Mint a signed `state` token for an in-progress OAuth2 handshake.

    Signed with `jwt_secret_key` — anyone holding that secret can forge a
    callback, which is the same trust boundary as session tokens. The token
    embeds the target `connector_id` and a short TTL so the popup has to
    complete within `oauth2_state_ttl_seconds` (default 10 min).
    """
    now = datetime.now(timezone.utc)
    exp = now + timedelta(
        seconds=ttl_seconds
        if ttl_seconds is not None
        else settings.oauth2_state_ttl_seconds
    )
    payload = {
        "sub": str(connector_id),
        "scope": _STATE_SCOPE,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def verify_state(token: str) -> uuid.UUID:
    """Verify a `state` token and return the connector_id it points to."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError as exc:
        raise OAuth2StateError(
            "oauth2 state expired — restart the connection flow"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise OAuth2StateError("invalid oauth2 state") from exc
    if payload.get("scope") != _STATE_SCOPE:
        raise OAuth2StateError("state token is not an oauth2 handshake token")
    sub = payload.get("sub")
    if not isinstance(sub, str):
        raise OAuth2StateError("oauth2 state missing connector id")
    try:
        return uuid.UUID(sub)
    except ValueError as exc:
        raise OAuth2StateError("oauth2 state has malformed connector id") from exc


def build_authorize_url(
    *,
    authorize_url: str,
    client_id: str,
    redirect_uri: str,
    scope: str | None,
    state: str,
) -> str:
    """Build the URL the user-agent visits to begin the IdP handshake."""
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if scope:
        params["scope"] = scope
    sep = "&" if "?" in authorize_url else "?"
    return f"{authorize_url}{sep}{urlencode(params)}"


async def exchange_code_for_tokens(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """POST to the IdP's token endpoint to swap an auth_code for tokens.

    Returns a credentials patch (`access_token`, optional `refresh_token`,
    optional `expires_at`). The caller merges this into the conector's
    stored credentials and persists.
    """
    # Send client credentials BOTH in the body and via HTTP Basic. RFC 6749
    # §2.3.1 specifies Basic but real providers diverge — some (Upwork)
    # demand body, others (RFC-strict ones) only accept Basic. Including
    # both keeps the request portable without per-provider configuration.
    # The User-Agent dodges Cloudflare's "looks like a bot" heuristic.
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {
        "Accept": "application/json",
        "User-Agent": "luna-core/oauth2",
    }
    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=15.0)
    try:
        response = await client.post(
            token_url,
            data=form,
            headers=headers,
            auth=(client_id, client_secret),
        )
    finally:
        if owned_client:
            await client.aclose()

    if response.status_code >= 400:
        raise ConnectorAuthError(
            f"oauth2 code exchange failed: HTTP {response.status_code} "
            f"{response.text[:300]}"
        )

    body = response.json()
    access_token = body.get("access_token")
    if not access_token:
        raise ConnectorAuthError(
            "oauth2 code exchange returned no access_token"
        )
    patch: dict[str, Any] = {"access_token": access_token}
    if body.get("refresh_token"):
        patch["refresh_token"] = body["refresh_token"]
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        patch["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        ).isoformat()
    return patch


__all__ = [
    "OAuth2StateError",
    "make_state",
    "verify_state",
    "build_authorize_url",
    "exchange_code_for_tokens",
]
