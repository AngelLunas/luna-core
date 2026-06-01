"""Outbound auth strategies for connector HTTP requests.

Every active `Connector` row has an `auth_type` that determines how the
registry attaches credentials to the outbound request. This module is the
single source of truth for that mapping so each strategy can evolve
independently (header conventions, token refresh, etc.) without polluting
the registry's execute() path.

Supported flavors:

  - **none**: no auth applied.
  - **api_key**: a static secret carried by the credentials blob. Three
    placements are accepted:
        scheme=bearer (default): `Authorization: Bearer <token>`
        scheme=header:           custom header like `X-API-Key: <token>`
        scheme=query:            appended as a query parameter
  - **basic**: HTTP Basic with `username` + `password` from the blob.
  - **oauth2**: standard refresh-token grant. If the cached `access_token`
    is missing or expired, we hit the connector's `token_url` with the
    `refresh_token` (or `client_credentials` grant when no refresh token
    exists), persist the new tokens back to the DB, and reuse them.

The credentials JSON shape per flavor is documented in `prepare()`'s
docstring so connector authors know what to seed.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.core.crypto import encrypt_json
from luna_core.models.connector import AuthType, Connector

logger = logging.getLogger(__name__)


# Tokens are refreshed slightly before their nominal expiry so a concurrent
# request that races a refresh doesn't get blocked behind a fresh token call.
_REFRESH_LEEWAY = timedelta(seconds=60)


# Per-connector locks gate parallel OAuth2 refreshes so two concurrent
# requests don't both hit the IdP and persist clobbering token blobs.
_refresh_locks: dict[Any, asyncio.Lock] = {}


@dataclass
class PreparedRequest:
    """Output of `prepare()` — the additions an authenticator applies."""

    headers: dict[str, str]
    params: dict[str, str]
    basic_auth: tuple[str, str] | None = None
    # When non-None, the registry should persist this back to the connector
    # row (encrypted) and update its in-memory copy. Currently only OAuth2
    # uses this path, after a token refresh.
    refreshed_credentials: dict[str, Any] | None = None


class ConnectorAuthError(RuntimeError):
    """Raised when credentials are missing/malformed for the given auth_type."""


async def prepare(
    connector: Connector,
    credentials: Mapping[str, Any] | None,
    *,
    db: AsyncSession | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> PreparedRequest:
    """Build the auth additions for a single outbound call.

    Parameters
    ----------
    connector:
        The connector row whose `auth_type` selects the strategy.
    credentials:
        Already-decrypted credentials blob. May be empty for `auth_type=none`.
    db:
        Optional async session used to persist refreshed OAuth2 tokens. If
        omitted, refreshed tokens are returned in `refreshed_credentials`
        but not written to the DB — the caller is responsible.
    http_client:
        Reusable httpx client. If omitted a short-lived one is created for
        the OAuth2 token call.

    Returns
    -------
    A `PreparedRequest` whose `headers` / `params` should be merged into
    the outbound request and whose `basic_auth`, if set, replaces httpx's
    `auth` argument.

    Expected `credentials` shape per auth_type
    ------------------------------------------
    api_key:
        {"token": "...", "scheme": "bearer"}                 # default
        {"token": "...", "scheme": "header", "header_name": "X-API-Key"}
        {"token": "...", "scheme": "query",  "param_name":  "api_key"}
    basic:
        {"username": "...", "password": "..."}
    oauth2:
        {
            "access_token":  "...",
            "refresh_token": "...",          # optional; if absent we use
                                             # client_credentials grant
            "expires_at":    "ISO-8601",     # optional
            "token_url":     "https://...",  # required to refresh
            "client_id":     "...",
            "client_secret": "...",
            "scope":         "read write",   # optional
        }
    none:
        {}  (any shape ignored)
    """
    creds = dict(credentials or {})
    match connector.auth_type:
        case AuthType.none:
            return PreparedRequest(headers={}, params={})
        case AuthType.api_key:
            return _prepare_api_key(creds)
        case AuthType.basic:
            return _prepare_basic(creds)
        case AuthType.oauth2:
            return await _prepare_oauth2(
                connector, creds, db=db, http_client=http_client
            )
    raise ConnectorAuthError(f"unsupported auth_type: {connector.auth_type}")


# ---------------------------------------------------------------------------
# api_key
# ---------------------------------------------------------------------------


def _prepare_api_key(creds: dict[str, Any]) -> PreparedRequest:
    token = creds.get("token") or creds.get("api_key")
    if not token:
        raise ConnectorAuthError(
            "api_key auth requires credentials.token (or .api_key)"
        )
    scheme = (creds.get("scheme") or "bearer").lower()

    if scheme == "bearer":
        return PreparedRequest(
            headers={"Authorization": f"Bearer {token}"}, params={}
        )
    if scheme == "header":
        header_name = creds.get("header_name") or "X-API-Key"
        return PreparedRequest(
            headers={str(header_name): str(token)}, params={}
        )
    if scheme == "query":
        param_name = creds.get("param_name") or "api_key"
        return PreparedRequest(
            headers={}, params={str(param_name): str(token)}
        )
    raise ConnectorAuthError(
        f"unknown api_key scheme '{scheme}' (expected bearer|header|query)"
    )


# ---------------------------------------------------------------------------
# basic
# ---------------------------------------------------------------------------


def _prepare_basic(creds: dict[str, Any]) -> PreparedRequest:
    username = creds.get("username")
    password = creds.get("password")
    if username is None or password is None:
        raise ConnectorAuthError(
            "basic auth requires credentials.username and credentials.password"
        )
    return PreparedRequest(
        headers={},
        params={},
        basic_auth=(str(username), str(password)),
    )


# ---------------------------------------------------------------------------
# oauth2
# ---------------------------------------------------------------------------


async def _prepare_oauth2(
    connector: Connector,
    creds: dict[str, Any],
    *,
    db: AsyncSession | None,
    http_client: httpx.AsyncClient | None,
) -> PreparedRequest:
    if _access_token_is_fresh(creds):
        return PreparedRequest(
            headers={"Authorization": f"Bearer {creds['access_token']}"},
            params={},
        )

    if not creds.get("token_url"):
        raise ConnectorAuthError(
            "oauth2 auth requires credentials.token_url to refresh"
        )

    lock = _refresh_locks.setdefault(connector.id, asyncio.Lock())
    async with lock:
        # Re-check inside the lock: a sibling coroutine may have refreshed
        # already while we were waiting.
        if _access_token_is_fresh(creds):
            return PreparedRequest(
                headers={"Authorization": f"Bearer {creds['access_token']}"},
                params={},
            )

        new_tokens = await _oauth2_refresh(creds, http_client=http_client)

    merged = {**creds, **new_tokens}
    if db is not None:
        await _persist_credentials(db, connector, merged)
        refreshed_for_caller = None
    else:
        refreshed_for_caller = merged

    return PreparedRequest(
        headers={"Authorization": f"Bearer {merged['access_token']}"},
        params={},
        refreshed_credentials=refreshed_for_caller,
    )


def _access_token_is_fresh(creds: Mapping[str, Any]) -> bool:
    token = creds.get("access_token")
    if not token:
        return False
    expires_at_raw = creds.get("expires_at")
    if not expires_at_raw:
        # No expiry tracked — optimistically use it. If the IdP rejects the
        # token, the surrounding HTTP call will surface a 401 and a future
        # iteration can wire a retry-on-401 here.
        return True
    try:
        expires_at = _parse_datetime(expires_at_raw)
    except ValueError:
        return False
    return datetime.now(timezone.utc) + _REFRESH_LEEWAY < expires_at


async def _oauth2_refresh(
    creds: Mapping[str, Any],
    *,
    http_client: httpx.AsyncClient | None,
) -> dict[str, Any]:
    """Hit the IdP's token endpoint and return a partial credentials patch.

    Uses `refresh_token` grant when a refresh token is present; falls back
    to `client_credentials` otherwise. The IdP response is normalized into
    fields the rest of this module understands: `access_token`,
    `refresh_token` (if rotated), `expires_at`.
    """
    token_url = str(creds["token_url"])
    refresh_token = creds.get("refresh_token")
    client_id = creds.get("client_id")
    client_secret = creds.get("client_secret")
    scope = creds.get("scope")

    if refresh_token:
        form: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": str(refresh_token),
        }
    else:
        form = {"grant_type": "client_credentials"}
    if scope:
        form["scope"] = str(scope)
    # Send client credentials BOTH in the body and via HTTP Basic. RFC 6749
    # says Basic is the canonical method (§2.3.1), but real-world providers
    # disagree: Upwork's token endpoint demands client_id/client_secret in
    # the body and errors out otherwise; others only read Basic. Including
    # both keeps the request portable without per-provider config. The
    # User-Agent dodges Cloudflare's "looks like a bot" heuristic.
    if client_id:
        form["client_id"] = str(client_id)
    if client_secret:
        form["client_secret"] = str(client_secret)
    headers = {
        "Accept": "application/json",
        "User-Agent": "luna-core/oauth2",
    }
    auth = (
        (str(client_id), str(client_secret))
        if client_id and client_secret
        else None
    )

    owned_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=15.0)
    try:
        response = await client.post(
            token_url, data=form, headers=headers, auth=auth
        )
    finally:
        if owned_client:
            await client.aclose()

    if response.status_code >= 400:
        raise ConnectorAuthError(
            f"oauth2 token refresh failed: HTTP {response.status_code} "
            f"{response.text[:300]}"
        )

    body = response.json()
    patch: dict[str, Any] = {"access_token": body["access_token"]}
    if body.get("refresh_token"):
        patch["refresh_token"] = body["refresh_token"]
    expires_in = body.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        patch["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        ).isoformat()
    return patch


async def force_refresh_oauth2(
    connector: Connector,
    credentials: Mapping[str, Any],
    *,
    db: AsyncSession | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Refresh OAuth2 tokens *regardless* of the cached `expires_at`.

    Used by the registry's retry-on-401 path when the upstream rejects what
    we thought was a fresh token (revoked, clock skew, IdP-side wipe).
    Acquires the per-connector lock so concurrent retries collapse onto a
    single refresh round-trip.

    Returns the merged credentials (existing + new tokens) and persists them
    to the DB when `db` is supplied — mutating the connector row in place
    so the registry's cached `credentials_encrypted` is updated too.
    """
    creds = dict(credentials or {})
    if not creds.get("token_url"):
        raise ConnectorAuthError(
            "oauth2 force-refresh requires credentials.token_url"
        )
    lock = _refresh_locks.setdefault(connector.id, asyncio.Lock())
    async with lock:
        new_tokens = await _oauth2_refresh(creds, http_client=http_client)
    merged = {**creds, **new_tokens}
    if db is not None:
        await _persist_credentials(db, connector, merged)
    return merged


async def _persist_credentials(
    db: AsyncSession, connector: Connector, credentials: dict[str, Any]
) -> None:
    """Encrypt the (refreshed) credentials and write them back to the DB.

    Two things happen here, deliberately decoupled:

    1) The in-memory ORM instance gets its `credentials_encrypted` mutated
       so the registry cache sees the new tokens on the next access
       without re-fetching from the DB.

    2) The actual DB write goes through a `core.connectors` UPDATE keyed
       by id. We do NOT call `db.add(connector)` because the same ORM
       object may already be attached to a different session (e.g. the
       request session that loaded it via `ensure_registered`) — SQLAlchemy
       refuses to re-attach across live sessions, which broke the
       "test operation" flow for OAuth2 conectores. A direct UPDATE
       sidesteps the identity-map check.
    """
    ciphertext = encrypt_json(credentials)
    connector.credentials_encrypted = ciphertext
    try:
        await db.execute(
            update(Connector)
            .where(Connector.id == connector.id)
            .values(credentials_encrypted=ciphertext)
        )
        await db.commit()
    except Exception:  # noqa: BLE001
        await db.rollback()
        raise


def _parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    text = str(raw)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


__all__ = [
    "ConnectorAuthError",
    "PreparedRequest",
    "prepare",
    "force_refresh_oauth2",
]
