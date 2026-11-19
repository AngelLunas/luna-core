"""Refresh token rotation: a session per sign-in, and a grace window for a
client that lost the answer to its own rotation."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from luna_core.core.security import hash_refresh_token
from luna_core.services import auth

pytestmark = pytest.mark.asyncio


class _Store:
    """Tokens in memory, behind the same helpers the service queries with."""

    def __init__(self) -> None:
        self.tokens: list = []
        self.user = SimpleNamespace(id=uuid.uuid4(), is_active=True)

    # AsyncSession surface used by the service
    def add(self, token) -> None:
        if token.revoked is None:
            token.revoked = False
        self.tokens.append(token)

    async def get(self, _model, _id):
        return self.user

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    # helpers patched into the service
    async def find(self, _db, token_hash):
        return next((t for t in self.tokens if t.token_hash == token_hash), None)

    async def is_live(self, _db, family_id) -> bool:
        return any(t.family_id == family_id and not t.revoked for t in self.tokens)

    async def revoke_family(self, _db, family_id, *, keep=None, commit=True) -> None:
        for t in self.tokens:
            if t.family_id == family_id and t.id != keep:
                t.revoked = True

    def by_raw(self, raw: str):
        h = hash_refresh_token(raw)
        return next(t for t in self.tokens if t.token_hash == h)


@pytest.fixture
def store(monkeypatch) -> _Store:
    s = _Store()
    monkeypatch.setattr(auth, "_find_refresh_token", s.find)
    monkeypatch.setattr(auth, "_family_is_live", s.is_live)
    monkeypatch.setattr(auth, "_revoke_family", s.revoke_family)
    return s


def _age_rotation(token, seconds: int) -> None:
    token.rotated_at = datetime.now(timezone.utc) - timedelta(seconds=seconds)


async def test_a_rotation_stays_in_the_sign_ins_family(store):
    first = await auth.issue_tokens(store, store.user)
    second = await auth.rotate_refresh_token(store, first.refresh_token)

    old, new = store.by_raw(first.refresh_token), store.by_raw(second.refresh_token)
    assert old.revoked and old.rotated_at is not None
    assert not new.revoked
    assert new.family_id == old.family_id == old.id


async def test_each_sign_in_opens_its_own_family(store):
    phone = await auth.issue_tokens(store, store.user)
    tablet = await auth.issue_tokens(store, store.user)
    assert store.by_raw(phone.refresh_token).family_id != store.by_raw(
        tablet.refresh_token
    ).family_id


async def test_a_reuse_right_after_rotating_gets_a_sibling(store):
    first = await auth.issue_tokens(store, store.user)
    lost = await auth.rotate_refresh_token(store, first.refresh_token)

    retry = await auth.rotate_refresh_token(store, first.refresh_token)

    sibling = store.by_raw(retry.refresh_token)
    assert not sibling.revoked
    assert sibling.family_id == store.by_raw(first.refresh_token).family_id
    # The answer the client lost is still valid until a sibling rotates.
    assert not store.by_raw(lost.refresh_token).revoked


async def test_rotating_the_kept_sibling_revokes_the_other(store):
    first = await auth.issue_tokens(store, store.user)
    lost = await auth.rotate_refresh_token(store, first.refresh_token)
    kept = await auth.rotate_refresh_token(store, first.refresh_token)

    await auth.rotate_refresh_token(store, kept.refresh_token)

    assert store.by_raw(lost.refresh_token).revoked


async def test_a_late_reuse_signs_out_that_device_only(store):
    phone = await auth.issue_tokens(store, store.user)
    tablet = await auth.issue_tokens(store, store.user)
    current = await auth.rotate_refresh_token(store, phone.refresh_token)
    _age_rotation(store.by_raw(phone.refresh_token), seconds=3600)

    with pytest.raises(HTTPException) as err:
        await auth.rotate_refresh_token(store, phone.refresh_token)

    assert err.value.status_code == 401
    assert store.by_raw(current.refresh_token).revoked
    assert not store.by_raw(tablet.refresh_token).revoked


async def test_an_expired_token_is_refused_without_revoking_anything(store):
    first = await auth.issue_tokens(store, store.user)
    other = await auth.issue_tokens(store, store.user)
    store.by_raw(first.refresh_token).expires_at = datetime.now(
        timezone.utc
    ) - timedelta(minutes=1)

    with pytest.raises(HTTPException) as err:
        await auth.rotate_refresh_token(store, first.refresh_token)

    assert err.value.status_code == 401
    assert not store.by_raw(other.refresh_token).revoked


async def test_the_grace_window_does_not_reopen_a_closed_session(store):
    first = await auth.issue_tokens(store, store.user)
    second = await auth.rotate_refresh_token(store, first.refresh_token)
    store.by_raw(second.refresh_token).revoked = True  # logout

    with pytest.raises(HTTPException) as err:
        await auth.rotate_refresh_token(store, first.refresh_token)
    assert err.value.status_code == 401
