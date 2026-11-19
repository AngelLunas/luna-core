"""Signing in with Google or Apple: which tokens pass, and which account each
identity reaches."""
from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from fastapi import HTTPException

from luna_core.core.config import settings
from luna_core.models.user import User
from luna_core.models.user_identity import UserIdentity
from luna_core.services import social_auth
from luna_core.services.social_auth import Provider, SocialClaims

pytestmark = pytest.mark.asyncio

GOOGLE_WEB = "web-client.apps.googleusercontent.com"
BUNDLE = "com.example.app"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(autouse=True)
def _providers(monkeypatch):
    monkeypatch.setattr(settings, "google_client_ids", f"{GOOGLE_WEB}, ios-client")
    monkeypatch.setattr(settings, "apple_bundle_ids", BUNDLE)
    signing = SimpleNamespace(
        get_signing_key_from_jwt=lambda _t: SimpleNamespace(key=_KEY.public_key())
    )
    monkeypatch.setattr(social_auth, "_jwks", lambda _url: signing)


def _token(**claims) -> str:
    now = int(time.time())
    body = {"iat": now, "exp": now + 600, "sub": "person-1", **claims}
    return jwt.encode(body, _KEY, algorithm="RS256")


# --- tokens -------------------------------------------------------------------


async def test_a_google_token_for_our_client_passes():
    claims = await social_auth.verify_google(
        _token(aud=GOOGLE_WEB, iss="https://accounts.google.com",
               email="a@b.com", email_verified=True)
    )
    assert (claims.provider, claims.subject, claims.email) == (
        Provider.GOOGLE, "person-1", "a@b.com"
    )
    assert claims.email_verified


async def test_a_google_token_for_another_app_is_refused():
    with pytest.raises(HTTPException) as err:
        await social_auth.verify_google(
            _token(aud="someone-else", iss="accounts.google.com", email="a@b.com")
        )
    assert err.value.status_code == 401


async def test_a_token_from_another_issuer_is_refused():
    with pytest.raises(HTTPException):
        await social_auth.verify_google(_token(aud=GOOGLE_WEB, iss="https://evil.example"))


async def test_a_token_signed_by_another_key_is_refused():
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(
        {"aud": GOOGLE_WEB, "iss": "accounts.google.com", "sub": "x",
         "iat": int(time.time()), "exp": int(time.time()) + 600},
        other, algorithm="RS256",
    )
    with pytest.raises(HTTPException):
        await social_auth.verify_google(forged)


async def test_an_apple_token_must_carry_the_apps_nonce():
    token = _token(aud=BUNDLE, iss="https://appleid.apple.com", nonce="n-1",
                   email="x@privaterelay.appleid.com", email_verified="true")
    claims = await social_auth.verify_apple(token, "n-1")
    assert claims.email_verified and claims.audience == BUNDLE
    with pytest.raises(HTTPException):
        await social_auth.verify_apple(token, "n-2")


async def test_a_provider_without_ids_is_off(monkeypatch):
    monkeypatch.setattr(settings, "google_client_ids", "")
    with pytest.raises(HTTPException) as err:
        await social_auth.verify_google(_token(aud=GOOGLE_WEB, iss="accounts.google.com"))
    assert err.value.status_code == 404


async def test_a_token_signs_in_once():
    class _Redis:
        seen: set = set()

        async def set(self, key, _v, nx, ex):
            if key in self.seen:
                return None
            self.seen.add(key)
            return True

    redis = _Redis()
    await social_auth.claim_token_once(redis, "tok")
    with pytest.raises(HTTPException):
        await social_auth.claim_token_once(redis, "tok")


# --- accounts -----------------------------------------------------------------


class _Db:
    def __init__(self) -> None:
        self.users: dict[uuid.UUID, User] = {}
        self.identities: list[UserIdentity] = []

    def add(self, obj) -> None:
        if isinstance(obj, User):
            obj.id = obj.id or uuid.uuid4()
            self.users[obj.id] = obj
        else:
            self.identities.append(obj)

    async def flush(self) -> None:
        for user in self.users.values():
            user.id = user.id or uuid.uuid4()

    async def commit(self) -> None:
        pass

    async def refresh(self, _obj) -> None:
        pass

    async def get(self, _model, key):
        return self.users.get(key)

    async def delete(self, obj) -> None:
        self.identities.remove(obj)

    def user(self, email: str, *, password: str | None = "hash", verified=True) -> User:
        user = User(id=uuid.uuid4(), email=email, password_hash=password,
                    is_active=True, is_verified=verified)
        self.users[user.id] = user
        return user


@pytest.fixture
def db(monkeypatch) -> _Db:
    store = _Db()

    async def identity(_db, provider, subject):
        return next((i for i in store.identities
                     if i.provider == provider.value and i.subject == subject), None)

    async def by_email(_db, email):
        return next((u for u in store.users.values() if u.email == email), None)

    async def revoke(_db, _user_id):
        pass

    monkeypatch.setattr(social_auth, "_identity", identity)
    monkeypatch.setattr(social_auth, "_user_by_email", by_email)
    monkeypatch.setattr(social_auth, "_revoke_sessions", revoke)
    return store


def _claims(email="grower@mail.com", verified=True, subject="g-1",
            provider=Provider.GOOGLE) -> SocialClaims:
    return SocialClaims(provider=provider, subject=subject, email=email,
                        email_verified=verified, audience=GOOGLE_WEB)


async def test_a_new_person_gets_a_verified_account_without_password(db):
    user = await social_auth.sign_in(db, _claims())
    assert user.email == "grower@mail.com"
    assert user.is_verified and user.password_hash is None
    assert [(i.provider, i.subject) for i in db.identities] == [("google", "g-1")]


async def test_the_same_email_reaches_the_account_it_already_has(db):
    existing = db.user("grower@mail.com")
    user = await social_auth.sign_in(db, _claims(email="Grower@Mail.com "))
    assert user is existing
    assert existing.password_hash == "hash"  # the password still works
    assert db.identities[0].user_id == existing.id


async def test_the_identity_finds_its_account_even_with_another_email(db):
    existing = db.user("grower@mail.com")
    await social_auth.sign_in(db, _claims())
    again = await social_auth.sign_in(db, _claims(email="new-address@mail.com"))
    assert again is existing


async def test_an_unproven_signup_loses_its_password_to_the_real_owner(db):
    squatter = db.user("grower@mail.com", password="their-hash", verified=False)
    user = await social_auth.sign_in(db, _claims())
    assert user is squatter
    assert user.password_hash is None and user.is_verified


async def test_an_unverified_provider_email_does_not_take_an_account(db):
    db.user("grower@mail.com")
    with pytest.raises(HTTPException) as err:
        await social_auth.sign_in(db, _claims(verified=False))
    assert err.value.status_code == 409


async def test_apple_revoking_consent_drops_the_identity(db, monkeypatch):
    user = db.user("x@privaterelay.appleid.com", password=None)
    db.identities.append(UserIdentity(user_id=user.id, provider="apple", subject="a-1"))
    payload = _token(aud=BUNDLE, iss="https://appleid.apple.com",
                     events=json.dumps({"type": "consent-revoked", "sub": "a-1"}))
    await social_auth.handle_apple_notification(db, payload)
    assert db.identities == []


async def test_the_apple_client_secret_is_signed_with_the_key(monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()
    monkeypatch.setattr(settings, "apple_team_id", "TEAM123456")
    monkeypatch.setattr(settings, "apple_key_id", "KEY1234567")
    # As Railway keeps it: the line breaks escaped.
    monkeypatch.setattr(settings, "apple_private_key", pem.replace("\n", "\\n"))
    secret = social_auth._apple_client_secret(BUNDLE)
    claims = jwt.decode(secret, key.public_key(), algorithms=["ES256"],
                        audience="https://appleid.apple.com")
    assert claims["iss"] == "TEAM123456" and claims["sub"] == BUNDLE
    assert jwt.get_unverified_header(secret)["kid"] == "KEY1234567"
