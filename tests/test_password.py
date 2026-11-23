"""Forgetting and changing a password: what each path sets, what it refuses,
and that every old session closes."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.core.security import hash_password, verify_password
from luna_core.models.email_verification_code import CodePurpose
from luna_core.schemas.user import UserRead
from luna_core.services import password as passwords
from luna_core.services.password import PasswordRefusal
from luna_core.services.verification import CodeCheck

pytestmark = pytest.mark.asyncio


class _Db:
    async def commit(self) -> None:
        pass


@pytest.fixture
def sessions(monkeypatch) -> list[uuid.UUID]:
    revoked: list[uuid.UUID] = []

    async def revoke(_db, user_id):
        revoked.append(user_id)

    monkeypatch.setattr(passwords, "_revoke_sessions", revoke)
    return revoked


def _user(password: str | None = "old-password", verified: bool = True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        email="grower@mail.com",
        password_hash=hash_password(password) if password else None,
        is_verified=verified,
        verified_at=None,
    )


def _code_check(monkeypatch, result: CodeCheck, seen: list | None = None):
    async def check(_db, _user, purpose, raw):
        if seen is not None:
            seen.append((purpose, raw))
        return result

    monkeypatch.setattr(passwords, "check_code", check)


async def test_the_right_current_password_sets_the_new_one(sessions):
    user = _user()
    refusal = await passwords.change_password(_Db(), user, "old-password", "new-password")
    assert refusal is None
    assert verify_password("new-password", user.password_hash)
    assert sessions == [user.id]


async def test_a_wrong_current_password_changes_nothing(sessions):
    user = _user()
    before = user.password_hash
    refusal = await passwords.change_password(_Db(), user, "guess", "new-password")
    assert refusal is PasswordRefusal.WRONG_PASSWORD
    assert user.password_hash == before
    assert sessions == []


async def test_an_account_without_password_has_none_to_change(sessions):
    refusal = await passwords.change_password(_Db(), _user(None), "x", "new-password")
    assert refusal is PasswordRefusal.PASSWORD_NOT_SET
    assert sessions == []


async def test_the_reset_code_sets_the_password_and_closes_sessions(monkeypatch, sessions):
    seen: list = []
    _code_check(monkeypatch, CodeCheck.MATCHED, seen)
    user = _user()
    check = await passwords.reset_password(_Db(), user, "123456", "new-password")
    assert check is CodeCheck.MATCHED
    assert seen == [(CodePurpose.PASSWORD_RESET, "123456")]
    assert verify_password("new-password", user.password_hash)
    assert sessions == [user.id]


async def test_the_reset_code_also_confirms_the_address(monkeypatch, sessions):
    _code_check(monkeypatch, CodeCheck.MATCHED)
    user = _user(verified=False)
    await passwords.reset_password(_Db(), user, "123456", "new-password")
    assert user.is_verified and user.verified_at is not None


@pytest.mark.parametrize(
    "result", [CodeCheck.INVALID, CodeCheck.EXPIRED, CodeCheck.NO_CODE, CodeCheck.TOO_MANY_ATTEMPTS]
)
async def test_a_refused_code_leaves_the_password(monkeypatch, sessions, result):
    _code_check(monkeypatch, result)
    user = _user()
    before = user.password_hash
    assert await passwords.reset_password(_Db(), user, "000000", "new-password") is result
    assert user.password_hash == before
    assert sessions == []


async def test_a_reset_gives_a_social_account_a_password(monkeypatch, sessions):
    _code_check(monkeypatch, CodeCheck.MATCHED)
    user = _user(None)
    await passwords.reset_password(_Db(), user, "123456", "new-password")
    assert verify_password("new-password", user.password_hash)


async def test_the_account_says_whether_it_has_a_password():
    from datetime import datetime, timezone

    from luna_core.models.user import User

    base = dict(id=uuid.uuid4(), email="a@b.com", role="user", is_active=True,
                is_verified=True, created_at=datetime.now(timezone.utc))
    assert UserRead.model_validate(User(**base, password_hash="h")).has_password is True
    assert UserRead.model_validate(User(**base, password_hash=None)).has_password is False
