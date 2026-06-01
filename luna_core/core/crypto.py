"""Symmetric encryption helpers for at-rest secrets (e.g. connector credentials)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from luna_core.core.config import settings


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = settings.encryption_key.encode("utf-8")
    return Fernet(key)


def encrypt_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _fernet().encrypt(raw).decode("utf-8")


def decrypt_json(token: str | None) -> dict[str, Any] | None:
    if token is None:
        return None
    try:
        raw = _fernet().decrypt(token.encode("utf-8"))
    except InvalidToken as exc:
        raise ValueError("invalid or tampered ciphertext") from exc
    return json.loads(raw.decode("utf-8"))


__all__ = ["encrypt_json", "decrypt_json"]
