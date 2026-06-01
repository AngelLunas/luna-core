"""Storage abstraction.

Backends store opaque bytes addressed by an internal `key`. The key is what
the application persists; URLs are derived on demand and may rotate (signed
URLs, CDN swaps). `upload()` returns a key, not a URL — callers that need a
URL must call `get_url(key)`.
"""
from __future__ import annotations

import abc


class BaseStorageBackend(abc.ABC):
    @abc.abstractmethod
    async def upload(self, file: bytes, path: str, mime_type: str) -> str:
        """Store bytes under `path` (used as the key). Returns the key."""

    @abc.abstractmethod
    async def download(self, key: str) -> bytes: ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None: ...

    @abc.abstractmethod
    async def get_url(self, key: str) -> str: ...


__all__ = ["BaseStorageBackend"]
