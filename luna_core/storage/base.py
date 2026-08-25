"""Storage abstraction.

Backends store opaque bytes addressed by an internal `key`. The key is what
the application persists; URLs are derived on demand and may rotate (signed
URLs, CDN swaps). `upload()` returns a key, not a URL — callers that need a
URL must call `get_url(key)`.
"""
from __future__ import annotations

import abc
import asyncio
from pathlib import Path


class BaseStorageBackend(abc.ABC):
    @abc.abstractmethod
    async def upload(self, file: bytes, path: str, mime_type: str) -> str:
        """Store bytes under `path` (used as the key). Returns the key."""

    @abc.abstractmethod
    async def download(self, key: str) -> bytes: ...

    # File-based variants for large objects (a 100 MB video must not sit in
    # RAM twice). The defaults go through bytes so every backend works; the
    # local backend overrides them with plain file copies.
    async def upload_path(self, path: Path, key: str, mime_type: str) -> str:
        data = await asyncio.to_thread(Path(path).read_bytes)
        return await self.upload(data, key, mime_type)

    async def download_to_path(self, key: str, dst: Path) -> None:
        data = await self.download(key)
        await asyncio.to_thread(Path(dst).write_bytes, data)

    def local_path(self, key: str) -> Path | None:
        """Where the object lives on THIS machine's filesystem, when it does
        (so a router can stream it with range support) — ``None`` for remote
        backends, whose ``get_url`` is the way to serve them."""
        return None

    @abc.abstractmethod
    async def delete(self, key: str) -> None: ...

    @abc.abstractmethod
    async def get_url(self, key: str) -> str: ...


__all__ = ["BaseStorageBackend"]
