"""Local-filesystem storage. Useful for dev and single-node deployments.

Files are written under `root_path / key`. URLs are produced by joining
`base_url` (typically the FastAPI static mount, e.g. `/static/storage`) with
the key — the host app is responsible for actually mounting that route at
`root_path`.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from luna_core.storage.base import BaseStorageBackend


class LocalStorageBackend(BaseStorageBackend):
    def __init__(self, root_path: str, base_url: str = "/static/storage"):
        self._root = Path(root_path).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url.rstrip("/")

    def _resolve(self, key: str) -> Path:
        target = (self._root / key).resolve()
        # prevent path traversal — the resolved path must remain under root
        if not str(target).startswith(str(self._root)):
            raise ValueError(f"key {key!r} resolves outside storage root")
        return target

    async def upload(self, file: bytes, path: str, mime_type: str) -> str:
        target = self._resolve(path)
        await asyncio.to_thread(_write_bytes, target, file)
        return path

    async def download(self, key: str) -> bytes:
        target = self._resolve(key)
        return await asyncio.to_thread(target.read_bytes)

    async def delete(self, key: str) -> None:
        target = self._resolve(key)
        await asyncio.to_thread(_safe_remove, target)

    async def get_url(self, key: str) -> str:
        return f"{self._base_url}/{key.lstrip('/')}"


def _write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        fh.write(data)


def _safe_remove(target: Path) -> None:
    try:
        os.remove(target)
    except FileNotFoundError:
        pass


__all__ = ["LocalStorageBackend"]
