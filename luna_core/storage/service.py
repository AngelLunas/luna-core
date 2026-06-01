"""Resolve a `BaseStorageBackend` from settings.

The host app calls `build_storage_backend(settings)` once at startup and
hands the resulting backend to whatever services need to read/write files.
We deliberately keep this thin — backend-specific kwargs come from a
small set of well-known settings; anything more exotic should be wired
by the host directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from luna_core.storage.base import BaseStorageBackend
from luna_core.storage.local import LocalStorageBackend
from luna_core.storage.r2 import R2StorageBackend
from luna_core.storage.s3 import S3StorageBackend

if TYPE_CHECKING:
    from luna_core.core.config import Settings


def build_storage_backend(settings: "Settings") -> BaseStorageBackend:
    backend = (settings.storage_backend or "local").lower()

    if backend == "local":
        return LocalStorageBackend(
            root_path=settings.storage_local_path,
            base_url=settings.storage_base_url or "/static/storage",
        )

    if backend == "s3":
        if not settings.storage_bucket:
            raise ValueError("storage_backend=s3 requires storage_bucket")
        return S3StorageBackend(
            bucket=settings.storage_bucket,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
            region=settings.storage_region,
            endpoint_url=settings.storage_endpoint_url,
            base_url=settings.storage_base_url,
        )

    if backend == "r2":
        if not settings.storage_bucket:
            raise ValueError("storage_backend=r2 requires storage_bucket")
        return R2StorageBackend(
            bucket=settings.storage_bucket,
            account_id=settings.storage_account_id,
            endpoint_url=settings.storage_endpoint_url,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
            base_url=settings.storage_base_url,
        )

    if backend == "gcs":
        raise NotImplementedError(
            "GCS backend not yet implemented — wire your own subclass of "
            "BaseStorageBackend and pass it directly."
        )

    raise ValueError(
        f"Unknown storage_backend {backend!r} — expected one of: local, s3, r2"
    )


__all__ = ["build_storage_backend"]
