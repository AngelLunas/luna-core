"""Storage backends — opaque bytes addressed by key.

Public API:

    from luna_core.storage import BaseStorageBackend, build_storage_backend

Concrete backends (`LocalStorageBackend`, `S3StorageBackend`,
`R2StorageBackend`) are importable too if the host needs to instantiate
one directly without going through settings.
"""
from luna_core.storage.base import BaseStorageBackend
from luna_core.storage.local import LocalStorageBackend
from luna_core.storage.r2 import R2StorageBackend
from luna_core.storage.s3 import S3StorageBackend
from luna_core.storage.service import build_storage_backend

__all__ = [
    "BaseStorageBackend",
    "LocalStorageBackend",
    "R2StorageBackend",
    "S3StorageBackend",
    "build_storage_backend",
]
