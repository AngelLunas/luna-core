"""S3-compatible storage (AWS S3, Cloudflare R2, MinIO, Backblaze B2, etc.).

boto3 is imported lazily so installing luna-core without the `[s3]` extra does
not pay for the dependency. Sync boto3 calls run in a thread pool so the event
loop stays responsive.
"""
from __future__ import annotations

import asyncio
from typing import Any

from luna_core.storage.base import BaseStorageBackend


class S3StorageBackend(BaseStorageBackend):
    def __init__(
        self,
        *,
        bucket: str,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        base_url: str | None = None,
        url_signature_expiry_seconds: int = 3600,
    ):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "boto3 is required for S3StorageBackend; install luna-core "
                "with the [s3] extra or `pip install boto3`."
            ) from exc

        self._bucket = bucket
        self._base_url = base_url.rstrip("/") if base_url else None
        self._expiry = url_signature_expiry_seconds
        session_kwargs: dict[str, Any] = {}
        if access_key:
            session_kwargs["aws_access_key_id"] = access_key
        if secret_key:
            session_kwargs["aws_secret_access_key"] = secret_key
        if region:
            session_kwargs["region_name"] = region
        client_kwargs: dict[str, Any] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        self._client = boto3.session.Session(**session_kwargs).client(
            "s3", **client_kwargs
        )

    async def upload(self, file: bytes, path: str, mime_type: str) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=path,
            Body=file,
            ContentType=mime_type,
        )
        return path

    async def download(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()

        return await asyncio.to_thread(_get)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object, Bucket=self._bucket, Key=key
        )

    async def get_url(self, key: str) -> str:
        if self._base_url:
            return f"{self._base_url}/{key.lstrip('/')}"
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=self._expiry,
        )


__all__ = ["S3StorageBackend"]
