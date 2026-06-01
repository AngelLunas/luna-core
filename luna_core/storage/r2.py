"""Cloudflare R2 storage — S3-compatible, just preconfigured endpoint URL.

R2 buckets are addressed as `https://<account_id>.r2.cloudflarestorage.com`.
Pass either `account_id` (we build the endpoint) or a full `endpoint_url`.
"""
from __future__ import annotations

from luna_core.storage.s3 import S3StorageBackend


class R2StorageBackend(S3StorageBackend):
    def __init__(
        self,
        *,
        bucket: str,
        account_id: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        url_signature_expiry_seconds: int = 3600,
    ):
        if endpoint_url is None:
            if not account_id:
                raise ValueError(
                    "R2StorageBackend requires either account_id or endpoint_url"
                )
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

        super().__init__(
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            region="auto",
            endpoint_url=endpoint_url,
            base_url=base_url,
            url_signature_expiry_seconds=url_signature_expiry_seconds,
        )


__all__ = ["R2StorageBackend"]
