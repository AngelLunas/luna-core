from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Luna Core"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Database (host app supplies the URL)
    database_url: str = "postgresql+asyncpg://luna:luna@localhost:5432/luna_core"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis (rate limiting, pub/sub, Celery broker)
    redis_url: str = "redis://localhost:6379/0"

    # Security
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 180  # 3 hours
    refresh_token_expire_days: int = 30

    # Credential encryption (Fernet — base64-encoded 32-byte key)
    encryption_key: str = Field(min_length=32)

    # Cookies
    refresh_cookie_name: str = "luna_refresh_token"
    refresh_cookie_secure: bool = False
    refresh_cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    refresh_cookie_domain: str | None = None

    # Rate limiting
    login_rate_limit_attempts: int = 5
    login_rate_limit_window_seconds: int = 300

    # Email verification. Off by default in luna-core (a registered user is
    # immediately usable); host apps opt in via env or a Settings subclass. When
    # on, ``get_current_user`` returns 403 ``email_not_verified`` on protected
    # routes until the user confirms their address.
    email_verification_required: bool = False
    # Verification is a short numeric code typed into the app (no link). Brute
    # force is bounded by the per-code attempt cap plus this short expiry.
    email_verification_code_ttl_minutes: int = 30
    email_verification_max_attempts: int = 5

    # Celery
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_default_queue: str = "luna_core"

    # Pub/sub channel template for run events
    run_event_channel_prefix: str = "luna_core:run_events"

    # Default Ollama settings used by the dev seed to bootstrap a baseline
    # LLMProvider row (core.llm_providers). Real chat providers — credentials,
    # base URL, model availability — live in the database and are managed
    # through the `/llm-providers` endpoints.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    ollama_api_key: str = "ollama"

    # LLM rate limiting and retries
    llm_rate_limit_rpm: int = 60
    llm_rate_limit_window_seconds: int = 60
    llm_max_retries: int = 3
    llm_retry_base_delay_seconds: float = 1.0

    # Streaming + abort signal TTLs (Redis seconds)
    run_stream_key_ttl_seconds: int = 3600
    run_abort_key_ttl_seconds: int = 60

    # Iteration concurrency ceiling. The user-declared `concurrency` on a
    # scratchpad iteration node is clamped down to this value at dispatch
    # time so a misconfigured node can't blow past what the host hardware
    # (or the LLM provider's rate limit) can handle. Defaults to 8 — safe
    # for a Raspberry Pi 5 8GB with NVMe; raise to 20 on a real server via
    # env var `LUNA_ITERATION_CONCURRENCY_MAX`.
    iteration_concurrency_max: int = 8

    # Embedding service (OpenAI-compatible HTTP endpoint — Ollama / TEI / vLLM)
    # Defaults assume a local Ollama serving mxbai-embed-large (1024 dims).
    # Host apps override via env vars / their own Settings subclass.
    embedding_api_key: str | None = "ollama"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "mxbai-embed-large"
    embedding_dimensions: int = 1024

    # MCP Server
    mcp_server_url: str = "http://localhost:8765"

    # OAuth2 (authorization_code) callback URL — the dashboard hosts a page at
    # this URL that finishes the handshake. Must match what's registered in
    # each OAuth2 provider (e.g. Upwork, Google) byte-for-byte. Override per
    # environment with LUNA_OAUTH2_CALLBACK_URL or via host Settings subclass.
    oauth2_callback_url: str = "http://localhost:5173/connectors/oauth2/callback"
    # `state` JWT lifetime — the user has this long to complete the popup
    # handshake before the state token expires.
    oauth2_state_ttl_seconds: int = 600

    # Storage backend (local | s3 | r2)
    storage_backend: Literal["local", "s3", "r2", "gcs"] = "local"
    storage_local_path: str = "./.storage"
    storage_base_url: str | None = None  # used by local + as CDN base for s3/r2
    storage_bucket: str | None = None
    storage_region: str | None = None
    storage_endpoint_url: str | None = None
    storage_account_id: str | None = None  # Cloudflare R2
    storage_access_key: str | None = None
    storage_secret_key: str | None = None

    # CORS (host apps may pass through)
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("refresh_cookie_domain", mode="before")
    @classmethod
    def _empty_to_none(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        return v

    @property
    def effective_celery_broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_celery_result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
