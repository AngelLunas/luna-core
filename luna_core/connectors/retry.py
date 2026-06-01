"""Per-operation retry policy for transient HTTP failures.

Some upstreams (notably APIs behind a CDN or with autoscaled backends —
Upwork's GraphQL endpoint is the canonical example) return spurious 404s
or 5xx responses on a small percentage of requests even when the request
is well-formed and the auth is valid. The same request retried a few
hundred milliseconds later succeeds without any client-side change.

This module defines a small declarative policy so operators can mark such
operations as retry-on-transient without baking provider-specific
heuristics into the executor. Defaults are sized to mask ~1-2 transient
edge failures without adding noticeable latency on the happy path.

OAuth2 401/403 are handled separately in the executor's auth-refresh
path (always one refresh + retry) and are intentionally ignored if listed
in `retry_on_status` — recovering from a stale token needs a refresh, not
just a wait.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_DEFAULT_INITIAL_DELAY_MS = 200
_DEFAULT_MULTIPLIER = 3.0
_DEFAULT_JITTER = True
# Hard cap so a misconfigured multiplier can't sleep a request for minutes.
_MAX_DELAY_MS = 30_000

# Auth-flow statuses are routed through the OAuth2 refresh path, not the
# generic retry loop. Listing them in retry_on_status is silently ignored.
_AUTH_STATUSES = frozenset({401, 403})


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy for a connector operation."""

    max_attempts: int
    retry_on_status: frozenset[int]
    initial_delay_ms: int = _DEFAULT_INITIAL_DELAY_MS
    multiplier: float = _DEFAULT_MULTIPLIER
    jitter: bool = _DEFAULT_JITTER

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay_ms < 0:
            raise ValueError("initial_delay_ms must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def should_retry(self, status_code: int | None) -> bool:
        """True when `status_code` is configured for retry and not an auth code."""
        if status_code is None or status_code in _AUTH_STATUSES:
            return False
        return status_code in self.retry_on_status

    def delay_seconds_for_retry(self, retry_index: int) -> float:
        """Sleep before the Nth retry (1-based: 1 = first retry after the original)."""
        if retry_index <= 0:
            return 0.0
        base = self.initial_delay_ms * (self.multiplier ** (retry_index - 1))
        capped = min(base, _MAX_DELAY_MS)
        if self.jitter:
            # Full jitter (AWS best practice): random in [0, capped]. Spreads
            # concurrent retries so a thundering herd doesn't hammer an
            # already-flaky upstream in lockstep.
            capped = random.uniform(0, capped)
        return capped / 1000.0


def parse_retry_policy(raw: Mapping[str, Any] | None) -> RetryPolicy | None:
    """Build a RetryPolicy from a JSONB-stored dict. None when unset/empty."""
    if not raw:
        return None
    statuses = raw.get("retry_on_status") or []
    # Strip auth statuses on read — they'd otherwise be silently ignored at
    # retry time, which is more confusing than dropping them at parse time.
    cleaned_statuses = frozenset(
        int(s) for s in statuses if int(s) not in _AUTH_STATUSES
    )
    return RetryPolicy(
        max_attempts=int(raw.get("max_attempts", 1)),
        retry_on_status=cleaned_statuses,
        initial_delay_ms=int(raw.get("initial_delay_ms", _DEFAULT_INITIAL_DELAY_MS)),
        multiplier=float(raw.get("multiplier", _DEFAULT_MULTIPLIER)),
        jitter=bool(raw.get("jitter", _DEFAULT_JITTER)),
    )
