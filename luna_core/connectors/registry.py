"""In-memory registry of active connectors + outbound HTTP executor.

Maintains a process-local cache of every active `Connector` + `Operation`
row so the engine resolves operation ids and performs outbound calls
without hitting Postgres on every request. `load_from_db()` repopulates
the cache; production hosts call it on startup and after admin changes.

`execute()` performs the actual HTTP request via httpx. Auth handling lives
in `luna_core.connectors.auth` so each strategy (api_key, basic, OAuth2
with refresh) can evolve without bloating this module. Path params declared
as `{name}` in `operation.path` are substituted from `input_data` and
removed from the payload; remaining keys go to the query string
(GET/DELETE) or the JSON body (POST/PUT/PATCH).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from luna_core.connectors.auth import (
    ConnectorAuthError,
    PreparedRequest,
    force_refresh_oauth2,
    prepare as prepare_auth,
)
from luna_core.connectors.retry import RetryPolicy, parse_retry_policy
from luna_core.core.crypto import decrypt_json
from luna_core.models.connector import AuthType, Connector, HTTPMethod, Operation

logger = logging.getLogger(__name__)


_PATH_PARAM_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
# A whole-value placeholder like "{user_id}" — when the entire string is just
# one placeholder we substitute the *typed* value (so numbers stay numbers).
_FULL_PARAM_RE = re.compile(r"^\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_BODY_METHODS = {HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH}
_DEFAULT_TIMEOUT_SECONDS = 30.0


class ConnectorExecutionError(RuntimeError):
    """Raised when a connector HTTP call fails (transport, 4xx, or 5xx)."""


# A `DbSessionFactory` is supplied to the registry so OAuth2 refreshes can
# persist new tokens without the caller having to thread a session through
# `execute()`. The host wires this from its own session factory at startup.
DbSessionFactory = Callable[[], "AsyncSession | Awaitable[AsyncSession]"]


@dataclass
class RegisteredConnector:
    connector: Connector
    operations: dict[uuid.UUID, Operation] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationCallResult:
    """Result of running an operation against its upstream service.

    Captures the full HTTP exchange — including 4xx/5xx responses —
    without raising, so callers (e.g. the test-from-UI endpoint) can
    render error responses verbatim instead of swallowing them.
    """

    ok: bool
    status_code: int | None
    latency_ms: int
    request_method: str
    request_url: str
    response: dict[str, Any] | None = None
    error: str | None = None


class ConnectorRegistry:
    def __init__(
        self,
        db_session_factory: DbSessionFactory | None = None,
    ) -> None:
        self._connectors: dict[uuid.UUID, RegisteredConnector] = {}
        self._operations: dict[uuid.UUID, uuid.UUID] = {}  # op_id -> connector_id
        self._lock = asyncio.Lock()
        self._db_session_factory = db_session_factory

    def set_db_session_factory(self, factory: DbSessionFactory) -> None:
        """Wire a session factory so OAuth2 refreshes can persist tokens.

        The factory must return an `AsyncSession` ready to be used as a
        context manager (e.g. `AsyncSessionLocal` from luna-core's db
        module).
        """
        self._db_session_factory = factory

    async def load_from_db(self, db: AsyncSession) -> None:
        async with self._lock:
            self._connectors.clear()
            self._operations.clear()

            connectors = (
                await db.execute(select(Connector).where(Connector.is_active.is_(True)))
            ).scalars().all()
            for connector in connectors:
                self._connectors[connector.id] = RegisteredConnector(connector=connector)

            operations = (
                await db.execute(
                    select(Operation).where(Operation.is_active.is_(True))
                )
            ).scalars().all()
            for operation in operations:
                bucket = self._connectors.get(operation.connector_id)
                if bucket is None:
                    continue
                bucket.operations[operation.id] = operation
                self._operations[operation.id] = operation.connector_id

    def register(
        self, connector: Connector, operations: list[Operation] | None = None
    ) -> None:
        bucket = self._connectors.setdefault(
            connector.id, RegisteredConnector(connector=connector)
        )
        bucket.connector = connector
        for op in operations or []:
            bucket.operations[op.id] = op
            self._operations[op.id] = connector.id

    def unregister_operation(self, operation_id: uuid.UUID) -> None:
        """Drop an operation from the in-memory cache. No-op if missing."""
        connector_id = self._operations.pop(operation_id, None)
        if connector_id is None:
            return
        bucket = self._connectors.get(connector_id)
        if bucket is not None:
            bucket.operations.pop(operation_id, None)

    def unregister_connector(self, connector_id: uuid.UUID) -> None:
        """Drop a connector and all of its operations from the cache."""
        bucket = self._connectors.pop(connector_id, None)
        if bucket is None:
            return
        for op_id in list(bucket.operations.keys()):
            self._operations.pop(op_id, None)

    async def ensure_registered(
        self, db: AsyncSession, operation_id: uuid.UUID
    ) -> Operation:
        """Return the operation, loading + registering it from DB if missing.

        Used by the test endpoint so users can exercise operations whose
        connector is inactive (otherwise `load_from_db` would have skipped
        them on startup).
        """
        if operation_id in self._operations:
            return self.get_operation(operation_id)
        result = await db.execute(
            select(Operation)
            .where(Operation.id == operation_id)
            .options(selectinload(Operation.connector))
        )
        operation = result.scalar_one_or_none()
        if operation is None:
            raise KeyError(f"operation {operation_id} not found")
        self.register(operation.connector, [operation])
        return operation

    def get_operation(self, operation_id: uuid.UUID) -> Operation:
        connector_id = self._operations.get(operation_id)
        if connector_id is None:
            raise KeyError(f"operation {operation_id} not registered")
        return self._connectors[connector_id].operations[operation_id]

    def get_connector_for(self, operation_id: uuid.UUID) -> Connector:
        connector_id = self._operations.get(operation_id)
        if connector_id is None:
            raise KeyError(f"operation {operation_id} not registered")
        return self._connectors[connector_id].connector

    def credentials_for(self, connector_id: uuid.UUID) -> dict[str, Any] | None:
        bucket = self._connectors.get(connector_id)
        if bucket is None:
            return None
        return decrypt_json(bucket.connector.credentials_encrypted)

    async def execute(
        self, operation_id: uuid.UUID, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the operation, raising on transport errors and HTTP 4xx/5xx.

        This is the contract the FlowRunner / MCP tools rely on — they want
        a clean dict on success and an exception on any failure. Test/UI
        callers should use `perform()` instead so they can render error
        responses verbatim.
        """
        result = await self.perform(operation_id, input_data)
        operation = self.get_operation(operation_id)
        connector = self.get_connector_for(operation_id)
        if result.error is not None:
            raise ConnectorExecutionError(
                f"connector {connector.name}.{operation.name} "
                f"transport error: {result.error}"
            )
        if result.status_code is not None and result.status_code >= 400:
            preview = ""
            if result.response is not None:
                preview = str(result.response)[:400]
            raise ConnectorExecutionError(
                f"connector {connector.name}.{operation.name} returned "
                f"HTTP {result.status_code}: {preview}"
            )
        return result.response or {}

    async def perform(
        self, operation_id: uuid.UUID, input_data: dict[str, Any]
    ) -> OperationCallResult:
        """Run the operation and return a structured result without raising.

        Captures transport errors and any HTTP status (including 4xx/5xx)
        so callers can inspect the full exchange. Path-parameter errors
        and unknown operation ids still raise (those are programmer errors,
        not upstream failures).
        """
        operation = self.get_operation(operation_id)
        connector = self.get_connector_for(operation_id)
        credentials = self.credentials_for(connector.id) or {}

        return await self._execute_http(
            connector=connector,
            credentials=credentials,
            method=operation.method,
            path=operation.path,
            parameters=list(operation.parameters or []),
            fixed_headers=dict(operation.fixed_headers or {}),
            fixed_body=operation.fixed_body,
            retry_policy=parse_retry_policy(operation.retry_policy),
            input_data=input_data,
        )

    async def perform_draft(
        self,
        *,
        connector: Connector,
        method: HTTPMethod | str,
        path: str,
        parameters: list[dict[str, Any]],
        fixed_headers: dict[str, str],
        fixed_body: dict[str, Any] | None,
        input_data: dict[str, Any],
        retry_policy: dict[str, Any] | None = None,
    ) -> OperationCallResult:
        """Run an unsaved operation definition against `connector`.

        Used by the test-draft endpoint so users can exercise an operation
        before persisting it. Credentials come from the connector's stored
        record; nothing about the operation is written to the database.
        """
        credentials = self.credentials_for(connector.id)
        if credentials is None:
            # Connector might not be in cache (e.g. inactive). Decrypt directly.
            credentials = decrypt_json(connector.credentials_encrypted) or {}
        return await self._execute_http(
            connector=connector,
            credentials=credentials,
            method=method,
            path=path,
            parameters=parameters,
            fixed_headers=fixed_headers,
            fixed_body=fixed_body,
            retry_policy=parse_retry_policy(retry_policy),
            input_data=input_data,
        )

    async def _execute_http(
        self,
        *,
        connector: Connector,
        credentials: dict[str, Any],
        method: HTTPMethod | str,
        path: str,
        parameters: list[dict[str, Any]],
        fixed_headers: dict[str, str],
        fixed_body: dict[str, Any] | None,
        input_data: dict[str, Any],
        retry_policy: RetryPolicy | None = None,
    ) -> OperationCallResult:
        method_enum = _coerce_method(method)
        # Merge parameter defaults into input_data BEFORE anything else
        # consumes it — path substitution, distribution, and template
        # interpolation all read from the merged dict, so a default value
        # behaves identically to a caller-supplied one regardless of
        # where the parameter is rendered (path/query/header/body).
        effective_input = _apply_parameter_defaults(parameters, input_data)
        url_path, _ = _substitute_path_params(path, dict(effective_input or {}))
        full_url = _join_url(connector.base_url, url_path)

        # Distribute the input by parameter `in`. Falls back to the legacy
        # "remaining payload → body|query" rule when no parameters are
        # declared so pre-migration operations keep working unchanged.
        query_vals, header_vals, body_vals = _distribute_input(
            method_enum, parameters, path, effective_input
        )

        # Fixed values use ALL of input_data for interpolation, not just the
        # leftover after path substitution — a `{user_id}` reference in
        # fixed_body should resolve to the user's input even when the same
        # field is also a path placeholder. Defaults are part of the
        # interpolation pool too so a `{cursor}` reference in fixed_body
        # picks up a parameter default the LLM never explicitly sent.
        interpolation_vars = dict(effective_input or {})
        interpolated_headers = {
            name: _interpolate(value, interpolation_vars)
            for name, value in fixed_headers.items()
        }
        interpolated_body = (
            _interpolate(fixed_body, interpolation_vars)
            if fixed_body is not None
            else None
        )

        # Avoid the double-write trap: any input key consumed by a
        # ``{placeholder}`` inside the body template is already inside the
        # interpolated body — letting the legacy "leftover → body" bucket
        # ALSO push it to the body root would corrupt strict payloads
        # (e.g. GraphQL, which rejects any top-level key other than
        # ``query``/``variables``/``operationName`` with a 404). Same idea
        # for headers, kept narrow until a use case appears.
        if fixed_body is not None and body_vals:
            consumed = _template_placeholders(fixed_body)
            if consumed:
                body_vals = {
                    k: v for k, v in body_vals.items() if k not in consumed
                }

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT_SECONDS) as client:
            # `credentials` may be replaced by a force-refresh between attempts.
            # The Connector ORM row is the same instance throughout, so the
            # in-memory registry cache picks up new ciphertext too via
            # `_persist_credentials`.
            current_creds = credentials
            prepared: PreparedRequest | None = None
            response: httpx.Response | None = None

            # Two retry budgets running in parallel:
            #   - `oauth2_refresh_used`: at most ONE refresh-and-retry on
            #     401/403, only for oauth2 connectors. Not gated by the
            #     policy — recovering from a stale token is auth-layer
            #     recovery, not transient-failure backoff.
            #   - `policy_retries_left`: configurable backoff loop for other
            #     statuses (e.g. flaky-edge 404s, 502/503/504). When no
            #     policy is set this is 0 → single attempt (legacy behavior).
            oauth2_refresh_used = False
            policy_retries_left = (
                retry_policy.max_attempts - 1 if retry_policy is not None else 0
            )
            policy_attempts_made = 0

            while True:
                try:
                    prepared = await self._prepare_auth(
                        connector, current_creds, client
                    )
                except ConnectorAuthError as exc:
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return OperationCallResult(
                        ok=False,
                        status_code=None,
                        latency_ms=latency_ms,
                        request_method=method_enum.value,
                        request_url=full_url,
                        response=None,
                        error=str(exc),
                    )

                # Defaults: a real User-Agent (httpx's default
                # `python-httpx/x.y` reliably trips Cloudflare bot
                # detection on protected APIs like Upwork's) and JSON
                # accept. Anything in fixed_headers or input params can
                # override these — the dict-merge order below makes that
                # explicit.
                final_headers: dict[str, Any] = {
                    "Accept": "application/json",
                    "User-Agent": "luna-core/connector",
                }
                # Stringify fixed-header interpolation results: HTTP headers
                # are text. (`_interpolate` can return non-strings when the
                # whole value is a {param} placeholder pointing at a typed.)
                for k, v in interpolated_headers.items():
                    final_headers[k] = "" if v is None else str(v)
                for k, v in header_vals.items():
                    final_headers[k] = "" if v is None else str(v)
                # Auth headers win — they're not user-overridable.
                final_headers.update(prepared.headers)

                params: dict[str, Any] = dict(prepared.params)
                params.update(
                    {k: v for k, v in query_vals.items() if v is not None}
                )

                request_kwargs: dict[str, Any] = {
                    "method": method_enum.value,
                    "url": full_url,
                    "headers": final_headers,
                }
                if method_enum in _BODY_METHODS:
                    final_body = _merge_body(interpolated_body, body_vals)
                    if final_body is not None:
                        request_kwargs["json"] = final_body
                if params:
                    request_kwargs["params"] = params
                if prepared.basic_auth is not None:
                    request_kwargs["auth"] = httpx.BasicAuth(
                        *prepared.basic_auth
                    )

                try:
                    response = await client.request(**request_kwargs)
                except httpx.HTTPError as exc:
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return OperationCallResult(
                        ok=False,
                        status_code=None,
                        latency_ms=latency_ms,
                        request_method=method_enum.value,
                        request_url=full_url,
                        response=None,
                        error=str(exc),
                    )

                # Auth recovery path: oauth2 only, at most once per call. If
                # the refresh itself fails we surface the original auth
                # error instead of silently returning the IdP's 401 — the
                # operator wants to know their credentials are bad.
                if (
                    not oauth2_refresh_used
                    and response.status_code in (401, 403)
                    and connector.auth_type == AuthType.oauth2
                ):
                    try:
                        current_creds = await self._force_oauth2_refresh(
                            connector, current_creds, client
                        )
                        oauth2_refresh_used = True
                        continue
                    except ConnectorAuthError as exc:
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        return OperationCallResult(
                            ok=False,
                            status_code=None,
                            latency_ms=latency_ms,
                            request_method=method_enum.value,
                            request_url=full_url,
                            response=None,
                            error=str(exc),
                        )

                # Transient-failure retry path: only when a policy is set
                # and the response status is on the configured retry list.
                if (
                    retry_policy is not None
                    and policy_retries_left > 0
                    and retry_policy.should_retry(response.status_code)
                ):
                    policy_attempts_made += 1
                    sleep_seconds = retry_policy.delay_seconds_for_retry(
                        policy_attempts_made
                    )
                    logger.info(
                        "connector %s retry %d/%d in %dms (status %d, %s)",
                        connector.name,
                        policy_attempts_made,
                        retry_policy.max_attempts - 1,
                        int(sleep_seconds * 1000),
                        response.status_code,
                        operation_label_or_path(path),
                    )
                    await asyncio.sleep(sleep_seconds)
                    policy_retries_left -= 1
                    continue

                break

        # `prepared` / `response` are guaranteed set by the loop above.
        assert prepared is not None and response is not None
        latency_ms = int((time.perf_counter() - started) * 1000)

        if prepared.refreshed_credentials is not None:
            await self._persist_refreshed_credentials(
                connector, prepared.refreshed_credentials
            )

        body = _decode_response(response)

        # On 4xx/5xx, log enough to debug the round-trip from the docker
        # logs without needing a stack trace. The error path upstream only
        # surfaces a 400-char preview of the decoded body, which is often
        # empty (e.g. Upwork returns 404 with no body) — what the operator
        # actually needs is the request we sent.
        if response.status_code >= 400:
            logger.warning(
                "connector %s.%s HTTP %d after %dms\n"
                "  request:  %s %s\n"
                "  headers:  %s\n"
                "  body:     %s\n"
                "  response: %s",
                connector.name,
                operation_label_or_path(path),
                response.status_code,
                latency_ms,
                method_enum.value,
                full_url,
                _redact_auth_headers(request_kwargs.get("headers") or {}),
                _truncate_for_log(request_kwargs.get("json")),
                _truncate_for_log(response.text if response.content else "(empty body)"),
            )

        return OperationCallResult(
            ok=response.status_code < 400,
            status_code=response.status_code,
            latency_ms=latency_ms,
            request_method=method_enum.value,
            request_url=full_url,
            response=body,
            error=None,
        )

    async def _force_oauth2_refresh(
        self,
        connector: Connector,
        credentials: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        """Refresh OAuth2 tokens and persist to DB if a session is available.

        Surfaces the new credentials dict so the caller can re-`_prepare_auth`
        in the same request. Persistence mutates the registry-cached
        connector row's `credentials_encrypted` in place.
        """
        db = await self._open_session()
        try:
            return await force_refresh_oauth2(
                connector,
                credentials,
                db=db,
                http_client=client,
            )
        finally:
            if db is not None:
                await db.close()

    async def _prepare_auth(
        self,
        connector: Connector,
        credentials: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> PreparedRequest:
        # If we have a session factory, hand a fresh session to the auth
        # layer so OAuth2 refreshes persist transactionally. Otherwise the
        # auth layer returns the patch in `refreshed_credentials` and we
        # persist it after the request.
        db = await self._open_session()
        try:
            return await prepare_auth(
                connector,
                credentials,
                db=db,
                http_client=client,
            )
        except ConnectorAuthError:
            # Surface auth failures as execution errors so flows see a
            # consistent exception type.
            raise
        finally:
            if db is not None:
                await db.close()

    async def _open_session(self) -> AsyncSession | None:
        if self._db_session_factory is None:
            return None
        result = self._db_session_factory()
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[no-any-return]
        return result  # type: ignore[return-value]

    async def _persist_refreshed_credentials(
        self, connector: Connector, credentials: dict[str, Any]
    ) -> None:
        if self._db_session_factory is None:
            logger.warning(
                "OAuth2 refresh happened for connector %s but no DB session "
                "factory is configured; new tokens are not persisted",
                connector.name,
            )
            return
        from luna_core.connectors.auth import _persist_credentials  # noqa: PLC0415

        session = await self._open_session()
        if session is None:
            return
        try:
            await _persist_credentials(session, connector, credentials)
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_placeholders(path: str) -> set[str]:
    return {m.group(1) for m in _PATH_PARAM_RE.finditer(path)}


def _template_placeholders(template: Any) -> set[str]:
    """Collect every `{name}` placeholder reachable inside a JSON-shaped
    template (strings, dicts, lists). Used to keep body keys that were
    already consumed by interpolation from leaking back into the request
    body root via the legacy bucketing fallback.

    Example: a GraphQL body template like
        {"variables": {"filter": {"skill_eq": "{skills}"}}}
    yields {"skills"} — so if the caller also passed ``skills`` as a
    top-level input arg, the registry won't drop it next to ``query`` and
    ``variables`` and trip the upstream parser.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for match in _PATH_PARAM_RE.finditer(node):
                out.add(match.group(1))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(template)
    return out


def _apply_parameter_defaults(
    parameters: list[dict[str, Any]],
    input_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a new dict with parameter defaults filled in for missing keys.

    Defaults apply ONLY when the input is missing the parameter key
    entirely. An explicit ``None`` or empty string the caller supplied
    is treated as intentional and passes through unchanged — this
    matches what JSON Schema's ``default`` keyword means everywhere
    else (a value used when the property is absent, not when it's
    present and falsy).

    Returns a fresh dict so callers can mutate freely. When parameters
    has no entries (legacy operations) this is a no-op clone.
    """
    out: dict[str, Any] = dict(input_data or {})
    if not parameters:
        return out
    for p in parameters:
        name = p.get("name")
        if not name or name in out:
            continue
        default = p.get("default")
        if default is None:
            continue
        out[name] = default
    return out


def _distribute_input(
    method: HTTPMethod,
    parameters: list[dict[str, Any]],
    path: str,
    input_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Sort input values into (query, header, body) buckets.

    Path params come out via `_substitute_path_params` on the rewritten URL
    and are excluded from these buckets — they only belong in the URL.

    Two regimes coexist:
      - `parameters` declared (new contract): each declared param's `in`
        decides the bucket. Input keys not declared in `parameters` are
        dropped silently (matches MCP's "tools only accept declared args").
      - `parameters` empty (legacy/back-compat): every input key that isn't
        consumed by the path goes to body (POST/PUT/PATCH) or query.
    """
    path_keys = _path_placeholders(path)
    payload = {k: v for k, v in (input_data or {}).items() if k not in path_keys}

    query: dict[str, Any] = {}
    header: dict[str, Any] = {}
    body: dict[str, Any] = {}

    if not parameters:
        # Legacy distribution: everything goes to body|query based on method.
        if method in _BODY_METHODS:
            return query, header, payload
        return payload, header, body

    for p in parameters:
        name = p.get("name")
        if not name or name not in payload:
            continue
        # `in` arrives as a JSON string ('path'|'query'|'body'|'header').
        # Path-destined params are already consumed via path_keys above;
        # ignoring them here keeps the contract "no duplication".
        dest = str(p.get("in", "body"))
        value = payload[name]
        if dest == "query":
            query[name] = value
        elif dest == "header":
            header[name] = value
        elif dest == "body":
            body[name] = value
        # path / unknown → ignored (path was already consumed)
    return query, header, body


def _interpolate(template: Any, variables: dict[str, Any]) -> Any:
    """Substitute `{name}` placeholders in arbitrary JSON-shaped data.

    Rules:
      - If a string is exactly `{name}` (no surrounding text), the typed
        value of `variables[name]` is returned — so numbers stay numbers,
        lists stay lists. Critical for body templates like `{"limit": "{n}"}`
        where `n` is an integer.
      - Mixed strings (`"prefix-{name}-suffix"`) get `str(value)` substitution.
        Missing variables leave the literal `{name}` untouched (so the user
        sees what didn't resolve).
      - dict / list values recurse. Other scalars pass through unchanged.
    """
    if isinstance(template, str):
        full = _FULL_PARAM_RE.match(template)
        if full and full.group(1) in variables:
            return variables[full.group(1)]

        def _repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in variables:
                value = variables[key]
                return "" if value is None else str(value)
            return match.group(0)

        return _PATH_PARAM_RE.sub(_repl, template)
    if isinstance(template, dict):
        return {k: _interpolate(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [_interpolate(v, variables) for v in template]
    return template


def _merge_body(
    fixed_body: Any,
    body_vals: dict[str, Any],
) -> Any:
    """Combine the interpolated body template with parameter-sourced fields.

    If both are present, parameter values overlay onto the template (so the
    caller's input can override constants when names collide — a common
    expectation when a body template ships sensible defaults).
    """
    if fixed_body is None and not body_vals:
        return None
    if fixed_body is None:
        return body_vals
    if not isinstance(fixed_body, dict):
        # Template is a list / scalar — caller knows what they want, ignore
        # body_vals rather than blindly mutating an alien shape.
        return fixed_body
    merged = dict(fixed_body)
    merged.update(body_vals)
    return merged


def _substitute_path_params(
    path: str, payload: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Replace `{name}` placeholders in `path` with values from `payload`.

    Returns the rewritten path plus a copy of `payload` with the consumed
    keys removed so they don't leak into the query string or body.
    """
    consumed: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in payload:
            raise ConnectorExecutionError(
                f"path parameter '{key}' missing from input"
            )
        value = payload[key]
        if value is None:
            raise ConnectorExecutionError(
                f"path parameter '{key}' must not be null"
            )
        consumed.append(key)
        return str(value)

    rewritten = _PATH_PARAM_RE.sub(_replace, path)
    remaining = {k: v for k, v in payload.items() if k not in consumed}
    return rewritten, remaining


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


# Diagnostics helpers for the 4xx/5xx logging branch in _execute_http.

def operation_label_or_path(path: str) -> str:
    """Best-effort label for a connector op when only `path` is in scope.
    Empty path means "the connector base URL itself" (typical for GraphQL)."""
    return path or "(root)"


_REDACT_HEADER_NAMES = frozenset(
    name.lower() for name in ("authorization", "x-api-key", "api-key", "cookie")
)


def _redact_auth_headers(headers: dict[str, Any]) -> dict[str, str]:
    """Return a shallow copy with sensitive header values masked. Keeps the
    header name visible so the operator sees that auth WAS attempted."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _REDACT_HEADER_NAMES:
            text = "" if v is None else str(v)
            out[k] = f"<redacted len={len(text)}>"
        else:
            out[k] = "" if v is None else str(v)
    return out


def _truncate_for_log(value: Any, *, limit: int = 2000) -> str:
    """JSON-serialise (or fall back to str) and truncate for log readability.
    Limit is generous enough to capture full GraphQL bodies + a real error
    payload, but short enough to keep the docker stdout sane."""
    if value is None:
        return "(none)"
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} bytes)"


def _coerce_method(method: HTTPMethod | str) -> HTTPMethod:
    if isinstance(method, HTTPMethod):
        return method
    try:
        return HTTPMethod(str(method).upper())
    except ValueError as exc:
        raise ConnectorExecutionError(f"unsupported HTTP method: {method}") from exc


def _decode_response(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    content_type = response.headers.get("content-type", "")
    looks_like_json = (
        "application/json" in content_type
        or response.content.startswith(b"{")
        or response.content.startswith(b"[")
    )
    if looks_like_json:
        try:
            body = response.json()
        except ValueError:
            body = None
        if body is not None:
            # Normalise list payloads under a stable key so downstream nodes
            # always see a dict.
            if isinstance(body, list):
                return {"items": body}
            return body
    return {"text": response.text}


_registry: ConnectorRegistry | None = None


def get_connector_registry() -> ConnectorRegistry:
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry


__all__ = [
    "ConnectorExecutionError",
    "ConnectorRegistry",
    "OperationCallResult",
    "get_connector_registry",
]
