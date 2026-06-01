"""Tests for ConnectorRegistry.perform() and the registry mutation helpers.

`perform()` is the test-friendly variant of `execute()`: it returns a
structured result (status, body, latency) without raising on transport
or 4xx/5xx so the UI test endpoint can render error responses verbatim.

These tests use httpx.MockTransport to avoid hitting a real upstream
service, and skip the DB entirely by registering Connector/Operation
ORM objects directly into the registry cache.
"""
from __future__ import annotations

import uuid

import httpx
import pytest

from luna_core.connectors import registry as registry_module
from luna_core.connectors.registry import ConnectorRegistry
from luna_core.models.connector import AuthType, Connector, HTTPMethod, Operation


def _make_connector(base_url: str = "https://example.test") -> Connector:
    c = Connector(
        name=f"conn-{uuid.uuid4().hex[:6]}",
        description="",
        auth_type=AuthType.none,
        base_url=base_url,
        credentials_encrypted=None,
        is_active=True,
    )
    c.id = uuid.uuid4()
    return c


def _make_operation(
    connector_id: uuid.UUID,
    *,
    name: str = "do_thing",
    method: HTTPMethod = HTTPMethod.GET,
    path: str = "/things",
    parameters: list[dict] | None = None,
    fixed_headers: dict[str, str] | None = None,
    fixed_body: dict | None = None,
) -> Operation:
    op = Operation(
        connector_id=connector_id,
        name=name,
        description="",
        method=method,
        path=path,
        input_schema={},
        output_schema={},
        parameters=parameters or [],
        fixed_headers=fixed_headers or {},
        fixed_body=fixed_body,
        is_active=True,
    )
    op.id = uuid.uuid4()
    return op


def _install_mock_transport(monkeypatch, handler) -> None:
    """Replace registry's httpx.AsyncClient with one bound to MockTransport.

    The registry instantiates a fresh AsyncClient inside perform(), so we
    patch the symbol it resolves at call time rather than the global
    httpx.AsyncClient (which would leak into other tests / modules).
    """
    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(registry_module.httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_perform_returns_response_body_on_2xx(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        return httpx.Response(200, json={"hello": "world"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(connector.id, path="/jobs")
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    assert result.ok is True
    assert result.status_code == 200
    assert result.response == {"hello": "world"}
    assert result.error is None
    assert seen["method"] == "GET"
    assert seen["url"].endswith("/jobs")


@pytest.mark.asyncio
async def test_perform_captures_4xx_without_raising(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "missing"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(connector.id)
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    assert result.ok is False
    assert result.status_code == 404
    assert result.response == {"error": "missing"}
    assert result.error is None


@pytest.mark.asyncio
async def test_perform_captures_transport_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(connector.id)
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    assert result.ok is False
    assert result.status_code is None
    assert result.response is None
    assert "nope" in (result.error or "")


@pytest.mark.asyncio
async def test_perform_substitutes_path_params_and_routes_remaining_to_body(
    monkeypatch,
):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(201, json={"ok": True})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id, method=HTTPMethod.POST, path="/jobs/{job_id}/proposals"
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(
        op.id, {"job_id": "abc-123", "cover_letter": "Hi", "bid_usd": 100}
    )

    assert result.ok is True
    assert "/jobs/abc-123/proposals" in seen["url"]
    # `job_id` is consumed by the path; only the remaining keys should land
    # in the JSON body.
    assert '"cover_letter":"Hi"' in seen["body"].replace(" ", "")
    assert "job_id" not in seen["body"]


def test_unregister_operation_removes_only_that_op():
    connector = _make_connector()
    op_a = _make_operation(connector.id, name="a")
    op_b = _make_operation(connector.id, name="b")
    reg = ConnectorRegistry()
    reg.register(connector, [op_a, op_b])

    reg.unregister_operation(op_a.id)

    with pytest.raises(KeyError):
        reg.get_operation(op_a.id)
    # op_b should still resolve.
    assert reg.get_operation(op_b.id).name == "b"


def test_unregister_connector_drops_all_operations():
    connector = _make_connector()
    op_a = _make_operation(connector.id, name="a")
    op_b = _make_operation(connector.id, name="b")
    reg = ConnectorRegistry()
    reg.register(connector, [op_a, op_b])

    reg.unregister_connector(connector.id)

    with pytest.raises(KeyError):
        reg.get_operation(op_a.id)
    with pytest.raises(KeyError):
        reg.get_operation(op_b.id)


def test_unregister_missing_operation_is_noop():
    reg = ConnectorRegistry()
    # Should not raise even though the id was never registered.
    reg.unregister_operation(uuid.uuid4())
    reg.unregister_connector(uuid.uuid4())


# ---------------------------------------------------------------------------
# parameters with `in` destinations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_distributes_inputs_by_in(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode() if request.content else ""
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        method=HTTPMethod.POST,
        path="/jobs/{job_id}/proposals",
        parameters=[
            {"name": "job_id", "type": "string", "in": "path"},
            {"name": "trace_id", "type": "string", "in": "header"},
            {"name": "page", "type": "integer", "in": "query"},
            {"name": "cover_letter", "type": "string", "in": "body"},
        ],
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(
        op.id,
        {
            "job_id": "abc-123",
            "trace_id": "trc-1",
            "page": 2,
            "cover_letter": "Hi",
        },
    )

    assert "/jobs/abc-123/proposals" in seen["url"]
    assert "page=2" in seen["url"]
    assert seen["headers"].get("trace_id") == "trc-1"
    # Only body-destined params land in the JSON body.
    assert '"cover_letter":"Hi"' in seen["body"].replace(" ", "")
    assert "job_id" not in seen["body"]
    assert "trace_id" not in seen["body"]


@pytest.mark.asyncio
async def test_perform_interpolates_fixed_headers(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        parameters=[{"name": "tenant", "type": "string", "in": "body"}],
        fixed_headers={
            "X-Tenant": "{tenant}",
            "X-Source": "luna-sentinel",
        },
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {"tenant": "acme"})

    assert seen["headers"]["x-tenant"] == "acme"
    assert seen["headers"]["x-source"] == "luna-sentinel"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_perform_does_not_leak_template_consumed_keys_to_body_root(
    monkeypatch,
):
    """An input key that's already consumed by a ``{placeholder}`` inside
    the body template must NOT also be appended at the body root by the
    legacy bucketing fallback.

    Regression test for an Upwork GraphQL operation: the body template
    routes ``{skills}`` into ``variables.marketPlaceJobFilter.skill_eq``,
    but the legacy "no parameters declared → leftover to body" path was
    also copying ``skills`` next to ``query``/``variables``, which made
    Upwork reject the request as malformed GraphQL (returned HTTP 404).
    """
    import json as _json

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = _json.loads(request.content.decode())
        return httpx.Response(200, json={"data": "ok"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        method=HTTPMethod.POST,
        path="/graphql",
        # Legacy mode: NO parameters declared. Without the fix the
        # registry would dump every input key into the body root.
        parameters=None,
        fixed_body={
            "query": "query S($f: F) { search(filter: $f) { id } }",
            "variables": {
                "f": {"skill_eq": "{skills}"},
            },
        },
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {"skills": ["three-js"]})

    body = seen["body"]
    # The template placeholder absorbed the value:
    assert body["variables"]["f"]["skill_eq"] == ["three-js"]
    # And the body root stays strictly GraphQL-shaped:
    assert set(body.keys()) == {"query", "variables"}


@pytest.mark.asyncio
async def test_perform_interpolates_fixed_body_preserving_types(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        method=HTTPMethod.POST,
        path="/search",
        parameters=[{"name": "limit", "type": "integer", "in": "body"}],
        # `{limit}` is the whole value → should arrive as an integer 25, not
        # a string. Mixed strings ("page-{limit}") use str() substitution.
        fixed_body={
            "filters": {"limit": "{limit}", "label": "page-{limit}"},
            "source": "luna",
        },
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {"limit": 25})

    body_text = seen["body"].replace(" ", "")
    assert '"limit":25' in body_text  # typed int preserved
    assert '"label":"page-25"' in body_text  # mixed string formatted
    assert '"source":"luna"' in body_text  # constant passes through


def test_parameters_to_input_schema_builds_jsonschema():
    from luna_core.schemas.connector import ParameterDef, ParameterIn, ParameterType
    from luna_core.services.connector import parameters_to_input_schema

    params = [
        ParameterDef(name="query", type=ParameterType.string, required=True),
        ParameterDef(
            name="limit",
            type=ParameterType.integer,
            description="Page size",
        ),
        ParameterDef(
            name="status",
            type=ParameterType.string,
            enum_values=["open", "closed"],
            in_=ParameterIn.query,
        ),
    ]
    schema = parameters_to_input_schema(params)

    assert schema["type"] == "object"
    assert schema["properties"]["query"] == {"type": "string"}
    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["limit"]["description"] == "Page size"
    assert schema["properties"]["status"]["enum"] == ["open", "closed"]
    assert schema["required"] == ["query"]


def test_parameters_to_input_schema_includes_default():
    # The `default` field surfaces in the JSON Schema so the LLM sees
    # the fallback in the tool definition (informational), and the
    # dispatcher applies the same value when the key is omitted.
    from luna_core.schemas.connector import ParameterDef, ParameterIn, ParameterType
    from luna_core.services.connector import parameters_to_input_schema

    params = [
        ParameterDef(
            name="cursor",
            type=ParameterType.string,
            in_=ParameterIn.query,
            default="0",
        ),
        ParameterDef(
            name="limit",
            type=ParameterType.integer,
            in_=ParameterIn.query,
            default=20,
        ),
        ParameterDef(name="no_default", type=ParameterType.string),
    ]
    schema = parameters_to_input_schema(params)
    assert schema["properties"]["cursor"]["default"] == "0"
    assert schema["properties"]["limit"]["default"] == 20
    # Params without a default must NOT carry the key — otherwise an
    # `"default": null` would mislead LLMs into sending null.
    assert "default" not in schema["properties"]["no_default"]


# ---------------------------------------------------------------------------
# OAuth2 retry-on-401
# ---------------------------------------------------------------------------


def _make_oauth2_connector(
    *,
    access_token: str = "stale-token",
    refresh_token: str = "rt-1",
    base_url: str = "https://api.example.test",
    token_url: str = "https://idp.example.test/token",
) -> Connector:
    from luna_core.core.crypto import encrypt_json
    from luna_core.models.connector import AuthType

    c = Connector(
        name=f"oauth-{uuid.uuid4().hex[:6]}",
        description="",
        auth_type=AuthType.oauth2,
        base_url=base_url,
        credentials_encrypted=encrypt_json(
            {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_url": token_url,
                "client_id": "cid",
                "client_secret": "csec",
                # No expires_at on purpose — forces the "optimistically use"
                # path so we hit the 401 retry instead of a proactive refresh.
            }
        ),
        is_active=True,
    )
    c.id = uuid.uuid4()
    return c


@pytest.mark.asyncio
async def test_oauth2_retries_on_401_after_force_refresh(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        auth = request.headers.get("Authorization")
        calls.append((url, auth))
        if "idp.example.test/token" in url:
            # IdP — refresh succeeds with a new access_token.
            return httpx.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "rt-2",
                    "expires_in": 3600,
                },
            )
        # Operation endpoint.
        if auth == "Bearer fresh-token":
            return httpx.Response(200, json={"ok": True})
        # First attempt with stale token → 401.
        return httpx.Response(401, json={"error": "token_expired"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_oauth2_connector()
    op = _make_operation(connector.id, path="/things")
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    assert result.ok is True
    assert result.status_code == 200
    assert result.response == {"ok": True}
    # Sequence: stale-token → 401 → refresh → fresh-token → 200. The IdP
    # call itself uses HTTP Basic for client credentials, not a Bearer, so
    # filter to bearers when checking the operation-call progression.
    bearers = [
        auth
        for _, auth in calls
        if auth is not None and auth.startswith("Bearer ")
    ]
    assert bearers == ["Bearer stale-token", "Bearer fresh-token"]
    # Verify the refresh actually hit the IdP between the two operation calls.
    op_attempts = [u for u, _ in calls if "things" in u]
    token_attempts = [u for u, _ in calls if "token" in u and "things" not in u]
    assert len(op_attempts) == 2
    assert len(token_attempts) == 1


@pytest.mark.asyncio
async def test_oauth2_does_not_retry_when_refresh_itself_fails(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "idp.example.test/token" in url:
            # IdP rejects the refresh (e.g. refresh_token revoked).
            return httpx.Response(401, json={"error": "invalid_grant"})
        return httpx.Response(401, json={"error": "token_expired"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_oauth2_connector()
    op = _make_operation(connector.id, path="/things")
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    # When the refresh itself fails we surface a transport-level error so
    # the operator sees "credentials are bad" instead of a silent 401.
    assert result.ok is False
    assert result.status_code is None
    assert result.error is not None
    assert "oauth2" in result.error.lower() or "invalid_grant" in result.error


@pytest.mark.asyncio
async def test_api_key_connector_does_not_retry_on_401(monkeypatch):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(401, json={"error": "bad_key"})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()  # auth_type=none, no retry path.
    op = _make_operation(connector.id, path="/things")
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    result = await reg.perform(op.id, {})

    # Single attempt — non-oauth2 connectors don't get the retry-on-401 path.
    assert result.status_code == 401
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# OAuth2 state token (JWT) round-trip
# ---------------------------------------------------------------------------


def test_oauth2_state_roundtrip():
    from luna_core.connectors.oauth2_flow import make_state, verify_state

    cid = uuid.uuid4()
    token = make_state(cid)
    assert verify_state(token) == cid


def test_oauth2_state_rejects_tampered_token():
    import pytest

    from luna_core.connectors.oauth2_flow import (
        OAuth2StateError,
        make_state,
        verify_state,
    )

    token = make_state(uuid.uuid4())
    tampered = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
    with pytest.raises(OAuth2StateError):
        verify_state(tampered)


def test_oauth2_state_rejects_expired_token():
    import pytest

    from luna_core.connectors.oauth2_flow import (
        OAuth2StateError,
        make_state,
        verify_state,
    )

    # ttl=-1 mints a token that's already past its `exp` claim.
    token = make_state(uuid.uuid4(), ttl_seconds=-1)
    with pytest.raises(OAuth2StateError):
        verify_state(token)


@pytest.mark.asyncio
async def test_oauth2_refresh_sends_client_id_in_both_body_and_basic(monkeypatch):
    """Regression: Upwork's token endpoint rejects refreshes that omit
    client_id from the body. RFC 6749 says Basic-Auth is canonical, but
    real providers diverge — we send both to stay portable.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "idp.example.test/token" in str(request.url):
            captured["body"] = request.content.decode()
            captured["auth"] = request.headers.get("Authorization")
            captured["ua"] = request.headers.get("User-Agent")
            return httpx.Response(
                200,
                json={"access_token": "new", "expires_in": 3600},
            )
        return httpx.Response(401)

    _install_mock_transport(monkeypatch, handler)

    connector = _make_oauth2_connector()
    op = _make_operation(connector.id, path="/things")
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {})

    # Body carries client_id + client_secret (Upwork-style).
    assert "client_id=cid" in captured["body"]
    assert "client_secret=csec" in captured["body"]
    # AND Basic Auth header is present (RFC 6749-style).
    assert captured["auth"] is not None and captured["auth"].startswith("Basic ")
    # Custom User-Agent dodges Cloudflare's bot blocklist.
    assert "luna-core" in (captured["ua"] or "")


def test_oauth2_build_authorize_url_encodes_params():
    from luna_core.connectors.oauth2_flow import build_authorize_url

    url = build_authorize_url(
        authorize_url="https://idp.example.test/auth",
        client_id="cid",
        redirect_uri="http://localhost:5173/connectors/oauth2/callback",
        scope="r:jobs w:proposals",
        state="abc.def.ghi",
    )
    assert url.startswith("https://idp.example.test/auth?")
    assert "response_type=code" in url
    assert "client_id=cid" in url
    # Spaces in scope must be URL-encoded so the IdP parses two scopes.
    assert "scope=r%3Ajobs+w%3Aproposals" in url
    assert "state=abc.def.ghi" in url


@pytest.mark.asyncio
async def test_perform_draft_runs_unsaved_definition(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(200, json={"draft": True})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    reg = ConnectorRegistry()
    # Connector lives in the cache because the dry-run path resolves
    # credentials_for() against the registered bucket. (It also has a
    # fallback to decrypt directly, but here we exercise the cache path.)
    reg.register(connector)

    result = await reg.perform_draft(
        connector=connector,
        method=HTTPMethod.POST,
        path="/echo",
        parameters=[{"name": "msg", "type": "string", "in": "body"}],
        fixed_headers={},
        fixed_body=None,
        input_data={"msg": "hello"},
    )

    assert result.ok is True
    assert result.response == {"draft": True}
    assert '"msg":"hello"' in seen["body"].replace(" ", "")


# ---------------------------------------------------------------------------
# parameter defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_applies_default_when_caller_omits_query_param(monkeypatch):
    # Mirrors the real-world Upwork pagination bug: the LLM keeps
    # forgetting to send `cursor=0` on the first turn. A declared
    # default makes the omission harmless — the dispatcher injects "0"
    # so the upstream receives a well-formed first-page query.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        path="/jobs",
        parameters=[
            {"name": "cursor", "type": "string", "in": "query", "default": "0"},
        ],
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {})
    assert "cursor=0" in seen["url"]


@pytest.mark.asyncio
async def test_perform_explicit_value_overrides_default(monkeypatch):
    # Defaults must not silently overwrite an explicit caller value —
    # otherwise paginated turns past the first would never advance.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        path="/jobs",
        parameters=[
            {"name": "cursor", "type": "string", "in": "query", "default": "0"},
        ],
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {"cursor": "42"})
    assert "cursor=42" in seen["url"]
    assert "cursor=0" not in seen["url"]


@pytest.mark.asyncio
async def test_perform_default_does_not_replace_explicit_empty_string(monkeypatch):
    # An explicit "" from the caller is intentional. Defaults apply for
    # MISSING keys only; otherwise an LLM that means "no cursor here"
    # would be silently rewritten and the next-page boundary would
    # shift in confusing ways.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        path="/jobs",
        parameters=[
            {"name": "cursor", "type": "string", "in": "query", "default": "0"},
        ],
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {"cursor": ""})
    # The empty cursor IS forwarded, the default does not kick in.
    assert "cursor=" in seen["url"]
    assert "cursor=0" not in seen["url"]


@pytest.mark.asyncio
async def test_perform_applies_default_for_path_param(monkeypatch):
    # Path placeholders also read from the merged dict — without
    # default application, an unsent path param would leave the literal
    # `{tenant}` in the URL and produce a 404 on the upstream.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        path="/orgs/{tenant}/jobs",
        parameters=[
            {"name": "tenant", "type": "string", "in": "path", "default": "acme"},
        ],
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {})
    assert "/orgs/acme/jobs" in seen["url"]


@pytest.mark.asyncio
async def test_perform_default_reaches_fixed_body_interpolation(monkeypatch):
    # fixed_body templates read from the same merged dict so a default
    # value resolves `{cursor}` just like an explicit caller-supplied
    # value would. Important for GraphQL-shaped connectors where the
    # query lives in fixed_body and pagination cursors substitute into it.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(200, json={})

    _install_mock_transport(monkeypatch, handler)

    connector = _make_connector()
    op = _make_operation(
        connector.id,
        method=HTTPMethod.POST,
        path="/graphql",
        parameters=[
            {"name": "cursor", "type": "string", "in": "body", "default": "0"},
        ],
        fixed_body={"query": "search(after: {cursor})"},
    )
    reg = ConnectorRegistry()
    reg.register(connector, [op])

    await reg.perform(op.id, {})
    assert "search(after: 0)" in seen["body"]
