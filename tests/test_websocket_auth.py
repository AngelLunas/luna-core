"""The run and flow stream WebSockets admit the same users as their REST reads.

Both sockets used to accept anyone who knew an id. They now authenticate
like ``CurrentUser`` (an access token, an active user, a verified email when
the host requires it) and the flow socket also checks ``flows:read``, like
``GET /flows/{id}``. A refused socket is closed with 1008 before it is
accepted, which the client sees as a 403 on the handshake.
"""
from __future__ import annotations

import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from luna_core.core import dependencies
from luna_core.core.config import settings
from luna_core.core.security import create_access_token
from luna_core.routers import flows as flows_router
from luna_core.routers import runs as runs_router
from luna_core.services import permission as permission_service

USER_ID = uuid.uuid4()
TARGET_ID = uuid.uuid4()


class FakeSession:
    """Just the ``db.get`` the helper calls, over one optional user."""

    def __init__(self, user: SimpleNamespace | None) -> None:
        self._user = user

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get(self, _model: object, key: uuid.UUID) -> SimpleNamespace | None:
        return self._user if self._user is not None and key == self._user.id else None


class AcceptingManager:
    """Stands in for the Redis-backed manager: accepts, says hello, closes."""

    async def connect(self, _key: uuid.UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"ok": True})
        await websocket.close()


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch):
    state = {
        "user": SimpleNamespace(id=USER_ID, is_active=True, is_verified=True),
        "permissions": {"flows:read"},
    }
    monkeypatch.setattr(dependencies, "AsyncSessionLocal", lambda: FakeSession(state["user"]))

    async def has_permission(_user: object, permission: str, _db: object) -> bool:
        return permission in state["permissions"]

    monkeypatch.setattr(permission_service, "has_permission", has_permission)
    monkeypatch.setattr(settings, "email_verification_required", False)
    monkeypatch.setattr(runs_router, "get_ws_manager", lambda: AcceptingManager())
    monkeypatch.setattr(flows_router, "get_flow_ws_manager", lambda: AcceptingManager())

    app = FastAPI()
    app.include_router(runs_router.router, prefix="/api/v1")
    app.include_router(flows_router.router, prefix="/api/v1")
    return TestClient(app), state


def _token(**kwargs: object) -> str:
    return create_access_token(USER_ID, **kwargs)


def _refused(client: TestClient, url: str, **kwargs: object) -> int:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(url, **kwargs) as ws:
            ws.receive_json()
    return exc.value.code


STREAMS = [f"/api/v1/runs/{TARGET_ID}/stream", f"/api/v1/flows/{TARGET_ID}/stream"]


@pytest.mark.parametrize("path", STREAMS)
def test_valid_token_in_query_connects(setup, path: str) -> None:
    client, _ = setup
    with client.websocket_connect(f"{path}?token={_token()}") as ws:
        assert ws.receive_json() == {"ok": True}


@pytest.mark.parametrize("path", STREAMS)
def test_bearer_header_connects(setup, path: str) -> None:
    client, _ = setup
    with client.websocket_connect(path, headers={"Authorization": f"Bearer {_token()}"}) as ws:
        assert ws.receive_json() == {"ok": True}


@pytest.mark.parametrize("path", STREAMS)
def test_missing_token_is_refused(setup, path: str) -> None:
    client, _ = setup
    assert _refused(client, path) == 1008


@pytest.mark.parametrize("path", STREAMS)
@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        create_access_token(USER_ID, expires_delta=timedelta(seconds=-5)),
        create_access_token(USER_ID, extra_claims={"type": "refresh"}),
        create_access_token("not-a-uuid"),
    ],
    ids=["garbage", "expired", "wrong-type", "bad-subject"],
)
def test_bad_token_is_refused(setup, path: str, token: str) -> None:
    client, _ = setup
    assert _refused(client, f"{path}?token={token}") == 1008


@pytest.mark.parametrize("path", STREAMS)
def test_inactive_or_missing_user_is_refused(setup, path: str) -> None:
    client, state = setup
    state["user"].is_active = False
    assert _refused(client, f"{path}?token={_token()}") == 1008
    state["user"] = None
    assert _refused(client, f"{path}?token={_token()}") == 1008


@pytest.mark.parametrize("path", STREAMS)
def test_unverified_user_is_refused_when_required(setup, path: str, monkeypatch) -> None:
    client, state = setup
    state["user"].is_verified = False
    monkeypatch.setattr(settings, "email_verification_required", True)
    assert _refused(client, f"{path}?token={_token()}") == 1008


def test_flow_stream_requires_flows_read(setup) -> None:
    client, state = setup
    state["permissions"] = set()
    assert _refused(client, f"/api/v1/flows/{TARGET_ID}/stream?token={_token()}") == 1008
    # Runs have no permission gate on REST, so none on the socket either.
    with client.websocket_connect(f"/api/v1/runs/{TARGET_ID}/stream?token={_token()}") as ws:
        assert ws.receive_json() == {"ok": True}
