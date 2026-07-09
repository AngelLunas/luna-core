"""The router retries transient provider failures (5xx, connection, or an error
streamed inside a 200 — the bare APIError OpenRouter raises mid-stream) but not
4xx client errors."""
from __future__ import annotations

import uuid

import httpx
import pytest
from openai import APIError, BadRequestError, InternalServerError

from luna_core.llm.router import LLMRouter


class _FakeProvider:
    def __init__(self, script: list) -> None:
        self._script = list(script)  # each item: exception to raise, or None = ok
        self.calls = 0

    async def complete(self, **_kwargs):
        self.calls += 1
        item = self._script.pop(0)
        if item is not None:
            raise item
        return [{"type": "text", "text": "ok"}]


def _router(provider: _FakeProvider, *, max_retries: int = 3) -> LLMRouter:
    r = LLMRouter(redis=None, session_factory=None, max_retries=max_retries)

    async def _resolve(_pid):
        return provider

    async def _no_rate_limit(_pid, _rid):
        return None

    r.resolve_chat_provider = _resolve  # type: ignore[assignment]
    r._enforce_rate_limit = _no_rate_limit  # type: ignore[assignment]
    r._backoff_delay = lambda _attempt: 0.0  # type: ignore[assignment]
    return r


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://x/v1/chat/completions")


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=_req())


async def _call(router: LLMRouter):
    return await router.complete(
        provider_id=uuid.uuid4(),
        messages=[],
        system="",
        tools=[],
        temperature=0.0,
        model="m",
        output_schema=None,
        run_id=uuid.uuid4(),
        node_id="n",
    )


@pytest.mark.asyncio
async def test_retries_bare_apierror_then_succeeds():
    # The OpenRouter mid-stream case: a bare APIError with no status_code.
    bare = APIError("stream error", request=_req(), body=None)
    prov = _FakeProvider([bare, None])
    out = await _call(_router(prov))
    assert out == [{"type": "text", "text": "ok"}]
    assert prov.calls == 2  # one retry


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds():
    prov = _FakeProvider(
        [InternalServerError("boom", response=_resp(503), body=None), None]
    )
    out = await _call(_router(prov))
    assert out == [{"type": "text", "text": "ok"}]
    assert prov.calls == 2


@pytest.mark.asyncio
async def test_does_not_retry_4xx():
    prov = _FakeProvider([BadRequestError("bad", response=_resp(400), body=None), None])
    with pytest.raises(BadRequestError):
        await _call(_router(prov))
    assert prov.calls == 1  # raised immediately, no retry


@pytest.mark.asyncio
async def test_gives_up_after_max_retries():
    prov = _FakeProvider([APIError("e", request=_req(), body=None)] * 10)
    with pytest.raises(APIError):
        await _call(_router(prov, max_retries=2))
    assert prov.calls == 3  # initial + 2 retries
