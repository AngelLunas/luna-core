"""The chat image resolver must know the photos of the message being SENT.

It is built before the runner persists that message, so a host scoping by
"latest stored user turn" would see the previous turn's photos (or none on a
fresh conversation). luna-core hands the request's media_ids to factories
that accept them and keeps calling older 4-argument factories as before.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from luna_core.routers.conversations import _image_resolver


def _request(factory):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        chat_image_resolver_factory=factory
    )))


@pytest.mark.asyncio
async def test_media_ids_reach_a_factory_that_accepts_them():
    seen = {}

    async def factory(db, user_id, agent, conversation, media_ids=None):
        seen["media_ids"] = media_ids
        return "resolver"

    mid = uuid.uuid4()
    out = await _image_resolver(
        _request(factory), None, None, None, uuid.uuid4(), media_ids=[mid]
    )
    assert out == "resolver"
    assert seen["media_ids"] == [mid]


@pytest.mark.asyncio
async def test_legacy_four_argument_factory_still_works():
    calls = []

    async def factory(db, user_id, agent, conversation):
        calls.append((db, user_id, agent, conversation))
        return "resolver"

    out = await _image_resolver(
        _request(factory), "db", "conv", "agent", "uid", media_ids=[uuid.uuid4()]
    )
    assert out == "resolver"
    assert calls == [("db", "uid", "agent", "conv")]


@pytest.mark.asyncio
async def test_no_media_ids_calls_factory_without_them():
    seen = {}

    async def factory(db, user_id, agent, conversation, media_ids=None):
        seen["media_ids"] = media_ids
        return None

    assert await _image_resolver(_request(factory), None, None, None, "uid") is None
    assert seen["media_ids"] is None


@pytest.mark.asyncio
async def test_no_hook_means_no_resolver():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert await _image_resolver(request, None, None, None, "uid") is None
