from __future__ import annotations

import pytest

from luna_core.services.context_sources import (
    DuplicateSourceError,
    UnknownSourceError,
    extract_context_sources,
    get_context_source,
    list_context_sources,
    register_context_source,
)


async def _loader(_ctx, _id):
    return {}


def test_extract_context_sources_finds_each_distinct_name():
    text = (
        "hello ${context.profile.name}, your project ${context.project.title}\n"
        "and again ${context.profile.skills.0}"
    )
    assert extract_context_sources(text) == ["profile", "project"]


def test_extract_context_sources_handles_empty_input():
    assert extract_context_sources("") == []
    assert extract_context_sources(None) == []


def test_extract_context_sources_ignores_non_context_refs():
    text = "uses ${inputs.foo} and ${outputs.bar.baz}"
    assert extract_context_sources(text) == []


def test_register_and_get():
    register_context_source(
        name="profile",
        description="freelancer profile",
        loader=_loader,
    )
    source = get_context_source("profile")
    assert source.name == "profile"
    assert source.id_implicit is False


def test_register_same_loader_is_idempotent():
    register_context_source(name="profile", description="", loader=_loader)
    # second call with same loader should not raise
    register_context_source(name="profile", description="", loader=_loader)


def test_register_conflicting_loader_raises():
    register_context_source(name="profile", description="", loader=_loader)

    async def other_loader(_ctx, _id):
        return {"different": True}

    with pytest.raises(DuplicateSourceError):
        register_context_source(name="profile", description="", loader=other_loader)


def test_get_unknown_raises():
    with pytest.raises(UnknownSourceError):
        get_context_source("nope")


def test_list_context_sources_sorted_by_name():
    register_context_source(name="zeta", description="", loader=_loader)
    register_context_source(name="alpha", description="", loader=_loader)
    register_context_source(name="middle", description="", loader=_loader)
    assert [s.name for s in list_context_sources()] == ["alpha", "middle", "zeta"]
