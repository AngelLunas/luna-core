"""Tests for ``${context.…}`` path resolution, including ``[*]`` syntax.

The instructions editor's chip palette emits ``field[*]`` for any array
property in a source's JSON schema — including arrays of scalars. The
resolver must recognise those segments instead of treating them as
literal dict keys (which was the original bug surfaced by the profile's
new list fields).
"""
from __future__ import annotations

from luna_core.engine.template_paths import resolve_path, tokenize_path


def test_tokenize_plain_dotted_path():
    assert tokenize_path("context.profile.name") == ["context", "profile", "name"]


def test_tokenize_terminal_star():
    assert tokenize_path("context.profile.skills[*]") == [
        "context",
        "profile",
        "skills",
        "[*]",
    ]


def test_tokenize_star_with_projection():
    assert tokenize_path("context.profile.work_experiences[*].title") == [
        "context",
        "profile",
        "work_experiences",
        "[*]",
        "title",
    ]


def test_resolve_scalar_path():
    state = {"context": {"profile": {"name": "Angel"}}}
    assert resolve_path(state, "context.profile.name") == "Angel"


def test_resolve_terminal_star_returns_whole_list():
    state = {"context": {"profile": {"skills": ["threejs", "python"]}}}
    assert resolve_path(state, "context.profile.skills[*]") == ["threejs", "python"]


def test_resolve_terminal_star_on_empty_list():
    state = {"context": {"profile": {"skills": []}}}
    assert resolve_path(state, "context.profile.skills[*]") == []


def test_resolve_star_with_projection():
    state = {
        "context": {
            "profile": {
                "work_experiences": [
                    {"title": "Lead", "company": "Luna"},
                    {"title": "Engineer", "company": "Acme"},
                ]
            }
        }
    }
    assert resolve_path(
        state, "context.profile.work_experiences[*].title"
    ) == ["Lead", "Engineer"]


def test_resolve_star_on_non_list_returns_none():
    # Asking for [*] over a dict is a chip-palette / user error — return
    # None so the formatter renders empty, matching missing-path behavior.
    state = {"context": {"profile": {"skills": {"not": "a list"}}}}
    assert resolve_path(state, "context.profile.skills[*]") is None


def test_resolve_missing_path_returns_none():
    state = {"context": {"profile": {"name": "Angel"}}}
    assert resolve_path(state, "context.profile.unknown") is None
    assert resolve_path(state, "context.profile.unknown[*]") is None


def test_resolve_empty_path_returns_state():
    state = {"foo": "bar"}
    assert resolve_path(state, "") == state
