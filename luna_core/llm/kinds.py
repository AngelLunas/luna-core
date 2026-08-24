"""Registry of LLM provider *kinds* — the one place that knows what each
implementation needs and how to present it.

Clients (the settings UI) must not learn kind names: they read the derived
fields ``requires_api_key`` / ``display_name`` off ``LLMProviderRead`` and
stay correct when a new kind (another CLI, a local runtime) is added here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderKindSpec:
    kind: str
    # Human label for providers of this kind when the row's own name is just
    # an identifier. None → the row name is presented (title-cased).
    label: str | None
    # Whether chat calls need an API key on the row. Kinds that authenticate
    # out-of-band (a logged-in CLI) don't.
    requires_api_key: bool
    # Whether the kind's models see attached images. True/False when the kind
    # settles it (every model behind the Claude CLI is multimodal); None when
    # it depends on the endpoint/model behind a generic HTTP row — the host
    # decides for those.
    vision: bool | None = None


PROVIDER_KINDS: dict[str, ProviderKindSpec] = {
    spec.kind: spec
    for spec in (
        ProviderKindSpec(
            kind="openai_compatible", label=None, requires_api_key=True
        ),
        ProviderKindSpec(
            kind="claude_cli",
            label="Claude Code CLI",
            requires_api_key=False,
            vision=True,
        ),
    )
}


def kind_spec(kind: str) -> ProviderKindSpec:
    return PROVIDER_KINDS[kind]


def display_name(kind: str, name: str) -> str:
    spec = PROVIDER_KINDS.get(kind)
    if spec is not None and spec.label:
        return spec.label
    return name[:1].upper() + name[1:]


__all__ = ["PROVIDER_KINDS", "ProviderKindSpec", "display_name", "kind_spec"]
