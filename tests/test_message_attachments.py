"""Media attached to a user turn is rendered for the model.

For a text model the provider renders each ``image`` content block as a text note
(so the agent knows a media is present and can pass its id to a tool). A
vision-native model will instead render it as an ``image_url`` part (M3).
"""
from __future__ import annotations

from luna_core.llm.providers.generic import _canonical_to_openai_messages


def test_image_block_rendered_as_text_note_alongside_text():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "crea la planta X"},
                {"type": "image", "media_id": "abc-123"},
            ],
        }
    ]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert "crea la planta X" in user[0]["content"]
    assert "media_id=abc-123" in user[0]["content"]


def test_image_only_message_still_produces_a_user_turn():
    messages = [{"role": "user", "content": [{"type": "image", "media_id": "x"}]}]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert "media_id=x" in user[0]["content"]


def test_no_image_blocks_unchanged():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hola"}]}]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert user[0]["content"] == "hola"


# --- vision path: a resolved image renders as an OpenAI image_url part -------

def test_resolved_image_rendered_as_image_url_part():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "que tiene esta hoja?"},
                {"type": "image", "media_id": "abc-123"},
            ],
        }
    ]
    out = _canonical_to_openai_messages(
        messages, system="", image_urls={"abc-123": "data:image/jpeg;base64,Zm9v"}
    )
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    parts = user[0]["content"]
    assert isinstance(parts, list)
    text_parts = [p for p in parts if p["type"] == "text"]
    image_parts = [p for p in parts if p["type"] == "image_url"]
    assert text_parts and "que tiene esta hoja?" in text_parts[0]["text"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,Zm9v"


def test_unresolved_image_falls_back_to_text_note():
    messages = [{"role": "user", "content": [{"type": "image", "media_id": "x"}]}]
    # Map present but this media_id is not in it → text-note path, no list.
    out = _canonical_to_openai_messages(messages, system="", image_urls={"other": "u"})
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert user[0]["content"] == "[image attached: media_id=x]"


def test_mixed_resolved_and_unresolved_images():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "mira"},
                {"type": "image", "media_id": "seen"},
                {"type": "image", "media_id": "unseen"},
            ],
        }
    ]
    out = _canonical_to_openai_messages(
        messages, system="", image_urls={"seen": "data:image/png;base64,YmFy"}
    )
    parts = [m for m in out if m["role"] == "user"][0]["content"]
    image_parts = [p for p in parts if p["type"] == "image_url"]
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert len(image_parts) == 1  # only the resolved one
    assert "mira" in text
    assert "media_id=unseen" in text  # the unresolved one rides as a note
