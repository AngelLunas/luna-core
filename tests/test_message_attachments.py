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
