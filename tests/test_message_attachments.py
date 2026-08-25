"""Media attached to a user turn is rendered for the model.

Every attached image gets an ``img-N`` label, numbered in order of appearance
across the conversation's user turns (the same order a host derives from its
stored messages, so tools can resolve the label back to the media row). A text
model sees ``[image attached: img-N]`` notes; a vision model sees the pixels
plus an ``[image attached: img-N (shown below)]`` note so it can still
reference the photo by label in tool calls. Raw media UUIDs are never shown.
"""
from __future__ import annotations

from luna_core.llm.providers.generic import (
    _canonical_to_openai_messages,
    _canonical_to_responses_input,
)


def test_image_block_rendered_as_label_note_alongside_text():
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
    assert "[image attached: img-1]" in user[0]["content"]
    assert "abc-123" not in user[0]["content"]  # raw media UUIDs never shown


def test_image_only_message_still_produces_a_user_turn():
    messages = [{"role": "user", "content": [{"type": "image", "media_id": "x"}]}]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert user[0]["content"] == "[image attached: img-1]"


def test_no_image_blocks_unchanged():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hola"}]}]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert user[0]["content"] == "hola"


def test_labels_number_across_turns_ignoring_tool_results():
    # Two photos in two user turns with a tool_result turn in between: the
    # counter spans the conversation and the tool_result must not shift it.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "primera foto"},
                {"type": "image", "media_id": "aaa"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "segunda foto"},
                {"type": "image", "media_id": "bbb"},
            ],
        },
    ]
    out = _canonical_to_openai_messages(messages, system="")
    notes = [
        m["content"]
        for m in out
        if m["role"] == "user" and "[image attached" in str(m["content"])
    ]
    assert "[image attached: img-1]" in notes[0]
    assert "[image attached: img-2]" in notes[1]


# --- vision path: a resolved image renders as an OpenAI image_url part -------

def test_resolved_image_rendered_as_image_url_part_with_label_note():
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
    # The rendered image still carries its label so the model can reference it.
    assert "[image attached: img-1 (shown below)]" in text_parts[0]["text"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,Zm9v"


def test_unresolved_image_falls_back_to_label_note():
    messages = [{"role": "user", "content": [{"type": "image", "media_id": "x"}]}]
    # Map present but this media_id is not in it → text-note path, no list.
    out = _canonical_to_openai_messages(messages, system="", image_urls={"other": "u"})
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert user[0]["content"] == "[image attached: img-1]"


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
    assert "[image attached: img-1 (shown below)]" in text  # the rendered one
    assert "[image attached: img-2]" in text  # the unresolved one, note only


# --- Responses API input: same labels, input_image parts ---------------------

def test_responses_input_labels_match_chat_completions():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "primera"},
                {"type": "image", "media_id": "aaa"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "segunda"},
                {"type": "image", "media_id": "bbb"},
            ],
        },
    ]
    items = _canonical_to_responses_input(
        messages, image_urls={"bbb": "data:image/png;base64,YmFy"}
    )
    users = [i for i in items if i.get("role") == "user"]
    first = "\n".join(p["text"] for p in users[0]["content"] if p["type"] == "input_text")
    second_texts = [p["text"] for p in users[1]["content"] if p["type"] == "input_text"]
    second_images = [p for p in users[1]["content"] if p["type"] == "input_image"]
    assert "[image attached: img-1]" in first
    assert any("[image attached: img-2 (shown below)]" in t for t in second_texts)
    assert len(second_images) == 1
    assert second_images[0]["image_url"] == "data:image/png;base64,YmFy"


# --- videos: vid-N, independent of img-N, never "watched" ---------------------


def test_video_block_rendered_as_vid_label_note():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "mira este clip"},
                {"type": "video", "media_id": "v-1"},
            ],
        }
    ]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert "[video attached: vid-1]" in user[0]["content"]
    assert "v-1" not in user[0]["content"]


def test_video_only_message_still_produces_a_user_turn():
    messages = [{"role": "user", "content": [{"type": "video", "media_id": "v"}]}]
    out = _canonical_to_openai_messages(messages, system="")
    user = [m for m in out if m["role"] == "user"]
    assert len(user) == 1
    assert user[0]["content"] == "[video attached: vid-1]"


def test_image_and_video_counters_are_independent():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "media_id": "a"},
                {"type": "video", "media_id": "b"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "media_id": "c"},
                {"type": "image", "media_id": "d"},
            ],
        },
    ]
    out = _canonical_to_openai_messages(messages, system="")
    notes = [m["content"] for m in out if m["role"] == "user"]
    assert notes[0] == "[image attached: img-1]\n[video attached: vid-1]"
    assert notes[1] == "[video attached: vid-2]\n[image attached: img-2]"


def test_resolved_video_renders_its_poster_as_an_image_with_the_first_frame_note():
    messages = [{"role": "user", "content": [{"type": "video", "media_id": "v"}]}]
    out = _canonical_to_openai_messages(
        messages, system="", image_urls={"v": "data:image/jpeg;base64,AAA="}
    )
    user = [m for m in out if m["role"] == "user"][0]
    parts = user["content"]
    assert parts[0] == {
        "type": "text",
        "text": "[video attached: vid-1 (first frame shown below)]",
    }
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA="}}


def test_responses_input_labels_videos_the_same_way():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hola"},
                {"type": "image", "media_id": "a"},
                {"type": "video", "media_id": "b"},
            ],
        }
    ]
    items = _canonical_to_responses_input(messages, image_urls={"b": "data:image/jpeg;base64,AAA="})
    parts = items[0]["content"]
    texts = [p["text"] for p in parts if p["type"] == "input_text"]
    assert texts == ["hola", "[image attached: img-1]", "[video attached: vid-1 (first frame shown below)]"]
    assert [p for p in parts if p["type"] == "input_image"] == [
        {"type": "input_image", "image_url": "data:image/jpeg;base64,AAA="}
    ]
