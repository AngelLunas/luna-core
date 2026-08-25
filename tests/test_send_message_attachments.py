"""``SendMessageRequest``: typed attachments alongside the legacy image ids."""
from __future__ import annotations

import uuid

import pytest

from luna_core.schemas.conversation import SendMessageRequest


def test_attachments_alone_satisfy_the_text_or_media_rule():
    mid = uuid.uuid4()
    req = SendMessageRequest(attachments=[{"type": "video", "media_id": mid}])
    assert req.attachment_blocks() == [{"type": "video", "media_id": str(mid)}]
    assert req.attachment_media_ids() == [mid]


def test_nothing_attached_and_no_text_is_rejected():
    with pytest.raises(ValueError):
        SendMessageRequest()


def test_legacy_media_ids_come_first_as_images_and_duplicates_collapse():
    a, b = uuid.uuid4(), uuid.uuid4()
    req = SendMessageRequest(
        new_message="x",
        media_ids=[a],
        attachments=[{"type": "image", "media_id": a}, {"type": "video", "media_id": b}],
    )
    assert req.attachment_blocks() == [
        {"type": "image", "media_id": str(a)},
        {"type": "video", "media_id": str(b)},
    ]


def test_unknown_attachment_type_is_rejected():
    with pytest.raises(ValueError):
        SendMessageRequest(attachments=[{"type": "gif", "media_id": uuid.uuid4()}])
