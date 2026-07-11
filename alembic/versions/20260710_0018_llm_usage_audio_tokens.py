"""llm_usage: audio_input_tokens split

Audio input tokens are priced differently from text input tokens, so the
ledger needs the split to compute real cost. Nullable — only audio features
(voice STT, audio-input chat) populate it.

Revision ID: 0018_llm_usage_audio_tokens
Revises: 0017_email_verification
Create Date: 2026-07-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018_llm_usage_audio_tokens"
down_revision: Union[str, None] = "0017_email_verification"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_usage",
        sa.Column("audio_input_tokens", sa.BigInteger(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("llm_usage", "audio_input_tokens", schema="core")
