"""refresh_tokens — a family per sign-in and the time of each rotation

``family_id`` groups the tokens one sign-in rotates through, so a replay
revokes that device's session only. Existing tokens each start their own
family. ``rotated_at`` lets a reuse seconds after a rotation get a fresh token
instead of a sign-out.

Revision ID: 0030_refresh_token_families
Revises: 0029_compaction_event_types
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0030_refresh_token_families"
down_revision = "0029_compaction_event_types"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "refresh_tokens",
        sa.Column("family_id", UUID(as_uuid=True), nullable=True),
        schema="core",
    )
    op.execute("UPDATE core.refresh_tokens SET family_id = id")
    op.alter_column("refresh_tokens", "family_id", nullable=False, schema="core")
    op.create_index(
        "ix_refresh_tokens_family_id",
        "refresh_tokens",
        ["family_id"],
        schema="core",
    )
    op.add_column(
        "refresh_tokens",
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("refresh_tokens", "rotated_at", schema="core")
    op.drop_index(
        "ix_refresh_tokens_family_id", table_name="refresh_tokens", schema="core"
    )
    op.drop_column("refresh_tokens", "family_id", schema="core")
