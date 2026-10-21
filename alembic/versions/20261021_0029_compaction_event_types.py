"""run_event_type — add the conversation compaction lifecycle values

``compaction_started`` / ``compaction_completed`` ride a conversation's pub/sub
channel only, but the Postgres enum must not drift from the Python enum (see
0020).

Downgrade is a no-op: Postgres cannot drop enum values.

Revision ID: 0029_compaction_event_types
Revises: 0028_llm_price_tiers
"""
from __future__ import annotations

from alembic import op

revision = "0029_compaction_event_types"
down_revision = "0028_llm_price_tiers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS 'compaction_started'"
    )
    op.execute(
        "ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS 'compaction_completed'"
    )


def downgrade() -> None:
    pass
