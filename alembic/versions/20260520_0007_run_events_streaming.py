"""run_events streaming types + flow_runs.cleared_at

Adds the four streaming-related enum values (agent_message_started,
agent_text_delta, agent_thinking_delta, agent_message_completed) plus
``run_cleared`` to ``core.run_event_type``, and the nullable
``cleared_at`` timestamp on ``core.flow_runs`` used by the soft-compact
delete (DELETE /runs/{id}).

Postgres ≥ 12 allows ``ALTER TYPE ... ADD VALUE`` inside a transaction
block as long as the new value is not used in the same transaction; this
migration only adds values and an unrelated column, so a plain ``op.execute``
is enough — no ``autocommit_block`` (which also breaks under alembic's
async runner, where the outer transaction it expects to exit isn't open).

Downgrade is intentionally a no-op for the enum side: Postgres has no
portable DROP VALUE, and rebuilding the type would force a destructive
rewrite of every row in core.run_events that references the new values.
The column is dropped on downgrade.

Revision ID: 0007_run_events_streaming
Revises: 0006_agent_required_sources
Create Date: 2026-05-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Keep ≤ 32 chars: alembic's default `alembic_version.version_num` column
# is VARCHAR(32), and a longer id makes `alembic upgrade` fail with a
# StringDataRightTruncationError when it tries to record the new head.
revision: str = "0007_run_events_streaming"
down_revision: Union[str, None] = "0006_agent_required_sources"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


NEW_ENUM_VALUES = (
    "agent_message_started",
    "agent_text_delta",
    "agent_thinking_delta",
    "agent_message_completed",
    "run_cleared",
)


def upgrade() -> None:
    for value in NEW_ENUM_VALUES:
        op.execute(
            f"ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS '{value}'"
        )

    op.add_column(
        "flow_runs",
        sa.Column("cleared_at", sa.DateTime(timezone=True), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("flow_runs", "cleared_at", schema="core")
    # Enum values are intentionally left in place: Postgres has no portable
    # DROP VALUE, and rebuilding the type would force a destructive rewrite
    # of every row in core.run_events that references the new values.
