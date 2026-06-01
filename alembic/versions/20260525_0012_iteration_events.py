"""run_events: iteration_started / iteration_completed / iteration_failed

Adds per-iteration lifecycle values to ``core.run_event_type``. These are
emitted once per item processed by a scratchpad-mode ai_agent node (both
sequential and parallel execution modes), and every event emitted from
inside an iteration carries the same ``iteration_id`` in its payload so
the UI can group them under the iteration block they belong to.

Pattern mirrors the streaming-types migration (0007): plain
``ALTER TYPE ... ADD VALUE`` inside the implicit transaction is fine
because the new values are not consumed in this same transaction.

Downgrade is intentionally a no-op for the enum side (no portable
DROP VALUE in Postgres; rebuilding would force a destructive rewrite
of every row that ever used the new values).

Revision ID: 0012_iteration_events
Revises: 0011_operations_retry_policy
Create Date: 2026-05-25
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0012_iteration_events"
down_revision: Union[str, None] = "0011_operations_retry_policy"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


NEW_ENUM_VALUES = (
    "iteration_started",
    "iteration_completed",
    "iteration_failed",
)


def upgrade() -> None:
    for value in NEW_ENUM_VALUES:
        op.execute(
            f"ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS '{value}'"
        )


def downgrade() -> None:
    # Enum values are intentionally left in place: Postgres has no portable
    # DROP VALUE, and rebuilding the type would force a destructive rewrite
    # of every row in core.run_events that references the new values.
    pass
