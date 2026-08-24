"""run_event_type — add the chat routing lifecycle values

``routing_started`` / ``routing_decided`` are emitted on a conversation's
pub/sub channel only (chat scopes never persist events), but the Postgres enum
backs the persisted flow path and must not drift from the Python enum — a
future flow that emits them would otherwise fail at INSERT.

Downgrade is a no-op: Postgres cannot drop enum values, and unused values are
harmless.

Revision ID: 0020_routing_event_types
Revises: 0019_tool_approval_agent
"""
from __future__ import annotations

from alembic import op

revision = "0020_routing_event_types"
down_revision = "0019_tool_approval_agent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS 'routing_started'"
    )
    op.execute(
        "ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS 'routing_decided'"
    )


def downgrade() -> None:
    pass
