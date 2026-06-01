"""operations: add retry_policy column for transient-failure backoff

Revision ID: 0011_operations_retry_policy
Revises: 0010_agent_system_tool_grants
Create Date: 2026-05-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_operations_retry_policy"
down_revision: Union[str, None] = "0010_agent_system_tool_grants"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NULL = no retry (legacy behavior). When set, the executor reads this
    # blob into luna_core.connectors.retry.RetryPolicy at call time and
    # applies exponential backoff for statuses in retry_on_status.
    op.add_column(
        "operations",
        sa.Column("retry_policy", postgresql.JSONB(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("operations", "retry_policy", schema="core")
