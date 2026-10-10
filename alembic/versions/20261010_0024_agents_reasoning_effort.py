"""agents: add reasoning_effort (how hard the model thinks before answering)

One of luna_core.llm.base.REASONING_EFFORTS, or NULL for the model's own
default — so every existing agent keeps behaving exactly as before.

Revision ID: 0024_agents_reasoning_effort
Revises: 0023_agents_builtin_tools
Create Date: 2026-10-10
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0024_agents_reasoning_effort"
down_revision: Union[str, None] = "0023_agents_builtin_tools"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column("reasoning_effort", sa.String(16), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("agents", "reasoning_effort", schema="core")
