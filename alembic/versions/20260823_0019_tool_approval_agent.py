"""tool_approvals.agent_name — who proposed the gated call

An approval row said which tool was proposed but not which agent proposed it,
so a client could only assume one. Records the agent's canonical name, so the
card can name the proposer instead of guessing.

Nullable, and left NULL on existing rows: who proposed them was never recorded,
and a client should fall back to a generic label rather than invent an author.

Revision ID: 0019_tool_approval_agent
Revises: 0018_llm_usage_audio_tokens
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019_tool_approval_agent"
down_revision = "0018_llm_usage_audio_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tool_approvals",
        sa.Column("agent_name", sa.String(length=255), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("tool_approvals", "agent_name", schema="core")
