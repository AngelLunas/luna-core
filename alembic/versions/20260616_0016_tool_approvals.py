"""tool_approvals + per-grant requires_approval + approval event types

Human-in-the-loop tool approval (chat). Adds:
  - ``tool_approval_required`` / ``tool_approval_resolved`` to core.run_event_type
    (ALTER TYPE ADD VALUE, same pattern as 0012 — not consumed in this tx).
  - ``requires_approval`` flag on both grant tables (per-agent, per-tool).
  - ``core.tool_approvals`` — the durable "intent to execute" a gated tool.

Downgrade leaves the enum values in place (Postgres has no portable DROP VALUE).

Revision ID: 0016_tool_approvals
Revises: 0015_llm_usage
Create Date: 2026-06-16
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016_tool_approvals"
down_revision: Union[str, None] = "0015_llm_usage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW_ENUM_VALUES = ("tool_approval_required", "tool_approval_resolved")


def upgrade() -> None:
    for value in NEW_ENUM_VALUES:
        op.execute(
            f"ALTER TYPE core.run_event_type ADD VALUE IF NOT EXISTS '{value}'"
        )

    op.add_column(
        "agent_operations",
        sa.Column(
            "requires_approval",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        schema="core",
    )
    op.add_column(
        "agent_system_tool_grants",
        sa.Column(
            "requires_approval",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        schema="core",
    )

    op.create_table(
        "tool_approvals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_use_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column(
            "tool_input",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "resolved_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "conversation_id", "tool_use_id", name="uq_tool_approvals_tool_use"
        ),
        schema="core",
    )
    op.create_index(
        "ix_tool_approvals_conv_status",
        "tool_approvals",
        ["conversation_id", "status"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tool_approvals_conv_status",
        table_name="tool_approvals",
        schema="core",
    )
    op.drop_table("tool_approvals", schema="core")
    op.drop_column("agent_system_tool_grants", "requires_approval", schema="core")
    op.drop_column("agent_operations", "requires_approval", schema="core")
    # Enum values intentionally left in place (no portable DROP VALUE).
