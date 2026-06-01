"""agent_system_tool_grants: per-agent grants for in-process system tools

Revision ID: 0010_agent_system_tool_grants
Revises: 0009_operation_parameters
Create Date: 2026-05-23

System tools (``stash_records``, ``yield_iteration``, etc.) live in the
in-process registry at ``luna_core.mcp.system_tools`` — not in
``core.operations`` — because they don't have a connector, HTTP method,
or URL. This table records which agent has access to which system tool
by *name*, mirroring how ``core.agent_operations`` records access to
connector-backed operations by id. Both are unioned by the AgentRunner
at run time to decide the agent's effective tool set.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_agent_system_tool_grants"
down_revision: Union[str, None] = "0009_operation_parameters"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_system_tool_grants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.UniqueConstraint(
            "agent_id", "tool_name", name="uq_agent_system_tool_grants_pair"
        ),
        schema="core",
    )
    op.create_index(
        "ix_agent_system_tool_grants_agent_id",
        "agent_system_tool_grants",
        ["agent_id"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_system_tool_grants_agent_id",
        table_name="agent_system_tool_grants",
        schema="core",
    )
    op.drop_table("agent_system_tool_grants", schema="core")
