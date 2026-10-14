"""agent_system_tool_grants: add deferred (granted, loaded on demand)

A grant marked deferred keeps the tool available without putting its
definition in every call: a provider that can search tools lists it in a
one-line index and loads it when the model asks; any other provider sends it
like every other tool. False on every existing grant — nothing changes until
an agent opts in.

Revision ID: 0026_agent_tool_grants_deferred
Revises: 0025_llm_usage_cache_write
Create Date: 2026-10-14
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0026_agent_tool_grants_deferred"
down_revision: Union[str, None] = "0025_llm_usage_cache_write"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_system_tool_grants",
        sa.Column("deferred", sa.Boolean(), nullable=False, server_default="false"),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("agent_system_tool_grants", "deferred", schema="core")
