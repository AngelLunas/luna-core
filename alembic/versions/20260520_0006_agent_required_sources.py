"""add core.agents.required_sources column

Stores the list of context source names referenced from
``agents.instructions`` (e.g. ``${context.profile.name}`` => ``['profile']``).
The list is recomputed server-side on every agent create/update so it
stays consistent with the template; we never let clients write it
directly.

Revision ID: 0006_agent_required_sources
Revises: 0005_agent_provider_ollama
Create Date: 2026-05-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_agent_required_sources"
down_revision: Union[str, None] = "0005_agent_provider_ollama"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column(
            "required_sources",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("agents", "required_sources", schema="core")
