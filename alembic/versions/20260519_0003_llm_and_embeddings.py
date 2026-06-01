"""llm & mcp phase: AgentMessage partial/thinking, embeddings table

Revision ID: 0003_llm_and_embeddings
Revises: 0002_flow_engine
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003_llm_and_embeddings"
down_revision: Union[str, None] = "0002_flow_engine"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_messages",
        sa.Column(
            "is_partial",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        schema="core",
    )
    op.add_column(
        "agent_messages",
        sa.Column("thinking", sa.Text(), nullable=True),
        schema="core",
    )

    op.create_table(
        "embeddings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("collection", sa.String(255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("vector", Vector(1024), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        schema="core",
    )
    op.create_index(
        "ix_embeddings_collection",
        "embeddings",
        ["collection"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_embeddings_collection", table_name="embeddings", schema="core"
    )
    op.drop_table("embeddings", schema="core")

    op.drop_column("agent_messages", "thinking", schema="core")
    op.drop_column("agent_messages", "is_partial", schema="core")
