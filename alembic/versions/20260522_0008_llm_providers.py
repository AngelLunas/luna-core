"""dynamic llm providers: drop agent_provider enum, add core.llm_providers + agents.llm_provider_id FK

Revision ID: 0008_llm_providers
Revises: 0007_run_events_streaming
Create Date: 2026-05-22
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_llm_providers"
down_revision: Union[str, None] = "0007_run_events_streaming"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_providers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("base_url", sa.String(1024), nullable=False),
        sa.Column("chat_url", sa.String(1024), nullable=True),
        sa.Column("models_url", sa.String(1024), nullable=True),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("name", name="uq_llm_providers_name"),
        schema="core",
    )
    op.create_index(
        "ix_llm_providers_name",
        "llm_providers",
        ["name"],
        unique=True,
        schema="core",
    )

    # Reset existing agents: switching from a static enum to a FK has no
    # data-preserving migration path (we can't synthesize provider rows
    # without API keys). Dev/seed data is recreated on next startup.
    op.execute("DELETE FROM core.agent_operations")
    op.execute("DELETE FROM core.agents")

    op.add_column(
        "agents",
        sa.Column(
            "llm_provider_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.llm_providers.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        schema="core",
    )
    op.create_index(
        "ix_agents_llm_provider_id",
        "agents",
        ["llm_provider_id"],
        schema="core",
    )

    op.drop_column("agents", "provider", schema="core")
    op.execute("DROP TYPE IF EXISTS core.agent_provider")


def downgrade() -> None:
    # Recreate the enum and column; existing rows are dropped (mirror of
    # the upgrade's reset).
    op.execute(
        "CREATE TYPE core.agent_provider AS ENUM "
        "('kimi', 'anthropic', 'openai', 'ollama')"
    )
    op.execute("DELETE FROM core.agent_operations")
    op.execute("DELETE FROM core.agents")
    op.add_column(
        "agents",
        sa.Column(
            "provider",
            postgresql.ENUM(
                name="agent_provider", schema="core", create_type=False
            ),
            nullable=False,
        ),
        schema="core",
    )

    op.drop_index(
        "ix_agents_llm_provider_id", table_name="agents", schema="core"
    )
    op.drop_column("agents", "llm_provider_id", schema="core")

    op.drop_index(
        "ix_llm_providers_name", table_name="llm_providers", schema="core"
    )
    op.drop_table("llm_providers", schema="core")
