"""user_identities — signing in with Google or Apple

An account can carry outside identities, one per provider account, and no
longer needs a password.

Revision ID: 0031_user_identities
Revises: 0030_refresh_token_families
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0031_user_identities"
down_revision = "0030_refresh_token_families"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_identities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("core.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("subject", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("provider_refresh_token", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "provider", "subject", name="uq_user_identities_provider_subject"
        ),
        schema="core",
    )
    op.create_index(
        "ix_user_identities_user_id", "user_identities", ["user_id"], schema="core"
    )
    op.alter_column("users", "password_hash", nullable=True, schema="core")


def downgrade() -> None:
    op.alter_column("users", "password_hash", nullable=False, schema="core")
    op.drop_index("ix_user_identities_user_id", table_name="user_identities", schema="core")
    op.drop_table("user_identities", schema="core")
