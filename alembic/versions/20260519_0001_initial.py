"""initial schema: core schema, users, refresh_tokens, pgvector extension

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
# "core" is the luna-core migration branch. luna-core and any host app
# (e.g. luna-sentinel) keep independent linear chains identified by their
# branch labels; alembic upgrade heads applies all branches.
branch_labels: Union[str, Sequence[str], None] = ("core",)
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE SCHEMA IF NOT EXISTS core")

    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
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
        sa.UniqueConstraint("email", name="uq_users_email"),
        schema="core",
    )
    op.create_index(
        "ix_users_email", "users", ["email"], unique=True, schema="core"
    )

    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "revoked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
        schema="core",
    )
    op.create_index(
        "ix_refresh_tokens_token_hash",
        "refresh_tokens",
        ["token_hash"],
        unique=True,
        schema="core",
    )
    op.create_index(
        "ix_refresh_tokens_user_id",
        "refresh_tokens",
        ["user_id"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_refresh_tokens_user_id", table_name="refresh_tokens", schema="core"
    )
    op.drop_index(
        "ix_refresh_tokens_token_hash", table_name="refresh_tokens", schema="core"
    )
    op.drop_table("refresh_tokens", schema="core")
    op.drop_index("ix_users_email", table_name="users", schema="core")
    op.drop_table("users", schema="core")
    op.execute("DROP SCHEMA IF EXISTS core CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
