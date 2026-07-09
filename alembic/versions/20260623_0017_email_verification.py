"""email verification: users.is_verified/verified_at + email_verification_codes

Adds email-ownership confirmation via a short numeric code typed into the app.
New columns on core.users and a single-use, hashed-code table (with an attempt
counter to bound brute force).

Existing accounts are backfilled to is_verified=true so enabling the gate doesn't
lock out anyone who registered before this — only NEW sign-ups start unverified.

Revision ID: 0017_email_verification
Revises: 0016_tool_approvals
Create Date: 2026-06-23
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_email_verification"
down_revision: Union[str, None] = "0016_tool_approvals"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        schema="core",
    )
    op.add_column(
        "users",
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        schema="core",
    )
    # Don't lock out anyone who registered before verification existed.
    op.execute("UPDATE core.users SET is_verified = true")

    op.create_table(
        "email_verification_codes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="core",
    )
    op.create_index(
        "ix_email_verification_codes_user_id",
        "email_verification_codes",
        ["user_id"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_email_verification_codes_user_id",
        table_name="email_verification_codes",
        schema="core",
    )
    op.drop_table("email_verification_codes", schema="core")
    op.drop_column("users", "verified_at", schema="core")
    op.drop_column("users", "is_verified", schema="core")
