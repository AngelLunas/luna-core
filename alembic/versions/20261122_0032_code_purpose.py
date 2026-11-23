"""email_verification_codes — the purpose of each code

The same table now holds the codes that reset a password. ``purpose`` keeps
the two apart, so asking for one never burns the other. Existing rows are
email verification codes.

Revision ID: 0032_code_purpose
Revises: 0031_user_identities
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0032_code_purpose"
down_revision = "0031_user_identities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "email_verification_codes",
        sa.Column(
            "purpose",
            sa.String(32),
            nullable=False,
            server_default="verify_email",
        ),
        schema="core",
    )


def downgrade() -> None:
    op.execute("DELETE FROM core.email_verification_codes WHERE purpose <> 'verify_email'")
    op.drop_column("email_verification_codes", "purpose", schema="core")
