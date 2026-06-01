"""roles & permissions: user.role column, core.permissions table

Revision ID: 0004_roles_and_permissions
Revises: 0003_llm_and_embeddings
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_roles_and_permissions"
down_revision: Union[str, None] = "0003_llm_and_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "role",
            sa.String(length=64),
            nullable=False,
            server_default="user",
        ),
        schema="core",
    )

    op.create_table(
        "permissions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("app", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("permission", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "app", "role", "permission", name="uq_permissions_app_role_permission"
        ),
        schema="core",
    )
    op.create_index(
        "ix_permissions_app", "permissions", ["app"], schema="core"
    )
    op.create_index(
        "ix_permissions_role", "permissions", ["role"], schema="core"
    )


def downgrade() -> None:
    op.drop_index("ix_permissions_role", table_name="permissions", schema="core")
    op.drop_index("ix_permissions_app", table_name="permissions", schema="core")
    op.drop_table("permissions", schema="core")
    op.drop_column("users", "role", schema="core")
