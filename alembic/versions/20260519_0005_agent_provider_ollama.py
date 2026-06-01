"""add 'ollama' value to core.agent_provider enum

Revision ID: 0005_agent_provider_ollama
Revises: 0004_roles_and_permissions
Create Date: 2026-05-19
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0005_agent_provider_ollama"
down_revision: Union[str, None] = "0004_roles_and_permissions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction block in
    # PostgreSQL < 12 and is finicky under Alembic's default transactional
    # DDL, so we run it in an autocommit block.
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE core.agent_provider ADD VALUE IF NOT EXISTS 'ollama'"
        )


def downgrade() -> None:
    # PostgreSQL has no DROP VALUE for enum types. Rolling back requires
    # recreating the type without 'ollama' and re-casting every column that
    # references it — intentionally left as a no-op since no rows can exist
    # for a value that was just added.
    pass
