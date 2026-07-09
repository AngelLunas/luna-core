"""conversations: optional owner (user_id) + title

Adds an optional owner and a human label to the conversation primitive so host
apps can expose conversations directly over an API and scope/list them per user.
``user_id`` is nullable — a domain-owned conversation (e.g. sentinel cover
letters) leaves it null; it does not couple the engine to any domain (``User``
is a core entity, ownership is generic multi-tenancy).

Revision ID: 0014_conversation_owner
Revises: 0013_conversations
Create Date: 2026-06-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0014_conversation_owner"
down_revision: Union[str, None] = "0013_conversations"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        schema="core",
    )
    op.add_column(
        "conversations",
        sa.Column("title", sa.String(length=255), nullable=True),
        schema="core",
    )
    op.create_index(
        "ix_core_conversations_user_id",
        "conversations",
        ["user_id"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_core_conversations_user_id",
        table_name="conversations",
        schema="core",
    )
    op.drop_column("conversations", "title", schema="core")
    op.drop_column("conversations", "user_id", schema="core")
