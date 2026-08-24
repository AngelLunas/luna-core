"""conversation_messages.agent_name — who authored an assistant turn

A multi-agent chat renders per-bubble identity (avatar + name). The live
stream already carries the agent on its events, but a reloaded transcript had
no attribution, so historic bubbles lost their author. Records the agent's
canonical name on assistant rows.

Nullable, and left NULL on existing rows: who wrote them was never recorded,
and a client should fall back to a generic label rather than invent an author.

Revision ID: 0021_conversation_message_agent
Revises: 0020_routing_event_types
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_conversation_message_agent"
down_revision = "0020_routing_event_types"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages",
        sa.Column("agent_name", sa.String(length=255), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("conversation_messages", "agent_name", schema="core")
