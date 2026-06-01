"""conversations + conversation_messages: persistent chat primitive

Adds a generic conversation/thread primitive to core. Unlike
``agent_messages`` (run-scoped audit log, purged on compaction), a
conversation survives the flow that created it and can be referenced
from any domain table (e.g. ``sentinel.cover_letters.conversation_id``).

The shape is intentionally naked: no FK back to agents or runs. The
owning domain table holds the FK to the conversation, not the other
way around — that lets the same primitive serve cover letters today,
debug threads tomorrow, interview prep after that, without adding
domain knowledge to core.

``content`` mirrors the JSONB array-of-blocks shape that ``AgentMessage``
already uses (text + tool_use + tool_result), so existing renderers and
serializers transfer 1:1.

Revision ID: 0013_conversations
Revises: 0012_iteration_events
Create Date: 2026-05-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_conversations"
down_revision: Union[str, None] = "0012_iteration_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="core",
    )

    op.create_table(
        "conversation_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        # Role mirrors AgentMessageRole values (user/assistant/system).
        # Tool calls/results live as content blocks inside an assistant or
        # user message — same pattern Anthropic uses — so we deliberately
        # don't add a separate "tool" role here.
        sa.Column(
            "role",
            sa.Enum(
                "user",
                "assistant",
                "system",
                name="conversation_message_role",
                schema="core",
                native_enum=True,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "content",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "is_partial",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "conversation_id", "sequence", name="uq_conversation_messages_sequence"
        ),
        schema="core",
    )
    op.create_index(
        "ix_conversation_messages_conv_seq",
        "conversation_messages",
        ["conversation_id", "sequence"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_conv_seq",
        table_name="conversation_messages",
        schema="core",
    )
    op.drop_table("conversation_messages", schema="core")
    op.drop_table("conversations", schema="core")
    op.execute("DROP TYPE IF EXISTS core.conversation_message_role")
