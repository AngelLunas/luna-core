"""llm_usage: generic token-usage ledger

One row per completed assistant turn with the real input/output/cached token
counts, keyed by the execution scope (flow_run_id or conversation_id — no FK,
polymorphic) and the assistant message_id. Raw infrastructure: no owner column;
host apps map scope → user/credits from their own data.

Revision ID: 0015_llm_usage
Revises: 0014_conversation_owner
Create Date: 2026-06-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_llm_usage"
down_revision: Union[str, None] = "0014_conversation_owner"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("scope_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column(
            "input_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "output_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("cached_input_tokens", sa.BigInteger(), nullable=True),
        sa.Column(
            "total_tokens", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="core",
    )
    op.create_index(
        "ix_llm_usage_scope", "llm_usage", ["scope_id"], schema="core"
    )


def downgrade() -> None:
    op.drop_index("ix_llm_usage_scope", table_name="llm_usage", schema="core")
    op.drop_table("llm_usage", schema="core")
