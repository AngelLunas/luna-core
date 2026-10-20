"""llm_usage: who and what each call was for; llm_prices: what tokens cost

``user_id`` / ``purpose`` / ``agent_name`` let a per-user budget sum a user's
spend and a report say what it went on (chat, a title, a look at photos, a
background review…) without mapping every kind of scope back to an owner.
NULL on rows from before. ``llm_prices`` holds the operator-entered price per
million tokens of each model, applied when the ledger is read.

Revision ID: 0027_usage_attribution_prices
Revises: 0026_agent_tool_grants_deferred
Create Date: 2026-10-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0027_usage_attribution_prices"
down_revision: Union[str, None] = "0026_agent_tool_grants_deferred"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_usage", sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True), schema="core")
    op.add_column("llm_usage", sa.Column("purpose", sa.String(64), nullable=True), schema="core")
    op.add_column("llm_usage", sa.Column("agent_name", sa.String(255), nullable=True), schema="core")
    op.create_index("ix_llm_usage_user_created", "llm_usage", ["user_id", "created_at"], schema="core")
    op.create_table(
        "llm_prices",
        sa.Column("model", sa.String(255), primary_key=True),
        sa.Column("input_per_mtok", sa.Numeric(12, 4), nullable=False),
        sa.Column("output_per_mtok", sa.Numeric(12, 4), nullable=False),
        sa.Column("cache_read_per_mtok", sa.Numeric(12, 4), nullable=True),
        sa.Column("cache_write_per_mtok", sa.Numeric(12, 4), nullable=True),
        sa.Column("audio_input_per_mtok", sa.Numeric(12, 4), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema="core",
    )


def downgrade() -> None:
    op.drop_table("llm_prices", schema="core")
    op.drop_index("ix_llm_usage_user_created", table_name="llm_usage", schema="core")
    op.drop_column("llm_usage", "agent_name", schema="core")
    op.drop_column("llm_usage", "purpose", schema="core")
    op.drop_column("llm_usage", "user_id", schema="core")
