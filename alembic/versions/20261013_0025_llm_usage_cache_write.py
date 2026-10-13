"""llm_usage: add cache_write_input_tokens (prompt-cache writes, apart)

A cache write costs more than a plain input token and a read far less; the
ledger kept reads apart but folded writes into input_tokens, so the cost of a
run could only be estimated. NULL on rows from before, and for providers that
do not report it.

Revision ID: 0025_llm_usage_cache_write
Revises: 0024_agents_reasoning_effort
Create Date: 2026-10-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0025_llm_usage_cache_write"
down_revision: Union[str, None] = "0024_agents_reasoning_effort"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_usage",
        sa.Column("cache_write_input_tokens", sa.BigInteger(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("llm_usage", "cache_write_input_tokens", schema="core")
