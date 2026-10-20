"""llm_prices: prompt-length tiers and where a price came from

A model priced by prompt length charges every token at a multiple once a
request's prompt passes a threshold; ``source`` tells a synced price from one
the operator set.

Revision ID: 0028_llm_price_tiers
Revises: 0027_usage_attribution_prices
Create Date: 2026-10-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0028_llm_price_tiers"
down_revision: Union[str, None] = "0027_usage_attribution_prices"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_prices", sa.Column("long_context_threshold", sa.Integer(), nullable=True), schema="core")
    op.add_column("llm_prices", sa.Column("long_context_multiplier", sa.Numeric(8, 4), nullable=True), schema="core")
    op.add_column("llm_prices", sa.Column("source", sa.String(32), nullable=False, server_default="manual"), schema="core")


def downgrade() -> None:
    op.drop_column("llm_prices", "source", schema="core")
    op.drop_column("llm_prices", "long_context_multiplier", schema="core")
    op.drop_column("llm_prices", "long_context_threshold", schema="core")
