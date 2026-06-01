"""operations: add parameters list + fixed_headers + fixed_body for visual editing

Revision ID: 0009_operation_parameters
Revises: 0008_llm_providers
Create Date: 2026-05-22
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_operation_parameters"
down_revision: Union[str, None] = "0008_llm_providers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # `parameters` is the source of truth for the visual operation editor.
    # `input_schema` (existing column) is derived from this list on save —
    # MCP tool definitions continue to read input_schema unchanged.
    op.add_column(
        "operations",
        sa.Column(
            "parameters",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        schema="core",
    )
    # Always-included headers (constants or `{param}` templates). Auth-related
    # headers still come from `credentials_encrypted` via the auth layer.
    op.add_column(
        "operations",
        sa.Column(
            "fixed_headers",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        schema="core",
    )
    # Optional JSON body template — merged on top of body-destined parameters
    # for POST/PUT/PATCH. `{param}` placeholders interpolate from input.
    op.add_column(
        "operations",
        sa.Column(
            "fixed_body",
            postgresql.JSONB(),
            nullable=True,
        ),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("operations", "fixed_body", schema="core")
    op.drop_column("operations", "fixed_headers", schema="core")
    op.drop_column("operations", "parameters", schema="core")
