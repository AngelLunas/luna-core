"""llm_providers.kind — which implementation drives a provider row

Until now every row was an OpenAI-compatible HTTP endpoint (GenericProvider).
The claude-cli provider shells out to the local Claude Code binary on
subscription auth instead, so rows need a discriminator the router can branch
on. Existing rows are HTTP providers by definition — the server default
"openai_compatible" backfills them correctly.

For kind=claude_cli rows, base_url holds the binary path and api_key stays
unused (auth is the machine's `claude login`).

Revision ID: 0022_llm_provider_kind
Revises: 0021_conversation_message_agent
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022_llm_provider_kind"
down_revision = "0021_conversation_message_agent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_providers",
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default="openai_compatible",
        ),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("llm_providers", "kind", schema="core")
