"""agents: add builtin_tools (provider built-in tools, e.g. web_search)

The ``Agent.builtin_tools`` column landed with the Responses API work
(``feat(llm): OpenAI Responses API path``) but never got a migration, so
any database migrated to 0018 fails every ``SELECT ... FROM core.agents``
with ``UndefinedColumnError``. This backfills the column exactly as the
model declares it: ``text[] NOT NULL DEFAULT '{}'``.

Revision ID: 0023_agents_builtin_tools
Revises: 0022_llm_provider_kind
Create Date: 2026-09-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_agents_builtin_tools"
down_revision: Union[str, None] = "0022_llm_provider_kind"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agents",
        sa.Column(
            "builtin_tools",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("agents", "builtin_tools", schema="core")
