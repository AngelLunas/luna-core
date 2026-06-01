"""flow engine: connectors, operations, agents, flows, runs, events, messages

Revision ID: 0002_flow_engine
Revises: 0001_initial
Create Date: 2026-05-19
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_flow_engine"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CONNECTOR_AUTH_TYPE = postgresql.ENUM(
    "none", "api_key", "oauth2", "basic",
    name="connector_auth_type",
    schema="core",
    create_type=False,
)
HTTP_METHOD = postgresql.ENUM(
    "GET", "POST", "PUT", "DELETE", "PATCH",
    name="http_method",
    schema="core",
    create_type=False,
)
AGENT_PROVIDER = postgresql.ENUM(
    "kimi", "anthropic", "openai",
    name="agent_provider",
    schema="core",
    create_type=False,
)
FLOW_RUN_STATUS = postgresql.ENUM(
    "pending", "running", "paused", "completed", "failed",
    name="flow_run_status",
    schema="core",
    create_type=False,
)
RUN_EVENT_TYPE = postgresql.ENUM(
    "flow_started", "flow_completed", "flow_failed",
    "node_started", "node_completed", "node_failed",
    "agent_thinking", "tool_called", "tool_result",
    "human_checkpoint", "human_response",
    name="run_event_type",
    schema="core",
    create_type=False,
)
AGENT_MESSAGE_ROLE = postgresql.ENUM(
    "system", "user", "assistant",
    name="agent_message_role",
    schema="core",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    CONNECTOR_AUTH_TYPE.create(bind, checkfirst=True)
    HTTP_METHOD.create(bind, checkfirst=True)
    AGENT_PROVIDER.create(bind, checkfirst=True)
    FLOW_RUN_STATUS.create(bind, checkfirst=True)
    RUN_EVENT_TYPE.create(bind, checkfirst=True)
    AGENT_MESSAGE_ROLE.create(bind, checkfirst=True)

    # connectors -------------------------------------------------------------
    op.create_table(
        "connectors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("auth_type", CONNECTOR_AUTH_TYPE, nullable=False, server_default="none"),
        sa.Column("base_url", sa.String(1024), nullable=False),
        sa.Column("credentials_encrypted", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_connectors_name"),
        schema="core",
    )
    op.create_index("ix_connectors_name", "connectors", ["name"], unique=True, schema="core")

    # operations -------------------------------------------------------------
    op.create_table(
        "operations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "connector_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.connectors.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("method", HTTP_METHOD, nullable=False),
        sa.Column("path", sa.String(1024), nullable=False),
        sa.Column("input_schema", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("output_schema", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("connector_id", "name", name="uq_operations_connector_name"),
        schema="core",
    )
    op.create_index("ix_operations_connector_id", "operations", ["connector_id"], schema="core")

    # agents -----------------------------------------------------------------
    op.create_table(
        "agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(255), nullable=False, server_default=""),
        sa.Column("instructions", sa.Text(), nullable=False, server_default=""),
        sa.Column("provider", AGENT_PROVIDER, nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("output_schema", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_agents_name"),
        schema="core",
    )
    op.create_index("ix_agents_name", "agents", ["name"], unique=True, schema="core")

    # agent_operations -------------------------------------------------------
    op.create_table(
        "agent_operations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.agents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "operation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.operations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.UniqueConstraint("agent_id", "operation_id", name="uq_agent_operations_pair"),
        schema="core",
    )
    op.create_index("ix_agent_operations_agent_id", "agent_operations", ["agent_id"], schema="core")
    op.create_index("ix_agent_operations_operation_id", "agent_operations", ["operation_id"], schema="core")

    # flows ------------------------------------------------------------------
    op.create_table(
        "flows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("definition", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_flows_name"),
        schema="core",
    )
    op.create_index("ix_flows_name", "flows", ["name"], unique=True, schema="core")

    # flow_runs --------------------------------------------------------------
    op.create_table(
        "flow_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "flow_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.flows.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", FLOW_RUN_STATUS, nullable=False, server_default="pending"),
        sa.Column("trigger", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("state", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema="core",
    )
    op.create_index("ix_flow_runs_flow_id", "flow_runs", ["flow_id"], schema="core")
    op.create_index("ix_flow_runs_status", "flow_runs", ["status"], schema="core")

    # run_events -------------------------------------------------------------
    op.create_table(
        "run_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "flow_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.flow_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("event_type", RUN_EVENT_TYPE, nullable=False),
        sa.Column("node_id", sa.String(255), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint("flow_run_id", "sequence", name="uq_run_events_run_sequence"),
        schema="core",
    )
    op.create_index(
        "ix_run_events_run_sequence",
        "run_events",
        ["flow_run_id", "sequence"],
        schema="core",
    )

    # agent_messages ---------------------------------------------------------
    op.create_table(
        "agent_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "flow_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("core.flow_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("node_id", sa.String(255), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("role", AGENT_MESSAGE_ROLE, nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("flow_run_id", "sequence", name="uq_agent_messages_run_sequence"),
        schema="core",
    )
    op.create_index(
        "ix_agent_messages_run_node",
        "agent_messages",
        ["flow_run_id", "node_id", "sequence"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index("ix_agent_messages_run_node", table_name="agent_messages", schema="core")
    op.drop_table("agent_messages", schema="core")

    op.drop_index("ix_run_events_run_sequence", table_name="run_events", schema="core")
    op.drop_table("run_events", schema="core")

    op.drop_index("ix_flow_runs_status", table_name="flow_runs", schema="core")
    op.drop_index("ix_flow_runs_flow_id", table_name="flow_runs", schema="core")
    op.drop_table("flow_runs", schema="core")

    op.drop_index("ix_flows_name", table_name="flows", schema="core")
    op.drop_table("flows", schema="core")

    op.drop_index("ix_agent_operations_operation_id", table_name="agent_operations", schema="core")
    op.drop_index("ix_agent_operations_agent_id", table_name="agent_operations", schema="core")
    op.drop_table("agent_operations", schema="core")

    op.drop_index("ix_agents_name", table_name="agents", schema="core")
    op.drop_table("agents", schema="core")

    op.drop_index("ix_operations_connector_id", table_name="operations", schema="core")
    op.drop_table("operations", schema="core")

    op.drop_index("ix_connectors_name", table_name="connectors", schema="core")
    op.drop_table("connectors", schema="core")

    bind = op.get_bind()
    AGENT_MESSAGE_ROLE.drop(bind, checkfirst=True)
    RUN_EVENT_TYPE.drop(bind, checkfirst=True)
    FLOW_RUN_STATUS.drop(bind, checkfirst=True)
    AGENT_PROVIDER.drop(bind, checkfirst=True)
    HTTP_METHOD.drop(bind, checkfirst=True)
    CONNECTOR_AUTH_TYPE.drop(bind, checkfirst=True)
