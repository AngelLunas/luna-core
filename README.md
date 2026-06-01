# Luna Core

> Installable Python library that powers Luna's agentic-automation backend — authentication, persistence, an LLM-agent flow engine, connectors, embeddings/RAG, and a Celery task layer.

**Luna Core is a library, not a service.** It ships routers, services, the flow
engine, an LLM router, an MCP layer, and a Celery app — but it does **not**
instantiate a FastAPI app, spawn `uvicorn`, run migrations on its own, or ship a
Dockerfile. A **host application** (such as `luna-sentinel`) owns the process: it
mounts the routers, builds the runtime collaborators, configures Celery, and runs
`alembic upgrade head` for the `core` schema.

---

## Table of contents

1. [What it is for](#what-it-is-for)
2. [Feature overview](#feature-overview)
3. [Architecture at a glance](#architecture-at-a-glance)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Quick start — wiring a host app](#quick-start--wiring-a-host-app)
8. [Database & migrations (Alembic)](#database--migrations-alembic)
9. [How it works](#how-it-works)
10. [Public API surface (Python)](#public-api-surface-python)
11. [HTTP API reference](#http-api-reference)
12. [Data model](#data-model)
13. [Running tests](#running-tests)
14. [Project layout](#project-layout)
15. [Operational notes](#operational-notes)

---

## What it is for

Luna Core is the reusable backend core for building **agentic automation
platforms** — systems where users visually compose *flows* (graphs of steps) that
combine:

- **AI agents** (LLM-driven steps with tools, structured output, and memory),
- **Connector actions** (authenticated HTTP calls to third-party APIs),
- **Conditional branching**, **human approval checkpoints**, and
- **Iteration** (looping an agent over a batch of records, sequentially or in
  parallel).

It handles the unglamorous-but-hard parts so the host app doesn't have to:
durable run state, resumable execution, real-time event streaming, credential
encryption, OAuth2 handshakes, rate limiting, role-based permissions, semantic
deduplication, embeddings/RAG, and cron-based scheduling.

The host application supplies the domain logic (context loaders, dedup checkers,
which routers to mount) and owns the process lifecycle. Luna Core supplies the
machinery.

---

## Feature overview

| Area | What you get |
| --- | --- |
| **Auth** | Email/password registration, bcrypt hashing, JWT access tokens, rotating refresh tokens (with replay-family revocation), refresh cookies |
| **RBAC** | Role → permission table, `require_permission(...)` dependency, seedable permission sets |
| **Flow engine** | Compiles a JSON `FlowDefinition` into a LangGraph `StateGraph`, with durable + resumable run state |
| **AI agents** | Multi-turn tool-calling loop, structured (JSON-schema) output, streaming, extended-thinking capture, message history inheritance |
| **LLM router** | DB-configured, OpenAI-compatible providers (OpenAI, Anthropic-compat, Ollama, vLLM, LM Studio, Kimi/Moonshot…), per-provider rate limiting, retries with backoff, mid-stream abort |
| **Connectors** | Visual HTTP operations with typed parameters, path/query/header/body mapping, fixed headers/body templating, retry policies, and 4 auth types (none / api_key / basic / oauth2) |
| **OAuth2** | Authorization-code popup flow + automatic refresh-token rotation on outbound calls |
| **System tools / MCP** | In-process tool registry (`stash_records`, `list_scratchpad`, `yield_iteration`) plus a FastMCP server builder and JSON-RPC MCP client |
| **Iteration** | Agent-driven (`yield_iteration`) loops with carry state, or runtime-driven loops over a Redis scratchpad collection (sequential or bounded-parallel) |
| **Embeddings / RAG / dedup** | pgvector storage, cosine search, context-string assembly, threshold-gated semantic dedup |
| **Streaming** | Per-run event timeline persisted to Postgres + Redis pub/sub fan-out to WebSocket clients with zero-loss reconnect |
| **Scheduling** | Cron rules in the user's timezone, dispatched once a minute by Celery beat |
| **Storage** | Pluggable backends: local filesystem, S3-compatible, Cloudflare R2 |

---

## Architecture at a glance

```
                ┌───────────────────────── Host application (e.g. luna-sentinel) ───────────────────────┐
                │  FastAPI app  ·  uvicorn  ·  Celery worker+beat  ·  alembic upgrade head  ·  seeds     │
                └───────────────┬─────────────────────────────────────────────────┬────────────────────┘
                                │ mounts routers / builds collaborators            │ registers worker factory
                                ▼                                                  ▼
   HTTP / WS  ───►  luna_core.routers  ──►  luna_core.services  ──►  models (SQLAlchemy / Postgres `core` schema)
                                │                     │
                                │                     ├─► FlowRunner (engine)  ──► LangGraph StateGraph
                                │                     │        │
                                │                     │        ├─► NodeExecutor ─► action / ai_agent / condition / …
                                │                     │        ├─► AgentRunner  ─► LLMRouter ─► GenericProvider (OpenAI-compatible)
                                │                     │        │                       └─► MCPClient / system tools
                                │                     │        └─► EventEmitter ─► Postgres RunEvent + Redis pub/sub
                                │                     │
                                │                     ├─► ConnectorRegistry (HTTP executor + auth + retry)
                                │                     └─► Embedding / RAG / Dedup (pgvector)
                                ▼
                        WebSocketManager  ◄── Redis pub/sub ──  EventEmitter
```

Three infrastructure dependencies underpin everything: **PostgreSQL** (durable
state, `core` schema, pgvector extension), **Redis** (rate limiting, run-state
cache, event pub/sub, scratchpad, Celery broker), and a **Celery worker + beat**
process (background flow execution and scheduling).

---

## Requirements

### Runtime

- **Python 3.11+** (async throughout)
- **PostgreSQL 14+** with the **`pgvector`** extension (the first migration creates
  the `core` schema and enables `vector`)
- **Redis 6+**
- A **Celery worker + beat** process managed by the host (background flow
  execution and scheduled triggers)
- An **OpenAI-compatible LLM endpoint** for agents and embeddings — e.g. a local
  [Ollama](https://ollama.com) (`qwen2.5:7b` + `mxbai-embed-large` by default),
  vLLM, LM Studio, or a hosted API. Configured per-provider in the database.

### Python dependencies

Pinned in [`pyproject.toml`](pyproject.toml) / [`requirements.txt`](requirements.txt):

| Dependency | Version | Role |
| --- | --- | --- |
| `fastapi` | 0.115.6 | Routers + WebSockets |
| `sqlalchemy[asyncio]` | 2.0.36 | Async ORM |
| `asyncpg` | 0.30.0 | Postgres driver |
| `alembic` | 1.14.0 | Migrations |
| `pgvector` | 0.3.6 | Vector column type |
| `pydantic` / `pydantic-settings` | 2.10.4 / 2.7.0 | Schemas + config |
| `PyJWT` | 2.10.1 | JWT signing |
| `bcrypt` | 4.2.1 | Password hashing |
| `cryptography` | 44.0.0 | Fernet credential encryption |
| `redis` | 5.2.1 | Cache / pub-sub / broker |
| `celery[redis]` | 5.4.0 | Background tasks + beat |
| `croniter` | 3.0.4 | Cron evaluation |
| `langgraph` | 0.2.74 | Flow state graph |
| `langgraph-checkpoint-postgres` | 2.0.13 | Optional Postgres checkpointer |
| `httpx` | 0.28.1 | Outbound HTTP (connectors, LLM, MCP) |
| `websockets` | 14.1 | WebSocket transport |
| `openai` | 1.59.6 | OpenAI-compatible client |
| `anthropic` | 0.42.0 | Anthropic SDK |
| `fastmcp` | 2.3.0 | MCP server builder |
| `jsonschema` | 4.23.0 | Structured-output validation |

**Dev extras** (`pip install -e ".[dev]"`): `pytest`, `pytest-asyncio`, `httpx`, `ruff`.

> **Optional:** the S3/R2 storage backends require `boto3` (imported lazily — add
> it to the host app's dependencies if you use cloud storage).

---

## Installation

```bash
# editable, from a sibling checkout
pip install -e ../luna-core

# with dev tooling
pip install -e "../luna-core[dev]"

# or straight from git
pip install "luna-core @ git+ssh://git@example.com/org/luna-core.git"
```

The package is typed (`py.typed` ships in the wheel), so downstream code gets full
type-checker support.

---

## Configuration

All configuration lives in `luna_core.core.config.Settings`
(`pydantic-settings`). It reads from **process environment variables** and,
optionally, a `.env` file. Variable names are **case-insensitive** and match the
field names **with no prefix** (e.g. the field `jwt_secret_key` reads from
`JWT_SECRET_KEY`). See [`.env.example`](.env.example) for a starting point.

> Note: a few code comments mention `LUNA_`-prefixed names, but no `env_prefix`
> is configured — the effective variable name is just the upper-cased field name.

### Required (no default — the process will fail to start without them)

| Variable | Notes |
| --- | --- |
| `JWT_SECRET_KEY` | ≥ 32-char secret used to sign access tokens **and** OAuth2 state tokens |
| `ENCRYPTION_KEY` | base64 Fernet key (32 bytes) for connector/provider credential encryption. Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |

### Full settings reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `Luna Core` | App label |
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `DEBUG` | `false` | Echo SQL when true |
| `API_V1_PREFIX` | `/api/v1` | Router mount prefix |
| `DATABASE_URL` | `postgresql+asyncpg://luna:luna@localhost:5432/luna_core` | Async DSN |
| `DATABASE_POOL_SIZE` | `10` | SQLAlchemy pool size |
| `DATABASE_MAX_OVERFLOW` | `20` | Pool overflow |
| `REDIS_URL` | `redis://localhost:6379/0` | Rate limit / pub-sub / broker |
| `JWT_ALGORITHM` | `HS256` | Token signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `180` | Access-token TTL (3 h) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `30` | Refresh-token TTL |
| `REFRESH_COOKIE_NAME` | `luna_refresh_token` | Cookie name |
| `REFRESH_COOKIE_SECURE` | `false` | HTTPS-only cookie |
| `REFRESH_COOKIE_SAMESITE` | `lax` | `lax` \| `strict` \| `none` |
| `REFRESH_COOKIE_DOMAIN` | *(empty → None)* | Cookie domain |
| `LOGIN_RATE_LIMIT_ATTEMPTS` | `5` | Max login attempts / window |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | `300` | Login window (5 min) |
| `CELERY_BROKER_URL` | *(falls back to `REDIS_URL`)* | Broker override |
| `CELERY_RESULT_BACKEND` | *(falls back to `REDIS_URL`)* | Result backend override |
| `CELERY_TASK_DEFAULT_QUEUE` | `luna_core` | Queue name |
| `RUN_EVENT_CHANNEL_PREFIX` | `luna_core:run_events` | Redis pub/sub channel prefix |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` | `http://localhost:11434/v1` / `qwen2.5:7b` / `ollama` | Defaults used by the dev seed to bootstrap a baseline `LLMProvider` row |
| `LLM_RATE_LIMIT_RPM` | `60` | Per-provider request/min ceiling |
| `LLM_RATE_LIMIT_WINDOW_SECONDS` | `60` | LLM rate-limit window |
| `LLM_MAX_RETRIES` | `3` | Retries on HTTP 429 |
| `LLM_RETRY_BASE_DELAY_SECONDS` | `1.0` | Backoff base delay |
| `RUN_STREAM_KEY_TTL_SECONDS` | `3600` | TTL for run-state/stream Redis keys |
| `RUN_ABORT_KEY_TTL_SECONDS` | `60` | TTL for abort-signal keys |
| `ITERATION_CONCURRENCY_MAX` | `8` | Hard ceiling on parallel iterations (clamps per-node `concurrency`; 8 is Raspberry-Pi-safe, raise to ~20 on servers) |
| `EMBEDDING_API_KEY` | `ollama` | Embedding endpoint key |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimension (must match the `embeddings.vector` column) |
| `MCP_SERVER_URL` | `http://localhost:8765` | Remote MCP server |
| `OAUTH2_CALLBACK_URL` | `http://localhost:5173/connectors/oauth2/callback` | Must match each provider's registered redirect URI byte-for-byte |
| `OAUTH2_STATE_TTL_SECONDS` | `600` | OAuth2 popup state lifetime |
| `STORAGE_BACKEND` | `local` | `local` \| `s3` \| `r2` (`gcs` not implemented) |
| `STORAGE_LOCAL_PATH` | `./.storage` | Local storage root |
| `STORAGE_BASE_URL` | *(none)* | CDN/base URL (local + cloud) |
| `STORAGE_BUCKET` / `STORAGE_REGION` / `STORAGE_ENDPOINT_URL` / `STORAGE_ACCOUNT_ID` / `STORAGE_ACCESS_KEY` / `STORAGE_SECRET_KEY` | *(none)* | Cloud storage params |
| `CORS_ORIGINS` | `[]` | Host app may pass through to FastAPI |

`settings.effective_celery_broker_url` and `settings.effective_celery_result_backend`
resolve to `REDIS_URL` when the dedicated Celery vars are unset.

---

## Quick start — wiring a host app

Luna Core ships no entry point; the host owns three processes (API, worker,
beat). A minimal host looks like this:

### 1. The FastAPI app

```python
from fastapi import FastAPI

from luna_core.core.config import settings
from luna_core.routers import (
    agents, auth, connectors, context_sources, dedup_checkers,
    flows, llm_providers, runs, system_tools, users,
)

app = FastAPI(title="luna-sentinel")

for router in (
    auth, users, connectors, agents, flows, runs,
    llm_providers, system_tools, context_sources, dedup_checkers,
):
    app.include_router(router.router, prefix=settings.api_v1_prefix)
```

### 2. The runtime collaborators (used by the engine)

```python
from luna_core.core.db import AsyncSessionLocal
from luna_core.core.redis import get_redis
from luna_core.connectors.registry import ConnectorRegistry
from luna_core.engine import FlowRunner
from luna_core.llm.router import LLMRouter
from luna_core.llm.providers.generic import GenericProvider
from luna_core.mcp.client import MCPClient

async def build_runner() -> FlowRunner:
    redis = get_redis()
    registry = ConnectorRegistry()
    async with AsyncSessionLocal() as db:
        await registry.load_from_db(db)

    llm_router = LLMRouter(
        redis=redis,
        session_factory=AsyncSessionLocal,
        embedding_provider=GenericProvider(),  # uses EMBEDDING_* settings
    )
    return FlowRunner(
        llm_router=llm_router,
        connector_registry=registry,
        mcp_client=MCPClient(),
        session_factory=AsyncSessionLocal,
    )
```

### 3. Register the worker factory and launch Celery

```python
from luna_core.tasks import set_runner_factory

set_runner_factory(build_runner)  # called at worker bootstrap
```

```bash
# worker (solo pool keeps the async loop + pools alive across tasks)
celery -A luna_core.tasks.celery_app worker -Q luna_core --pool=solo

# beat (fires luna_core.scheduler_tick every minute)
celery -A luna_core.tasks.celery_app beat
```

### 4. (Optional) install built-in system tools & seed permissions

```python
from luna_core.mcp.system_tools import install_builtins
from luna_core.seeds.permissions import seed_core_permissions

install_builtins()  # registers stash_records / list_scratchpad / yield_iteration
# await seed_core_permissions(db)
```

---

## Database & migrations (Alembic)

Luna Core owns the **`core`** Postgres schema and ships its own Alembic
environment. As a host app you don't write these migrations — you **run** them and
point Alembic at your database. Everything you need is already configured in
[`alembic.ini`](alembic.ini) and [`alembic/env.py`](alembic/env.py).

### How the Alembic environment is wired

The `env.py` is pre-configured so you generally don't touch it:

- **Connection URL comes from settings, not the ini.** `alembic.ini` leaves
  `sqlalchemy.url` blank; `env.py` overrides it with
  `settings.database_url` at runtime. So Alembic targets whatever `DATABASE_URL`
  the process sees — set that one env var and you're done.
- **Async engine.** Migrations run through `async_engine_from_config` + asyncpg,
  matching the app's runtime driver.
- **Schema-aware.** `include_schemas=True` and `version_table_schema="core"` keep
  the `alembic_version` bookkeeping table inside the `core` schema, so Luna Core's
  migration history never collides with other services that share the database.
- **Autogenerate-ready.** `target_metadata = Base.metadata` and
  `import luna_core.models` mean every model is registered; `compare_type=True`
  detects column-type changes.

### Running migrations

From the host process (or this repo during development), with `DATABASE_URL`
exported (or a `.env` present):

```bash
# apply everything up to the latest revision
alembic upgrade head

# inspect state
alembic current          # currently applied revision
alembic history --verbose

# step up/down
alembic upgrade +1
alembic downgrade -1
```

> The very first migration creates the `core` schema **and** runs
> `CREATE EXTENSION IF NOT EXISTS vector`. The Postgres role in `DATABASE_URL`
> therefore needs permission to create schemas and extensions (or have an operator
> pre-create the `vector` extension once).

### Integrating with a host that has its own migrations

If your host app keeps its own Alembic tree, you have two clean options:

1. **Run Luna Core's Alembic separately** (recommended). Invoke it with its own
   config so the two version tables stay independent:
   ```bash
   alembic -c $(python -c "import luna_core, pathlib; \
     print(pathlib.Path(luna_core.__file__).parent.parent / 'alembic.ini')") \
     upgrade head
   ```
   Because Luna Core's version table lives in the `core` schema and the host's
   typically lives in `public`, they never conflict — you simply run both
   `upgrade head` commands at deploy time.

2. **Treat `core` as managed-by-Luna-Core.** Never autogenerate against Luna
   Core's tables from the host tree; let Luna Core own that schema end-to-end and
   only reference its tables read-only from your domain models.

### Creating new revisions (when you fork/extend Luna Core)

```bash
# autogenerate from model changes (review the output — autogen isn't perfect)
alembic revision --autogenerate -m "add my column"

# empty revision for hand-written DDL / data migrations
alembic revision -m "backfill something"
```

Generated files are named `YYYYMMDD_HHMM_<slug>.py` (UTC) per the `file_template`
in `alembic.ini`, which keeps the `alembic/versions/` directory chronologically
sorted.

---

## How it works

### The flow engine

`FlowRunner` (`luna_core/engine/runner.py`) is the heart of the system. Given a
`FlowRun` row it:

1. **Builds a LangGraph `StateGraph`** from the flow's `FlowDefinition`. Each node
   becomes a graph node wrapping `NodeExecutor.execute()`; edges become direct or
   conditional transitions; `START → entry_point` is added and every terminal node
   is wired to `END`.
2. **Reconstructs initial state** — from the Redis snapshot at `run_state:{run_id}`
   (crash recovery) or the DB `FlowRun.state`.
3. **Streams execution** via `compiled.astream(state, config={"thread_id": run_id})`.
   The graph uses a `MemorySaver` checkpointer by default (Postgres checkpointer
   wiring is available).
4. **Persists state** after each node: the live snapshot is cached in Redis;
   on pause/complete/fail it is mirrored to `FlowRun.state` and the Redis keys
   are cleared.
5. **Handles control-flow exceptions**: `HumanCheckpointInterrupt` → status
   `paused`; `AbortSignalError` → status `failed`; any other exception → `failed`
   with the error captured.

The shared state is a `FlowState` TypedDict. Two channels — `outputs` and
`context` — use a **merge reducer** so nodes *accumulate* into them rather than
overwriting; `run_id`, `flow_id`, `inputs`, and `trigger` are immutable for the
run.

### Flow definition format

A flow is a JSON document (validated by the `FlowDefinition` Pydantic schema):

```json
{
  "entry_point": "fetch_jobs",
  "trigger": { "type": "schedule", "schedules": [
    { "cron": "0 */6 * * *", "tz": "Europe/Madrid", "inputs": {} }
  ]},
  "inputs": [
    { "name": "keyword", "type": "string", "required": true }
  ],
  "nodes": [
    { "id": "fetch_jobs",  "type": "action",   "name": "Fetch Jobs",
      "config": { "operation_id": "..." } },
    { "id": "score_agent", "type": "ai_agent", "name": "Score Agent",
      "config": { "agent_id": "...", "output_key": "score_result" } },
    { "id": "approval",    "type": "human_checkpoint", "name": "Approve",
      "config": { "message": "Review and approve scored jobs" } }
  ],
  "edges": [
    { "from": "fetch_jobs",  "to": "score_agent" },
    { "from": "score_agent", "to": "approval",
      "condition": { "field": "outputs.score_result.recommendation",
                     "operator": "eq", "value": "apply" } },
    { "from": "score_agent", "to": "__end__",
      "condition": { "field": "outputs.score_result.recommendation",
                     "operator": "eq", "value": "skip" } }
  ],
  "layout": { "fetch_jobs": { "x": 0, "y": 0 } }
}
```

- `inputs` declares the trigger payload contract (typed, with defaults); incoming
  runs are validated and coerced against it.
- `trigger.cron` is a legacy single-cron field, auto-promoted into `schedules[0]`.
- `layout` stores node positions for the visual editor.

### Node types

`NodeType` ∈ `action` · `ai_agent` · `condition` · `human_checkpoint` · `trigger`
· `output`. Each has a handler in `NodeExecutor`:

| Type | Behavior |
| --- | --- |
| **`action`** | Runs a **connector operation** (`config.operation_id`) or a **catalog system tool** (`config.system_tool_name`). Resolves templated inputs, emits `tool_called` → executes → emits `tool_result`, returns `{node_id: result}`. |
| **`ai_agent`** | Loads the agent, its context sources, and (optionally) inherited message history; resolves templates in role/instructions; runs the agent loop. Has a dedicated **iterative** path (see [Iteration](#iteration-loops-over-records)). Returns `{node_id: output}` and merges any loaded `context`. |
| **`condition`** | No-op node; branching happens on its outgoing **edges**. |
| **`human_checkpoint`** | Emits `human_checkpoint` and raises `HumanCheckpointInterrupt` to pause the run until a human responds. |
| **`trigger`** | No-op start marker. |
| **`output`** | If `config.select` is a list, copies those keys into the canonical `outputs` dict. |

### Edges and conditions

Edges may be unconditional or carry a `FlowEdgeCondition`:

```json
{ "field": "outputs.score.value", "operator": "gte", "value": 0.8 }
```

The `field` is a dotted path resolved against the run state. Operators:
`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `contains`. A `None` lookup evaluates
to `False` (safe navigation). When several conditional edges leave a node, they're
compiled into a LangGraph conditional router with a default fallback. Use the
`__end__` target to terminate a branch.

### Templating

Anywhere a node config takes a value, you can interpolate run state with
`${...}` paths (note: `${}`, **not** `{{}}` — the latter is reserved for the
frontend). Resolution is implemented in `engine/template_paths.py`:

- `${inputs.keyword}` — trigger inputs
- `${outputs.node_id}` / `${outputs.score.value}` — prior node outputs
- `${context.<source>.<field>}` — loaded context sources
- `${trigger.user_id}` / `${trigger.timestamp}` / `${trigger.source}`
- `${iteration.item.<field>}`, `${iteration.index}`, `${iteration.carry.<field>}`

Path grammar supports list projection: `items[*]` returns the whole list, and
`items[*].name` projects a field across each element. Missing paths resolve to an
empty string in string templates (or `None` in value resolution).

### The agent loop

`AgentRunner` (`luna_core/engine/agent.py`) drives one agent step:

1. **Assemble the tool set** from three tiers — *context tools* injected by the
   caller (e.g. `yield_iteration`, always allowed), *catalog system tools* (the
   in-process registry, filtered by the agent's grants), and *MCP tools* (filtered
   by grants). On a name collision, system tools win.
2. **Build the message history** — persisted `AgentMessage`s (optionally inherited
   from another node), plus the new user turn.
3. **Loop** up to `MAX_TOOL_ITERATIONS` (16): call `LLMRouter.complete(...)`; if
   the assistant returns tool-use blocks, dispatch each (system tool in-process, or
   MCP via `MCPClient.call_tool`), append `tool_result` blocks, and continue. Tool
   errors come back as `is_error` blocks so the model can recover.
4. **Finalize** — concatenate text blocks; if the agent has an `output_schema`,
   parse the text as JSON and validate it with `jsonschema`; return the validated
   object (or plain text).

A system tool flagged **`terminal`** short-circuits the loop on success and hands
its captured arguments back to the caller — this is the mechanism behind
agent-driven iteration.

### Iteration (loops over records)

The `ai_agent` node has an iterative mode with two sources:

- **Agent-yield** (`_iterate_with_agent_yield`): the agent itself drives the loop
  by calling the terminal `yield_iteration` tool each turn, passing `next_carry`
  (loop state carried forward, shaped by a declared `carry_schema`), an `append`
  array (accumulated into the node's `items` output), and a `done` flag. Each turn
  runs with a fresh context window. Exits on `done`, on `max_iterations`, or when
  the agent stops yielding.

- **Scratchpad** (`_iterate_with_scratchpad`): the runtime drives the loop over a
  named Redis **scratchpad** collection (records previously staged via the
  `stash_records` tool). It snapshots the record IDs up front (a point-in-time
  view), then processes each record either **sequentially** (a failure aborts) or
  **in parallel** (bounded by a semaphore = `min(node.concurrency, ITERATION_CONCURRENCY_MAX)`,
  with an `on_iteration_error` policy of `continue` or `cancel_siblings`). Each
  iteration runs in its own `AsyncSession`, emits `iteration_started` /
  `iteration_completed` / `iteration_failed`, and drops the record on success.

Per-iteration events carry an `iteration_id` (propagated via a `ContextVar` so it
survives `asyncio` task boundaries), which lets the UI group them into accordions.

### LLM router and providers

`LLMRouter` (`luna_core/llm/router.py`) resolves a chat provider from the
`llm_providers` DB table by `provider_id`, caching the built provider keyed on the
row's `updated_at` (so edits rebuild it). It enforces a **per-provider** fixed-window
rate limit (`LLM_RATE_LIMIT_RPM`) and **retries on HTTP 429** with exponential
backoff. Aborts (`AbortSignalError`) are never retried.

`GenericProvider` (`luna_core/llm/providers/generic.py`) is a single
**OpenAI-compatible** client that talks to OpenAI, Anthropic (via the compat
layer), Ollama, vLLM, LM Studio, Kimi/Moonshot, and TEI for embeddings. It:

- streams completions, pushing **text** and **thinking** (`reasoning_content`)
  deltas to Redis and publishing `agent_text_delta` / `agent_thinking_delta`
  events;
- accumulates tool-call deltas and returns canonical content blocks
  (`thinking` → `text` → `tool_use`);
- checks a Redis **abort key** before/within streaming and persists partial output
  as `AgentMessage(is_partial=True)` before raising `AbortSignalError`;
- supports structured output via the OpenAI `json_schema` response format;
- maintains crash-recovery stream keys so a mid-flight message can be reconstructed.

The `BaseLLMProvider` protocol defines `complete(...)` and `embed(...)`;
`ToolDefinition` is the provider-agnostic tool spec.

### Connectors

A **Connector** is an authenticated HTTP base; an **Operation** is one endpoint on
it. Operations are defined visually with typed **parameters** (`ParameterDef`:
name, type, `in` ∈ path/query/header/body, required, default, enum, nested
properties). `ConnectorRegistry` (`luna_core/connectors/registry.py`) caches active
connectors and executes operations:

1. apply parameter defaults; substitute `{param}` placeholders in the path;
2. distribute remaining inputs into query / header / body buckets by their `in`;
3. interpolate `fixed_headers` / `fixed_body` templates;
4. attach auth, send via `httpx`, and decode the response (JSON when hinted, else
   text; bare lists normalized to `{"items": [...]}`).

`execute()` raises `ConnectorExecutionError` on failure; `perform()` returns a
structured `OperationCallResult` (status, latency, request URL, response, error)
without raising; `perform_draft()` tests an unsaved operation.

**Auth types** (`luna_core/connectors/auth.py`): `none`, `api_key`
(bearer / custom header / query-param schemes), `basic`, and `oauth2`. OAuth2
auto-refreshes with a 60-second freshness leeway, recovers from a 401/403 by
forcing a refresh and retrying once, and serializes refreshes per connector with a
lock. **Retry policy** (`retry.py`) is a declarative `RetryPolicy` with exponential
backoff + full jitter; auth statuses (401/403) are intentionally handled by the
OAuth2 path rather than retried here.

### System tools & MCP

**System tools** are in-process Python tools the agent can call, kept in a
`SystemToolRegistry`. Each tool has a `scope`: **catalog** (toggleable per agent,
granted via `AgentSystemToolGrant`) or **context** (auto-injected by a runtime).
Built-ins (registered by `install_builtins()`):

| Tool | Scope | Terminal | Purpose |
| --- | --- | --- | --- |
| `stash_records` | catalog | no | Stage free-form records into a named scratchpad collection (optional schema validation + dedup). |
| `list_scratchpad` | catalog | no | Read back all records in a collection. |
| `yield_iteration` | context | **yes** | Close one iteration of an agent-driven loop (`next_carry`, `append`, `done`). |

**MCP**: `MCPServerBuilder` materializes active connector Operations — plus any
host-registered internal tools — into a single **FastMCP** server, with live
`refresh_from_db` / `attach` / `detach`. `MCPClient` is a JSON-RPC HTTP client
(`tools/list`, `tools/call`) that wraps remote errors into soft
`ToolCallResult(is_error=True)` values.

### Embeddings, RAG and dedup

- **`EmbeddingService`** — `embed(text)`, `upsert(text, collection, metadata)`,
  cosine `search(query, collection, k, filter)` over the pgvector `embeddings`
  table.
- **`RAGService`** — `build_context(queries, collections, k_per_query)` runs the
  query × collection product, dedupes by text, returns a prompt-ready string.
- **`DedupService`** — threshold-gated (default 0.9) semantic dedup;
  `metadata.entity_id` points back to the caller's domain row.

### Events, streaming and WebSockets

Every run produces an ordered **event timeline**. `EventEmitter` persists each
`RunEvent` (monotonic `sequence` per run) to Postgres **and** publishes it to a
Redis channel `{RUN_EVENT_CHANNEL_PREFIX}:{run_id}`. High-frequency token deltas
are published to pub/sub only.

`RunEventType` values:

```
flow_started · flow_completed · flow_failed
node_started · node_completed · node_failed
agent_thinking
agent_message_started · agent_text_delta · agent_thinking_delta · agent_message_completed
tool_called · tool_result
human_checkpoint · human_response
iteration_started · iteration_completed · iteration_failed
run_cleared
```

`WebSocketManager` keeps **one** Redis `SUBSCRIBE` per run regardless of client
count; each client gets its own async queue. On connect it sends a **snapshot**
before live frames for **zero-loss reconnect**, and supports per-`iteration_id`
subscribe/unsubscribe control frames.

### Human checkpoints (pause/resume)

A `human_checkpoint` node emits the event, raises `HumanCheckpointInterrupt`, and
`FlowRunner` saves state and marks the run `paused`. `POST /runs/{id}/resume`
records the reply as an `AgentMessage(role=user)`, emits `human_response`, and
re-enters the graph from the LangGraph checkpoint.

### Scheduling

`scheduling.dispatch_due_runs(db)` runs **once a minute** via the Celery beat task
`luna_core.scheduler_tick`. It evaluates each schedule rule's cron in **its own
IANA timezone** (DST handled by `croniter`) and dispatches due runs. Rules carry
the owning `user_id`; `preview_next_runs(cron, tz, count)` powers the "next fire
times" UI.

### Authentication & permissions

Passwords are bcrypt-hashed; login issues a JWT **access token** (3 h) plus a
*hashed* **refresh token** (30 d) set as an HTTP-only cookie. Refresh rotation
detects replay (reusing an old token revokes the family). `get_current_user`
validates the `Bearer` token; `require_permission("flows:create")` gates endpoints
against the role→permission table. There is **no role hierarchy** — `admin` carries
the full permission list (seeded by `seed_core_permissions`).

### Storage backends

`build_storage_backend(settings)` returns a `BaseStorageBackend`
(`upload → key`, `download`, `delete`, `get_url`) chosen by `STORAGE_BACKEND`:
**local** (filesystem, path-traversal-guarded), **s3** (boto3, presigned URLs), or
**r2** (S3 subclass, endpoint auto-built from `STORAGE_ACCOUNT_ID`). `gcs` raises
`NotImplementedError`.

---

## Public API surface (Python)

Everything a host needs is re-exported from the top-level package
(`luna_core/__init__.py`); importing it triggers **no** side effects.

```python
from luna_core import (
    AgentRunner, EventEmitter, NodeExecutor, FlowRunner, WebSocketManager,
    BaseLLMProvider, GenericProvider, LLMRouter, ToolDefinition,
    LLMRateLimitError, AbortSignalError,
    MCPClient, MCPServerBuilder,
    DedupService, EmbeddingService, RAGService,
    BaseStorageBackend, build_storage_backend,
    __version__,
)
```

Other commonly imported entry points:

```python
from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal, get_db, Base, engine
from luna_core.core.redis import get_redis, close_redis
from luna_core.routers import (auth, users, connectors, agents, flows, runs,
                               llm_providers, system_tools, context_sources, dedup_checkers)
from luna_core.tasks import celery_app, set_runner_factory
from luna_core.connectors.registry import ConnectorRegistry
from luna_core.services.context_sources import register_context_source
from luna_core.mcp.system_tools import install_builtins
from luna_core.seeds.permissions import seed_core_permissions
```

---

## HTTP API reference

All paths are prefixed with `settings.api_v1_prefix` (default `/api/v1`). Send the
access token as `Authorization: Bearer <token>`. Permissions in brackets are
enforced via `require_permission(...)`.

Conventions used below: 🔓 = public · 🔑 = bearer token · 🍪 = refresh cookie.

### Auth — `/auth`

| Method | Path | Access |
| --- | --- | --- |
| POST | `/register` | 🔓 |
| POST | `/login` | 🔓 (rate-limited per IP, 5 / 5 min) |
| POST | `/refresh` | 🍪 |
| POST | `/logout` | 🍪 |
| GET | `/me` | 🔑 |

**`POST /auth/register`** — body:
```json
{ "email": "dev@example.com", "password": "at-least-8-chars" }
```
**`POST /auth/login`** — same body shape. Both return `201`/`200`:
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 10800,
  "user": { "id": "f1c2...", "email": "dev@example.com",
            "role": "user", "is_active": true, "created_at": "2026-05-31T10:00:00Z" }
}
```
A `luna_refresh_token` HTTP-only cookie is set on the `/auth` path. `POST /auth/refresh`
reads that cookie and returns a fresh `TokenResponse` (rotating the refresh token);
reusing a revoked token revokes the whole family (`401`). `GET /auth/me` returns the
bare `UserRead`. Login over its limit returns `429` with a `Retry-After` header.

### Users — `/users`  · [`users:manage`]

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/` | List `UserRead[]` |
| PUT | `/{user_id}` | Body `{ "role": "admin" }` → updated `UserRead` |

### Connectors — `/connectors`

| Method | Path | Permission |
| --- | --- | --- |
| POST | `/` | `connectors:create` |
| GET | `/` | `connectors:read` |
| GET | `/{id}` | `connectors:read` |
| PATCH | `/{id}` | `connectors:update` |
| DELETE | `/{id}` | `connectors:delete` (→ `204`) |
| GET | `/oauth2/config` | 🔓 |
| POST | `/{id}/oauth2/start` | `connectors:update` |
| POST | `/oauth2/callback` | state JWT |
| GET | `/operations/{operation_id}` | `connectors:read` |
| POST | `/{id}/operations` | `connectors:create` |
| GET | `/{id}/operations` | `connectors:read` |
| PATCH | `/{id}/operations/{operation_id}` | `connectors:update` |
| DELETE | `/{id}/operations/{operation_id}` | `connectors:delete` (→ `204`) |
| POST | `/{id}/operations/test-draft` | `connectors:test` |
| POST | `/{id}/operations/{operation_id}/test` | `connectors:test` |

**`POST /connectors`** — body (`credentials` is encrypted at rest; `auth_type` ∈
`none`/`api_key`/`basic`/`oauth2`):
```json
{
  "name": "GitHub",
  "description": "GitHub REST API",
  "auth_type": "api_key",
  "base_url": "https://api.github.com",
  "credentials": { "token": "ghp_...", "scheme": "bearer" },
  "is_active": true
}
```
Response `201` (note: the secret never comes back — only `has_credentials`):
```json
{
  "id": "9b1d...", "name": "GitHub", "description": "GitHub REST API",
  "auth_type": "api_key", "base_url": "https://api.github.com",
  "has_credentials": true, "oauth2_connected": null, "is_active": true,
  "created_at": "2026-05-31T10:00:00Z", "updated_at": "2026-05-31T10:00:00Z"
}
```

**`POST /connectors/{id}/operations`** — a parameter's `in` ∈
`path`/`query`/`header`/`body`; `type` ∈ `string`/`integer`/`number`/`boolean`/`array`/`object`:
```json
{
  "name": "list_issues",
  "description": "List repository issues",
  "method": "GET",
  "path": "/repos/{owner}/{repo}/issues",
  "parameters": [
    { "name": "owner", "type": "string", "in": "path", "required": true },
    { "name": "repo",  "type": "string", "in": "path", "required": true },
    { "name": "state", "type": "string", "in": "query", "required": false,
      "default": "open", "enum_values": ["open", "closed", "all"] }
  ],
  "fixed_headers": { "Accept": "application/vnd.github+json" },
  "retry_policy": { "max_attempts": 3, "retry_on_status": [502, 503, 504],
                    "initial_delay_ms": 200, "multiplier": 3.0, "jitter": true },
  "output_schema": {},
  "is_active": true
}
```
The `input_schema` is **derived** from `parameters` (don't hand-write it unless you
fall back to the legacy escape hatch with `parameters: []`).

**`POST /connectors/{id}/operations/{operation_id}/test`** — body `{ "input": { "owner": "octocat", "repo": "hello", "state": "open" } }`.
Returns the raw call result, including 4xx/5xx (it does **not** throw):
```json
{
  "ok": true, "status_code": 200, "latency_ms": 412,
  "request_method": "GET",
  "request_url": "https://api.github.com/repos/octocat/hello/issues?state=open",
  "response": { "items": [ ... ] }, "error": null
}
```

### Agents — `/agents`

| Method | Path | Permission |
| --- | --- | --- |
| POST | `/preview-instructions` | `agents:read` |
| POST | `/` | `agents:create` |
| GET | `/` | `agents:read` — `?include=system_tools` |
| GET | `/{id}` | `agents:read` — `?include=operations,system_tools` |
| PUT | `/{id}` | `agents:update` |
| DELETE | `/{id}` | `agents:delete` |
| POST | `/{id}/operations` | `agents:update` |
| GET | `/{id}/operations` | `agents:read` |
| POST | `/{id}/system-tools` | `agents:update` |
| GET | `/{id}/system-tools` | `agents:read` |

**`POST /agents`** — body (`temperature` 0.0–2.0; `output_schema` optional JSON
Schema for structured output):
```json
{
  "name": "Job Scorer",
  "role": "You score job posts for fit.",
  "instructions": "Score this job for ${context.profile.headline}. Job: ${inputs.job}",
  "llm_provider_id": "7a3e...",
  "model": "qwen2.5:7b",
  "temperature": 0.4,
  "output_schema": {
    "type": "object",
    "properties": { "recommendation": { "type": "string", "enum": ["apply", "skip"] },
                    "score": { "type": "number" } },
    "required": ["recommendation"]
  }
}
```
Response `201` includes `required_sources` (auto-extracted from `${context.*}`
references in `instructions`):
```json
{
  "id": "c4d5...", "name": "Job Scorer", "role": "You score job posts for fit.",
  "instructions": "Score this job for ${context.profile.headline}. Job: ${inputs.job}",
  "llm_provider_id": "7a3e...", "model": "qwen2.5:7b", "temperature": 0.4,
  "output_schema": { "...": "..." }, "required_sources": ["profile"],
  "created_at": "2026-05-31T10:00:00Z", "updated_at": "2026-05-31T10:00:00Z"
}
```
**`POST /agents/{id}/operations`** — `{ "operation_ids": ["..."] }` (min 1).
**`POST /agents/{id}/system-tools`** — `{ "tool_names": ["stash_records"] }`
(empty list clears all grants). **`POST /agents/preview-instructions`** —
`{ "instructions": "...", "source_bindings": { "profile": "<id>" } }` returns
`{ "resolved": "...", "required_sources": [...], "diagnostics": [...] }`.

### Flows — `/flows`

| Method | Path | Permission |
| --- | --- | --- |
| POST | `/` | `flows:create` |
| GET | `/` | `flows:read` |
| GET | `/{id}` | `flows:read` |
| POST | `/validate` | `flows:read` (always `200`; errors in body) |
| POST | `/preview-schedule` | `flows:read` |
| PUT | `/{id}` | — (merges other users' schedules) |
| DELETE | `/{id}` | `flows:delete` (→ `204`) |
| GET | `/{id}/runs` | `flows:read` — `?limit=20&offset=0` |
| POST | `/{id}/run` | — (→ `202`) |
| WS | `/{id}/stream` | — |

**`POST /flows`** — `{ "name": "...", "description": "...", "definition": { …FlowDefinition… }, "is_active": true }`
(see [Flow definition format](#flow-definition-format)). Returns `FlowRead` with the
stored `definition`.

**`POST /flows/validate`** — `{ "definition": { … } }` → `{ "ok": false, "errors": ["entry_point 'x' is not a node", …] }`
without a 4xx, so editors can show inline diagnostics.

**`POST /flows/preview-schedule`** — `{ "cron": "0 */6 * * *", "tz": "Europe/Madrid", "count": 5 }`
→ `{ "next_runs": ["2026-05-31T12:00:00Z", …] }`.

**`POST /flows/{id}/run`** — optional body (auto-stamped with the caller's
`user_id`); inputs are validated against the flow's `inputs` contract:
```json
{ "inputs": { "keyword": "python" }, "source": "manual", "metadata": {} }
```
Returns `202` with the pending `FlowRunRead` (and enqueues `run_flow_task`):
```json
{
  "id": "run-8f...", "flow_id": "flow-1...", "status": "pending",
  "trigger": { "inputs": { "keyword": "python" }, "source": "manual" },
  "state": {}, "started_at": null, "completed_at": null,
  "cleared_at": null, "created_at": "2026-05-31T10:00:00Z"
}
```
**WS `/flows/{id}/stream`** emits `{ "event": "run_created" | "run_status_changed", "run": { …FlowRunRead… } }`.

### Runs — `/runs`  · 🔑 all endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/{run_id}` | `FlowRunRead` (status + state snapshot) |
| GET | `/{run_id}/events` | `?since_sequence=<int>`, `?iteration_id=<uuid>` |
| GET | `/{run_id}/messages` | `?node_id=<id>` |
| POST | `/{run_id}/resume` | → `202` |
| POST | `/{run_id}/abort` | → `202` |
| DELETE | `/{run_id}` | Soft-compact (purge events+messages, keep row) |
| WS | `/{run_id}/stream` | Snapshot + live event/message stream |

**`GET /runs/{run_id}/events`** → `RunEventRead[]` ordered by `sequence`:
```json
[
  { "id": "ev-1", "flow_run_id": "run-8f...", "sequence": 1,
    "timestamp": "2026-05-31T10:00:01Z", "event_type": "flow_started",
    "node_id": null, "payload": {} },
  { "id": "ev-2", "flow_run_id": "run-8f...", "sequence": 2,
    "timestamp": "2026-05-31T10:00:02Z", "event_type": "node_started",
    "node_id": "score_agent", "payload": { "type": "ai_agent", "name": "Score Agent" } }
]
```
Pass `?since_sequence=N` to page forward; `?iteration_id=<uuid>` to fetch only one
iteration's sub-events.

**`POST /runs/{run_id}/resume`** — answer a `human_checkpoint`:
```json
{ "response": "approved", "metadata": { "by": "dev@example.com" } }
```
Returns `202` with the run flipping back to `running`. **`POST /runs/{run_id}/abort`**
takes no body and sets a Redis abort key the engine honors mid-stream.

### Read-only registries

| Method | Path | Permission | Returns |
| --- | --- | --- | --- |
| GET | `/system-tools` | `agents:read` | `{ name, description, input_schema }[]` (catalog scope only) |
| GET | `/context-sources` | `agents:read` | `{ name, description, id_implicit, schema }[]` |
| GET | `/dedup-checkers` | `agents:read` | `{ name, label, description, required_fields[] }[]` |

These reflect what the host registered at startup (`register_context_source`,
dedup-checker registry, `install_builtins`).

### LLM providers — `/llm-providers`

| Method | Path | Permission |
| --- | --- | --- |
| POST | `/` | `llm_providers:create` |
| GET | `/` | `llm_providers:read` |
| GET | `/{id}` | `llm_providers:read` |
| PUT | `/{id}` | `llm_providers:update` |
| DELETE | `/{id}` | `llm_providers:delete` (→ `409` if an agent uses it) |
| GET | `/{id}/models` | `llm_providers:read` |

**`POST /llm-providers`** — `api_key` is encrypted at rest:
```json
{
  "name": "OpenAI", "base_url": "https://api.openai.com/v1",
  "chat_url": null, "models_url": null, "api_key": "sk-...", "is_active": true
}
```
Response exposes `has_api_key` (never the key). On update, `api_key` semantics are:
omit = keep, a string = replace, `""` = clear. **`GET /llm-providers/{id}/models`**
proxies the upstream `/models` endpoint → `{ "provider_id": "...", "models": [ { "id": "gpt-4o", "owned_by": "openai" }, … ] }`.

**Status codes** across the API: `201` create · `202` async accepted · `204`
delete · `400` validation · `401` auth · `403` permission · `404` missing · `409`
conflict/in-use · `429` rate-limited · `502` upstream error · `503` collaborator
not wired.

---

## Data model

All tables live in the PostgreSQL **`core`** schema. Primary keys are UUIDs;
timestamps are timezone-aware.

### Identity & access

- **`users`** — `id`, `email` (unique), `password_hash`, `role` (default `user`),
  `is_active`, `created_at`. → `refresh_tokens`.
- **`refresh_tokens`** — `id`, `user_id` → users (cascade), `token_hash` (unique),
  `expires_at`, `revoked`, `created_at`.
- **`permissions`** — `id`, `app`, `role`, `permission`, unique `(app, role, permission)`.

### Connectors

- **`connectors`** — `id`, `name` (unique), `description`, `auth_type`
  (`AuthType`: none / api_key / oauth2 / basic), `base_url`, `credentials_encrypted`
  (Fernet), `is_active`, timestamps. → `operations`.
- **`operations`** — `id`, `connector_id` → connectors (cascade), `name`,
  `method` (`HTTPMethod`: GET/POST/PUT/DELETE/PATCH), `path`, `input_schema` (JSONB,
  derived from parameters), `output_schema`, `parameters` (JSONB list — source of
  truth for the editor), `fixed_headers`, `fixed_body`, `retry_policy` (JSONB,
  nullable), `is_active`, unique `(connector_id, name)`. → `agent_operations`.

### Agents

- **`agents`** — `id`, `name` (unique), `role`, `instructions` (template),
  `llm_provider_id` → llm_providers (restrict), `model`, `temperature` (0.7),
  `output_schema` (JSONB), `required_sources` (text[] — computed from instruction
  templates), timestamps.
- **`agent_operations`** — join `agent_id` ↔ `operation_id` (both cascade), unique
  pair.
- **`agent_system_tool_grants`** — `agent_id` → agents (cascade), `tool_name`
  (string, not an FK — the registry is authoritative), unique `(agent_id, tool_name)`.
- **`llm_providers`** — `id`, `name` (unique), `base_url`, `chat_url` (nullable),
  `models_url` (nullable), `api_key_encrypted` (Fernet JSON), `is_active`,
  timestamps.

### Flows & runs

- **`flows`** — `id`, `name` (unique), `description`, `definition` (JSONB),
  `is_active`, timestamps. → `flow_runs`.
- **`flow_runs`** — `id`, `flow_id` → flows (cascade), `status` (`FlowRunStatus`:
  pending / running / paused / completed / failed), `trigger` (JSONB), `state`
  (JSONB), `started_at`, `completed_at`, `cleared_at` (soft-compaction marker),
  `created_at`. → `run_events`, `agent_messages`.
- **`run_events`** — `id`, `flow_run_id` → flow_runs (cascade), `sequence`
  (bigint), `timestamp`, `event_type` (`RunEventType`), `node_id`, `payload`
  (JSONB), unique `(flow_run_id, sequence)`.
- **`agent_messages`** — `id`, `flow_run_id` → flow_runs (cascade), `node_id`,
  `sequence`, `role` (`AgentMessageRole`: system / user / assistant), `content`
  (JSONB block list), `is_partial`, `thinking` (text), `created_at`, unique
  `(flow_run_id, sequence)`.

### Embeddings & conversations

- **`embeddings`** — `id`, `collection`, `text`, `vector` (`Vector(1024)`),
  `metadata` (JSONB), `created_at`. Standalone, collection-scoped (pgvector).
- **`conversations`** / **`conversation_messages`** — a reusable persistent-chat
  primitive (survives the flow that created it); messages mirror the `AgentMessage`
  content shape with `(conversation_id, sequence)` ordering.

### Entity relationships

```
User 1─* RefreshToken          Connector 1─* Operation *─* Agent (via AgentOperation)
User  ·  Permission (by role)  LLMProvider 1─* Agent     Agent 1─* AgentSystemToolGrant
Flow 1─* FlowRun 1─* RunEvent
                  └─* AgentMessage
Conversation 1─* ConversationMessage         Embedding (standalone, by collection)
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The suite (`tests/`) covers the engine (templates, iteration, scratchpad,
in-flight snapshot), the connector registry, system tools, event-emitter retries,
the WebSocket iteration filter, dedup, and context sources. Tests use
`pytest-asyncio`.

---

## Project layout

```
luna-core/
├── luna_core/
│   ├── core/          config, db, redis, security, crypto, rate_limit, dependencies
│   ├── models/        User, RefreshToken, Permission, Connector, Operation, Agent,
│   │                  AgentOperation, AgentSystemToolGrant, LLMProvider, Flow,
│   │                  FlowRun, RunEvent, AgentMessage, Embedding, Conversation(+Message)
│   ├── schemas/       Pydantic v2 request/response models
│   ├── services/      auth, agent, flow, connector, embedding, rag, dedup,
│   │                  scratchpad, context_sources, permission, scheduling, llm_provider
│   ├── routers/       auth, users, connectors, agents, flows, runs, llm_providers,
│   │                  system_tools, context_sources, dedup_checkers
│   ├── engine/        runner, nodes, agent, iteration, iteration_context,
│   │                  emitter, websocket, template_paths
│   ├── llm/           base, router, providers/generic
│   ├── mcp/           client, builder, schemas, system_tools/{registry, stash_records,
│   │                  list_scratchpad, yield_iteration}
│   ├── connectors/    registry, auth, retry, oauth2_flow
│   ├── dedup/         registry, node_config
│   ├── storage/       base, service, local, s3, r2
│   ├── seeds/         permissions
│   └── tasks.py       celery_app + run_flow / resume_flow / trigger_scheduled_run / scheduler_tick
├── alembic/ · alembic.ini
├── tests/
├── pyproject.toml · requirements.txt · .env.example
```

---

## Operational notes

- **Celery worker pool:** run with `--pool=solo`. Tasks bridge Celery's sync world
  to async code via a **persistent event loop per worker**; forking would
  invalidate the shared DB pool, Redis client, and httpx clients. The worker builds
  its `FlowRunner` once via the factory registered with `set_runner_factory(...)`.
- **Beat granularity:** scheduling resolution is **one minute** (a single
  `scheduler_tick` beat entry scans all flows).
- **State durability:** in-flight run state lives in Redis (`run_state:{run_id}`,
  TTL `RUN_STREAM_KEY_TTL_SECONDS`) and is mirrored to `flow_runs.state` on
  pause/complete/fail; a crash mid-run can recover from the Redis snapshot.
- **Soft compaction:** `DELETE /runs/{id}` purges a terminal run's events and
  messages, emits `run_cleared`, and keeps the run row for audit/metrics.
- **Abort:** `POST /runs/{id}/abort` sets a short-lived Redis abort key; the LLM
  stream and parallel iterations check it and unwind cleanly, persisting partial
  output.
- **Embedding dimension:** the `embeddings.vector` column is fixed at **1024**; if
  you change `EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS`, the column (and a migration)
  must match.
- **Host responsibilities:** mounting routers, creating the FastAPI app + CORS,
  building runtime collaborators, registering context sources / dedup checkers /
  system tools, running migrations, seeding permissions, and managing process
  lifecycle. Luna Core deliberately does none of these automatically.

---

*Luna Core — version 0.3.0.*
