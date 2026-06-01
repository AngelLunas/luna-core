"""Celery app + tasks. The host application is responsible for booting the
Celery worker / beat process; luna-core simply exports a configured app and
the tasks it registers."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from celery import Celery
from celery.schedules import crontab

from luna_core.core.config import settings
from luna_core.core.db import AsyncSessionLocal
from luna_core.core.redis import get_redis
from luna_core.engine.runner import FlowRunner

logger = logging.getLogger(__name__)

# Host applications register an async factory here so the Celery tasks can
# build a FlowRunner with the right collaborators (LLMRouter, MCPClient,
# ConnectorRegistry). Without it the tasks fall back to a bare FlowRunner()
# which can only execute trivial flows — any node that needs a connector or
# LLM will fail. The result is cached per worker process.
RunnerFactory = Callable[[], Awaitable[FlowRunner]]
_runner_factory: RunnerFactory | None = None
_cached_runner: FlowRunner | None = None


def set_runner_factory(factory: RunnerFactory) -> None:
    """Register the async builder the worker process uses to construct its
    `FlowRunner`. Call from the host's Celery bootstrap module (e.g.
    `sentinel.worker`)."""
    global _runner_factory, _cached_runner
    _runner_factory = factory
    _cached_runner = None


async def _get_runner() -> FlowRunner:
    global _cached_runner
    if _cached_runner is not None:
        return _cached_runner
    if _runner_factory is None:
        logger.warning(
            "no FlowRunner factory registered; falling back to bare "
            "FlowRunner() — nodes requiring connectors / LLMs will fail. "
            "Register one with luna_core.tasks.set_runner_factory()."
        )
        _cached_runner = FlowRunner()
    else:
        _cached_runner = await _runner_factory()
    return _cached_runner

celery_app = Celery(
    "luna_core",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
)
celery_app.conf.update(
    task_default_queue=settings.celery_task_default_queue,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    timezone="UTC",
    enable_utc=True,
)

# Beat fires ``scheduler_tick`` every minute. The tick is the only beat entry
# we need: it scans the DB and dispatches per-flow ``trigger_scheduled_run``
# tasks for any rule that's due. Doing it that way (instead of registering one
# beat entry per flow) means schedules edited via the API take effect on the
# next minute boundary without restarting beat.
celery_app.conf.beat_schedule = {
    "scheduler-tick": {
        "task": "luna_core.scheduler_tick",
        "schedule": crontab(minute="*"),
    },
}


_persistent_loop: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Reuse one event loop for the lifetime of the worker process.

    `asyncio.run()` creates a fresh loop per call and closes it on exit, which
    invalidates anything attached to the previous loop — including the cached
    FlowRunner's Redis client, the asyncpg connection pool, and any httpx
    clients held by providers. Reusing one loop keeps those handles alive and
    valid across tasks (which is also why `--pool=solo` is required: prefork
    pools would each need their own loop).
    """
    global _persistent_loop
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_persistent_loop)
    return _persistent_loop


def _run_async(coro):  # noqa: ANN001
    return _get_loop().run_until_complete(coro)


@celery_app.task(name="luna_core.run_flow")
def run_flow_task(flow_run_id: str) -> str:
    """Celery entrypoint that drives FlowRunner.run() against a fresh
    AsyncSession + Redis client. The FlowRun row must already exist (created
    by the HTTP trigger endpoint or the scheduler) — the worker only
    executes it; it does not create a new row."""
    async def _execute() -> uuid.UUID:
        runner = await _get_runner()
        redis = get_redis()
        async with AsyncSessionLocal() as db:
            return await runner.run(db, redis, uuid.UUID(flow_run_id))

    run_id = _run_async(_execute())
    return str(run_id)


@celery_app.task(name="luna_core.resume_flow")
def resume_flow_task(
    flow_run_id: str,
    human_response: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    async def _execute() -> uuid.UUID:
        runner = await _get_runner()
        redis = get_redis()
        async with AsyncSessionLocal() as db:
            return await runner.resume(
                db, redis, uuid.UUID(flow_run_id), human_response, metadata
            )

    run_id = _run_async(_execute())
    return str(run_id)


@celery_app.task(name="luna_core.trigger_scheduled_run")
def trigger_scheduled_run_task(
    flow_id: str, trigger_data: dict[str, Any] | None = None
) -> str:
    """Wrapper used by Celery beat: create the FlowRun row, then enqueue the
    actual execution. Beat itself can't call async code, so this lives as a
    sync Celery task that runs a short async coroutine to insert the row,
    then delegates the heavy work to `run_flow`.
    """
    from luna_core.services.flow import create_flow_run

    async def _create() -> uuid.UUID:
        redis = get_redis()
        async with AsyncSessionLocal() as db:
            run = await create_flow_run(
                db, uuid.UUID(flow_id), trigger_data, redis=redis
            )
            return run.id

    run_id = _run_async(_create())
    run_flow_task.delay(str(run_id))
    return str(run_id)


@celery_app.task(name="luna_core.scheduler_tick")
def scheduler_tick() -> int:
    """Per-minute sweep: dispatch any schedule rule whose next fire is due.

    Returns the number of runs enqueued (handy for logs / metrics)."""

    async def _scan() -> int:
        from luna_core.services.scheduling import dispatch_due_runs

        async with AsyncSessionLocal() as db:
            return await dispatch_due_runs(db)

    return _run_async(_scan())


__all__ = [
    "celery_app",
    "run_flow_task",
    "resume_flow_task",
    "trigger_scheduled_run_task",
    "scheduler_tick",
    "set_runner_factory",
    "RunnerFactory",
]
