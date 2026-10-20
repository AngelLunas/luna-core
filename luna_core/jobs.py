"""Background jobs: one class per job, registered on the shared Celery app.

A job states what Celery needs to know as class attributes — its task
``name``, the ``queue`` it runs on, its beat ``schedule``, how long a queued
pass stays worth running (``expires``), time limits, retries — and does its
work in ``async run``. ``register_job`` turns the class into a Celery task
(a sync entry point that drives ``run`` on the worker's event loop), adds its
beat entry when it has a schedule and its route when it has a queue, all in
one place. A host collects its jobs with ``discover_jobs(package)`` — one
module per job — so adding a job is adding a file.

Callers enqueue through the class (``MyJob.enqueue(...)``): the route knows the
queue, so no call site repeats it. Task names are the stable contract: a
message already queued, or a beat entry already firing, finds its task by
name across deploys.
"""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, ClassVar

from celery.result import AsyncResult
from celery.schedules import schedule as CelerySchedule

from luna_core.tasks import _run_async, celery_app


class Job(ABC):
    """A unit of background work. Subclass, set ``name``, implement ``run``,
    decorate with ``register_job``."""

    #: The Celery task name — stable: queued messages and beat find it by name.
    name: ClassVar[str]
    #: The queue it runs on; ``None`` is the default queue.
    queue: ClassVar[str | None] = None
    #: Seconds between passes, a ``timedelta`` or a ``crontab``; ``None`` runs
    #: only when enqueued.
    schedule: ClassVar[float | timedelta | CelerySchedule | None] = None
    #: A scheduled pass that waited longer than this (seconds) is dropped, so
    #: an outage never replays hours of stale passes.
    expires: ClassVar[float | None] = None
    soft_time_limit: ClassVar[int | None] = None
    time_limit: ClassVar[int | None] = None
    max_retries: ClassVar[int | None] = None

    #: The Celery task, set by ``register_job``.
    task: ClassVar[Any]

    @abstractmethod
    async def run(self, *args: Any) -> Any:
        """The work. Arguments are what ``enqueue`` was given (JSON-able)."""

    @classmethod
    def enqueue(cls, *args: Any, **options: Any) -> AsyncResult:
        """Queue one run with ``args``; ``options`` go to ``apply_async``."""
        return cls.task.apply_async(args=list(args), **options)


def register_job(cls: type[Job]) -> type[Job]:
    """Make ``cls`` a Celery task with its route and beat entry."""
    options = {
        key: getattr(cls, key)
        for key in ("soft_time_limit", "time_limit", "max_retries")
        if getattr(cls, key) is not None
    }

    def entry(*args: Any) -> Any:
        return _run_async(cls().run(*args))

    entry.__name__ = entry.__qualname__ = cls.__name__
    entry.__doc__ = cls.__doc__
    entry.__module__ = cls.__module__
    cls.task = celery_app.task(name=cls.name, **options)(entry)

    if cls.queue:
        celery_app.conf.task_routes = {
            **(celery_app.conf.task_routes or {}),
            cls.name: {"queue": cls.queue},
        }
    if cls.schedule is not None:
        beat: dict[str, Any] = {"task": cls.name, "schedule": cls.schedule}
        if cls.expires is not None:
            beat["options"] = {"expires": cls.expires}
        celery_app.conf.beat_schedule = {
            **(celery_app.conf.beat_schedule or {}),
            cls.name: beat,
        }
    return cls


def discover_jobs(package: str) -> list[str]:
    """Import every module under ``package`` (registering the jobs they
    declare); returns the module names imported."""
    root = importlib.import_module(package)
    imported = []
    for module in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        importlib.import_module(module.name)
        imported.append(module.name)
    return imported


__all__ = ["Job", "discover_jobs", "register_job"]
