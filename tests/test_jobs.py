"""A job is one class: its task name, queue, schedule and limits as attributes,
its work in ``run``. ``register_job`` makes the Celery task, the beat entry and
the route from them — nothing for a host to merge by hand into the app's
config, where a forgotten ``**`` silently dropped every other entry."""
from __future__ import annotations

import pytest

from luna_core.jobs import Job, discover_jobs, register_job
from luna_core.tasks import celery_app


@pytest.fixture
def clean_conf():
    beat, routes = dict(celery_app.conf.beat_schedule or {}), dict(celery_app.conf.task_routes or {})
    yield
    celery_app.conf.beat_schedule, celery_app.conf.task_routes = beat, routes


def test_a_scheduled_job_gets_its_task_beat_entry_and_route(clean_conf):
    before = dict(celery_app.conf.beat_schedule)

    @register_job
    class Sweep(Job):
        name = "test.sweep"
        queue = "slow"
        schedule = 30.0
        expires = 25.0
        time_limit = 60

        async def run(self, n: int = 1) -> int:
            return n * 2

    assert celery_app.tasks["test.sweep"].name == Sweep.task.name == "test.sweep"
    assert Sweep.task.time_limit == 60
    assert celery_app.conf.task_routes["test.sweep"] == {"queue": "slow"}
    assert celery_app.conf.beat_schedule["test.sweep"] == {
        "task": "test.sweep", "schedule": 30.0, "options": {"expires": 25.0},
    }
    # Every entry registered before it is still there.
    assert before.items() <= celery_app.conf.beat_schedule.items()
    # The task runs the job's coroutine to completion.
    assert Sweep.task.run(21) == 42


def test_an_unscheduled_job_is_only_enqueued(clean_conf, monkeypatch):
    @register_job
    class OnDemand(Job):
        name = "test.on_demand"

        async def run(self, item: str) -> str:
            return item

    assert "test.on_demand" not in celery_app.conf.beat_schedule
    sent = {}
    monkeypatch.setattr(OnDemand.task, "apply_async", lambda **kw: sent.update(kw))
    OnDemand.enqueue("a", countdown=5)
    assert sent == {"args": ["a"], "countdown": 5}


def test_a_job_without_run_cannot_be_built():
    class Incomplete(Job):
        name = "test.incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_discovery_imports_every_module_of_a_package():
    assert "luna_core.schemas.user" in discover_jobs("luna_core.schemas")
