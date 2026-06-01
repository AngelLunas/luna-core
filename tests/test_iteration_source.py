"""Tests for the iteration-source layer.

Two surfaces are covered:
  - pure helpers in ``engine/iteration.py`` (source normalization,
    scratchpad collection validation)
  - ``_iterate_with_scratchpad`` on NodeExecutor — exercised against
    in-memory stubs that replace the LLM/MCP/agent fetch so we can
    assert the loop's bookkeeping (read every snapshot id, drop each
    after run, respect max_iterations) without standing up a full
    runtime.
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna_core.engine.iteration import (
    ITERATION_SOURCE_AGENT_YIELD,
    ITERATION_SOURCE_SCRATCHPAD,
    IterationSourceError,
    format_stash_schema_addendum,
    resolve_iteration_source,
    resolve_scratchpad_collection,
)
from luna_core.engine.nodes import NodeExecutor


# ---- Pure helpers -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ITERATION_SOURCE_AGENT_YIELD),
        ("", ITERATION_SOURCE_AGENT_YIELD),
        ("garbage", ITERATION_SOURCE_AGENT_YIELD),
        (ITERATION_SOURCE_AGENT_YIELD, ITERATION_SOURCE_AGENT_YIELD),
        (ITERATION_SOURCE_SCRATCHPAD, ITERATION_SOURCE_SCRATCHPAD),
    ],
)
def test_resolve_iteration_source_normalizes_to_known_tokens(raw, expected):
    assert resolve_iteration_source(raw) == expected


def test_resolve_scratchpad_collection_happy_path():
    cfg = {"collection": "pending_review"}
    assert resolve_scratchpad_collection(cfg, node_id="n1") == "pending_review"


@pytest.mark.parametrize(
    "bad_cfg",
    [None, "string", 42, {}, {"collection": ""}, {"collection": None}, {"other": "x"}],
)
def test_resolve_scratchpad_collection_rejects_invalid(bad_cfg):
    with pytest.raises(IterationSourceError):
        resolve_scratchpad_collection(bad_cfg, node_id="n1")


def test_format_stash_schema_addendum_returns_none_when_empty():
    # Both shapes the caller might pass: missing schema and empty list.
    assert format_stash_schema_addendum(None) is None
    assert format_stash_schema_addendum([]) is None


def test_format_stash_schema_addendum_renders_required_and_nullable_flags():
    schema = [
        {"name": "title", "type": "string"},
        {"name": "location", "type": "string", "nullable": True},
        {"name": "skills", "type": "array"},
        {"name": "count", "type": "integer", "default": 0},
    ]
    out = format_stash_schema_addendum(schema)
    assert out is not None
    # Header is present so the LLM has an unambiguous section anchor.
    assert "# Stash records contract" in out
    # Fields without nullable=True render as required.
    assert "- title: string (required)" in out
    # Nullable rendering is explicit.
    assert "- location: string (nullable)" in out
    # Defaults surface when set so the agent knows what's used when absent.
    assert "- count: integer (required, default=0)" in out
    # Trailing guidance gives the agent a hint about the retry contract.
    assert "tool error" in out


def test_resolve_scratchpad_collection_error_mentions_node_id():
    with pytest.raises(IterationSourceError, match="my_node"):
        resolve_scratchpad_collection({}, node_id="my_node")


# ---- _iterate_with_scratchpad ------------------------------------------


class _FakeScratchpad:
    """Tracks list_ids / get / drop calls so tests can assert ordering
    and side effects without a real Redis. Records start populated;
    drop removes them.
    """

    def __init__(self, records: dict[str, dict[str, Any]]):
        self._records = dict(records)
        self.dropped: list[str] = []

    async def list_ids(self, _run_id, _collection):
        return list(self._records.keys())

    async def get(self, _run_id, _collection, record_id):
        return self._records.get(record_id)

    async def drop(self, _run_id, _collection, record_id):
        if record_id in self._records:
            del self._records[record_id]
            self.dropped.append(record_id)
            return True
        return False


class _FakeRedis:
    """Tiny async stand-in for redis.asyncio.Redis covering just the
    surface ``_execute_single_scratchpad_iteration`` touches: ``exists``
    for the abort key pre-check. Aborted-by-default is False; tests can
    set ``aborted = True`` to simulate the operator clicking Abort
    between dispatching the iteration and the semaphore letting it run.
    """

    def __init__(self) -> None:
        self.aborted = False

    async def exists(self, _key: str) -> int:
        return 1 if self.aborted else 0


def _make_executor(
    scratchpad: _FakeScratchpad,
    *,
    redis: _FakeRedis | None = None,
) -> tuple[NodeExecutor, list[dict[str, Any]], _FakeRedis]:
    """Construct a NodeExecutor with just enough wiring to exercise the
    scratchpad loop. The LLM/MCP/DB collaborators are mocks — the loop
    body doesn't care what they return, only that they're called.

    Returns the executor plus an ``emitted`` list the caller can inspect
    to assert the per-iteration lifecycle events (iteration_started /
    iteration_completed / iteration_failed) were fired with the right
    payloads, and the fake redis (so a test can flip the abort flag
    mid-run if it wants to).
    """
    emitter = MagicMock()
    emitter.flow_run_id = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
    emitted: list[dict[str, Any]] = []

    async def fake_emit(event_type, node_id=None, payload=None):
        emitted.append(
            {
                "event_type": event_type,
                "node_id": node_id,
                "payload": dict(payload) if payload else {},
            }
        )

    emitter.emit = fake_emit
    fake_redis = redis or _FakeRedis()
    executor = NodeExecutor(
        emitter=emitter,
        db=AsyncMock(),
        redis=fake_redis,
        llm_router=MagicMock(),
        mcp_client=MagicMock(),
    )
    # ScratchpadStore is constructed inside the helper from self._redis;
    # patch the class so all instances route to our fake.
    return executor, emitted, fake_redis


async def _run_scratchpad_with(
    scratchpad: _FakeScratchpad,
    iteration_cfg: dict[str, Any],
    max_iterations: int = 50,
    *,
    fake_run=None,
    redis: _FakeRedis | None = None,
):
    """Run ``_iterate_with_scratchpad`` against an in-memory fake.

    Pass ``fake_run`` to override the stubbed AgentRunner.run — useful
    for parallel-mode tests that need to observe concurrency (e.g. by
    blocking on an event) or to inject failures. Pass ``redis`` to use
    a pre-built ``_FakeRedis`` (e.g. with ``aborted=True``).
    """
    executor, emitted, fake_redis = _make_executor(scratchpad, redis=redis)
    fake_node = MagicMock(id="scorer", config={})
    runner_calls: list[dict[str, Any]] = []

    if fake_run is None:

        async def _default_run(**kwargs):
            runner_calls.append(kwargs)
            return {}

        fake_run = _default_run
    else:
        original = fake_run

        async def _tracking(**kwargs):
            runner_calls.append(kwargs)
            return await original(**kwargs)

        fake_run = _tracking

    # ``_execute_single_scratchpad_iteration`` now opens a per-iteration
    # ``AsyncSessionLocal()`` and builds an EventEmitter against it for
    # transaction isolation under parallel mode. The tests don't want a
    # real DB roundtrip, so we patch both:
    #  - AsyncSessionLocal returns an async context manager yielding a
    #    throwaway AsyncMock session.
    #  - EventEmitter is patched so the per-iteration instance routes
    #    its emit() calls into the same ``emitted`` list the executor's
    #    outer emitter writes to. That keeps the existing assertions
    #    (one started + one completed per item, etc.) valid without
    #    distinguishing which emitter emitted what.
    class _FakeSessionCM:
        async def __aenter__(self):
            return AsyncMock()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _fake_emitter_factory(db, redis_, run_id):
        m = MagicMock()
        m.flow_run_id = run_id

        async def _record(event_type, node_id=None, payload=None):
            emitted.append(
                {
                    "event_type": event_type,
                    "node_id": node_id,
                    "payload": dict(payload) if payload else {},
                }
            )

        m.emit = _record
        return m

    with (
        patch(
            "luna_core.engine.nodes.AsyncSessionLocal",
            new=_FakeSessionCM,
        ),
        patch(
            "luna_core.engine.nodes.EventEmitter",
            side_effect=_fake_emitter_factory,
        ),
        patch(
            "luna_core.engine.nodes.ScratchpadStore",
            return_value=scratchpad,
        ),
        patch(
            "luna_core.engine.nodes.AgentRunner",
            return_value=MagicMock(run=fake_run),
        ),
        patch.object(executor, "_load_history", new=AsyncMock(return_value=[])),
        patch(
            "luna_core.engine.nodes._resolve_prompt",
            return_value="prompt",
        ),
        patch(
            "luna_core.engine.nodes.build_system_prompt",
            return_value="system",
        ),
    ):
        agent = MagicMock(role="", instructions="")
        result = await executor._iterate_with_scratchpad(
            node=fake_node,
            state={"inputs": {}},
            iteration_cfg=iteration_cfg,
            agent=agent,
            inherit_from=[],
            include_tool_interactions=True,
            loaded_context={},
            max_iterations=max_iterations,
        )
    return result, runner_calls, emitted


@pytest.mark.asyncio
async def test_scratchpad_processes_every_item_and_drops_each():
    scratchpad = _FakeScratchpad(
        {"job-a": {"title": "A"}, "job-b": {"title": "B"}, "job-c": {"title": "C"}}
    )
    result, runner_calls, emitted = await _run_scratchpad_with(
        scratchpad, {"source_config": {"collection": "pending"}}
    )
    assert result["processed"] == 3
    assert result["failed"] == 0
    assert result["snapshot_size"] == 3
    assert result["exit_reason"] == "exhausted"
    assert result["collection"] == "pending"
    assert result["execution"] == "sequential"
    # Every snapshot id was dropped, in the deterministic sorted order
    # the helper applies before iterating.
    assert sorted(scratchpad.dropped) == sorted(["job-a", "job-b", "job-c"])
    assert scratchpad.dropped == sorted(scratchpad.dropped)
    # The agent was invoked once per item.
    assert len(runner_calls) == 3
    # Lifecycle events: one started + one completed per processed item.
    started = [e for e in emitted if e["event_type"].name == "iteration_started"]
    completed = [e for e in emitted if e["event_type"].name == "iteration_completed"]
    assert len(started) == 3
    assert len(completed) == 3
    # Each iteration_id is a stable UUID propagated from started to completed.
    started_ids = {e["payload"]["iteration_id"] for e in started}
    completed_ids = {e["payload"]["iteration_id"] for e in completed}
    assert started_ids == completed_ids
    assert all("duration_ms" in e["payload"] for e in completed)


@pytest.mark.asyncio
async def test_scratchpad_empty_collection_is_a_clean_noop():
    scratchpad = _FakeScratchpad({})
    result, runner_calls, emitted = await _run_scratchpad_with(
        scratchpad, {"source_config": {"collection": "pending"}}
    )
    assert result == {
        "processed": 0,
        "skipped_missing": 0,
        "skipped_aborted": 0,
        "failed": 0,
        "snapshot_size": 0,
        "collection": "pending",
        "exit_reason": "exhausted",
        "execution": "sequential",
    }
    assert runner_calls == []
    assert scratchpad.dropped == []
    assert emitted == []


@pytest.mark.asyncio
async def test_scratchpad_respects_max_iterations_cap():
    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(10)}
    )
    result, runner_calls, _emitted = await _run_scratchpad_with(
        scratchpad,
        {"source_config": {"collection": "pending"}},
        max_iterations=3,
    )
    assert result["processed"] == 3
    assert result["snapshot_size"] == 10
    assert result["exit_reason"] == "max_iterations"
    # The first 3 (sorted) were dropped; the rest remain in scratchpad
    # for a future run to consume.
    assert scratchpad.dropped == ["job-00", "job-01", "job-02"]
    assert len(runner_calls) == 3


@pytest.mark.asyncio
async def test_scratchpad_missing_record_counted_as_skipped():
    # Simulate the scratchpad reporting an id but returning None on get
    # (raced with a concurrent drop / TTL expiry).
    scratchpad = _FakeScratchpad({"job-x": {"title": "X"}})

    real_get = scratchpad.get

    async def get_returning_none_for_y(_run_id, _collection, record_id):
        if record_id == "job-y":
            return None
        return await real_get(_run_id, _collection, record_id)

    scratchpad.get = get_returning_none_for_y  # type: ignore[assignment]

    async def list_with_extra(_run_id, _collection):
        return ["job-x", "job-y"]

    scratchpad.list_ids = list_with_extra  # type: ignore[assignment]

    result, runner_calls, _emitted = await _run_scratchpad_with(
        scratchpad, {"source_config": {"collection": "pending"}}
    )
    assert result["processed"] == 1
    assert result["skipped_missing"] == 1
    assert result["snapshot_size"] == 2
    assert result["exit_reason"] == "exhausted"
    # job-y was never dropped (we never got hold of it); job-x was.
    assert scratchpad.dropped == ["job-x"]
    assert len(runner_calls) == 1


@pytest.mark.asyncio
async def test_scratchpad_invalid_source_config_raises_node_error():
    from luna_core.engine.nodes import NodeExecutionError

    scratchpad = _FakeScratchpad({})
    with pytest.raises(NodeExecutionError, match="source_config"):
        await _run_scratchpad_with(
            scratchpad, {"source_config": "not a dict"}
        )


# ---- parallel execution mode -------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_parallel_processes_every_item():
    """Parallel mode walks the same snapshot, drops every record, and
    emits the same lifecycle envelope as sequential — just with the
    runs happening concurrently."""
    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(6)}
    )
    result, runner_calls, emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 3,
        },
    )
    assert result["processed"] == 6
    assert result["failed"] == 0
    assert result["execution"] == "parallel"
    assert sorted(scratchpad.dropped) == [f"job-{i:02d}" for i in range(6)]
    assert len(runner_calls) == 6
    started = [e for e in emitted if e["event_type"].name == "iteration_started"]
    completed = [e for e in emitted if e["event_type"].name == "iteration_completed"]
    assert len(started) == 6
    assert len(completed) == 6


@pytest.mark.asyncio
async def test_scratchpad_parallel_runs_concurrently_under_semaphore():
    """Sanity check on actual concurrency: with concurrency=3 and 6 items,
    we should observe at most 3 agent runs in flight at any moment.
    Without parallelism the max would be 1."""
    import asyncio

    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(6)}
    )
    in_flight = 0
    max_in_flight = 0
    gate = asyncio.Event()

    async def slow_run(**_kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield once so all parallel starters reach this point before any
        # exits — otherwise the first task could finish before the second
        # starts and the max would never rise above 1.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        gate.set()
        return {}

    result, _runner_calls, _emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 3,
        },
        fake_run=slow_run,
    )
    assert result["processed"] == 6
    assert 2 <= max_in_flight <= 3, f"max_in_flight={max_in_flight}"


@pytest.mark.asyncio
async def test_scratchpad_parallel_continue_records_failed_outcomes():
    """on_iteration_error=continue: a failing iteration becomes a 'failed'
    outcome; siblings still process to completion."""
    scratchpad = _FakeScratchpad(
        {"job-a": {"i": 0}, "job-b": {"i": 1}, "job-c": {"i": 2}}
    )

    async def maybe_fail(**kwargs):
        ctx = kwargs.get("extra_call_context") or {}
        if ctx.get("iteration_item_id") == "job-b":
            raise RuntimeError("boom")
        return {}

    result, _runner_calls, emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 3,
            "on_iteration_error": "continue",
        },
        fake_run=maybe_fail,
    )
    assert result["processed"] == 2
    assert result["failed"] == 1
    failed_events = [
        e for e in emitted if e["event_type"].name == "iteration_failed"
    ]
    assert len(failed_events) == 1
    assert failed_events[0]["payload"]["item_id"] == "job-b"
    # The failing record stays in the scratchpad (drop only happens on
    # success), the other two are consumed.
    assert "job-b" not in scratchpad.dropped
    assert sorted(scratchpad.dropped) == ["job-a", "job-c"]


@pytest.mark.asyncio
async def test_scratchpad_parallel_continue_emits_failed_on_cancelled_error():
    """CancelledError (BaseException, not Exception) must still produce an
    iteration_failed event — otherwise the UI shows the iteration as
    'running' forever and the scratchpad record is never drained.
    """
    import asyncio as _asyncio

    scratchpad = _FakeScratchpad(
        {"job-a": {"i": 0}, "job-b": {"i": 1}, "job-c": {"i": 2}}
    )

    async def maybe_cancel(**kwargs):
        ctx = kwargs.get("extra_call_context") or {}
        if ctx.get("iteration_item_id") == "job-b":
            raise _asyncio.CancelledError("simulated cancel")
        return {}

    result, _runner_calls, emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 3,
            "on_iteration_error": "continue",
        },
        fake_run=maybe_cancel,
    )
    # The cancelled iteration is counted in the outcome tags so the
    # operator sees the real total of failures, not a phantom success.
    assert result["processed"] == 2
    assert result["failed"] == 1
    failed_events = [
        e for e in emitted if e["event_type"].name == "iteration_failed"
    ]
    assert len(failed_events) == 1, "CancelledError must surface as iteration_failed"
    assert failed_events[0]["payload"]["item_id"] == "job-b"
    assert "CancelledError" in failed_events[0]["payload"]["error"]
    # The cancelled record stays in the scratchpad — drop only happens
    # on a successful return; this is what lets a retry pick it up.
    assert "job-b" not in scratchpad.dropped


@pytest.mark.asyncio
async def test_scratchpad_parallel_cancel_siblings_raises_node_error():
    """on_iteration_error=cancel_siblings: first failure cancels the rest
    and surfaces as NodeExecutionError."""
    from luna_core.engine.nodes import NodeExecutionError

    scratchpad = _FakeScratchpad(
        {"job-a": {"i": 0}, "job-b": {"i": 1}}
    )

    async def always_fail(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(NodeExecutionError, match="cancel_siblings"):
        await _run_scratchpad_with(
            scratchpad,
            {
                "source_config": {"collection": "pending"},
                "execution": "parallel",
                "concurrency": 2,
                "on_iteration_error": "cancel_siblings",
            },
            fake_run=always_fail,
        )


@pytest.mark.asyncio
async def test_scratchpad_parallel_clamps_concurrency_to_settings_max(
    monkeypatch,
):
    """concurrency=999 in node config is clamped down to
    settings.iteration_concurrency_max so a misconfigured flow can't run
    away from the host hardware."""
    from luna_core.engine import iteration as iteration_module

    monkeypatch.setattr(
        iteration_module.settings, "iteration_concurrency_max", 2
    )

    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(5)}
    )

    import asyncio

    in_flight = 0
    max_in_flight = 0

    async def slow_run(**_kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1
        return {}

    result, _runner_calls, _emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 999,
        },
        fake_run=slow_run,
    )
    assert result["processed"] == 5
    assert max_in_flight <= 2


@pytest.mark.asyncio
async def test_scratchpad_parallel_iteration_id_distinct_per_item():
    """Each item gets its own iteration_id; the events for one item all
    share that id (started + completed)."""
    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(4)}
    )
    _result, _runner_calls, emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 4,
        },
    )
    started = [e for e in emitted if e["event_type"].name == "iteration_started"]
    completed = [
        e for e in emitted if e["event_type"].name == "iteration_completed"
    ]
    started_ids = {e["payload"]["iteration_id"] for e in started}
    completed_ids = {e["payload"]["iteration_id"] for e in completed}
    # Four distinct iteration_ids, matched 1:1 between started and completed.
    assert len(started_ids) == 4
    assert started_ids == completed_ids


# ---- abort behaviour ---------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_parallel_pre_check_skips_iterations_when_aborted():
    """If the abort key is already set when an iteration's semaphore
    slot opens, that iteration must skip without burning an LLM call —
    every item ends up as 'skipped_aborted', no agent runs, no
    iteration_started events emitted."""
    scratchpad = _FakeScratchpad(
        {f"job-{i:02d}": {"i": i} for i in range(5)}
    )
    redis = _FakeRedis()
    redis.aborted = True  # Set before the run begins

    result, runner_calls, emitted = await _run_scratchpad_with(
        scratchpad,
        {
            "source_config": {"collection": "pending"},
            "execution": "parallel",
            "concurrency": 3,
        },
        redis=redis,
    )
    assert result["processed"] == 0
    assert result["skipped_aborted"] == 5
    assert result["failed"] == 0
    # No LLM calls happened — the pre-check short-circuited every task.
    assert runner_calls == []
    # And no iteration lifecycle events were emitted: a skipped
    # iteration never opens a block on the timeline.
    assert emitted == []
    # The records stay in the scratchpad for a future run to consume.
    assert scratchpad.dropped == []


@pytest.mark.asyncio
async def test_scratchpad_parallel_continue_promotes_abort_to_cancel_all():
    """on_iteration_error=continue normally absorbs failures and lets
    siblings keep running. AbortSignalError is the exception: it's a
    flow-level signal and must propagate out of the executor, not be
    swallowed as one more 'failed' outcome (otherwise the abort cascade
    keeps starting fresh LLM streams while the user is waiting for it
    to stop)."""
    from luna_core.llm.base import AbortSignalError

    scratchpad = _FakeScratchpad(
        {"job-a": {"i": 0}, "job-b": {"i": 1}, "job-c": {"i": 2}}
    )

    async def maybe_abort(**kwargs):
        ctx = kwargs.get("extra_call_context") or {}
        if ctx.get("iteration_item_id") == "job-b":
            raise AbortSignalError(
                uuid.UUID("00000000-0000-0000-0000-0000000000aa"), "scorer"
            )
        return {}

    with pytest.raises(AbortSignalError):
        await _run_scratchpad_with(
            scratchpad,
            {
                "source_config": {"collection": "pending"},
                "execution": "parallel",
                "concurrency": 3,
                "on_iteration_error": "continue",
            },
            fake_run=maybe_abort,
        )


@pytest.mark.asyncio
async def test_scratchpad_parallel_cancel_siblings_lets_abort_through_raw():
    """In cancel_siblings mode every non-abort failure is wrapped as
    NodeExecutionError. AbortSignalError must bypass that wrapping so
    the outer FlowRunner sees an AbortSignalError (and emits flow_failed
    with reason=aborted) instead of a generic NodeExecutionError."""
    from luna_core.llm.base import AbortSignalError

    scratchpad = _FakeScratchpad({"job-a": {"i": 0}, "job-b": {"i": 1}})

    async def always_abort(**_kwargs):
        raise AbortSignalError(
            uuid.UUID("00000000-0000-0000-0000-0000000000aa"), "scorer"
        )

    with pytest.raises(AbortSignalError):
        await _run_scratchpad_with(
            scratchpad,
            {
                "source_config": {"collection": "pending"},
                "execution": "parallel",
                "concurrency": 2,
                "on_iteration_error": "cancel_siblings",
            },
            fake_run=always_abort,
        )
