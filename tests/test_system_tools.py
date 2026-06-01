"""Tests for the system tool registry and built-in handlers.

Hermetic: each test constructs an isolated SystemToolRegistry (so the
process-wide default isn't poked) and an in-memory FakeRedis matching
the surface ScratchpadStore actually uses.
"""
from __future__ import annotations

import json
import uuid

import pytest

from luna_core.mcp.system_tools import SystemTool, SystemToolRegistry, install_builtins
from luna_core.mcp.system_tools import list_scratchpad as list_scratchpad_module
from luna_core.mcp.system_tools import stash_records as stash_records_module
from luna_core.mcp.system_tools import yield_iteration as yield_iteration_module


# ---- FakeRedis (same shape as test_scratchpad's; duplicated on purpose
# so each test file stays self-contained) ----------------------------------


class FakeRedis:
    def __init__(self) -> None:
        self._strings: dict[str, bytes] = {}
        self._sets: dict[str, set[bytes]] = {}

    async def set(self, key, value, *, ex=None):
        self._strings[key] = value.encode("utf-8")

    async def get(self, key):
        return self._strings.get(key)

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self._strings:
                del self._strings[key]
                removed += 1
            if key in self._sets:
                del self._sets[key]
                removed += 1
        return removed

    async def expire(self, key, seconds):
        pass

    async def sadd(self, key, *members):
        bucket = self._sets.setdefault(key, set())
        added = 0
        for m in members:
            b = m.encode("utf-8") if isinstance(m, str) else m
            if b not in bucket:
                bucket.add(b)
                added += 1
        return added

    async def srem(self, key, *members):
        bucket = self._sets.get(key)
        if bucket is None:
            return 0
        removed = 0
        for m in members:
            b = m.encode("utf-8") if isinstance(m, str) else m
            if b in bucket:
                bucket.discard(b)
                removed += 1
        if not bucket:
            del self._sets[key]
        return removed

    async def smembers(self, key):
        return set(self._sets.get(key, set()))

    async def scard(self, key):
        return len(self._sets.get(key, set()))


# ---- Registry basics -----------------------------------------------------


@pytest.fixture
def registry() -> SystemToolRegistry:
    return SystemToolRegistry()


async def _noop_handler(args, *, call_context):
    return {"ok": True}


def _make_tool(name: str, scope: str = "catalog", terminal: bool = False) -> SystemTool:
    return SystemTool(
        name=name,
        description="",
        input_schema={"type": "object"},
        handler=_noop_handler,
        scope=scope,  # type: ignore[arg-type]
        terminal=terminal,
    )


def test_register_then_get_roundtrip(registry):
    tool = _make_tool("foo")
    registry.register(tool)
    assert registry.get("foo") is tool


def test_get_unknown_returns_none(registry):
    assert registry.get("ghost") is None


def test_duplicate_registration_raises(registry):
    registry.register(_make_tool("foo"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_tool("foo"))


def test_get_many_skips_unknowns_and_preserves_order(registry):
    registry.register(_make_tool("a"))
    registry.register(_make_tool("c"))
    fetched = registry.get_many(["a", "ghost", "c"])
    assert [t.name for t in fetched] == ["a", "c"]


def test_list_catalog_excludes_context_tools(registry):
    registry.register(_make_tool("cat1", scope="catalog"))
    registry.register(_make_tool("ctx1", scope="context"))
    registry.register(_make_tool("cat2", scope="catalog"))
    catalog = {t.name for t in registry.list_catalog()}
    assert catalog == {"cat1", "cat2"}


def test_list_all_returns_both_scopes(registry):
    registry.register(_make_tool("cat1", scope="catalog"))
    registry.register(_make_tool("ctx1", scope="context"))
    assert {t.name for t in registry.list_all()} == {"cat1", "ctx1"}


def test_install_builtins_registers_known_tools(registry):
    install_builtins(registry)
    by_name = {t.name: t for t in registry.list_all()}
    assert set(by_name) == {"stash_records", "list_scratchpad", "yield_iteration"}
    assert by_name["stash_records"].scope == "catalog"
    assert by_name["stash_records"].terminal is False
    assert by_name["list_scratchpad"].scope == "catalog"
    assert by_name["list_scratchpad"].terminal is False
    assert by_name["yield_iteration"].scope == "context"
    assert by_name["yield_iteration"].terminal is True


def test_install_builtins_into_default_registry_is_idempotent_via_engine_import():
    # The engine package installs builtins as a side effect of import and
    # swallows the duplicate-registration error. Re-importing must not
    # raise — this guards against the swallow being accidentally removed.
    from luna_core.mcp.system_tools import get_default_registry

    reg = get_default_registry()
    names = {t.name for t in reg.list_all()}
    assert "stash_records" in names
    assert "list_scratchpad" in names
    assert "yield_iteration" in names


# ---- stash_records handler ------------------------------------------------


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def call_context(redis: FakeRedis):
    return {
        "redis": redis,
        "flow_run_id": uuid.UUID("00000000-0000-0000-0000-000000000077"),
        "node_id": "agent_1",
    }


@pytest.mark.asyncio
async def test_stash_records_writes_to_scratchpad(call_context, redis):
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"x": 1}, {"x": 2}]},
        call_context=call_context,
    )
    assert result["stashed"] == 2
    assert result["collection"] == "pending"
    assert len(result["ids"]) == 2
    # Verify the records actually landed.
    run_id = call_context["flow_run_id"]
    rk_0 = f"scratchpad:{run_id}:records:pending:{result['ids'][0]}"
    payload = redis._strings[rk_0]
    assert json.loads(payload) == {"x": 1}


@pytest.mark.asyncio
async def test_stash_records_honors_explicit_record_ids(call_context):
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [{"a": 1}, {"a": 2}],
            "record_ids": ["job-7", None],
        },
        call_context=call_context,
    )
    assert result["ids"][0] == "job-7"
    assert result["ids"][1] != "job-7"


@pytest.mark.asyncio
async def test_stash_records_empty_batch_returns_zero(call_context):
    result = await stash_records_module.handler(
        {"collection": "pending", "records": []},
        call_context=call_context,
    )
    assert result == {"stashed": 0, "collection": "pending", "ids": []}


@pytest.mark.asyncio
async def test_stash_records_missing_redis_raises_runtime_error():
    # Programming error in the dispatcher — must surface, not be masked
    # as an agent-visible tool result.
    with pytest.raises(RuntimeError, match="call_context"):
        await stash_records_module.handler(
            {"collection": "pending", "records": [{"x": 1}]},
            call_context={"flow_run_id": uuid.uuid4()},
        )


@pytest.mark.asyncio
async def test_stash_records_bad_collection_returns_error_dict(call_context):
    result = await stash_records_module.handler(
        {"collection": "", "records": [{"x": 1}]},
        call_context=call_context,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_stash_records_non_list_records_returns_error_dict(call_context):
    result = await stash_records_module.handler(
        {"collection": "pending", "records": "not a list"},
        call_context=call_context,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_stash_records_invalid_collection_name_returns_error_dict(call_context):
    # Validation comes from ScratchpadStore — handler should surface it
    # as a soft error so the agent can retry with a better name.
    result = await stash_records_module.handler(
        {"collection": "BadName", "records": [{"x": 1}]},
        call_context=call_context,
    )
    assert "error" in result


# ---- stash_records idempotency (already_stashed) -------------------------
#
# A confused agent can call stash_records twice with the same record
# inside one turn — the handler must NOT re-insert and must surface the
# duplicate under already_stashed so the agent stops retrying. Two
# scopes of duplication: across calls (record was stashed earlier in
# the turn) and within a single call (same record sent twice in the
# records array).


@pytest.mark.asyncio
async def test_stash_records_second_call_reports_already_stashed(call_context):
    first = await stash_records_module.handler(
        {"collection": "decisions", "records": [{"id": "j-1", "score": 0.9}]},
        call_context=call_context,
    )
    assert first["stashed"] == 1
    assert "already_stashed" not in first

    second = await stash_records_module.handler(
        {"collection": "decisions", "records": [{"id": "j-1", "score": 0.9}]},
        call_context=call_context,
    )
    assert second["stashed"] == 0
    assert second["ids"] == []
    assert len(second["already_stashed"]) == 1
    assert second["already_stashed"][0]["id"] == first["ids"][0]


@pytest.mark.asyncio
async def test_stash_records_duplicates_within_one_call_reported(call_context):
    # Same record twice in one records array → first wins, second is
    # reported as already_stashed within this call.
    result = await stash_records_module.handler(
        {
            "collection": "decisions",
            "records": [{"id": "j-1", "x": 1}, {"id": "j-1", "x": 1}],
        },
        call_context=call_context,
    )
    assert result["stashed"] == 1
    assert len(result["already_stashed"]) == 1
    assert result["already_stashed"][0]["id"] == result["ids"][0]


@pytest.mark.asyncio
async def test_stash_records_explicit_record_id_idempotent_across_calls(call_context):
    # When the agent passes a stable record_id, repeated calls with
    # the same id are no-ops regardless of payload changes — the
    # already_stashed branch protects against the second SET.
    await stash_records_module.handler(
        {
            "collection": "decisions",
            "records": [{"score": 0.9}],
            "record_ids": ["job-7"],
        },
        call_context=call_context,
    )
    second = await stash_records_module.handler(
        {
            "collection": "decisions",
            "records": [{"score": 0.99}],  # different payload, same id
            "record_ids": ["job-7"],
        },
        call_context=call_context,
    )
    assert second["stashed"] == 0
    assert second["already_stashed"][0]["id"] == "job-7"


@pytest.mark.asyncio
async def test_stash_records_mixed_new_and_already_stashed_split_correctly(call_context):
    # Pre-stash one record, then send a batch with that one + a new
    # one. The new one is inserted; the old one is reported under
    # already_stashed.
    await stash_records_module.handler(
        {
            "collection": "decisions",
            "records": [{"id": "j-1"}],
            "record_ids": ["j-1"],
        },
        call_context=call_context,
    )
    result = await stash_records_module.handler(
        {
            "collection": "decisions",
            "records": [{"id": "j-1"}, {"id": "j-2"}],
            "record_ids": ["j-1", "j-2"],
        },
        call_context=call_context,
    )
    assert result["stashed"] == 1
    assert result["ids"] == ["j-2"]
    assert len(result["already_stashed"]) == 1
    assert result["already_stashed"][0]["id"] == "j-1"


# ---- stash_records record_schema validation ------------------------------


@pytest.mark.asyncio
async def test_stash_records_validates_against_declared_schema(call_context):
    # When the runtime injects a stash_record_schema, the handler enforces
    # presence + type per field. Failures come back structured so the
    # agent can pinpoint which record/field broke.
    call_context["stash_record_schema"] = [
        {"name": "title", "type": "string"},
        {"name": "salary", "type": "number", "nullable": True},
    ]
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [
                {"title": "Senior Engineer", "salary": 120000},
                {"title": "Junior"},  # missing salary — even though nullable,
                                        # the field is required to be present
            ],
        },
        call_context=call_context,
    )
    assert "error" in result
    assert "details" in result
    assert {"record_index": 1, "field": "salary", "reason": "required field missing"} in result["details"]


@pytest.mark.asyncio
async def test_stash_records_accepts_null_for_nullable_field(call_context):
    call_context["stash_record_schema"] = [
        {"name": "title", "type": "string"},
        {"name": "salary", "type": "number", "nullable": True},
    ]
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [{"title": "Remote-only", "salary": None}],
        },
        call_context=call_context,
    )
    assert "error" not in result
    assert result["stashed"] == 1


@pytest.mark.asyncio
async def test_stash_records_rejects_null_for_non_nullable_field(call_context):
    call_context["stash_record_schema"] = [
        {"name": "title", "type": "string"},
    ]
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"title": None}]},
        call_context=call_context,
    )
    assert "error" in result
    assert any(
        e.get("reason", "").startswith("field is null but schema declares it non-nullable")
        for e in result["details"]
    )


@pytest.mark.asyncio
async def test_stash_records_rejects_wrong_type(call_context):
    call_context["stash_record_schema"] = [
        {"name": "count", "type": "integer"},
    ]
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"count": "not an int"}]},
        call_context=call_context,
    )
    assert "error" in result
    assert result["details"][0]["reason"].startswith("expected integer")


@pytest.mark.asyncio
async def test_stash_records_bool_is_not_int(call_context):
    # Python treats bool as int but the schema editor models them as
    # distinct types; the validator follows the editor's mental model.
    call_context["stash_record_schema"] = [
        {"name": "count", "type": "integer"},
    ]
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"count": True}]},
        call_context=call_context,
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_stash_records_without_schema_is_permissive(call_context):
    # When no schema is in call_context, anything goes — the back-compat
    # path for flows that predate the declarative shape.
    assert "stash_record_schema" not in call_context
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"anything": "goes", "even": 42}]},
        call_context=call_context,
    )
    assert "error" not in result
    assert result["stashed"] == 1


@pytest.mark.asyncio
async def test_stash_records_validation_short_circuits_before_write(call_context, redis):
    # A validation failure must NOT touch the scratchpad — otherwise
    # a half-valid batch could leak partial state.
    call_context["stash_record_schema"] = [
        {"name": "title", "type": "string"},
    ]
    await stash_records_module.handler(
        {"collection": "pending", "records": [{"title": 42}]},
        call_context=call_context,
    )
    assert redis._strings == {}
    assert redis._sets == {}


# ---- list_scratchpad handler ---------------------------------------------


@pytest.mark.asyncio
async def test_list_scratchpad_returns_every_record(call_context):
    await stash_records_module.handler(
        {"collection": "decisions", "records": [{"x": 1}, {"x": 2}]},
        call_context=call_context,
    )
    result = await list_scratchpad_module.handler(
        {"collection": "decisions"}, call_context=call_context
    )
    assert result["collection"] == "decisions"
    assert result["count"] == 2
    payloads = sorted(item["record"]["x"] for item in result["records"])
    assert payloads == [1, 2]
    for item in result["records"]:
        assert isinstance(item["id"], str) and item["id"]


@pytest.mark.asyncio
async def test_list_scratchpad_empty_collection_returns_zero(call_context):
    result = await list_scratchpad_module.handler(
        {"collection": "decisions"}, call_context=call_context
    )
    assert result == {"collection": "decisions", "count": 0, "records": []}


@pytest.mark.asyncio
async def test_list_scratchpad_scoped_to_flow_run(redis):
    # A read on flow_run B must not see records stashed by flow_run A,
    # even with the same collection name — scratchpad keys include the
    # run id.
    run_a = {"redis": redis, "flow_run_id": uuid.UUID("00000000-0000-0000-0000-0000000000aa")}
    run_b = {"redis": redis, "flow_run_id": uuid.UUID("00000000-0000-0000-0000-0000000000bb")}
    await stash_records_module.handler(
        {"collection": "shared", "records": [{"belongs_to": "a"}]},
        call_context=run_a,
    )
    result_b = await list_scratchpad_module.handler(
        {"collection": "shared"}, call_context=run_b
    )
    assert result_b["count"] == 0


@pytest.mark.asyncio
async def test_list_scratchpad_missing_redis_raises_runtime_error():
    with pytest.raises(RuntimeError, match="call_context"):
        await list_scratchpad_module.handler(
            {"collection": "decisions"},
            call_context={"flow_run_id": uuid.uuid4()},
        )


@pytest.mark.asyncio
async def test_list_scratchpad_bad_collection_returns_error_dict(call_context):
    result = await list_scratchpad_module.handler(
        {"collection": ""}, call_context=call_context
    )
    assert "error" in result


@pytest.mark.asyncio
async def test_list_scratchpad_invalid_collection_name_returns_error_dict(call_context):
    result = await list_scratchpad_module.handler(
        {"collection": "BadName"}, call_context=call_context
    )
    assert "error" in result


# ---- yield_iteration handler ---------------------------------------------


@pytest.mark.asyncio
async def test_yield_iteration_returns_ack_with_iteration_index():
    result = await yield_iteration_module.handler(
        {"next_carry": {"cursor": "abc"}, "done": False},
        call_context={"iteration_index": 3},
    )
    assert result == {"ok": True, "iteration": 3, "done": False}


@pytest.mark.asyncio
async def test_yield_iteration_done_signal_surfaces_in_ack():
    result = await yield_iteration_module.handler(
        {"next_carry": {}, "done": True},
        call_context={"iteration_index": 7},
    )
    assert result["done"] is True
