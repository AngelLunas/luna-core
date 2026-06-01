"""Tests for the dedup checker registry, the node-config resolver, and
the stash_records integration with a bound checker.

Hermetic: each test constructs an isolated ``DedupCheckerRegistry`` so
the process-wide default isn't poked. The stash_records integration
tests reuse the FakeRedis shape from test_system_tools to keep this
file self-contained.
"""
from __future__ import annotations

import uuid

import pytest

from luna_core.dedup.node_config import (
    StashDedupConfigError,
    build_call_context_checker,
    format_stash_dedup_addendum,
    resolve_stash_dedup_binding,
)
from luna_core.dedup.registry import (
    DEDUP_FIELD_PRIMITIVE_TYPES,
    DedupChecker,
    DedupCheckerRegistry,
    DedupFieldSpec,
    DedupVerdict,
    get_default_registry,
)
from luna_core.mcp.system_tools import stash_records as stash_records_module


# ---- FakeRedis (same shape as test_system_tools') ------------------------


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
def registry() -> DedupCheckerRegistry:
    return DedupCheckerRegistry()


async def _noop_handler(records, *, call_context):
    return [None] * len(records)


def _make_checker(
    name: str = "test.checker",
    *,
    required_fields: list[DedupFieldSpec] | None = None,
    label: str = "",
) -> DedupChecker:
    return DedupChecker(
        name=name,
        label=label,
        description="",
        required_fields=required_fields or [
            DedupFieldSpec(name="external_id", type="string"),
        ],
        handler=_noop_handler,
    )


def test_register_then_get_roundtrip(registry):
    checker = _make_checker()
    registry.register(checker)
    assert registry.get("test.checker") is checker


def test_get_unknown_returns_none(registry):
    assert registry.get("ghost") is None


def test_duplicate_registration_raises(registry):
    registry.register(_make_checker())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_checker())


def test_list_all_returns_every_registration(registry):
    registry.register(_make_checker("a"))
    registry.register(_make_checker("b"))
    assert {c.name for c in registry.list_all()} == {"a", "b"}


def test_default_registry_is_singleton():
    first = get_default_registry()
    second = get_default_registry()
    assert first is second


# ---- DedupFieldSpec validation ------------------------------------------


def test_field_spec_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        DedupFieldSpec(name="")


def test_field_spec_rejects_unknown_type():
    with pytest.raises(ValueError, match="must be one of"):
        DedupFieldSpec(name="x", type="weird-type")


def test_field_spec_accepts_every_primitive():
    for primitive in DEDUP_FIELD_PRIMITIVE_TYPES:
        DedupFieldSpec(name="x", type=primitive)


# ---- Resolver -----------------------------------------------------------


def test_resolver_returns_none_when_no_stash_block(registry):
    assert resolve_stash_dedup_binding(None, node_id="n1", registry=registry) is None
    assert resolve_stash_dedup_binding({}, node_id="n1", registry=registry) is None


def test_resolver_returns_none_when_no_dedup_key(registry):
    assert (
        resolve_stash_dedup_binding(
            {"record_schema": []}, node_id="n1", registry=registry
        )
        is None
    )


def test_resolver_raises_on_non_dict_dedup(registry):
    with pytest.raises(StashDedupConfigError, match="must be an object"):
        resolve_stash_dedup_binding(
            {"dedup": "no"}, node_id="n1", registry=registry
        )


def test_resolver_raises_on_missing_checker_name(registry):
    with pytest.raises(StashDedupConfigError, match="must be a non-empty string"):
        resolve_stash_dedup_binding(
            {"dedup": {"checker": ""}}, node_id="n1", registry=registry
        )


def test_resolver_raises_on_unknown_checker(registry):
    with pytest.raises(StashDedupConfigError, match="not a registered"):
        resolve_stash_dedup_binding(
            {"dedup": {"checker": "ghost", "fields": {}}},
            node_id="n1",
            registry=registry,
        )


def test_resolver_raises_on_missing_required_field_mapping(registry):
    registry.register(_make_checker())
    with pytest.raises(StashDedupConfigError, match="external_id"):
        resolve_stash_dedup_binding(
            {"dedup": {"checker": "test.checker", "fields": {}}},
            node_id="n1",
            registry=registry,
        )


def test_resolver_accepts_minimal_valid_config(registry):
    registry.register(_make_checker())
    binding = resolve_stash_dedup_binding(
        {"dedup": {"checker": "test.checker", "fields": {"external_id": "ext"}}},
        node_id="n1",
        registry=registry,
    )
    assert binding is not None
    assert binding.checker.name == "test.checker"
    assert binding.field_map == {"external_id": "ext"}


def test_resolver_allows_missing_optional_fields(registry):
    registry.register(
        _make_checker(
            required_fields=[
                DedupFieldSpec(name="external_id", type="string"),
                DedupFieldSpec(name="title", type="string", optional=True),
            ]
        )
    )
    binding = resolve_stash_dedup_binding(
        {"dedup": {"checker": "test.checker", "fields": {"external_id": "ext"}}},
        node_id="n1",
        registry=registry,
    )
    assert binding is not None
    assert binding.field_map == {"external_id": "ext"}


def test_resolver_rejects_non_string_field_mapping(registry):
    registry.register(_make_checker())
    with pytest.raises(StashDedupConfigError, match="must be a string"):
        resolve_stash_dedup_binding(
            {
                "dedup": {
                    "checker": "test.checker",
                    "fields": {"external_id": 42},
                }
            },
            node_id="n1",
            registry=registry,
        )


# ---- Bound checker (projection) -----------------------------------------


@pytest.mark.asyncio
async def test_bound_checker_projects_records_before_calling_handler(registry):
    captured: list[list[dict]] = []

    async def capture_handler(records, *, call_context):
        captured.append(records)
        return [None] * len(records)

    registry.register(
        DedupChecker(
            name="cap.checker",
            label="",
            description="",
            required_fields=[
                DedupFieldSpec(name="external_id", type="string"),
                DedupFieldSpec(name="source", type="string"),
            ],
            handler=capture_handler,
        )
    )
    binding = resolve_stash_dedup_binding(
        {
            "dedup": {
                "checker": "cap.checker",
                "fields": {"external_id": "ext", "source": "src"},
            }
        },
        node_id="n1",
        registry=registry,
    )
    bound = build_call_context_checker(binding)
    verdicts = await bound(
        [
            {"ext": "job-1", "src": "upwork", "title": "ignored"},
            {"ext": "job-2", "src": "upwork", "title": "ignored"},
        ],
        call_context={},
    )
    assert verdicts == [None, None]
    assert captured == [
        [
            {"external_id": "job-1", "source": "upwork"},
            {"external_id": "job-2", "source": "upwork"},
        ]
    ]


@pytest.mark.asyncio
async def test_bound_checker_flags_records_missing_required_field(registry):
    async def boom_handler(records, *, call_context):
        # Should never run for records flagged invalid by the projection.
        assert records == []
        return []

    registry.register(
        DedupChecker(
            name="check2",
            label="",
            description="",
            required_fields=[DedupFieldSpec(name="external_id", type="string")],
            handler=boom_handler,
        )
    )
    binding = resolve_stash_dedup_binding(
        {"dedup": {"checker": "check2", "fields": {"external_id": "ext"}}},
        node_id="n1",
        registry=registry,
    )
    bound = build_call_context_checker(binding)
    verdicts = await bound(
        [{"title": "no external id here"}],
        call_context={},
    )
    assert verdicts[0] is not None
    assert verdicts[0].match_kind == "invalid_record"
    assert "ext" in verdicts[0].reason


@pytest.mark.asyncio
async def test_bound_checker_splices_verdicts_back_in_order(registry):
    async def alternating(records, *, call_context):
        out: list[DedupVerdict | None] = []
        for idx, _r in enumerate(records):
            if idx % 2 == 0:
                out.append(DedupVerdict(match_kind="exact", existing_id="x"))
            else:
                out.append(None)
        return out

    registry.register(
        DedupChecker(
            name="alt",
            label="",
            description="",
            required_fields=[DedupFieldSpec(name="external_id", type="string")],
            handler=alternating,
        )
    )
    binding = resolve_stash_dedup_binding(
        {"dedup": {"checker": "alt", "fields": {"external_id": "ext"}}},
        node_id="n1",
        registry=registry,
    )
    bound = build_call_context_checker(binding)
    verdicts = await bound(
        [
            {"ext": "a"},
            {"no_ext_here": True},   # invalid_record (index 1)
            {"ext": "b"},
            {"ext": "c"},
        ],
        call_context={},
    )
    assert verdicts[0].match_kind == "exact"
    assert verdicts[1].match_kind == "invalid_record"
    assert verdicts[2] is None
    assert verdicts[3].match_kind == "exact"


@pytest.mark.asyncio
async def test_bound_checker_raises_on_count_mismatch(registry):
    async def bad_handler(records, *, call_context):
        return []  # always returns nothing — programming error

    registry.register(
        DedupChecker(
            name="bad",
            label="",
            description="",
            required_fields=[DedupFieldSpec(name="external_id", type="string")],
            handler=bad_handler,
        )
    )
    binding = resolve_stash_dedup_binding(
        {"dedup": {"checker": "bad", "fields": {"external_id": "ext"}}},
        node_id="n1",
        registry=registry,
    )
    bound = build_call_context_checker(binding)
    with pytest.raises(RuntimeError, match="0 verdicts for 1"):
        await bound([{"ext": "a"}], call_context={})


# ---- format_stash_dedup_addendum ----------------------------------------


def test_addendum_none_when_binding_absent():
    assert format_stash_dedup_addendum(None) is None


def test_addendum_lists_fields_and_explains_duplicates(registry):
    registry.register(
        _make_checker(
            "explainer",
            required_fields=[
                DedupFieldSpec(
                    name="external_id",
                    type="string",
                    description="platform id",
                ),
            ],
            label="My Checker",
        )
    )
    binding = resolve_stash_dedup_binding(
        {
            "dedup": {
                "checker": "explainer",
                "fields": {"external_id": "ext"},
            }
        },
        node_id="n1",
        registry=registry,
    )
    text = format_stash_dedup_addendum(binding)
    assert text is not None
    assert "My Checker" in text
    assert "record.ext → external_id" in text
    assert "duplicates" in text
    assert "iteration quota" in text


# ---- stash_records handler with dedup checker ---------------------------


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def call_context(redis: FakeRedis):
    return {
        "redis": redis,
        "flow_run_id": uuid.UUID("00000000-0000-0000-0000-000000000099"),
        "node_id": "agent_1",
    }


def _always_dup_checker(*, existing_id="job-existing"):
    async def _bound(records, *, call_context):
        return [
            DedupVerdict(
                match_kind="exact",
                existing_id=existing_id,
                reason="seen before",
            )
            for _ in records
        ]

    return _bound


def _never_dup_checker():
    async def _bound(records, *, call_context):
        return [None] * len(records)

    return _bound


@pytest.mark.asyncio
async def test_stash_records_writes_when_checker_returns_none(call_context, redis):
    call_context["stash_dedup_checker"] = _never_dup_checker()
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"ext": "a"}, {"ext": "b"}]},
        call_context=call_context,
    )
    assert result["stashed"] == 2
    assert "duplicates" not in result


@pytest.mark.asyncio
async def test_stash_records_skips_when_checker_flags_duplicate(call_context, redis):
    call_context["stash_dedup_checker"] = _always_dup_checker()
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"ext": "a"}, {"ext": "b"}]},
        call_context=call_context,
    )
    assert result["stashed"] == 0
    assert result["ids"] == []
    assert len(result["duplicates"]) == 2
    assert result["duplicates"][0] == {
        "record_index": 0,
        "match_kind": "exact",
        "reason": "seen before",
        "existing_id": "job-existing",
    }
    # Nothing should have landed in Redis.
    assert redis._strings == {}


@pytest.mark.asyncio
async def test_stash_records_mixed_dedup_keeps_only_new(call_context, redis):
    async def selective(records, *, call_context):
        return [
            None,
            DedupVerdict(match_kind="exact", existing_id="x", reason="dup"),
            None,
        ]

    call_context["stash_dedup_checker"] = selective
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [{"ext": "a"}, {"ext": "b"}, {"ext": "c"}],
            "record_ids": ["a", "b", "c"],
        },
        call_context=call_context,
    )
    assert result["stashed"] == 2
    assert set(result["ids"]) == {"a", "c"}
    assert len(result["duplicates"]) == 1
    assert result["duplicates"][0]["record_index"] == 1
    assert result["duplicates"][0]["existing_id"] == "x"


@pytest.mark.asyncio
async def test_stash_records_dedup_runs_after_schema_validation(call_context):
    # Schema failure must short-circuit before dedup so the agent
    # gets the schema error (more actionable) instead of "duplicate".
    call_context["stash_record_schema"] = [
        {"name": "ext", "type": "string"}
    ]
    call_context["stash_dedup_checker"] = _always_dup_checker()
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"wrong_field": 1}]},
        call_context=call_context,
    )
    assert "error" in result
    assert "details" in result
    assert "duplicates" not in result


@pytest.mark.asyncio
async def test_stash_records_raises_on_checker_count_mismatch(call_context):
    async def bad(records, *, call_context):
        return []  # ignores input

    call_context["stash_dedup_checker"] = bad
    with pytest.raises(RuntimeError, match="verdicts for"):
        await stash_records_module.handler(
            {"collection": "pending", "records": [{"x": 1}]},
            call_context=call_context,
        )


@pytest.mark.asyncio
async def test_stash_records_without_checker_behaves_unchanged(call_context, redis):
    # Back-compat: when no checker is injected the handler stashes
    # everything and never adds a "duplicates" key.
    assert "stash_dedup_checker" not in call_context
    result = await stash_records_module.handler(
        {"collection": "pending", "records": [{"x": 1}]},
        call_context=call_context,
    )
    assert result["stashed"] == 1
    assert "duplicates" not in result


@pytest.mark.asyncio
async def test_stash_records_empty_record_ids_treated_as_omitted(call_context):
    # LLMs following the JSON schema commonly send record_ids=[] when
    # they have no ids for the batch. Without this normalization we'd
    # index into a 0-length list later in the dedup branch and crash.
    call_context["stash_dedup_checker"] = _never_dup_checker()
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [{"x": 1}, {"x": 2}],
            "record_ids": [],
        },
        call_context=call_context,
    )
    assert result["stashed"] == 2
    assert len(result["ids"]) == 2


@pytest.mark.asyncio
async def test_stash_records_empty_record_ids_without_dedup(call_context):
    # Same normalization must apply when no checker is in play —
    # otherwise the ScratchpadStore length check would still trip.
    assert "stash_dedup_checker" not in call_context
    result = await stash_records_module.handler(
        {
            "collection": "pending",
            "records": [{"x": 1}, {"x": 2}],
            "record_ids": [],
        },
        call_context=call_context,
    )
    assert result["stashed"] == 2
