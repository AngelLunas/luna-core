"""Tests for ScratchpadStore.

Uses an in-memory FakeRedis (the same pattern as test_inflight_snapshot)
to keep the suite hermetic. We only stub the redis-py methods the store
actually calls; if a new test needs more surface area, extend the fake
rather than reaching for a heavier dependency.
"""
from __future__ import annotations

import json
import uuid

import pytest

from luna_core.services.scratchpad import ScratchpadError, ScratchpadStore


class FakeRedis:
    """Minimal async Redis double covering the methods ScratchpadStore uses.

    Mirrors redis-py's default of returning bytes (decode_responses=False)
    so the store exercises its bytes-decode branches the same way it
    would in production.
    """

    def __init__(self) -> None:
        self._strings: dict[str, bytes] = {}
        self._sets: dict[str, set[bytes]] = {}
        # Per-key TTL last applied — tests use this to assert the store
        # actually sets expirations on everything it writes.
        self.ttls: dict[str, int] = {}

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self._strings[key] = value.encode("utf-8")
        if ex is not None:
            self.ttls[key] = ex

    async def get(self, key: str) -> bytes | None:
        return self._strings.get(key)

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self._strings:
                del self._strings[key]
                removed += 1
            if key in self._sets:
                del self._sets[key]
                removed += 1
        return removed

    async def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds

    async def sadd(self, key: str, *members: str) -> int:
        bucket = self._sets.setdefault(key, set())
        added = 0
        for m in members:
            b = m.encode("utf-8") if isinstance(m, str) else m
            if b not in bucket:
                bucket.add(b)
                added += 1
        return added

    async def srem(self, key: str, *members: str) -> int:
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

    async def smembers(self, key: str) -> set[bytes]:
        return set(self._sets.get(key, set()))

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def store(redis: FakeRedis) -> ScratchpadStore:
    # Pin a non-default TTL so we can assert it's actually applied to
    # every key the store touches.
    return ScratchpadStore(redis, ttl_seconds=42)


@pytest.fixture
def run_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


# ---- basic write/read --------------------------------------------------


@pytest.mark.asyncio
async def test_stash_then_get_roundtrip(store, run_id):
    rid = await store.stash(run_id, "pending", {"title": "X", "url": "u"})
    assert rid  # non-empty
    fetched = await store.get(run_id, "pending", rid)
    assert fetched == {"title": "X", "url": "u"}


@pytest.mark.asyncio
async def test_stash_with_explicit_record_id_is_idempotent(store, run_id):
    rid_a = await store.stash(run_id, "pending", {"v": 1}, record_id="job-7")
    rid_b = await store.stash(run_id, "pending", {"v": 2}, record_id="job-7")
    assert rid_a == "job-7" == rid_b
    # Last write wins; only one entry in the collection.
    assert await store.count(run_id, "pending") == 1
    assert (await store.get(run_id, "pending", "job-7")) == {"v": 2}


@pytest.mark.asyncio
async def test_stash_without_id_hashes_payload_deterministically(store, run_id):
    rid_a = await store.stash(run_id, "pending", {"a": 1, "b": 2})
    rid_b = await store.stash(run_id, "pending", {"b": 2, "a": 1})  # key order
    assert rid_a == rid_b, "canonical hash must be order-insensitive"
    assert await store.count(run_id, "pending") == 1


@pytest.mark.asyncio
async def test_distinct_payloads_get_distinct_hashes(store, run_id):
    rid_a = await store.stash(run_id, "pending", {"x": 1})
    rid_b = await store.stash(run_id, "pending", {"x": 2})
    assert rid_a != rid_b
    assert await store.count(run_id, "pending") == 2


# ---- batch -------------------------------------------------------------


@pytest.mark.asyncio
async def test_stash_batch_returns_ids_in_input_order(store, run_id):
    ids = await store.stash_batch(
        run_id,
        "pending",
        [{"i": 1}, {"i": 2}, {"i": 3}],
    )
    assert len(ids) == 3
    assert (await store.get(run_id, "pending", ids[0])) == {"i": 1}
    assert (await store.get(run_id, "pending", ids[2])) == {"i": 3}


@pytest.mark.asyncio
async def test_stash_batch_with_mixed_explicit_and_hashed_ids(store, run_id):
    ids = await store.stash_batch(
        run_id,
        "pending",
        [{"i": 1}, {"i": 2}, {"i": 3}],
        record_ids=["one", None, "three"],
    )
    assert ids[0] == "one"
    assert ids[2] == "three"
    assert ids[1] != "one" and ids[1] != "three"
    assert await store.count(run_id, "pending") == 3


@pytest.mark.asyncio
async def test_stash_batch_empty_is_noop(store, run_id):
    ids = await store.stash_batch(run_id, "pending", [])
    assert ids == []
    assert await store.count(run_id, "pending") == 0


@pytest.mark.asyncio
async def test_stash_batch_rejects_mismatched_record_ids_length(store, run_id):
    with pytest.raises(ScratchpadError, match="record_ids length"):
        await store.stash_batch(
            run_id,
            "pending",
            [{"i": 1}, {"i": 2}],
            record_ids=["only-one"],
        )


@pytest.mark.asyncio
async def test_stash_batch_rejects_non_dict_record(store, run_id):
    with pytest.raises(ScratchpadError, match="expected dict"):
        await store.stash_batch(run_id, "pending", [{"ok": True}, ["not", "a", "dict"]])  # type: ignore[list-item]


# ---- get / drop / count edges -----------------------------------------


@pytest.mark.asyncio
async def test_get_missing_returns_none(store, run_id):
    assert await store.get(run_id, "pending", "nope") is None


@pytest.mark.asyncio
async def test_count_empty_collection_is_zero(store, run_id):
    assert await store.count(run_id, "pending") == 0


@pytest.mark.asyncio
async def test_list_ids_empty_collection_returns_empty(store, run_id):
    assert await store.list_ids(run_id, "pending") == []


@pytest.mark.asyncio
async def test_drop_removes_and_decrements_count(store, run_id):
    rid = await store.stash(run_id, "pending", {"x": 1})
    assert await store.count(run_id, "pending") == 1
    assert await store.drop(run_id, "pending", rid) is True
    assert await store.count(run_id, "pending") == 0
    assert await store.get(run_id, "pending", rid) is None


@pytest.mark.asyncio
async def test_drop_missing_returns_false(store, run_id):
    assert await store.drop(run_id, "pending", "ghost") is False


@pytest.mark.asyncio
async def test_list_records_returns_id_record_pairs(store, run_id):
    rid_a = await store.stash(run_id, "pending", {"i": 1})
    rid_b = await store.stash(run_id, "pending", {"i": 2})
    pairs = dict(await store.list_records(run_id, "pending"))
    assert set(pairs.keys()) == {rid_a, rid_b}
    assert pairs[rid_a] == {"i": 1}
    assert pairs[rid_b] == {"i": 2}


@pytest.mark.asyncio
async def test_list_records_skips_missing_payload(store, redis, run_id):
    # Simulate the index being slightly out of sync with records (the
    # store guards against this in production via TTL alignment, but the
    # list_records contract still has to filter out ghosts gracefully).
    rid = await store.stash(run_id, "pending", {"i": 1})
    await redis.delete(f"scratchpad:{run_id}:records:pending:{rid}")
    pairs = await store.list_records(run_id, "pending")
    assert pairs == []


# ---- isolation ---------------------------------------------------------


@pytest.mark.asyncio
async def test_different_runs_do_not_collide(store):
    run_a = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
    run_b = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
    await store.stash(run_a, "pending", {"r": "a"}, record_id="shared")
    await store.stash(run_b, "pending", {"r": "b"}, record_id="shared")
    assert (await store.get(run_a, "pending", "shared")) == {"r": "a"}
    assert (await store.get(run_b, "pending", "shared")) == {"r": "b"}
    assert await store.count(run_a, "pending") == 1
    assert await store.count(run_b, "pending") == 1


@pytest.mark.asyncio
async def test_different_collections_in_same_run_do_not_collide(store, run_id):
    await store.stash(run_id, "alpha", {"x": 1}, record_id="shared")
    await store.stash(run_id, "beta", {"x": 2}, record_id="shared")
    assert (await store.get(run_id, "alpha", "shared")) == {"x": 1}
    assert (await store.get(run_id, "beta", "shared")) == {"x": 2}


# ---- collection name validation ---------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "Has-Hyphen", "1starts_with_digit", "has space", "has:colon", "MIXEDcase"],
)
@pytest.mark.asyncio
async def test_invalid_collection_name_rejected(store, run_id, bad):
    with pytest.raises(ScratchpadError, match="invalid collection name"):
        await store.stash(run_id, bad, {"x": 1})


@pytest.mark.parametrize(
    "ok",
    ["pending", "pending_review", "p", "_underscore_start", "x123"],
)
@pytest.mark.asyncio
async def test_valid_collection_names_accepted(store, run_id, ok):
    rid = await store.stash(run_id, ok, {"x": 1})
    assert (await store.get(run_id, ok, rid)) == {"x": 1}


# ---- TTL is applied to every key the store touches --------------------


@pytest.mark.asyncio
async def test_ttl_set_on_record_index_and_collections_keys(store, redis, run_id):
    rid = await store.stash(run_id, "pending", {"x": 1}, record_id="job-1")
    rk = f"scratchpad:{run_id}:records:pending:{rid}"
    ik = f"scratchpad:{run_id}:index:pending"
    ck = f"scratchpad:{run_id}:collections"
    assert redis.ttls[rk] == 42
    assert redis.ttls[ik] == 42
    assert redis.ttls[ck] == 42


# ---- clear -------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_collection_removes_all_and_returns_count(store, run_id):
    await store.stash_batch(run_id, "pending", [{"i": 1}, {"i": 2}, {"i": 3}])
    assert await store.clear_collection(run_id, "pending") == 3
    assert await store.count(run_id, "pending") == 0
    assert await store.list_ids(run_id, "pending") == []


@pytest.mark.asyncio
async def test_clear_collection_empty_returns_zero(store, run_id):
    assert await store.clear_collection(run_id, "pending") == 0


@pytest.mark.asyncio
async def test_clear_run_wipes_every_collection_the_run_touched(store, run_id):
    await store.stash_batch(run_id, "alpha", [{"i": 1}, {"i": 2}])
    await store.stash_batch(run_id, "beta", [{"i": 3}])
    removed = await store.clear_run(run_id)
    assert removed == 3
    assert await store.count(run_id, "alpha") == 0
    assert await store.count(run_id, "beta") == 0


# ---- bytes from Redis are decoded correctly ---------------------------


@pytest.mark.asyncio
async def test_list_ids_decodes_bytes_into_strings(store, run_id):
    await store.stash(run_id, "pending", {"x": 1}, record_id="job-7")
    ids = await store.list_ids(run_id, "pending")
    assert all(isinstance(i, str) for i in ids)
    assert ids == ["job-7"]


@pytest.mark.asyncio
async def test_get_tolerates_corrupted_payload(store, redis, run_id):
    rid = await store.stash(run_id, "pending", {"x": 1})
    rk = f"scratchpad:{run_id}:records:pending:{rid}"
    # Overwrite the record with non-JSON garbage; production has never
    # produced this, but log-and-return-None is the contract for the
    # unusual case where it would.
    redis._strings[rk] = b"not json{"
    assert await store.get(run_id, "pending", rid) is None


@pytest.mark.asyncio
async def test_get_tolerates_non_dict_payload(store, redis, run_id):
    rid = await store.stash(run_id, "pending", {"x": 1})
    rk = f"scratchpad:{run_id}:records:pending:{rid}"
    redis._strings[rk] = json.dumps(["array", "not", "dict"]).encode("utf-8")
    assert await store.get(run_id, "pending", rid) is None
