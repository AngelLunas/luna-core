"""Run-scoped opaque record store backed by Redis.

A Scratchpad is a transient key-value store partitioned by ``flow_run_id``
and further namespaced into user-defined ``collection`` buckets. The
store has zero domain knowledge — records are arbitrary dicts; the
caller decides what they mean. Host apps (e.g. luna-sentinel staging
normalized jobs before scoring) compose business semantics on top via
flow configuration: the synthetic ``stash_records`` tool wraps this
service and is the typical writer; downstream nodes iterating with
``iteration.source = "scratchpad"`` are the typical readers.

Key layout (flat namespacing, mirrors the existing ``run_state`` /
``stream`` key conventions in ``luna_core/llm/base.py``):

  scratchpad:{run}:records:{collection}:{id}   -> JSON-encoded record
  scratchpad:{run}:index:{collection}          -> SET of record ids
  scratchpad:{run}:collections                 -> SET of collection names

The per-collection index lets ``list``/``count``/``drop`` work without
a SCAN; the per-run collections set lets the runner clean up everything
this run wrote when it terminates, without enumerating Redis. Both
indices and records share the same TTL — keeping them in lockstep is
intentional so a partially-expired collection can't leave dangling
entries in the index.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any

from redis.asyncio import Redis

from luna_core.core.config import settings

logger = logging.getLogger(__name__)

# Collection names land in Redis key paths and chip labels in the UI —
# constrain them the same way we constrain other identifier-like names
# (carry field names, flow input names). Lowercase + digits + underscore,
# must not start with a digit. Rejecting colons/whitespace is the load-
# bearing piece; the rest is consistency.
_VALID_COLLECTION_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

# Truncated blake2b is plenty — the keyspace is per-run, so the chance of
# two distinct records colliding inside one collection of (realistically)
# hundreds of items is negligible. Hex output keeps keys safe to log.
_HASH_DIGEST_BYTES = 16


class ScratchpadError(ValueError):
    """Raised for caller mistakes (bad collection name, non-dict record).

    Subclasses ``ValueError`` so callers can ``except ValueError`` without
    importing this module if they don't care to distinguish.
    """


class ScratchpadStore:
    """Run-scoped opaque record store.

    The instance is cheap — it holds nothing but a Redis client and a
    TTL. Construct one per request/task; do not share long-lived state.
    All public methods accept ``flow_run_id`` explicitly so a single
    store instance can serve concurrent runs.

    TTL defaults to ``settings.run_stream_key_ttl_seconds`` so scratchpad
    data evaporates on the same schedule as the rest of a run's
    transient state. Pass an explicit ``ttl_seconds`` to override (e.g.,
    in tests or for a host app that wants a shorter window).
    """

    def __init__(self, redis: Redis, *, ttl_seconds: int | None = None):
        self._redis = redis
        self._ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else settings.run_stream_key_ttl_seconds
        )

    # ---- public API --------------------------------------------------------

    async def stash(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
        record: dict[str, Any],
        *,
        record_id: str | None = None,
    ) -> str:
        """Save one record. Returns the id used.

        If ``record_id`` is provided, it is used verbatim — that's the
        idempotency hook: the caller can derive an id from the remote
        source (e.g. ``remote_job.id``) so re-running the producing
        agent doesn't pile up duplicates. If omitted, a deterministic
        hash of the record's canonical JSON is used — same payload
        always produces the same id.

        Existing records with the same id are overwritten silently
        (records are immutable from the agent's POV; mutating means
        explicitly re-stashing).
        """
        return (await self.stash_batch(flow_run_id, collection, [record], record_ids=[record_id]))[0]

    @staticmethod
    def compute_target_ids(
        records: list[dict[str, Any]],
        record_ids: list[str | None] | None = None,
    ) -> list[str]:
        """Compute the ids ``stash_batch`` would assign — without writing.

        Same rules: explicit id wins, otherwise a deterministic
        canonical-JSON hash of the record. Callers (e.g. the
        ``stash_records`` system tool) use this to detect a record
        already present in the collection BEFORE issuing a redundant
        SET — so the agent can be told "this was already stashed,
        don't retry" instead of silently rewriting the same payload.
        """
        if record_ids is not None and len(record_ids) != len(records):
            raise ScratchpadError(
                f"record_ids length ({len(record_ids)}) does not match "
                f"records length ({len(records)})"
            )
        ids: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ScratchpadError(
                    f"record at index {index} is {type(record).__name__}, expected dict"
                )
            explicit = record_ids[index] if record_ids is not None else None
            if explicit is not None and not isinstance(explicit, str):
                raise ScratchpadError(
                    f"record_id at index {index} is {type(explicit).__name__}, expected str"
                )
            ids.append(explicit if explicit else _hash_record(record))
        return ids

    async def stash_batch(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
        records: list[dict[str, Any]],
        *,
        record_ids: list[str | None] | None = None,
    ) -> list[str]:
        """Save many records in one round. Returns the ids in input order.

        ``record_ids``, if supplied, must be the same length as
        ``records``; each entry is either an explicit id or ``None``
        meaning "hash the payload". Mixing is allowed so a caller can
        provide ids for the records that have a stable remote id and
        let the rest auto-hash.
        """
        _validate_collection_name(collection)
        if record_ids is not None and len(record_ids) != len(records):
            raise ScratchpadError(
                f"record_ids length ({len(record_ids)}) does not match "
                f"records length ({len(records)})"
            )
        if not records:
            return []

        run_id = _run_id_str(flow_run_id)
        ids: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ScratchpadError(
                    f"record at index {index} is {type(record).__name__}, expected dict"
                )
            explicit = record_ids[index] if record_ids is not None else None
            if explicit is not None and not isinstance(explicit, str):
                raise ScratchpadError(
                    f"record_id at index {index} is {type(explicit).__name__}, expected str"
                )
            ids.append(explicit if explicit else _hash_record(record))

        # Write records, then update indices. We do not use a pipeline:
        # the scratchpad is run-scoped, contention is bounded by how
        # many tools the same agent runs in parallel, and partial
        # failures are recoverable (re-stash is idempotent). If we ever
        # see torn writes show up in practice, swap this loop for a
        # transaction.
        for record_id, record in zip(ids, records):
            await self._redis.set(
                _record_key(run_id, collection, record_id),
                json.dumps(record, default=str),
                ex=self._ttl_seconds,
            )
        await self._redis.sadd(_index_key(run_id, collection), *ids)
        await self._redis.expire(_index_key(run_id, collection), self._ttl_seconds)
        await self._redis.sadd(_collections_key(run_id), collection)
        await self._redis.expire(_collections_key(run_id), self._ttl_seconds)
        return ids

    async def get(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
        record_id: str,
    ) -> dict[str, Any] | None:
        """Return the record or ``None`` if it isn't present.

        Missing records are not an error — callers iterating concurrently
        may legitimately race with a ``drop`` from a peer.
        """
        _validate_collection_name(collection)
        run_id = _run_id_str(flow_run_id)
        raw = await self._redis.get(_record_key(run_id, collection, record_id))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "scratchpad record %s/%s/%s is not valid JSON; treating as missing",
                run_id,
                collection,
                record_id,
            )
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "scratchpad record %s/%s/%s is %s, expected dict; treating as missing",
                run_id,
                collection,
                record_id,
                type(payload).__name__,
            )
            return None
        return payload

    async def list_ids(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
    ) -> list[str]:
        """Return every record id currently in the collection.

        Order is not guaranteed (Redis SETs are unordered). Callers that
        need stable ordering should sort or carry an ordering hint
        inside the records themselves.
        """
        _validate_collection_name(collection)
        run_id = _run_id_str(flow_run_id)
        raw = await self._redis.smembers(_index_key(run_id, collection))
        return [m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in raw]

    async def list_records(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(id, record)`` pairs for every entry in the collection.

        Entries whose payload has gone missing (e.g. expired earlier than
        their index entry because of clock skew) are filtered out.
        """
        ids = await self.list_ids(flow_run_id, collection)
        results: list[tuple[str, dict[str, Any]]] = []
        for record_id in ids:
            payload = await self.get(flow_run_id, collection, record_id)
            if payload is not None:
                results.append((record_id, payload))
        return results

    async def drop(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
        record_id: str,
    ) -> bool:
        """Remove one record. Returns ``True`` if it existed."""
        _validate_collection_name(collection)
        run_id = _run_id_str(flow_run_id)
        removed = await self._redis.delete(
            _record_key(run_id, collection, record_id)
        )
        await self._redis.srem(_index_key(run_id, collection), record_id)
        return bool(removed)

    async def count(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
    ) -> int:
        """How many records currently live in the collection."""
        _validate_collection_name(collection)
        run_id = _run_id_str(flow_run_id)
        return int(await self._redis.scard(_index_key(run_id, collection)))

    async def clear_collection(
        self,
        flow_run_id: uuid.UUID | str,
        collection: str,
    ) -> int:
        """Drop every record in a collection. Returns the number removed.

        Useful when a flow wants to reset a staging area between phases.
        """
        _validate_collection_name(collection)
        run_id = _run_id_str(flow_run_id)
        ids = await self.list_ids(flow_run_id, collection)
        if not ids:
            return 0
        keys = [_record_key(run_id, collection, rid) for rid in ids]
        await self._redis.delete(*keys)
        await self._redis.delete(_index_key(run_id, collection))
        await self._redis.srem(_collections_key(run_id), collection)
        return len(ids)

    async def clear_run(self, flow_run_id: uuid.UUID | str) -> int:
        """Drop every collection this run has touched. Returns total records
        removed.

        Designed for the runner's end-of-run cleanup path so a single
        call wipes whatever the flow stashed regardless of how many
        collections it used.
        """
        run_id = _run_id_str(flow_run_id)
        raw = await self._redis.smembers(_collections_key(run_id))
        collections = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in raw
        ]
        total = 0
        for collection in collections:
            total += await self.clear_collection(flow_run_id, collection)
        await self._redis.delete(_collections_key(run_id))
        return total


# ---- key builders ---------------------------------------------------------

def _record_key(run_id: str, collection: str, record_id: str) -> str:
    return f"scratchpad:{run_id}:records:{collection}:{record_id}"


def _index_key(run_id: str, collection: str) -> str:
    return f"scratchpad:{run_id}:index:{collection}"


def _collections_key(run_id: str) -> str:
    return f"scratchpad:{run_id}:collections"


# ---- helpers --------------------------------------------------------------

def _run_id_str(flow_run_id: uuid.UUID | str) -> str:
    return str(flow_run_id)


def _validate_collection_name(name: str) -> None:
    if not isinstance(name, str) or not _VALID_COLLECTION_NAME.match(name):
        raise ScratchpadError(
            f"invalid collection name {name!r}: must match {_VALID_COLLECTION_NAME.pattern}"
        )


def _hash_record(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=_HASH_DIGEST_BYTES
    ).hexdigest()


__all__ = ["ScratchpadError", "ScratchpadStore"]
