from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.engine.emitter import publish_flow_run_event
from luna_core.llm.base import delta_event_id, inflight_meta_key, stream_key
from luna_core.models.event import (
    AgentMessage,
    AgentMessageRole,
    RunEvent,
    RunEventType,
)
from luna_core.models.flow import Flow, FlowRun, FlowRunStatus
from luna_core.schemas.flow import (
    FlowCreate,
    FlowDefinition,
    FlowInputDef,
    FlowRunRead,
    FlowRunTrigger,
    FlowUpdate,
)


class FlowNotFound(LookupError):
    pass


class FlowRunNotFound(LookupError):
    pass


class DuplicateFlow(ValueError):
    pass


class FlowInputValidationError(ValueError):
    """Raised when a trigger payload doesn't satisfy the flow's input schema.

    ``errors`` is a {field_name: message} map suitable for surfacing
    directly to the caller (e.g. a 400 response body).
    """

    def __init__(self, errors: dict[str, str]):
        super().__init__(f"flow input validation failed: {errors}")
        self.errors = errors


def _coerce_input_value(value: Any, target_type: str) -> Any:
    """Coerce a JSON-decoded value into the declared input type.

    Accepts what JSON naturally produces (numbers, bools, strings, dicts).
    Booleans are NOT accepted as integers/numbers because in JSON they're
    a distinct type — silently coercing them tends to hide caller bugs.
    """
    if value is None:
        return None
    match target_type:
        case "string":
            if not isinstance(value, str):
                raise ValueError(f"expected string, got {type(value).__name__}")
            return value
        case "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"expected integer, got {type(value).__name__}")
            return value
        case "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expected number, got {type(value).__name__}")
            return value
        case "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"expected boolean, got {type(value).__name__}")
            return value
        case "object":
            if not isinstance(value, dict):
                raise ValueError(f"expected object, got {type(value).__name__}")
            return value
        case "array":
            if not isinstance(value, list):
                raise ValueError(f"expected array, got {type(value).__name__}")
            return value
    raise ValueError(f"unsupported input type: {target_type}")


def validate_flow_inputs(
    inputs: list[FlowInputDef], payload: dict[str, Any] | None
) -> dict[str, Any]:
    """Validate a caller-supplied inputs dict against the flow's input schema.

    Returns the validated dict (with defaults applied) on success; raises
    ``FlowInputValidationError`` with field-level messages on failure.
    """
    errors: dict[str, str] = {}
    resolved: dict[str, Any] = {}
    payload = payload or {}
    by_name = {spec.name: spec for spec in inputs}

    for key in payload:
        if key not in by_name:
            errors[key] = "unknown input"

    for spec in inputs:
        if spec.name in payload:
            try:
                resolved[spec.name] = _coerce_input_value(
                    payload[spec.name], spec.type
                )
            except ValueError as exc:
                errors[spec.name] = str(exc)
            continue
        if spec.required:
            errors[spec.name] = "required input is missing"
        elif spec.default is not None:
            resolved[spec.name] = spec.default

    if errors:
        raise FlowInputValidationError(errors)
    return resolved


def _stamp_schedule_owner(
    definition: dict[str, Any], user_id: uuid.UUID
) -> dict[str, Any]:
    """Force ``user_id`` on every schedule rule in ``definition`` (in-place).

    Rules belong to the user who saved them; the client never picks the owner
    — the server stamps it. Returns the mutated dict for fluent use.
    """
    trigger = definition.get("trigger") or {}
    rules = trigger.get("schedules") or []
    if not rules:
        return definition
    user_id_str = str(user_id)
    for rule in rules:
        if isinstance(rule, dict):
            rule["user_id"] = user_id_str
    return definition


def _definition_for_user(
    definition: dict[str, Any], user_id: uuid.UUID
) -> dict[str, Any]:
    """Return a copy of ``definition`` with ``trigger.schedules`` filtered to
    only the rules owned by ``user_id``. Rules without a ``user_id`` (legacy)
    are dropped from the read view as well — they wouldn't fire anyway, and
    surfacing them would mislead the editor into thinking they're its own.
    """
    out = dict(definition)
    trigger = out.get("trigger")
    if not isinstance(trigger, dict):
        return out
    out["trigger"] = dict(trigger)
    rules = out["trigger"].get("schedules") or []
    user_id_str = str(user_id)
    out["trigger"]["schedules"] = [
        r for r in rules if isinstance(r, dict) and r.get("user_id") == user_id_str
    ]
    return out


def flow_for_user(flow: Flow, user_id: uuid.UUID) -> Flow:
    """Return ``flow`` with its definition narrowed to ``user_id``'s schedules.

    The returned object is detached-ish — we replace ``definition`` with a
    filtered dict in memory so ``FlowRead.model_validate(flow)`` serialises
    only what the user is allowed to see. The DB row is unaffected.
    """
    flow.definition = _definition_for_user(flow.definition or {}, user_id)
    return flow


def _merge_user_schedules(
    existing_definition: dict[str, Any],
    incoming_definition: dict[str, Any],
    user_id: uuid.UUID,
) -> dict[str, Any]:
    """Merge ``incoming_definition`` into ``existing_definition``'s schedules:
    other users' rules are preserved verbatim, the calling user's rules are
    replaced wholesale by whatever ``incoming_definition`` brings (already
    stamped with this user's id).
    """
    merged = dict(incoming_definition)
    existing_trigger = (existing_definition or {}).get("trigger") or {}
    existing_rules = existing_trigger.get("schedules") or []
    user_id_str = str(user_id)
    foreign = [
        r for r in existing_rules
        if isinstance(r, dict) and r.get("user_id") and r.get("user_id") != user_id_str
    ]
    incoming_trigger = merged.get("trigger") or {}
    mine = incoming_trigger.get("schedules") or []
    merged_trigger = dict(incoming_trigger)
    merged_trigger["schedules"] = [*mine, *foreign]
    merged["trigger"] = merged_trigger
    return merged


async def create_flow(
    db: AsyncSession, payload: FlowCreate, *, user_id: uuid.UUID
) -> Flow:
    definition = payload.definition.model_dump(by_alias=True)
    _stamp_schedule_owner(definition, user_id)
    flow = Flow(
        name=payload.name,
        description=payload.description,
        definition=definition,
        is_active=payload.is_active,
    )
    db.add(flow)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateFlow(payload.name) from exc
    await db.refresh(flow)
    return flow


async def list_flows(db: AsyncSession) -> list[Flow]:
    result = await db.execute(select(Flow).order_by(Flow.created_at.desc()))
    return list(result.scalars().all())


async def get_flow(db: AsyncSession, flow_id: uuid.UUID) -> Flow:
    flow = await db.get(Flow, flow_id)
    if flow is None:
        raise FlowNotFound(str(flow_id))
    return flow


async def update_flow(
    db: AsyncSession,
    flow_id: uuid.UUID,
    payload: FlowUpdate,
    *,
    user_id: uuid.UUID,
) -> Flow:
    flow = await get_flow(db, flow_id)
    data = payload.model_dump(exclude_unset=True)
    if "definition" in data and data["definition"] is not None:
        # re-validate as FlowDefinition then dump (preserves `from` alias)
        defn = FlowDefinition.model_validate(data["definition"])
        incoming = defn.model_dump(by_alias=True)
        # Stamp ownership on the caller's rules then merge with the persisted
        # definition's other-user rules so a write from one user never
        # destroys another user's schedules.
        _stamp_schedule_owner(incoming, user_id)
        flow.definition = _merge_user_schedules(
            flow.definition or {}, incoming, user_id
        )
        data.pop("definition")
    for field, value in data.items():
        setattr(flow, field, value)
    await db.commit()
    await db.refresh(flow)
    return flow


async def delete_flow(db: AsyncSession, flow_id: uuid.UUID) -> None:
    """Hard-delete a flow and every dependent FlowRun/event/message via the
    cascade on ``Flow.runs``. We don't soft-delete — the editor needs the name
    slot freed (``Flow.name`` is unique) and there's no audit case here that
    justifies keeping the row around.
    """
    flow = await get_flow(db, flow_id)
    await db.delete(flow)
    await db.commit()


class FlowDefinitionInvalid(ValueError):
    """Raised by ``validate_definition`` when the graph itself is malformed.

    ``errors`` is a list of human-readable strings — the editor groups them in
    a single panel without trying to parse paths back into form fields.
    """

    def __init__(self, errors: list[str]):
        super().__init__(f"flow definition invalid: {errors}")
        self.errors = errors


def validate_definition(definition: FlowDefinition) -> None:
    """Cross-field sanity checks Pydantic doesn't cover.

    Pydantic already enforces per-field types, ``FlowInputDef`` uniqueness,
    and (with the model validator) that ``layout`` keys match node ids. Here
    we additionally enforce graph-level invariants: entry point exists,
    edges reference real nodes, node ids are unique, and required config
    fields per node type are present. Anything that would surface as a
    NodeExecutionError at runtime is caught upfront so the editor can show
    it before save.
    """
    errors: list[str] = []
    node_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for node in definition.nodes:
        if node.id in node_ids:
            duplicate_ids.append(node.id)
        node_ids.add(node.id)
    if duplicate_ids:
        errors.append(f"duplicate node ids: {sorted(set(duplicate_ids))!r}")

    if definition.entry_point not in node_ids:
        errors.append(
            f"entry_point {definition.entry_point!r} does not match any node id"
        )

    for i, edge in enumerate(definition.edges):
        if edge.from_ not in node_ids:
            errors.append(f"edge[{i}].from {edge.from_!r} references unknown node")
        if edge.to not in node_ids:
            errors.append(f"edge[{i}].to {edge.to!r} references unknown node")

    for node in definition.nodes:
        match node.type:
            case "action":
                if not node.config.get("operation_id"):
                    errors.append(
                        f"node {node.id!r} (action) missing config.operation_id"
                    )
            case "ai_agent":
                if not node.config.get("agent_id"):
                    errors.append(
                        f"node {node.id!r} (ai_agent) missing config.agent_id"
                    )
                inherit = node.config.get("inherit_history_from")
                if inherit is not None:
                    if not isinstance(inherit, list) or not all(
                        isinstance(x, str) for x in inherit
                    ):
                        errors.append(
                            f"node {node.id!r}: inherit_history_from must be a list of node ids"
                        )
                    else:
                        for ref in inherit:
                            if ref == node.id:
                                errors.append(
                                    f"node {node.id!r}: cannot inherit from itself"
                                )
                            elif ref not in node_ids:
                                errors.append(
                                    f"node {node.id!r}: inherit_history_from references unknown node {ref!r}"
                                )
            case _:
                # condition / human_checkpoint / trigger / output have no hard
                # config requirements at the schema layer.
                pass

    if errors:
        raise FlowDefinitionInvalid(errors)


async def create_flow_run(
    db: AsyncSession,
    flow_id: uuid.UUID,
    trigger: FlowRunTrigger | dict[str, Any] | None = None,
    *,
    redis: Redis | None = None,
) -> FlowRun:
    flow = await get_flow(db, flow_id)
    if not flow.is_active:
        raise ValueError(f"flow {flow_id} is inactive")
    trigger_payload: dict[str, Any]
    if trigger is None:
        trigger_payload = {}
    elif isinstance(trigger, FlowRunTrigger):
        trigger_payload = trigger.model_dump()
    else:
        trigger_payload = trigger

    # Validate trigger.inputs against the flow's declared input schema and
    # persist the coerced values back into the trigger payload so the runner
    # picks them up via state.inputs.
    definition = FlowDefinition.model_validate(flow.definition)
    trigger_payload["inputs"] = validate_flow_inputs(
        definition.inputs, trigger_payload.get("inputs") or {}
    )

    run = FlowRun(
        flow_id=flow_id,
        status=FlowRunStatus.pending,
        trigger=trigger_payload,
        state={},
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    if redis is not None:
        await publish_flow_run_event(
            redis,
            flow_id,
            "run_created",
            FlowRunRead.model_validate(run).model_dump(mode="json"),
        )
    return run


async def get_flow_run(db: AsyncSession, run_id: uuid.UUID) -> FlowRun:
    run = await db.get(FlowRun, run_id)
    if run is None:
        raise FlowRunNotFound(str(run_id))
    return run


async def list_flow_runs(
    db: AsyncSession,
    flow_id: uuid.UUID,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[FlowRun]:
    """Return runs for a flow ordered newest-first. Callers paginate with
    ``limit``/``offset`` — the API exposes this as the load-more cursor.
    """
    await get_flow(db, flow_id)
    stmt = (
        select(FlowRun)
        .where(FlowRun.flow_id == flow_id)
        .order_by(FlowRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def set_run_status(
    db: AsyncSession,
    run_id: uuid.UUID,
    status: FlowRunStatus,
    state: dict[str, Any] | None = None,
    *,
    redis: Redis | None = None,
) -> FlowRun:
    run = await get_flow_run(db, run_id)
    run.status = status
    if state is not None:
        run.state = state
    now = datetime.now(timezone.utc)
    if status == FlowRunStatus.running and run.started_at is None:
        run.started_at = now
    if status in (FlowRunStatus.completed, FlowRunStatus.failed):
        run.completed_at = now
    await db.commit()
    await db.refresh(run)
    if redis is not None:
        await publish_flow_run_event(
            redis,
            run.flow_id,
            "run_status_changed",
            FlowRunRead.model_validate(run).model_dump(mode="json"),
        )
    return run


async def list_run_events(
    db: AsyncSession,
    run_id: uuid.UUID,
    since_sequence: int | None = None,
    redis: Redis | None = None,
    iteration_id: str | None = None,
) -> list[RunEvent]:
    """Return the full historical event stream for a run.

    The wire shape is the same one clients see live over Redis pub/sub —
    every persisted event plus *synthesized* agent_text_delta /
    agent_thinking_delta rows rebuilt from the assistant AgentMessages.
    During streaming the provider broadcasts deltas chunk-by-chunk but
    never writes them to ``run_events`` (the canonical record is the
    single AgentMessage row); here we hand back one delta per message
    carrying the full text so the frontend's groupEvents reducer sees
    the same shape it saw live and renders the conversation identically.

    When ``redis`` is supplied, the synthesis also covers turns that are
    still mid-stream: for any ``agent_message_started`` without a matching
    AgentMessage we rehydrate a synthetic delta from the per-node chunk
    cache (``stream_key``) so a client opening the page mid-run sees the
    accumulated prefix instead of an empty bubble.
    """
    await get_flow_run(db, run_id)
    persisted_stmt = (
        select(RunEvent)
        .where(RunEvent.flow_run_id == run_id)
        .order_by(RunEvent.sequence.asc())
    )
    persisted = list((await db.execute(persisted_stmt)).scalars().all())

    synthesized = await _synthesize_message_deltas(
        db, run_id, persisted, redis=redis
    )
    if synthesized:
        combined = persisted + synthesized
        combined.sort(key=lambda e: e.sequence)
    else:
        combined = persisted

    if since_sequence is not None:
        combined = [e for e in combined if e.sequence > since_sequence]
    if iteration_id is not None:
        # Only return events tagged for this iteration. Used by the
        # dashboard when a user expands one iteration accordion: it
        # backfills the historical sub-events (agent messages, tool
        # calls) belonging to that iteration without dragging the full
        # node history along. The lifecycle envelope events themselves
        # (iteration_started / _completed / _failed for this id) are
        # included so the panel can render the header from one round
        # trip; other iterations' events are filtered out by id.
        combined = [
            e
            for e in combined
            if (e.payload or {}).get("iteration_id") == iteration_id
        ]
    return combined


async def _synthesize_message_deltas(
    db: AsyncSession,
    run_id: uuid.UUID,
    persisted: list[RunEvent],
    redis: Redis | None = None,
) -> list[RunEvent]:
    """Rebuild transient agent_text_delta / agent_thinking_delta rows from
    the persisted assistant AgentMessages (and, with ``redis``, from the
    per-node chunk cache for turns that are still mid-stream).

    The returned RunEvent instances are *not* added to the session — they
    only exist long enough to be serialized by the API layer. Sequence is
    ``started.sequence + 1`` so each synthetic delta sorts immediately
    after the agent_message_started it belongs to; collisions with
    ``agent_message_completed`` (which may also land at that sequence
    when no real deltas were persisted) are harmless because the frontend
    dedupes by ``id`` and stable-sorts by ``sequence``. The id is a
    deterministic uuid5 of the message id so repeated reads stay
    idempotent on the client side — and so the in-flight synthesis below
    converges to the same id as the completed-message synthesis once the
    turn lands in the DB.
    """
    started_by_msg: dict[str, RunEvent] = {}
    for ev in persisted:
        if ev.event_type != RunEventType.agent_message_started:
            continue
        msg_id = (ev.payload or {}).get("message_id")
        if isinstance(msg_id, str):
            started_by_msg[msg_id] = ev

    if not started_by_msg:
        return []

    msg_stmt = (
        select(AgentMessage)
        .where(AgentMessage.flow_run_id == run_id)
        .where(AgentMessage.role == AgentMessageRole.assistant)
    )
    messages = list((await db.execute(msg_stmt)).scalars().all())
    completed_msg_ids = {str(m.id) for m in messages}

    synthesized: list[RunEvent] = []
    for msg in messages:
        started = started_by_msg.get(str(msg.id))
        if started is None:
            # No matching started — either a legacy run from before the
            # delta-persistence optimization (in which case the real
            # deltas are already in ``persisted``) or an orphan message
            # from a restarted stream. Either way, leave it alone.
            continue
        synth_seq = started.sequence + 1
        # Propagate the iteration tag from the started event so the
        # synthesized text/thinking blocks route to the same iteration
        # block in the UI as their lifecycle envelope. Without this,
        # ``GET /events?iteration_id=`` drops the synth (no tag → no
        # match) and the historical message bubble appears empty in the
        # iteration accordion while the live deltas (which DO carry the
        # tag) trickle in separately.
        iteration_id_tag = _iteration_id_from_event(started)

        text_parts = [
            block.get("text", "")
            for block in (msg.content or [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        full_text = "".join(text_parts)
        # ``complete: true`` is the signal to the client reducer that
        # this delta is the canonical post-completion text for the
        # message — not a streaming chunk. The reducer replaces the
        # accumulated text with this value and ignores any subsequent
        # per-chunk deltas it might see for the same message_id.
        #
        # Without this flag the synth would APPEND its full text on top
        # of whatever the live deltas already accumulated, and because
        # the synth's sequence equals ``started.sequence + 1`` (same as
        # live delta 0), JS's stable sort can interleave it anywhere
        # between live deltas of the same message. In parallel mode
        # that produces the "chaos" the dashboard renders: chunks +
        # full text mixed in unpredictable order.
        if full_text:
            synthesized.append(
                _build_synthetic_event(
                    run_id=run_id,
                    node_id=msg.node_id,
                    sequence=synth_seq,
                    timestamp=started.timestamp,
                    event_type=RunEventType.agent_text_delta,
                    stable_key=f"text-delta:{msg.id}",
                    payload=_with_iteration_id(
                        {
                            "message_id": str(msg.id),
                            "chunk_index": 0,
                            "text": full_text,
                            "complete": True,
                        },
                        iteration_id_tag,
                    ),
                )
            )
        if msg.thinking:
            synthesized.append(
                _build_synthetic_event(
                    run_id=run_id,
                    node_id=msg.node_id,
                    sequence=synth_seq,
                    timestamp=started.timestamp,
                    event_type=RunEventType.agent_thinking_delta,
                    stable_key=f"thinking-delta:{msg.id}",
                    payload=_with_iteration_id(
                        {
                            "message_id": str(msg.id),
                            "chunk_index": 0,
                            "text": msg.thinking,
                            "complete": True,
                        },
                        iteration_id_tag,
                    ),
                )
            )

    if redis is not None:
        for msg_id, started in started_by_msg.items():
            if msg_id in completed_msg_ids:
                continue
            node_id = started.node_id
            if not node_id:
                continue
            iteration_id_tag = _iteration_id_from_event(started)
            # Per-message read — never sees a sibling iteration's chunks
            # even when both turns share an ai_agent node id.
            chunks = await _read_inflight_chunks(redis, run_id, msg_id)
            for chunk in chunks:
                kind = chunk["kind"]
                event_type = (
                    RunEventType.agent_text_delta
                    if kind == "text"
                    else RunEventType.agent_thinking_delta
                )
                synthesized.append(
                    _build_inflight_chunk_event(
                        run_id=run_id,
                        node_id=node_id,
                        sequence=started.sequence + chunk["sequence_offset"],
                        timestamp=started.timestamp,
                        event_type=event_type,
                        message_id=msg_id,
                        kind=kind,
                        chunk_index=chunk["chunk_index"],
                        text=chunk["text"],
                        iteration_id=iteration_id_tag,
                    )
                )

    return synthesized


def _iteration_id_from_event(event: RunEvent) -> str | None:
    """Pull ``iteration_id`` from an event's payload when present.

    Returned to the caller as a separate value (rather than threaded
    through whole payload dicts) so the synth helpers can decide
    whether to include the field — keeps payloads from carrying
    explicit ``iteration_id: null`` for non-iterative runs.
    """
    payload = event.payload or {}
    value = payload.get("iteration_id")
    return value if isinstance(value, str) else None


def _with_iteration_id(
    payload: dict[str, Any], iteration_id: str | None
) -> dict[str, Any]:
    """Return ``payload`` augmented with ``iteration_id`` when set.

    Mutates and returns the same dict to avoid an extra copy on the hot
    path. Non-iterative runs (iteration_id is None) get an unchanged
    payload — no spurious key landing in the wire shape.
    """
    if iteration_id is not None:
        payload["iteration_id"] = iteration_id
    return payload


def _build_inflight_chunk_event(
    *,
    run_id: uuid.UUID,
    node_id: str,
    sequence: int,
    timestamp: datetime,
    event_type: RunEventType,
    message_id: str,
    kind: str,
    chunk_index: int,
    text: str,
    iteration_id: str | None = None,
) -> RunEvent:
    """Build a single synthetic chunk-level delta event.

    Id matches what the live publisher would produce for the same
    (message_id, kind, chunk_index) — so when a reconnecting client
    merges REST backfill, WS snapshot, and live deltas, every overlap is
    deduped by id with no per-chunk-index dedup table needed downstream.
    """
    return _build_synthetic_event(
        run_id=run_id,
        node_id=node_id,
        sequence=sequence,
        timestamp=timestamp,
        event_type=event_type,
        stable_key=f"delta:{message_id}:{kind}:{chunk_index}",
        payload=_with_iteration_id(
            {
                "message_id": message_id,
                "chunk_index": chunk_index,
                "text": text,
            },
            iteration_id,
        ),
    )


async def _read_inflight_chunks(
    redis: Redis, run_id: uuid.UUID, message_id: str
) -> list[dict[str, Any]]:
    """Return one structured record per chunk currently in the stream cache.

    Each record carries ``kind``, ``text``, ``chunk_index`` (kind-specific
    counter matching what the live publisher used) and ``sequence_offset``
    (1-based INCR offset, so absolute sequence is
    ``started_seq + sequence_offset``). The order matches RPUSH order,
    which is also the order the live publisher allocated sequences in.
    Empty if the stream cache is absent (turn finished or never started).

    Keyed by ``message_id`` (not ``node_id``) — parallel iterations of
    the same ai_agent node each own a separate cache, so this read only
    returns chunks for the turn whose message_id we asked about. See
    ``stream_key`` for the rationale.
    """
    raw = await redis.lrange(stream_key(run_id, message_id), 0, -1)
    chunks: list[dict[str, Any]] = []
    text_count = 0
    thinking_count = 0
    for index, entry in enumerate(raw):
        if isinstance(entry, bytes):
            entry = entry.decode("utf-8")
        try:
            data = json.loads(entry)
        except (TypeError, json.JSONDecodeError):
            continue
        kind = data.get("kind")
        text = data.get("text", "")
        if not isinstance(text, str):
            continue
        if kind == "text":
            chunk_index = text_count
            text_count += 1
        elif kind == "thinking":
            chunk_index = thinking_count
            thinking_count += 1
        else:
            continue
        chunks.append(
            {
                "kind": kind,
                "text": text,
                "chunk_index": chunk_index,
                "sequence_offset": index + 1,
            }
        )
    return chunks


async def build_run_stream_snapshot(
    redis: Redis, run_id: uuid.UUID
) -> list[str]:
    """Build initial WebSocket frames for a new subscriber.

    Reads every ``stream_meta:msg:{run_id}:*`` key to find assistant
    turns that are currently mid-stream and emits one synthetic delta
    frame per chunk already in the per-message stream cache, matching
    exactly what the live publisher would have broadcast (same
    deterministic id, same sequence, same chunk_index). The client
    reducer dedupes by id, so any chunk that arrives both via this
    snapshot AND via live pub/sub — whether because of a race during
    connect or because the client already had it from before a
    disconnect — collapses to a single entry in the timeline with no
    double-counted text.

    Meta is keyed per ``message_id`` so concurrent turns of the same
    ai_agent node (parallel iteration) each get their own snapshot
    entry instead of overwriting each other. ``node_id`` is carried in
    the meta payload because we can no longer parse it out of the key.

    Returns an empty list when no streams are in flight for this run.
    """
    # The ``message_id`` placeholder for scan only matches the
    # post-message_id segment; pre-message_id parts of the key are
    # literal. Using inflight_meta_key with "*" gives us the right
    # pattern without hardcoding the prefix here.
    pattern = inflight_meta_key(run_id, "*")
    keys: list[str] = []
    async for raw_key in redis.scan_iter(match=pattern):
        if isinstance(raw_key, bytes):
            raw_key = raw_key.decode("utf-8")
        keys.append(raw_key)

    if not keys:
        return []

    frames: list[str] = []
    for key in keys:
        raw_meta = await redis.get(key)
        if not raw_meta:
            continue
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8")
        try:
            meta = json.loads(raw_meta)
        except (TypeError, json.JSONDecodeError):
            continue
        message_id = meta.get("message_id")
        started_seq = meta.get("started_seq")
        timestamp_iso = meta.get("timestamp")
        # ``node_id`` lives in the meta payload now (the key encodes
        # message_id, not node_id). Falls back to splitting the key
        # only for back-compat with pre-fix meta payloads that might
        # still be in Redis after a deploy (TTL keeps them around for
        # an hour).
        node_id = meta.get("node_id")
        if not isinstance(node_id, str):
            node_id = key.split(":")[-1]
        if not isinstance(message_id, str) or not isinstance(started_seq, int):
            continue
        # Optional — only present when the turn was started inside an
        # iteration scope (see _write_inflight_meta). Snapshot frames
        # carry this through so the WS filter on the receiving side
        # routes them to the right iteration block.
        meta_iteration_id = meta.get("iteration_id")
        iteration_id_tag = (
            meta_iteration_id if isinstance(meta_iteration_id, str) else None
        )
        chunks = await _read_inflight_chunks(redis, run_id, message_id)
        if not chunks:
            continue
        for chunk in chunks:
            kind = chunk["kind"]
            event_type = (
                RunEventType.agent_text_delta
                if kind == "text"
                else RunEventType.agent_thinking_delta
            )
            frames.append(
                _serialize_synthetic_frame(
                    run_id=run_id,
                    node_id=node_id,
                    sequence=started_seq + chunk["sequence_offset"],
                    timestamp_iso=timestamp_iso,
                    event_type=event_type,
                    event_id=delta_event_id(
                        message_id, kind, chunk["chunk_index"]
                    ),
                    payload=_with_iteration_id(
                        {
                            "message_id": message_id,
                            "chunk_index": chunk["chunk_index"],
                            "text": chunk["text"],
                        },
                        iteration_id_tag,
                    ),
                )
            )

    return frames


def _serialize_synthetic_frame(
    *,
    run_id: uuid.UUID,
    node_id: str | None,
    sequence: int,
    timestamp_iso: Any,
    event_type: RunEventType,
    event_id: uuid.UUID,
    payload: dict[str, Any],
) -> str:
    if not isinstance(timestamp_iso, str):
        timestamp_iso = datetime.now(timezone.utc).isoformat()
    frame = {
        "id": str(event_id),
        "flow_run_id": str(run_id),
        "sequence": sequence,
        "timestamp": timestamp_iso,
        "event_type": event_type.value,
        "node_id": node_id,
        "payload": payload,
    }
    return json.dumps(frame)


def _build_synthetic_event(
    *,
    run_id: uuid.UUID,
    node_id: str | None,
    sequence: int,
    timestamp: datetime,
    event_type: RunEventType,
    stable_key: str,
    payload: dict[str, Any],
) -> RunEvent:
    ev = RunEvent(
        flow_run_id=run_id,
        sequence=sequence,
        timestamp=timestamp,
        event_type=event_type,
        node_id=node_id,
        payload=payload,
    )
    ev.id = uuid.uuid5(uuid.NAMESPACE_OID, stable_key)
    return ev


class RunNotTerminal(ValueError):
    """Raised when trying to clear a run that is still running/pending/paused."""


async def clear_run_data(db: AsyncSession, run_id: uuid.UUID) -> FlowRun:
    """Soft-compact a terminal run: keep the FlowRun row (so we still know it
    happened, how long it ran, how it finished) but wipe the streaming
    payloads — every RunEvent and every AgentMessage. Idempotent on already
    cleared runs.
    """
    run = await get_flow_run(db, run_id)
    if run.status not in (FlowRunStatus.completed, FlowRunStatus.failed):
        raise RunNotTerminal(
            f"flow run {run_id} is not terminal (status={run.status.value})"
        )
    if run.cleared_at is not None:
        return run

    await db.execute(delete(RunEvent).where(RunEvent.flow_run_id == run_id))
    await db.execute(delete(AgentMessage).where(AgentMessage.flow_run_id == run_id))
    now = datetime.now(timezone.utc)
    run.cleared_at = now
    # Drop a single trailing marker so consumers know *something* used to be
    # here and when it was wiped.
    marker = RunEvent(
        flow_run_id=run_id,
        sequence=1,
        timestamp=now,
        event_type=RunEventType.run_cleared,
        node_id=None,
        payload={"cleared_at": now.isoformat()},
    )
    db.add(marker)
    await db.commit()
    await db.refresh(run)
    return run


async def list_run_messages(
    db: AsyncSession, run_id: uuid.UUID, node_id: str | None = None
) -> list[AgentMessage]:
    await get_flow_run(db, run_id)
    stmt = select(AgentMessage).where(AgentMessage.flow_run_id == run_id)
    if node_id is not None:
        stmt = stmt.where(AgentMessage.node_id == node_id)
    stmt = stmt.order_by(AgentMessage.sequence.asc())
    result = await db.execute(stmt)
    return list(result.scalars().all())
