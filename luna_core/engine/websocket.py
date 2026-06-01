from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from luna_core.engine.emitter import run_event_channel

logger = logging.getLogger(__name__)


ChannelFn = Callable[[uuid.UUID], str]
SnapshotFn = Callable[[uuid.UUID], Awaitable[list[str]]]

# Event types that constitute the per-iteration lifecycle envelope. Always
# forwarded to every connected client regardless of their subscription set,
# so the UI can render an iteration accordion header (started / done / failed)
# even when the user hasn't expanded it. The noisy sub-events emitted *inside*
# an iteration (agent_text_delta, tool_called, agent_message_completed, …)
# carry an ``iteration_id`` payload field and are filtered per-client.
ITERATION_LIFECYCLE_TYPES: frozenset[str] = frozenset(
    {"iteration_started", "iteration_completed", "iteration_failed"}
)


class _PumpState:
    """Per-key pump bookkeeping. ``subscribed`` is set once the Redis
    SUBSCRIBE command has been acknowledged; any client that connects
    afterwards is guaranteed that no pub/sub message published between
    its connect and snapshot read is lost — those messages land in its
    per-client queue and are forwarded right after the snapshot frames.
    Any overlap between snapshot and live frames is collapsed downstream
    by the client reducer via deterministic event ids."""

    __slots__ = ("task", "subscribed")

    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None
        self.subscribed: asyncio.Event = asyncio.Event()


class _ClientState:
    """Per-WebSocket bookkeeping. The queue decouples the pub/sub pump
    from the WS send loop so a slow client can't backpressure other
    clients sharing the same pump.

    ``iteration_subscriptions`` holds the iteration_ids this client has
    asked to receive sub-events for. It's populated by the recv loop
    when the client sends ``{"type": "subscribe", "scope": "iteration",
    "id": "<uuid>"}`` and cleared by the matching unsubscribe. Empty
    by default — a fresh client gets the lifecycle envelope only,
    keeping the per-run firehose readable when many parallel iterations
    are emitting in flight. A lock guards mutations so the recv loop
    and fanout don't race on read/write.
    """

    __slots__ = ("queue", "iteration_subscriptions", "subs_lock")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.iteration_subscriptions: set[str] = set()
        self.subs_lock: asyncio.Lock = asyncio.Lock()


class WebSocketManager:
    """Bridges Redis pub/sub channels into FastAPI WebSocket clients.

    Each key (run_id or flow_id) gets at most one Redis subscriber regardless
    of how many clients are connected. Incoming messages are fanned out into
    per-client async queues so a slow client can't backpressure others.

    Zero-loss reconnect contract: when ``snapshot_fn`` is supplied, every
    new client first awaits the pump's SUBSCRIBE ack, then computes its
    snapshot (e.g. the in-flight stream cache rehydrated as synthetic
    delta frames with deterministic ids matching their live counterparts),
    then sends the snapshot frames before any live pub/sub frame. Any
    live frame published between subscribe and snapshot computation lands
    in the per-client queue and is drained right after; overlaps with the
    snapshot are deduped by id at the client. This eliminates both the
    "lost prefix on mid-stream reconnect" and "duplicated chunks across
    the overlap window" failure modes without any server-side dedup
    bookkeeping.
    """

    def __init__(
        self,
        redis: Redis,
        channel_fn: ChannelFn = run_event_channel,
        snapshot_fn: SnapshotFn | None = None,
    ) -> None:
        self._redis = redis
        self._channel_fn = channel_fn
        self._snapshot_fn = snapshot_fn
        self._clients: dict[uuid.UUID, dict[WebSocket, _ClientState]] = (
            defaultdict(dict)
        )
        self._pumps: dict[uuid.UUID, _PumpState] = {}
        self._lock = asyncio.Lock()

    async def connect(self, key: uuid.UUID, websocket: WebSocket) -> None:
        await websocket.accept()
        state = _ClientState()
        async with self._lock:
            self._clients[key][websocket] = state
            pump = self._pumps.get(key)
            if pump is None:
                pump = _PumpState()
                self._pumps[key] = pump
                pump.task = asyncio.create_task(self._pump(key, pump))

        # Wait for the SUBSCRIBE to be active before snapshotting — anything
        # published after this point will be queued for us by ``_fanout``.
        await pump.subscribed.wait()

        snapshot_frames: list[str] = []
        if self._snapshot_fn is not None:
            try:
                snapshot_frames = await self._snapshot_fn(key)
            except Exception:
                logger.exception("snapshot_fn failed for %s", key)
                snapshot_frames = []

        send_task = asyncio.create_task(
            self._send_loop(websocket, state, snapshot_frames)
        )
        recv_task = asyncio.create_task(self._recv_loop(websocket, state))
        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    logger.debug("websocket task error for %s: %r", key, exc)
        finally:
            await self._remove(key, websocket)

    async def _send_loop(
        self,
        websocket: WebSocket,
        state: _ClientState,
        snapshot_frames: list[str],
    ) -> None:
        # 1. Snapshot first, in order — captured AFTER subscribe so anything
        #    in flight is preserved in the queue rather than silently lost.
        #    Snapshot frames pass through the same iteration filter the
        #    fanout uses: a fresh client has no iteration subscriptions
        #    yet, so any in-flight iteration deltas captured by
        #    snapshot_fn would otherwise leak through. The client can
        #    backfill the specific iteration it expands via REST
        #    `/runs/{id}/events?iteration_id=...`.
        for frame in snapshot_frames:
            if not self._frame_allowed(frame, state):
                continue
            await websocket.send_text(frame)
        # 2. Drain anything the pump enqueued during snapshot computation;
        #    duplicates with the snapshot (same deterministic id) are
        #    collapsed by the client reducer, so no server-side filter
        #    needed here. The fanout already applied the iteration
        #    subscription filter before enqueueing.
        while not state.queue.empty():
            await websocket.send_text(state.queue.get_nowait())
        # 3. Steady-state forward.
        while True:
            frame = await state.queue.get()
            await websocket.send_text(frame)

    @staticmethod
    def _frame_allowed(frame: str, state: _ClientState) -> bool:
        """Decide whether one frame may reach ``state``'s client.

        Mirrors the rule in ``_fanout`` but used on the snapshot path
        (which is sent once at connect, outside the fanout). Frames
        without an iteration_id, or carrying a lifecycle iteration event
        type, always pass. Sub-events tagged with an iteration_id
        require an active subscription on that id.
        """
        iteration_id, event_type = _extract_routing_keys(frame)
        if iteration_id is None or event_type in ITERATION_LIFECYCLE_TYPES:
            return True
        return iteration_id in state.iteration_subscriptions

    async def _recv_loop(
        self, websocket: WebSocket, state: _ClientState
    ) -> None:
        """Consume control frames from the client.

        Protocol (one JSON object per frame):

            {"type": "subscribe",   "scope": "iteration", "id": "<uuid>"}
            {"type": "unsubscribe", "scope": "iteration", "id": "<uuid>"}

        Subscribing to an iteration_id tells the fanout to forward
        sub-events tagged with that id to this client. Unsubscribing
        removes it. Lifecycle envelope events
        (iteration_started / _completed / _failed) and events without
        an iteration_id are always forwarded — the subscription set
        only gates the noisy in-iteration sub-events.

        Malformed frames are tolerated and dropped: the client is on
        its own conscience to send valid JSON. The legacy "send any
        text, server ignores" contract is preserved for clients that
        never opt into the protocol.
        """
        try:
            while True:
                raw = await websocket.receive_text()
                await self._handle_control_message(state, raw)
        except WebSocketDisconnect:
            return

    async def _handle_control_message(
        self, state: _ClientState, raw: str
    ) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        msg_type = msg.get("type")
        scope = msg.get("scope")
        target_id = msg.get("id")
        if (
            msg_type not in ("subscribe", "unsubscribe")
            or scope != "iteration"
            or not isinstance(target_id, str)
            or not target_id
        ):
            return
        async with state.subs_lock:
            if msg_type == "subscribe":
                state.iteration_subscriptions.add(target_id)
            else:
                state.iteration_subscriptions.discard(target_id)

    async def _remove(self, key: uuid.UUID, websocket: WebSocket) -> None:
        async with self._lock:
            clients = self._clients.get(key)
            if clients is None:
                return
            clients.pop(websocket, None)
            if clients:
                return
            self._clients.pop(key, None)
            pump = self._pumps.pop(key, None)
        if pump is not None and pump.task is not None:
            pump.task.cancel()
            try:
                await pump.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _pump(self, key: uuid.UUID, state: _PumpState) -> None:
        channel = self._channel_fn(key)
        pubsub: PubSub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        state.subscribed.set()
        try:
            async for message in pubsub.listen():
                if message is None or message.get("type") != "message":
                    continue
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                self._fanout(key, data)
        except asyncio.CancelledError:
            raise
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    def _fanout(self, key: uuid.UUID, payload: str) -> None:
        """Dispatch one pub/sub frame to every connected client, applying
        per-client iteration filtering.

        We parse the JSON once here (a few microseconds) instead of N
        times in the send loop, and the result is reused for every
        client sharing this pump. Frames that aren't valid JSON or that
        carry no ``iteration_id`` flow to everyone — that covers
        lifecycle events (flow_*, node_*, run_cleared) and any sub-event
        emitted outside an iteration scope.
        """
        iteration_id, event_type = _extract_routing_keys(payload)

        # Fast path: anything not tagged with an iteration_id, plus the
        # per-iteration lifecycle envelope, goes to every client. Only
        # tagged sub-events face the subscription gate.
        broadcast_to_all = iteration_id is None or (
            event_type in ITERATION_LIFECYCLE_TYPES
        )

        for client_state in list(self._clients.get(key, {}).values()):
            if not broadcast_to_all:
                if iteration_id not in client_state.iteration_subscriptions:
                    continue
            try:
                client_state.queue.put_nowait(payload)
            except Exception:  # noqa: BLE001
                logger.debug("queue put failed for %s", key)


def _extract_routing_keys(payload: str) -> tuple[str | None, str | None]:
    """Pull ``(iteration_id, event_type)`` from a serialized event frame.

    Returns ``(None, None)`` when the payload isn't a JSON object or
    when the keys are missing — the fanout then treats the frame as
    universally broadcastable. Kept as a free function so it can be
    unit-tested without standing up the manager.
    """
    try:
        obj = json.loads(payload)
    except (TypeError, ValueError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    event_type = obj.get("event_type")
    inner = obj.get("payload")
    if not isinstance(inner, dict):
        return None, event_type if isinstance(event_type, str) else None
    iteration_id = inner.get("iteration_id")
    return (
        iteration_id if isinstance(iteration_id, str) else None,
        event_type if isinstance(event_type, str) else None,
    )
