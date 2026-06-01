"""Fires scheduled flows that are due.

Invoked once per minute by :func:`luna_core.tasks.scheduler_tick` (registered
in ``celery_app.conf.beat_schedule``). The dispatcher walks active flows whose
trigger type is ``schedule``; for each rule it asks croniter "what is the next
fire strictly after the last one?" and dispatches when that timestamp is in
the past.

Cron expressions are interpreted in the rule's IANA timezone, not in UTC. The
frontend stores the user's wall-clock time verbatim alongside the IANA zone
(e.g. ``cron='30 9 * * 1,3'`` + ``tz='Europe/Madrid'``) and croniter handles
DST transitions natively when evaluated against tz-aware datetimes. Storing
pre-converted UTC crons would silently drift twice a year.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from luna_core.models.flow import Flow, FlowRun

logger = logging.getLogger(__name__)


def _resolve_tz(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        logger.warning("unknown timezone %r — falling back to UTC", tz)
        return ZoneInfo("UTC")


async def _last_scheduled_fire(
    db: AsyncSession, flow_id: uuid.UUID, rule_index: int
) -> datetime | None:
    """Most recent scheduled FlowRun for ``flow_id`` matching this rule.

    Rule 0 also matches legacy runs (created before this feature, when the
    trigger metadata didn't carry ``rule_index``) so we don't double-fire on
    the first tick after upgrade. Rule N>0 requires an exact match.
    """
    stmt = (
        select(FlowRun.created_at)
        .where(FlowRun.flow_id == flow_id)
        .where(FlowRun.trigger["source"].astext == "schedule")
        .order_by(FlowRun.created_at.desc())
        .limit(1)
    )
    if rule_index > 0:
        stmt = stmt.where(
            FlowRun.trigger["metadata"]["rule_index"].astext == str(rule_index)
        )
    return (await db.execute(stmt)).scalar_one_or_none()


async def dispatch_due_runs(db: AsyncSession) -> int:
    # Lazy import: tasks.py imports this module, so a top-level import would
    # create a cycle.
    from luna_core.tasks import trigger_scheduled_run_task

    now = datetime.now(timezone.utc)
    stmt = select(Flow).where(
        Flow.is_active.is_(True),
        Flow.definition["trigger"]["type"].astext == "schedule",
    )
    flows = (await db.execute(stmt)).scalars().all()

    dispatched = 0
    for flow in flows:
        trigger = (flow.definition or {}).get("trigger") or {}
        rules = trigger.get("schedules") or []
        # Legacy flows that only carry ``trigger.cron`` are materialised to a
        # single rule so both shapes flow through the same code path.
        if not rules and trigger.get("cron"):
            rules = [{"cron": trigger["cron"], "tz": "UTC"}]

        for idx, rule in enumerate(rules):
            cron_expr = rule.get("cron")
            if not cron_expr or not croniter.is_valid(cron_expr):
                continue
            # Each rule is bound to the user who authored it; the dispatched
            # run inherits that identity via trigger.metadata.user_id so
            # id_implicit context sources (profile, user, …) resolve cleanly.
            # Schedules saved before this field existed are skipped — they'd
            # blow up at the first id_implicit loader anyway.
            user_id = rule.get("user_id")
            if not user_id:
                logger.warning(
                    "skipping flow %s rule %d: missing user_id (re-save the schedule)",
                    flow.id,
                    idx,
                )
                continue
            tz = _resolve_tz(rule.get("tz", "UTC"))
            last_fire = await _last_scheduled_fire(db, flow.id, idx)
            # First-ever evaluation looks one minute back so a cron whose slot
            # has just passed still fires on the very first tick. We don't
            # backfill further than that — the goal is "from now on", not
            # "replay the year".
            base_utc = last_fire or (now - timedelta(minutes=1))
            base_local = base_utc.astimezone(tz)
            next_fire = croniter(cron_expr, base_local).get_next(datetime)
            next_fire_utc = next_fire.astimezone(timezone.utc)
            if next_fire_utc <= now:
                rule_inputs = rule.get("inputs") or {}
                trigger_scheduled_run_task.delay(
                    str(flow.id),
                    {
                        "source": "schedule",
                        # Per-rule inputs feed `state.inputs`; missing keys
                        # get the declared defaults via validate_flow_inputs
                        # inside create_flow_run.
                        "inputs": rule_inputs,
                        "metadata": {
                            "rule_index": idx,
                            "scheduled_for": next_fire_utc.isoformat(),
                            "user_id": str(user_id),
                        },
                    },
                )
                dispatched += 1
                logger.info(
                    "dispatched flow %s rule %d (user %s) for %s",
                    flow.id,
                    idx,
                    user_id,
                    next_fire_utc.isoformat(),
                )
    return dispatched


def preview_next_runs(cron_expr: str, tz: str, count: int) -> list[datetime]:
    """Return ``count`` upcoming fire times in UTC for the given cron + tz."""
    if not croniter.is_valid(cron_expr):
        raise ValueError(f"invalid cron expression: {cron_expr!r}")
    zone = _resolve_tz(tz)
    now_local = datetime.now(timezone.utc).astimezone(zone)
    it = croniter(cron_expr, now_local)
    out: list[datetime] = []
    for _ in range(count):
        nxt = it.get_next(datetime)
        out.append(nxt.astimezone(timezone.utc))
    return out
