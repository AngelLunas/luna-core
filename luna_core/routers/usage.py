"""The operator's view of what the models cost: prices in, report out.

``/llm-prices`` holds what each model's tokens cost (entered by an admin from
the provider's pricing page — never written in code). ``/usage/report`` groups
the ledger by who (``user_id``), what (``purpose``, ``agent_name``), which model
and which day, with each group's cost at those prices.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import select

from luna_core.core.dependencies import DBSession, require_permission
from luna_core.models.pricing import LLMPrice
from luna_core.schemas.usage import LLMPriceRead, LLMPriceWrite, UsageReportRow
from luna_core.services.usage import REPORT_GROUPS, usage_report

router = APIRouter(tags=["usage"])


@router.get(
    "/llm-prices",
    response_model=list[LLMPriceRead],
    dependencies=[require_permission("usage:read")],
)
async def list_prices(db: DBSession) -> list[LLMPriceRead]:
    rows = (await db.execute(select(LLMPrice).order_by(LLMPrice.model))).scalars().all()
    return [LLMPriceRead.model_validate(r) for r in rows]


@router.put(
    "/llm-prices/{model:path}",
    response_model=LLMPriceRead,
    dependencies=[require_permission("llm_prices:update")],
)
async def put_price(model: str, payload: LLMPriceWrite, db: DBSession) -> LLMPriceRead:
    """Set a model's price (the id exactly as the ledger records it)."""
    row = await db.get(LLMPrice, model)
    if row is None:
        row = LLMPrice(model=model, **payload.model_dump())
        db.add(row)
    else:
        for field, value in payload.model_dump().items():
            setattr(row, field, value)
    await db.commit()
    await db.refresh(row)
    return LLMPriceRead.model_validate(row)


@router.delete(
    "/llm-prices/{model:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    dependencies=[require_permission("llm_prices:update")],
)
async def delete_price(model: str, db: DBSession) -> Response:
    row = await db.get(LLMPrice, model)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no price for that model")
    await db.delete(row)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/usage/report",
    response_model=list[UsageReportRow],
    dependencies=[require_permission("usage:read")],
)
async def report(
    db: DBSession,
    since: datetime | None = None,
    until: datetime | None = None,
    group_by: str = Query(default="user_id,purpose", description=f"Comma-separated: {', '.join(REPORT_GROUPS)}"),
    user_id: uuid.UUID | None = None,
) -> list[UsageReportRow]:
    """Calls, tokens and cost by group, dearest first. Default window: the
    last 7 days."""
    groups = tuple(g.strip() for g in group_by.split(",") if g.strip())
    try:
        rows = await usage_report(
            db,
            since=since or datetime.now(UTC) - timedelta(days=7),
            until=until,
            group_by=groups,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return [UsageReportRow(**r) for r in rows]
