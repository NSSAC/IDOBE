"""Segment review API: experts see a full series with a method's proposed outbreak windows and
accept, adjust or reject each one, or add outbreaks the method missed."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..auth import admin_user, current_user
from ..config import settings
from ..db import get_db
from ..models import ProposedSegment, RunSeries, SegmentationRun, SegmentReview, Series, User
from ..segments import REVIEW_EXPORT_COLUMNS, export_rows, latest_review_ids, pick_next_series, validate_items, week
from .admin import _csv_response

router = APIRouter(prefix="/api/seg", tags=["segments"])
MAX_TIME_SPENT_MS = 24 * 3600 * 1000


class ItemIn(BaseModel):
    proposed_id: int | None = None
    verdict: Literal["accept", "adjust", "reject", "added"]
    start_idx: int | None = None
    end_idx: int | None = None


class ReviewIn(BaseModel):
    run_id: int
    series_id: int
    items: list[ItemIn] = Field(default_factory=list, max_length=500)
    series_unusable: bool = False
    skipped: bool = False
    confidence: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=5000)
    time_spent_ms: int | None = Field(default=None, ge=0)


def _run(db: Session, run_id: int | None) -> SegmentationRun:
    stmt = select(SegmentationRun).where(SegmentationRun.is_active.is_(True))
    stmt = stmt.where(SegmentationRun.id == run_id) if run_id else stmt.order_by(SegmentationRun.id.desc())
    run = db.execute(stmt.limit(1)).scalar_one_or_none()
    if run is None:
        raise HTTPException(404, "No active segmentation run" if not run_id else "Run not found or inactive")
    return run


def _review_out(r: SegmentReview | None):
    if r is None:
        return None
    return {"id": r.id, "items": r.items, "series_unusable": r.series_unusable, "skipped": r.skipped,
            "confidence": r.confidence, "comment": r.comment, "created_at": r.created_at}


def series_payload(db: Session, run: SegmentationRun, s: Series, user_id: int) -> dict:
    props = db.execute(select(ProposedSegment).where(ProposedSegment.run_id == run.id,
                                                     ProposedSegment.series_id == s.id)
                       .order_by(ProposedSegment.start_idx)).scalars().all()
    mine = db.execute(select(SegmentReview).where(SegmentReview.run_id == run.id, SegmentReview.series_id == s.id,
                                                  SegmentReview.user_id == user_id)
                      .order_by(SegmentReview.id.desc()).limit(1)).scalar_one_or_none()
    n = len(s.values)
    return {
        "run": {"id": run.id, "name": run.name},
        "series": {"id": s.id, "key": s.series_key, "disease": s.disease, "location": s.location,
                   "event": s.event, "geo_level": s.geo_level, "source": s.source,
                   "dates": [week(s, i) for i in range(n)], "values": s.values},
        "proposals": [{"id": p.id, "start_idx": p.start_idx, "end_idx": p.end_idx, "score": p.score,
                       "peak_idx": p.peak_idx, "core_start_idx": p.core_start_idx,
                       "core_end_idx": p.core_end_idx} for p in props],
        "my_review": _review_out(mine),
    }


@router.get("/runs")
def runs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    out = []
    for run in db.execute(select(SegmentationRun).where(SegmentationRun.is_active.is_(True))
                          .order_by(SegmentationRun.id.desc())).scalars():
        n_series = db.execute(select(func.count()).select_from(RunSeries).where(RunSeries.run_id == run.id)).scalar_one()
        mine = db.execute(select(func.count(func.distinct(SegmentReview.series_id)))
                          .where(SegmentReview.run_id == run.id, SegmentReview.user_id == user.id)).scalar_one()
        out.append({"id": run.id, "name": run.name, "method": run.method, "n_series": n_series, "reviewed_by_me": mine})
    return out


@router.get("/next")
def next_series(run_id: int | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _run(db, run_id)
    sid = pick_next_series(db, run.id, user.id, settings.target_annotations)
    if sid is None:
        return {"done": True, "run": {"id": run.id, "name": run.name}}
    return {"done": False, **series_payload(db, run, db.get(Series, sid), user.id)}


@router.get("/series/{series_id}")
def get_series(series_id: int, run_id: int | None = None, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    run = _run(db, run_id)
    s = db.get(Series, series_id)
    in_run = db.get(RunSeries, (run.id, series_id))
    if s is None or in_run is None:
        raise HTTPException(404, "Series not found in this run")
    return series_payload(db, run, s, user.id)


@router.post("/reviews", status_code=201)
def create_review(body: ReviewIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _run(db, body.run_id)
    s = db.get(Series, body.series_id)
    if s is None or db.get(RunSeries, (run.id, s.id)) is None:
        raise HTTPException(404, "Series not found in this run")
    props = db.execute(select(ProposedSegment).where(ProposedSegment.run_id == run.id,
                                                     ProposedSegment.series_id == s.id)).scalars().all()
    if body.skipped:
        items, confidence = [], None
    else:
        if body.confidence is None:
            raise HTTPException(422, "confidence (1-5) is required unless skipping")
        try:
            items = validate_items([i.model_dump() for i in body.items], props, len(s.values),
                                   complete=not body.series_unusable)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        confidence = body.confidence
    rev = SegmentReview(
        run_id=run.id, series_id=s.id, user_id=user.id, items=items, series_unusable=body.series_unusable,
        skipped=body.skipped, confidence=confidence, comment=(body.comment or "").strip() or None,
        time_spent_ms=min(body.time_spent_ms, MAX_TIME_SPENT_MS) if body.time_spent_ms is not None else None,
        app_version=__version__,
    )
    db.add(rev)
    db.commit()
    return _review_out(rev)


@router.get("/my-reviews")
def my_reviews(run_id: int | None = None, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=200),
               user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _run(db, run_id)
    latest = latest_review_ids(run.id)
    base = (select(SegmentReview, Series).join(latest, latest.c.id == SegmentReview.id)
            .join(Series, Series.id == SegmentReview.series_id).where(SegmentReview.user_id == user.id))
    total = db.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    rows = db.execute(base.order_by(SegmentReview.id.desc()).offset(offset).limit(limit)).all()

    def summary(r: SegmentReview) -> str:
        if r.skipped:
            return "Skipped"
        if r.series_unusable:
            return "Series unusable"
        c = {v: sum(1 for i in r.items if i["verdict"] == v) for v in ("accept", "adjust", "reject", "added")}
        return ", ".join(f"{n} {v}" for v, n in c.items() if n) or "No outbreaks"

    return {"total": total, "items": [
        {"series_id": s.id, "disease": s.disease, "location": s.location, "event": s.event,
         "summary": summary(r), "confidence": r.confidence, "created_at": r.created_at} for r, s in rows]}


@router.get("/export.csv")
def export(run_id: int | None = None, _: User = Depends(admin_user), db: Session = Depends(get_db)):
    """Each expert's current review, one row per decision; input to pipeline/consensus.py."""
    return _csv_response(REVIEW_EXPORT_COLUMNS, export_rows(db, run_id), "segment_reviews.csv")
