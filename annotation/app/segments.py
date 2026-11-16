"""Segment review: queue, validation and export. The HTTP layer is in routes/segments.py."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import ProposedSegment, RunSeries, SegmentationRun, SegmentReview, Series, User

# Columns read by pipeline/consensus.py.
REVIEW_EXPORT_COLUMNS = [
    "series_id", "run_method", "reviewer", "proposed_segment_id", "verdict", "start_date", "end_date",
    "confidence", "series_unusable", "review_id", "run_name", "comment", "created_at",
]


def latest_review_ids(run_id: int | None = None):
    """Subquery: id of each user's current review per (run, series)."""
    stmt = select(func.max(SegmentReview.id).label("id")).group_by(
        SegmentReview.run_id, SegmentReview.series_id, SegmentReview.user_id)
    if run_id is not None:
        stmt = stmt.where(SegmentReview.run_id == run_id)
    return stmt.subquery()


def week(series: Series, i: int | None):
    return None if i is None else series.start_date + timedelta(weeks=i)


def pick_next_series(db: Session, run_id: int, user_id: int, target: int) -> int | None:
    """A random series of the run that this user hasn't reviewed and that has fewer than `target`
    completed (not skipped) reviews."""
    latest = latest_review_ids(run_id)
    done = (
        select(SegmentReview.series_id, func.count().label("n"))
        .join(latest, latest.c.id == SegmentReview.id)
        .where(SegmentReview.skipped.is_(False))
        .group_by(SegmentReview.series_id)
        .subquery()
    )
    mine = select(SegmentReview.series_id).where(SegmentReview.run_id == run_id, SegmentReview.user_id == user_id)
    stmt = (
        select(RunSeries.series_id)
        .outerjoin(done, done.c.series_id == RunSeries.series_id)
        .where(RunSeries.run_id == run_id, func.coalesce(done.c.n, 0) < target,
               RunSeries.series_id.not_in(mine))
        .order_by(func.coalesce(done.c.n, 0).desc(), func.random())
        .limit(1)
    )
    return db.execute(stmt).scalar_one_or_none()


def validate_items(items: list[dict], proposals: list[ProposedSegment], n_weeks: int, *, complete: bool) -> list[dict]:
    """Check a review's decisions; returns them normalized. Raises ValueError with a readable message."""
    by_id = {p.id: p for p in proposals}
    seen: set[int] = set()
    out = []
    for it in items:
        pid, verdict = it.get("proposed_id"), it["verdict"]
        start, end = it.get("start_idx"), it.get("end_idx")
        if verdict == "added":
            if pid is not None:
                raise ValueError("an added outbreak cannot refer to a proposed segment")
        else:
            if pid not in by_id:
                raise ValueError(f"segment {pid} is not proposed for this series")
            if pid in seen:
                raise ValueError(f"segment {pid} has more than one decision")
            seen.add(pid)
        if verdict == "reject":
            start = end = None
        elif verdict == "accept":
            start, end = by_id[pid].start_idx, by_id[pid].end_idx
        else:
            if start is None or end is None or not (0 <= start <= end < n_weeks):
                raise ValueError(f"boundaries {start}..{end} are outside the series (0..{n_weeks - 1})")
            if verdict == "adjust" and (start, end) == (by_id[pid].start_idx, by_id[pid].end_idx):
                verdict = "accept"
        out.append({"proposed_id": pid, "verdict": verdict, "start_idx": start, "end_idx": end})
    if complete and seen != set(by_id):
        missing = sorted(set(by_id) - seen)
        raise ValueError(f"every proposed segment needs a decision; missing {missing}")
    return out


def export_rows(db: Session, run_id: int | None = None):
    """Each expert's current review, one row per decision (the format pipeline/consensus.py reads).

    A review with no decisions (e.g. "no outbreaks here" on a series without proposals) still
    yields one row with an empty verdict, so the reviewer counts toward that series."""
    latest = latest_review_ids(run_id)
    q = (
        select(SegmentReview, User, Series, SegmentationRun)
        .join(latest, latest.c.id == SegmentReview.id)
        .join(User, User.id == SegmentReview.user_id)
        .join(Series, Series.id == SegmentReview.series_id)
        .join(SegmentationRun, SegmentationRun.id == SegmentReview.run_id)
        .where(SegmentReview.skipped.is_(False))
        .order_by(SegmentReview.id)
    )
    keys = dict(db.execute(select(ProposedSegment.id, ProposedSegment.segment_key)).all())
    for rev, user, s, run in db.execute(q).all():
        base = [s.series_key, run.method, user.username]
        tail = [rev.confidence, int(rev.series_unusable), rev.id, run.name, rev.comment, rev.created_at.isoformat()]
        if not rev.items:
            yield base + ["", "", "", ""] + tail
        for it in rev.items:
            yield base + [keys.get(it["proposed_id"], ""), it["verdict"],
                          week(s, it["start_idx"]), week(s, it["end_idx"])] + tail
