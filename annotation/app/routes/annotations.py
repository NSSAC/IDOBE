from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import __version__
from ..analysis import latest_ids_subquery
from ..auth import current_user
from ..db import get_db
from ..config import settings
from ..ml.train import retrain_if_due
from ..models import Annotation, Assignment, Outbreak, User
from ..schemas import AnnotationIn, AnnotationOut

router = APIRouter(prefix="/api", tags=["annotations"])

MAX_TIME_SPENT_MS = 24 * 3600 * 1000


def served_as(db: Session, user_id: int, outbreak_id: int) -> str | None:
    """Why this series was in front of the expert: a revisit of their own answer, or the queue's reason."""
    mine = (Annotation.user_id == user_id) & (Annotation.outbreak_id == outbreak_id)
    if db.execute(select(Annotation.id).where(mine).limit(1)).first():
        return "revisit"
    reason = db.execute(
        select(Assignment.reason)
        .where(Assignment.user_id == user_id, Assignment.outbreak_id == outbreak_id)
        .order_by(Assignment.id.desc()).limit(1)
    ).scalar_one_or_none()
    return reason or "lookup"


@router.post("/annotations", response_model=AnnotationOut, status_code=201)
def create_annotation(body: AnnotationIn, request: Request, background: BackgroundTasks,
                      user: User = Depends(current_user), db: Session = Depends(get_db)):
    if db.get(Outbreak, body.outbreak_id) is None:
        raise HTTPException(404, "Outbreak not found")
    ann = Annotation(
        served_as=served_as(db, user.id, body.outbreak_id),
        outbreak_id=body.outbreak_id,
        user_id=user.id,
        label=body.label,
        confidence=body.confidence,
        comment=body.comment,
        time_spent_ms=min(body.time_spent_ms, MAX_TIME_SPENT_MS) if body.time_spent_ms is not None else None,
        client_info=(request.headers.get("user-agent") or "")[:512] or None,
        app_version=__version__,
    )
    db.add(ann)
    db.commit()
    if settings.auto_retrain and body.label in ("outbreak", "not_outbreak"):
        background.add_task(retrain_if_due)   # runs after the response; returns at once if not due
    return AnnotationOut.of(ann)


@router.get("/my-annotations")
def my_annotations(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                   user: User = Depends(current_user), db: Session = Depends(get_db)):
    """This user's current answer per outbreak, most recent first."""
    latest = latest_ids_subquery()
    base = (
        select(Annotation, Outbreak)
        .join(latest, Annotation.id == latest.c.id)
        .join(Outbreak, Outbreak.id == Annotation.outbreak_id)
        .where(Annotation.user_id == user.id)
    )
    total = db.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    rows = db.execute(base.order_by(Annotation.id.desc()).offset(offset).limit(limit)).all()
    return {
        "total": total,
        "items": [
            {**AnnotationOut.of(a).model_dump(), "disease": o.disease, "location": o.location,
             "event": o.event, "start_date": o.start_date}
            for a, o in rows
        ],
    }
