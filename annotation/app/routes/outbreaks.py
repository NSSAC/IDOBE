from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import settings
from ..db import get_db
from ..models import Annotation, Assignment, Outbreak, User
from ..queue import pick_next
from ..schemas import OutbreakOut

router = APIRouter(prefix="/api", tags=["outbreaks"])


def my_current_annotation(db: Session, user_id: int, outbreak_id: int) -> Annotation | None:
    return db.execute(
        select(Annotation)
        .where(Annotation.user_id == user_id, Annotation.outbreak_id == outbreak_id)
        .order_by(Annotation.id.desc())
        .limit(1)
    ).scalar_one_or_none()


@router.get("/next")
def next_outbreak(user: User = Depends(current_user), db: Session = Depends(get_db)):
    oid, reason, run_id = pick_next(db, user.id, settings.target_annotations, settings.queue_order)
    if oid is None:
        return {"done": True, "outbreak": None}
    # The reason is logged, never shown: experts must not know the model is unsure about a series.
    db.add(Assignment(user_id=user.id, outbreak_id=oid, reason=reason, model_run_id=run_id))
    db.commit()
    return {"done": False, "outbreak": OutbreakOut.of(db.get(Outbreak, oid))}


@router.get("/outbreaks/{outbreak_id}", response_model=OutbreakOut)
def get_outbreak(outbreak_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    outbreak = db.get(Outbreak, outbreak_id)
    if outbreak is None:
        raise HTTPException(404, "Outbreak not found")
    return OutbreakOut.of(outbreak, my_current_annotation(db, user.id, outbreak_id))
