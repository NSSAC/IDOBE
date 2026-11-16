import csv
import io
from collections import Counter
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..analysis import agreement_summary, current_annotations, vote_counts_subquery
from ..auth import admin_user, hash_password
from ..config import settings
from ..db import get_db
from ..ml.results import RESULT_COLUMNS, series_results
from ..ml.train import is_training, latest_run, readiness, retrain_if_due
from ..models import Annotation, ModelPrediction, ModelRun, Outbreak, User
from ..queue import early_stop_active
from ..schemas import UserCreateIn, UserUpdateIn

router = APIRouter(prefix="/api/admin", tags=["admin"])

EXPORT_COLUMNS = [
    "annotation_id", "outbreak_id", "disease", "location", "event", "start_date", "end_date",
    "username", "label", "confidence", "comment", "time_spent_ms", "served_as", "created_at", "app_version",
]


def export_row(ann: Annotation, user: User, outbreak: Outbreak) -> list:
    return [
        ann.id, ann.outbreak_id, outbreak.disease, outbreak.location, outbreak.event,
        outbreak.start_date, outbreak.end_date, user.username, ann.label, ann.confidence,
        ann.comment, ann.time_spent_ms, ann.served_as, ann.created_at.isoformat(), ann.app_version,
    ]


@router.get("/stats")
def stats(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    total_outbreaks = db.execute(select(func.count()).select_from(Outbreak)).scalar_one()
    counts = vote_counts_subquery()
    coverage = Counter(dict(db.execute(select(counts.c.n, func.count()).group_by(counts.c.n)).all()))
    coverage[0] = total_outbreaks - sum(coverage.values())

    rows = current_annotations(db)
    per_user: dict[str, Counter] = {}
    last_active: dict[str, str] = {}
    for ann, user, _ in rows:
        per_user.setdefault(user.username, Counter())[ann.label] += 1
    for username, ts in db.execute(
        select(User.username, func.max(Annotation.created_at)).join(Annotation).group_by(User.username)
    ).all():
        last_active[username] = ts.isoformat() if ts else None

    return {
        "target_annotations": settings.target_annotations,
        "total_outbreaks": total_outbreaks,
        "complete": sum(v for k, v in coverage.items() if k >= settings.target_annotations),
        "coverage": {str(k): v for k, v in sorted(coverage.items())},
        "labels": Counter(ann.label for ann, _, _ in rows),
        "annotators": [
            {"username": u, "labels": dict(c), "last_active": last_active.get(u)}
            for u, c in sorted(per_user.items())
        ],
        "agreement": agreement_summary(rows),
    }


@router.get("/export")
def export(scope: Literal["current", "all"] = Query("current"), _: User = Depends(admin_user),
           db: Session = Depends(get_db)):
    """CSV of each user's current answers, or of every row ever written (scope=all)."""
    if scope == "current":
        rows = current_annotations(db)
    else:
        rows = db.execute(
            select(Annotation, User, Outbreak).join(User).join(Outbreak).order_by(Annotation.id)
        ).all()
    return _csv_response(EXPORT_COLUMNS, (export_row(*r) for r in rows), f"annotations_{scope}.csv")


def _csv_response(header: list, rows, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/model")
def model_status(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    run = latest_run(db)
    history = db.execute(select(ModelRun).order_by(ModelRun.id.desc()).limit(20)).scalars().all()
    served = Counter({(source or "unknown"): count for source, count in db.execute(
        select(Annotation.served_as, func.count())
        .where(Annotation.label.in_(("outbreak", "not_outbreak", "unsure")))
        .group_by(Annotation.served_as)
    ).all()})
    counts = vote_counts_subquery()
    n = func.coalesce(counts.c.n, 0)
    uncertain = db.execute(
        select(Outbreak.id, Outbreak.disease, Outbreak.location, Outbreak.event, Outbreak.start_date,
               ModelPrediction.p_outbreak, n)
        .join(ModelPrediction, ModelPrediction.outbreak_id == Outbreak.id)
        .outerjoin(counts, counts.c.outbreak_id == Outbreak.id)
        .where(n < settings.target_annotations)
        .order_by(ModelPrediction.uncertainty.desc()).limit(15)
    ).all()
    return {
        "queue_order": settings.queue_order,
        "readiness": readiness(db),
        "training": is_training(),
        "retrain_every": settings.retrain_every,
        "early_stop": {"active": early_stop_active(run),
                       "min_auc": settings.early_stop_min_auc, "min_labels": settings.early_stop_min_labels,
                       "confident_uncertainty": settings.confident_uncertainty},
        "run": None if run is None else {
            "id": run.id, "created_at": run.created_at.isoformat(), "model_type": run.model_type,
            "n_votes": run.n_votes, "n_outbreaks": run.n_outbreaks, "train_seconds": run.train_seconds,
            "metrics": run.metrics,
        },
        "history": [
            {"id": r.id, "created_at": r.created_at.isoformat(), "model_type": r.model_type, "n_votes": r.n_votes,
             "cv_auc": r.metrics.get("cv_auc"),
             "random_sample": r.metrics.get("evaluation", {}).get("random_sample", {})}
            for r in history
        ],
        "served": served,
        "most_uncertain": [
            {"id": i, "disease": d, "location": loc, "event": e, "start_date": s, "p_outbreak": round(p, 3), "n_labels": k}
            for i, d, loc, e, s, p, k in uncertain
        ],
    }


@router.post("/retrain")
def retrain(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    r = readiness(db)
    if not r["ready"]:
        raise HTTPException(400, f"Not enough labels yet: {r['n_outbreak']} Outbreak and {r['n_not_outbreak']} "
                                 f"Not outbreak answers; need {r['min_labels']} with at least {r['min_per_class']} of each")
    if is_training():
        raise HTTPException(409, "Training is already running")
    run = retrain_if_due(force=True)
    if run is None:
        raise HTTPException(500, "Training failed; see the server log")
    return {"run_id": run.id, "model_type": run.model_type, "cv_auc": run.metrics.get("cv_auc")}


@router.get("/predictions")
def predictions(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    """Every series with its expert answers, model score and final label."""
    rows = series_results(db)
    return _csv_response(RESULT_COLUMNS, ([r[c] for c in RESULT_COLUMNS] for r in rows), "series_labels.csv")


def _user_row(u: User, n_labels: int = 0) -> dict:
    return {"username": u.username, "display_name": u.display_name, "affiliation": u.affiliation,
            "expertise": u.expertise, "is_admin": u.is_admin, "is_active": u.is_active,
            "created_at": u.created_at.isoformat(), "n_labels": n_labels}


@router.get("/users")
def list_users(_: User = Depends(admin_user), db: Session = Depends(get_db)):
    labels = dict(db.execute(
        select(Annotation.user_id, func.count(func.distinct(Annotation.outbreak_id)))
        .where(Annotation.label != "skip").group_by(Annotation.user_id)
    ).all())
    users = db.execute(select(User).order_by(User.username)).scalars()
    return [_user_row(u, labels.get(u.id, 0)) for u in users]


@router.post("/users", status_code=201)
def create_user(body: UserCreateIn, _: User = Depends(admin_user), db: Session = Depends(get_db)):
    if db.execute(select(User.id).where(User.username == body.username)).first():
        raise HTTPException(409, f"User {body.username!r} already exists")
    user = User(username=body.username, password_hash=hash_password(body.password),
                display_name=body.display_name, affiliation=body.affiliation, expertise=body.expertise,
                is_admin=body.is_admin)
    db.add(user)
    db.commit()
    return _user_row(user)


@router.patch("/users/{username}")
def update_user(username: str, body: UserUpdateIn, me: User = Depends(admin_user), db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None:
        raise HTTPException(404, "User not found")
    if user.id == me.id and (body.is_active is False or body.is_admin is False):
        raise HTTPException(400, "You can't deactivate or remove admin rights from your own account")
    if body.password is not None:
        user.password_hash = hash_password(body.password)
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    db.commit()
    return _user_row(user)
