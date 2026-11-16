"""Chooses the next outbreak to show an expert."""

import random

from sqlalchemy import and_, func, not_, or_, select
from sqlalchemy.orm import Session

from .analysis import vote_counts_subquery
from .config import settings
from .models import Annotation, ModelPrediction, ModelRun, Outbreak


def _candidates(user_id: int, target: int, settled_below: float | None = None):
    """Outbreaks this user hasn't labeled or skipped that still have fewer than `target` labels.

    With `settled_below`, also drop outbreaks the model is confident about (uncertainty below the
    threshold) whose labels so far all agree with it: one confirming expert is enough for those.
    """
    counts = vote_counts_subquery()
    n = func.coalesce(counts.c.n, 0)
    seen = select(Annotation.outbreak_id).where(Annotation.user_id == user_id)
    uncertainty = func.coalesce(ModelPrediction.uncertainty, 1.0)
    stmt = (
        select(Outbreak.id, n.label("n"), uncertainty.label("uncertainty"))
        .outerjoin(counts, counts.c.outbreak_id == Outbreak.id)
        .outerjoin(ModelPrediction, ModelPrediction.outbreak_id == Outbreak.id)
        .where(n < target, Outbreak.id.not_in(seen))
    )
    if settled_below is not None:
        p = func.coalesce(ModelPrediction.p_outbreak, 0.5)
        agrees = or_(
            and_(p >= 0.5, func.coalesce(counts.c.n_outbreak, 0) == n),
            and_(p < 0.5, func.coalesce(counts.c.n_not_outbreak, 0) == n),
        )
        stmt = stmt.where(not_(and_(n >= 1, uncertainty < settled_below, agrees)))
    return stmt, n


def early_stop_active(run: ModelRun | None) -> bool:
    """Whether one expert agreeing with a confident model is enough (see Settings)."""
    if run is None:
        return False
    cv_auc = (run.metrics or {}).get("cv_auc") or 0
    return run.n_votes >= settings.early_stop_min_labels and cv_auc >= settings.early_stop_min_auc


def _latest_run(db: Session) -> ModelRun | None:
    return db.execute(select(ModelRun).order_by(ModelRun.id.desc()).limit(1)).scalar_one_or_none()


def pick_next(db: Session, user_id: int, target: int, order: str = "active", *,
              random_fraction: float | None = None, top_k: int | None = None,
              rng: random.Random = random) -> tuple[int | None, str, int | None]:
    """Returns (outbreak id or None, why it was picked, model run id used).

    order="active": random until a model has been trained; then, with probability `random_fraction`,
        a random series, otherwise a random pick among the `top_k` series the model is least sure about.
    order="random": any eligible series, uniformly.
    order="spread": least-labeled series first, so coverage grows evenly.
    order="complete": most-labeled series first, so series reach `target` sooner.
    Ties are broken at random so experts working at the same time rarely get the same series.
    """
    if order == "active":
        run = _latest_run(db)
        if run is not None:
            random_fraction = settings.active_random_fraction if random_fraction is None else random_fraction
            top_k = settings.active_top_k if top_k is None else top_k
            settled = settings.confident_uncertainty if early_stop_active(run) else None
            stmt, _ = _candidates(user_id, target, settled)
            if rng.random() < random_fraction:
                oid = db.execute(stmt.order_by(func.random()).limit(1)).scalar_one_or_none()
                return oid, "random", run.id
            top = stmt.order_by(stmt.selected_columns.uncertainty.desc()).limit(top_k).subquery()
            oid = db.execute(select(top.c.id).order_by(func.random()).limit(1)).scalar_one_or_none()
            return oid, "uncertain", run.id
        order = "random"

    stmt, n = _candidates(user_id, target)
    if order == "random":
        ordering = (func.random(),)
    else:
        ordering = (n.desc() if order == "complete" else n.asc(), func.random())
    return db.execute(stmt.order_by(*ordering).limit(1)).scalar_one_or_none(), "random", None


def next_outbreak_id(db: Session, user_id: int, target: int, order: str = "random") -> int | None:
    return pick_next(db, user_id, target, order)[0]


def queue_size(db: Session, user_id: int, target: int) -> int:
    stmt, _ = _candidates(user_id, target)
    return db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
