"""Train on expert answers, score every series, and store the scores for the queue to use.

Each expert's current Outbreak / Not outbreak answer is one training row, weighted by its confidence
(Unsure and Skip are left out). Two small models are cross-validated with folds grouped by series and
the one with the lower log loss is kept. Accuracy is reported twice: on all labeled series (which
over-represents hard cases, because the queue seeks them out) and on series whose first label came from
a random draw. The latter describes the sampled eligible pool, not a fixed population holdout.
"""

import logging
import threading
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from ..analysis import current_annotations
from ..config import settings
from ..db import SessionLocal
from ..models import Annotation, ModelPrediction, ModelRun, Outbreak
from .features import feature_matrix

log = logging.getLogger(__name__)

CANDIDATES = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
    "boosted_trees": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10, l2_regularization=1.0,
        random_state=0,
    ),
}

_feature_cache: dict = {}
_train_lock = threading.Lock()


def _fit(name: str, X, y, w):
    model = CANDIDATES[name]()
    if name == "logistic":
        return model.fit(X, y, logisticregression__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


def features(db: Session):
    """Feature matrix for all series, cached until the outbreaks table changes."""
    key = tuple(db.execute(select(func.count(), func.max(Outbreak.loaded_at)).select_from(Outbreak)).one())
    if _feature_cache.get("key") != key:
        outbreaks = db.execute(select(Outbreak).order_by(Outbreak.id)).scalars().all()
        _feature_cache.update(key=key, value=feature_matrix(outbreaks))
    return _feature_cache["value"]


def collect_votes(db: Session) -> list[tuple[int, int, float]]:
    """(outbreak_id, 1 if Outbreak else 0, weight) for each expert's current Outbreak / Not outbreak answer."""
    return [
        (a.outbreak_id, int(a.label == "outbreak"), (a.confidence or 3) / 5)
        for a, _user, _outbreak in current_annotations(db)
        if a.label in ("outbreak", "not_outbreak")
    ]


def first_label_source(db: Session) -> dict[int, str | None]:
    """How each labeled series was first served (its earliest non-skip annotation's served_as)."""
    first = (
        select(func.min(Annotation.id).label("id"))
        .where(Annotation.label != "skip")
        .group_by(Annotation.outbreak_id)
        .subquery()
    )
    return dict(db.execute(select(Annotation.outbreak_id, Annotation.served_as).join(first, Annotation.id == first.c.id)).all())


def readiness(db: Session) -> dict:
    votes = collect_votes(db)
    n_pos = sum(v[1] for v in votes)
    n_neg = len(votes) - n_pos
    return {
        "n_votes": len(votes), "n_outbreak": n_pos, "n_not_outbreak": n_neg,
        "min_labels": settings.model_min_labels, "min_per_class": settings.model_min_per_class,
        "ready": len(votes) >= settings.model_min_labels and min(n_pos, n_neg) >= settings.model_min_per_class,
    }


def _scores(y_true: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    out = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    out["accuracy"] = float(np.average((p >= 0.5) == y_true, weights=w))
    out["share_outbreak"] = float(np.average(y_true, weights=w))
    if len(set(y_true.tolist())) == 2:
        out["auc"] = float(roc_auc_score(y_true, p, sample_weight=w))
        out["log_loss"] = float(log_loss(y_true, np.clip(p, 1e-4, 1 - 1e-4), sample_weight=w))
    return out


def train_and_predict(db: Session) -> ModelRun | None:
    """Train if there are enough labels; store the run and replace all predictions. Returns the run."""
    if not readiness(db)["ready"]:
        return None
    started = time.perf_counter()
    ids, X_all, names = features(db)
    row_of = {oid: i for i, oid in enumerate(ids)}
    votes = [v for v in collect_votes(db) if v[0] in row_of]
    groups = np.array([v[0] for v in votes])
    y = np.array([v[1] for v in votes])
    w = np.array([v[2] for v in votes])
    X = X_all[[row_of[g] for g in groups]]

    # Cross-validate each candidate; folds never split one series' answers.
    pos_series = len(set(groups[y == 1].tolist()))
    neg_series = len(set(groups[y == 0].tolist()))
    n_splits = min(5, pos_series, neg_series)
    cv, oof = {}, {}
    if n_splits >= 2:
        splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y, groups))
        for name in CANDIDATES:
            p = np.empty(len(y))
            for train_idx, test_idx in splits:
                if len(set(y[train_idx].tolist())) < 2:
                    p[test_idx] = y[train_idx].mean()
                else:
                    p[test_idx] = _fit(name, X[train_idx], y[train_idx], w[train_idx]).predict_proba(X[test_idx])[:, 1]
            oof[name] = p
            cv[name] = _scores(y, p, w)
        best = min(cv, key=lambda k: cv[k].get("log_loss", float("inf")))
    else:
        best = "logistic"

    model = _fit(best, X, y, w)
    p_all = model.predict_proba(X_all)[:, 1]
    uncertainty = 1 - np.abs(2 * p_all - 1)

    # Series-level evaluation against the expert majority, using out-of-fold predictions.
    tally: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    oof_by_series: dict[int, float] = {}
    for i, (g, label) in enumerate(zip(groups.tolist(), y.tolist())):
        tally[g][label] += 1
        if best in oof:
            oof_by_series[g] = float(oof[best][i])
    consensus = {g: int(c[1] > c[0]) for g, c in tally.items() if c[0] != c[1]}
    source = first_label_source(db)
    evaluation = {}
    for subset, keep in (("labeled", lambda g: True), ("random_sample", lambda g: source.get(g) == "random")):
        gs = [g for g in consensus if keep(g) and g in oof_by_series]
        evaluation[subset] = _scores(np.array([consensus[g] for g in gs]), np.array([oof_by_series[g] for g in gs]))

    top_features = []
    if best == "logistic":
        coefs = model.named_steps["logisticregression"].coef_[0]
        top_features = [{"feature": names[i], "weight": float(coefs[i])} for i in np.argsort(-np.abs(coefs))[:10]]

    metrics = {
        "selected": best,
        "cv": cv,
        "cv_auc": cv.get(best, {}).get("auc"),
        "evaluation": evaluation,
        "predicted": {
            "n": int(len(p_all)),
            "mean_p": float(p_all.mean()),
            "n_predicted_outbreak": int((p_all >= 0.5).sum()),
            "n_uncertain": int((uncertainty >= 0.5).sum()),     # 0.25 < p < 0.75
            "histogram": np.histogram(p_all, bins=10, range=(0, 1))[0].tolist(),
        },
        "top_features": top_features,
    }
    run = ModelRun(model_type=best, n_votes=int(len(y)), n_outbreaks=int(len(set(groups.tolist()))),
                   train_seconds=round(time.perf_counter() - started, 3), metrics=metrics)
    db.add(run)
    db.flush()
    db.execute(delete(ModelPrediction))
    db.execute(insert(ModelPrediction), [
        {"outbreak_id": oid, "run_id": run.id, "p_outbreak": float(p), "uncertainty": float(u)}
        for oid, p, u in zip(ids, p_all, uncertainty)
    ])
    db.commit()
    log.info("model run %s: %s on %s answers, cv auc %s", run.id, best, run.n_votes, metrics["cv_auc"])
    return run


def latest_run(db: Session) -> ModelRun | None:
    return db.execute(select(ModelRun).order_by(ModelRun.id.desc()).limit(1)).scalar_one_or_none()


def retrain_if_due(force: bool = False) -> ModelRun | None:
    """Retrain when `retrain_every` new answers have arrived since the last run (or always, if forced).

    Safe to call after every annotation: returns immediately if a training run is already in progress.
    """
    if not _train_lock.acquire(blocking=False):
        return None
    try:
        with SessionLocal() as db:
            if not force:
                last = latest_run(db)
                if last is not None and len(collect_votes(db)) - last.n_votes < settings.retrain_every:
                    return None
            return train_and_predict(db)
    except Exception:
        log.exception("model training failed")
        return None
    finally:
        _train_lock.release()


def is_training() -> bool:
    return _train_lock.locked()
