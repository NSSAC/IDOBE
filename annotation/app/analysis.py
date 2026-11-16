"""Queries over current (latest-per-user) annotations and inter-annotator agreement.

Shared by the admin API and scripts/export_annotations.py.
"""

from collections import Counter, defaultdict

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .models import VOTE_LABELS, Annotation, Outbreak, User


def latest_ids_subquery():
    """Ids of each user's current annotation per outbreak (append-only table, so highest id wins)."""
    return (
        select(func.max(Annotation.id).label("id"))
        .group_by(Annotation.outbreak_id, Annotation.user_id)
        .subquery()
    )


def vote_counts_subquery():
    """Per outbreak, how many users currently hold a non-skip label, and how many of each answer."""
    latest = latest_ids_subquery()
    return (
        select(
            Annotation.outbreak_id,
            func.count().label("n"),
            func.sum(case((Annotation.label == "outbreak", 1), else_=0)).label("n_outbreak"),
            func.sum(case((Annotation.label == "not_outbreak", 1), else_=0)).label("n_not_outbreak"),
        )
        .join(latest, Annotation.id == latest.c.id)
        .where(Annotation.label != "skip")
        .group_by(Annotation.outbreak_id)
        .subquery()
    )


def current_annotations(db: Session):
    """Rows of (Annotation, User, Outbreak) for every user's current answer, ordered by outbreak."""
    latest = latest_ids_subquery()
    stmt = (
        select(Annotation, User, Outbreak)
        .join(latest, Annotation.id == latest.c.id)
        .join(User, User.id == Annotation.user_id)
        .join(Outbreak, Outbreak.id == Annotation.outbreak_id)
        .order_by(Annotation.outbreak_id, Annotation.user_id)
    )
    return db.execute(stmt).all()


def fleiss_kappa(items: list[Counter]) -> float | None:
    """Fleiss' kappa over items rated by >= 2 raters; allows a different rater count per item."""
    items = [c for c in items if sum(c.values()) >= 2]
    if not items:
        return None
    totals: Counter = Counter()
    p_items = []
    for c in items:
        n = sum(c.values())
        totals.update(c)
        p_items.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    grand = sum(totals.values())
    p_bar = sum(p_items) / len(p_items)
    p_e = sum((totals[k] / grand) ** 2 for k in totals)
    if p_e >= 1:
        return None
    return (p_bar - p_e) / (1 - p_e)


def agreement_summary(rows) -> dict:
    """Kappa and majority-vote stats, overall and per disease, from current_annotations() rows."""
    by_outbreak: dict[int, Counter] = defaultdict(Counter)
    disease_of: dict[int, str] = {}
    for ann, _user, outbreak in rows:
        if ann.label in VOTE_LABELS:
            by_outbreak[ann.outbreak_id][ann.label] += 1
            disease_of[ann.outbreak_id] = outbreak.disease

    def summarize(ids) -> dict:
        multi = [by_outbreak[i] for i in ids if sum(by_outbreak[i].values()) >= 2]
        unanimous = sum(1 for c in multi if len(c) == 1)
        return {
            "outbreaks_with_2plus_labels": len(multi),
            "unanimous": unanimous,
            "fleiss_kappa": fleiss_kappa(multi),
        }

    per_disease: dict[str, list[int]] = defaultdict(list)
    for oid, disease in disease_of.items():
        per_disease[disease].append(oid)
    return {
        "overall": summarize(by_outbreak.keys()),
        "by_disease": {d: summarize(ids) for d, ids in sorted(per_disease.items())},
    }


def majority_label(counts: Counter) -> str | None:
    """Most common vote label, or None if there are no votes or the top labels are tied."""
    top = counts.most_common(2)
    if not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return None
    return top[0][0]
