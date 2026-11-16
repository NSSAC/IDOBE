"""One row per series combining expert answers with the latest model score."""

from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..analysis import current_annotations, majority_label
from ..models import ModelPrediction, Outbreak

RESULT_COLUMNS = [
    "outbreak_id", "disease", "location", "event", "start_date", "end_date",
    "n_outbreak", "n_not_outbreak", "n_unsure", "expert_label", "p_outbreak", "uncertainty",
    "final_label", "label_source",
]


def series_results(db: Session) -> list[dict]:
    """Expert majority where there is one (ties and Unsure-only don't count), otherwise the model's call."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for ann, _user, _outbreak in current_annotations(db):
        if ann.label != "skip":
            votes[ann.outbreak_id][ann.label] += 1
    preds = {p.outbreak_id: p for p in db.execute(select(ModelPrediction)).scalars()}
    rows = []
    for o in db.execute(select(Outbreak).order_by(Outbreak.id)).scalars():
        c = votes.get(o.id, Counter())
        expert = majority_label(Counter({k: c[k] for k in ("outbreak", "not_outbreak") if c[k]}))
        pred = preds.get(o.id)
        if expert:
            final, source = expert, "expert"
        elif pred is not None:
            final, source = ("outbreak" if pred.p_outbreak >= 0.5 else "not_outbreak"), "model"
        else:
            final, source = None, None
        rows.append({
            "outbreak_id": o.id, "disease": o.disease, "location": o.location, "event": o.event,
            "start_date": o.start_date, "end_date": o.end_date,
            "n_outbreak": c["outbreak"], "n_not_outbreak": c["not_outbreak"], "n_unsure": c["unsure"],
            "expert_label": expert,
            "p_outbreak": round(pred.p_outbreak, 4) if pred else None,
            "uncertainty": round(pred.uncertainty, 4) if pred else None,
            "final_label": final, "label_source": source,
        })
    return rows
