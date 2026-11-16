"""Export annotations and agreement statistics.

Writes, into --out-dir (default: annotation/exports/):
  annotations_current.csv   each expert's current answer per outbreak
  annotations_all.csv       every row ever written, including edits (the audit trail)
  consensus.csv             per outbreak: vote counts, majority label, mean confidence
  agreement.json            Fleiss' kappa and unanimity, overall and per disease
  series_labels.csv         every series: expert answers, model score, final label and its source
  model.json                the latest model run's metrics (if a model has been trained)

    python scripts/export_annotations.py [--out-dir DIR]
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import _path  # noqa: F401
from sqlalchemy import select

from app.analysis import agreement_summary, current_annotations, majority_label
from app.config import ANNOTATION_DIR
from app.db import SessionLocal
from app.ml.results import RESULT_COLUMNS, series_results
from app.ml.train import latest_run
from app.models import VOTE_LABELS, Annotation, Outbreak, User
from app.routes.admin import EXPORT_COLUMNS, export_row


def write_csv(path: Path, header: list, rows) -> int:
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def consensus_rows(current):
    votes: dict[int, Counter] = defaultdict(Counter)
    confidences: dict[int, list[int]] = defaultdict(list)
    meta: dict[int, Outbreak] = {}
    for ann, _user, outbreak in current:
        meta[ann.outbreak_id] = outbreak
        if ann.label in VOTE_LABELS:
            votes[ann.outbreak_id][ann.label] += 1
            confidences[ann.outbreak_id].append(ann.confidence)
    for oid in sorted(votes):
        c, o, conf = votes[oid], meta[oid], confidences[oid]
        yield [oid, o.disease, o.location, o.event, o.start_date, sum(c.values()),
               *(c[label] for label in VOTE_LABELS), majority_label(c),
               round(sum(conf) / len(conf), 2)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ANNOTATION_DIR / "exports")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        current = current_annotations(db)
        everything = db.execute(
            select(Annotation, User, Outbreak).join(User).join(Outbreak).order_by(Annotation.id)
        ).all()
        results = series_results(db)
        run = latest_run(db)

    n_cur = write_csv(args.out_dir / "annotations_current.csv", EXPORT_COLUMNS, (export_row(*r) for r in current))
    n_all = write_csv(args.out_dir / "annotations_all.csv", EXPORT_COLUMNS, (export_row(*r) for r in everything))
    n_con = write_csv(
        args.out_dir / "consensus.csv",
        ["outbreak_id", "disease", "location", "event", "start_date", "n_votes",
         *(f"n_{label}" for label in VOTE_LABELS), "majority_label", "mean_confidence"],
        consensus_rows(current),
    )
    agreement = agreement_summary(current)
    (args.out_dir / "agreement.json").write_text(json.dumps(agreement, indent=2))

    write_csv(args.out_dir / "series_labels.csv", RESULT_COLUMNS, ([r[c] for c in RESULT_COLUMNS] for r in results))
    if run is not None:
        (args.out_dir / "model.json").write_text(json.dumps(
            {"run_id": run.id, "created_at": run.created_at.isoformat(), "model_type": run.model_type,
             "n_votes": run.n_votes, "n_outbreaks": run.n_outbreaks, "metrics": run.metrics}, indent=2))
    sources = Counter(r["label_source"] for r in results)

    print(f"wrote {args.out_dir}: {n_cur} current, {n_all} total rows, {n_con} outbreaks in consensus")
    print(f"final labels: {sources['expert']} from experts, {sources['model']} from the model, "
          f"{sources[None]} unlabeled")
    overall = agreement["overall"]
    kappa = overall["fleiss_kappa"]
    print(f"outbreaks with 2+ labels: {overall['outbreaks_with_2plus_labels']}, "
          f"unanimous: {overall['unanimous']}, Fleiss' kappa: {'n/a' if kappa is None else f'{kappa:.3f}'}")


if __name__ == "__main__":
    main()
