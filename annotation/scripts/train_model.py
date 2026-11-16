"""Train the active-learning model now and print how it did.

The web app also retrains on its own every RETRAIN_EVERY answers; use this for a manual run or cron.

    python scripts/train_model.py
"""

import argparse
import json

import _path  # noqa: F401

from app.db import SessionLocal
from app.ml.train import readiness, retrain_if_due


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run = retrain_if_due(force=True)
    if run is None:
        with SessionLocal() as db:
            r = readiness(db)
        if not r["ready"]:
            print(f"Not enough labels yet: {r['n_outbreak']} Outbreak and {r['n_not_outbreak']} Not outbreak answers; "
                  f"need {r['min_labels']} in total with at least {r['min_per_class']} of each.")
        else:
            print("Training failed or is already running; see the log.")
        return
    m = run.metrics
    print(f"run {run.id}: {run.model_type} on {run.n_votes} answers ({run.n_outbreaks} series) in {run.train_seconds}s")
    print("cross-validation:", json.dumps(m["cv"], indent=2))
    print("evaluation vs expert majority:", json.dumps(m["evaluation"], indent=2))
    p = m["predicted"]
    print(f"predicted outbreak: {p['n_predicted_outbreak']} of {p['n']}; uncertain (0.25<p<0.75): {p['n_uncertain']}")


if __name__ == "__main__":
    main()
