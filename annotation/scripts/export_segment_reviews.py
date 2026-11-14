"""Export each expert's current segment reviews and the expert consensus windows.

Writes into --out-dir (default annotation/exports/):
  segment_reviews.csv     one row per expert decision (accept/adjust/reject/added)
  consensus_segments.csv  accepted windows in the pipeline segments format, ready for
                          `python -m pipeline.build_benchmark SERIES.csv consensus_segments.csv`

    python scripts/export_segment_reviews.py [--run-id N] [--min-reviewers 2] [--out-dir DIR]
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import _path  # noqa: F401

from app.config import ANNOTATION_DIR, REPO_ROOT
from app.db import SessionLocal
from app.segments import REVIEW_EXPORT_COLUMNS, export_rows

sys.path.insert(0, str(REPO_ROOT))
from pipeline.consensus import consensus  # noqa: E402
from pipeline.formats import write_segments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--min-reviewers", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=ANNOTATION_DIR / "exports")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        rows = list(export_rows(db, a.run_id))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(REVIEW_EXPORT_COLUMNS)
    w.writerows(rows)
    (a.out_dir / "segment_reviews.csv").write_text(buf.getvalue())
    records = list(csv.DictReader(io.StringIO(buf.getvalue())))
    segs, stats = consensus(records, min_reviewers=a.min_reviewers)
    write_segments(a.out_dir / "consensus_segments.csv", segs, ["n_reviewers", "source_segment_id", "origin"])
    print(f"{len(rows)} decisions; {len(segs)} consensus segments;",
          ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
