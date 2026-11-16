"""Load (or refresh) outbreak time series from the IDOBE CSV into the database.

Idempotent: rows are keyed on unique_id, so re-running only inserts new outbreaks and updates
changed ones. Annotations are never touched.

    python scripts/load_outbreaks.py [--csv PATH]
"""

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import _path  # noqa: F401
from sqlalchemy import select

from app.config import REPO_ROOT
from app.db import SessionLocal
from app.models import Annotation, Outbreak, utcnow

DEFAULT_CSV = REPO_ROOT / "raw_data" / "outbreaks_disease_location.csv"
WEEK_COLUMNS = [str(i) for i in range(60)]
FIELDS = ("disease", "location", "event", "start_date", "end_date", "duration", "total", "values")


def parse_values(row: dict) -> list[float | None]:
    raw = [row.get(c, "") for c in WEEK_COLUMNS]
    while raw and raw[-1] == "":
        raw.pop()
    return [float(v) if v != "" else None for v in raw]


def parse_row(row: dict, source_file: str) -> dict:
    start = date.fromisoformat(row["start_date"])
    end = date.fromisoformat(row["end_date"])
    values = parse_values(row)
    if not values:
        raise ValueError("no weekly values")
    span = (end - start).days // 7 + 1
    # Value i is the week ending start_date + 7*i, so the series can't run past end_date.
    if start + timedelta(weeks=len(values) - 1) > end:
        raise ValueError(f"{len(values)} values but start..end spans only {span} weeks")
    return {
        "id": int(row["unique_id"]),
        "disease": row["disease"].strip(),
        "location": row["location"].strip(),
        "event": row["event"].strip(),
        "start_date": start,
        "end_date": end,
        "duration": int(row["duration"]),
        "total": float(row["total"]) if row.get("total") else None,
        "values": values,
        "source_file": source_file,
        "_short": len(values) < span,
    }


def load(csv_path: Path, db) -> dict:
    try:
        source_file = str(csv_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source_file = csv_path.name
    with open(csv_path, newline="") as f:
        records = [parse_row(r, source_file) for r in csv.DictReader(f)]

    existing = {o.id: o for o in db.execute(select(Outbreak)).scalars()}
    annotated = set(db.execute(select(Annotation.outbreak_id).distinct()).scalars())
    stats = {"rows": len(records), "inserted": 0, "updated": 0, "unchanged": 0,
             "short_series": 0, "updated_but_annotated": []}
    for rec in records:
        stats["short_series"] += rec.pop("_short")
        current = existing.get(rec["id"])
        if current is None:
            db.add(Outbreak(**rec))
            stats["inserted"] += 1
        elif any(getattr(current, k) != rec[k] for k in FIELDS):
            for k in FIELDS + ("source_file",):
                setattr(current, k, rec[k])
            current.loaded_at = utcnow()
            stats["updated"] += 1
            if rec["id"] in annotated:
                stats["updated_but_annotated"].append(rec["id"])
        else:
            stats["unchanged"] += 1
    db.commit()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    with SessionLocal() as db:
        stats = load(args.csv, db)
    changed = stats.pop("updated_but_annotated")
    print(", ".join(f"{k}={v}" for k, v in stats.items()))
    if stats["short_series"]:
        print(f"note: {stats['short_series']} series end before their end_date (data stops early); "
              "they are loaded as-is.")
    if changed:
        print(f"WARNING: {len(changed)} already-annotated outbreaks changed: {changed[:20]}"
              f"{' ...' if len(changed) > 20 else ''}")


if __name__ == "__main__":
    main()
