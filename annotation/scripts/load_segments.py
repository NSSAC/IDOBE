"""Load full series and a segmentation method's proposed outbreaks into the app for expert review.

    # Files from the pipeline (see pipeline/formats.py):
    python scripts/load_segments.py --series SERIES.csv --segments SEGMENTS.csv --run-name NAME
    # The current published windows (epidemicKabu), rebuilt into full series:
    python scripts/load_segments.py --legacy [--diseases COVID-19,INFLUENZA,ILI,RSV]
    # Run a registered method on the series first, then load its output:
    python scripts/load_segments.py --series SERIES.csv --method seasonal-threshold

Series are keyed on series_id and shared across runs; a series whose values change is updated
(and reported if it already has reviews). Each run is loaded once; use a new --run-name for a new
method or parameter setting. Reviews are never touched.
"""

import argparse
import sys
from pathlib import Path

import _path  # noqa: F401
from sqlalchemy import select

from app.config import REPO_ROOT
from app.db import SessionLocal
from app.models import ProposedSegment, RunSeries, SegmentationRun, SegmentReview, Series, utcnow

sys.path.insert(0, str(REPO_ROOT))
from pipeline.formats import read_segments, read_series  # noqa: E402
from pipeline.segment import run as run_method  # noqa: E402
from pipeline.segmentation import METHODS, legacy  # noqa: E402

DEFAULT_LEGACY_CSV = REPO_ROOT / "raw_data" / "outbreaks_disease_location.csv"
RESPIRATORY = "COVID-19,INFLUENZA,ILI,RSV"


def upsert_series(db, series: dict) -> tuple[dict[str, Series], list[str]]:
    existing = {s.series_key: s for s in db.execute(
        select(Series).where(Series.series_key.in_(list(series)))).scalars()}
    reviewed = set(db.execute(select(Series.series_key).join(SegmentReview, SegmentReview.series_id == Series.id)
                              .distinct()).scalars())
    changed = []
    out = {}
    for key, s in series.items():
        fields = dict(source=s.source or None, disease=s.disease, location=s.location,
                      geo_level=s.geo_level or None, event=s.event, start_date=s.start_date, values=s.values)
        row = existing.get(key)
        if row is None:
            row = Series(series_key=key, **fields)
            db.add(row)
        elif any(getattr(row, k) != v for k, v in fields.items()):
            for k, v in fields.items():
                setattr(row, k, v)
            row.loaded_at = utcnow()
            if key in reviewed:
                changed.append(key)
        out[key] = row
    db.flush()
    return out, changed


def load(db, series: dict, segments: list, run_name: str, source_file: str | None) -> dict:
    if db.execute(select(SegmentationRun).where(SegmentationRun.name == run_name)).scalar_one_or_none():
        raise SystemExit(f"run {run_name!r} already exists; choose another --run-name")
    methods = {(s.method, s.method_version) for s in segments}
    if len(methods) > 1:
        raise SystemExit(f"segments file mixes methods {sorted(methods)}; load one method per run")
    method, version = methods.pop() if methods else ("unknown", "")
    rows, changed = upsert_series(db, series)
    run = SegmentationRun(name=run_name, method=method, method_version=version or None, source_file=source_file)
    db.add(run)
    db.flush()
    db.add_all(RunSeries(run_id=run.id, series_id=r.id) for r in rows.values())
    missing = 0
    for seg in segments:
        s = series.get(seg.series_id)
        if s is None:
            missing += 1
            continue
        idx = lambda d: None if d is None else s.index_of(d)  # noqa: E731
        db.add(ProposedSegment(run_id=run.id, series_id=rows[seg.series_id].id, segment_key=seg.segment_id,
                               start_idx=idx(seg.start_date), end_idx=idx(seg.end_date), score=seg.score,
                               peak_idx=idx(seg.peak_date), core_start_idx=idx(seg.core_start_date),
                               core_end_idx=idx(seg.core_end_date)))
    db.commit()
    return {"run": run_name, "run_id": run.id, "series": len(series), "segments": len(segments) - missing,
            "segments_without_series": missing, "changed_reviewed_series": changed}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", type=Path)
    ap.add_argument("--segments", type=Path)
    ap.add_argument("--method", choices=sorted(METHODS), help="segment --series with this method first")
    ap.add_argument("--legacy", nargs="?", const=DEFAULT_LEGACY_CSV, type=Path,
                    help="rebuild series and windows from the published wide CSV")
    ap.add_argument("--diseases", default=RESPIRATORY, help="with --legacy; 'all' keeps every disease")
    ap.add_argument("--run-name")
    a = ap.parse_args()

    if a.legacy:
        diseases = None if a.diseases.lower() == "all" else {d.strip().upper() for d in a.diseases.split(",")}
        series, segments, stats = legacy.stitch(a.legacy, diseases)
        print(", ".join(f"{k}={v}" for k, v in stats.items()))
        name, src = a.run_name or f"{legacy.METHOD} ({a.diseases})", a.legacy.name
    elif a.series and (a.segments or a.method):
        series = read_series(a.series)
        segments = read_segments(a.segments) if a.segments else run_method(series, a.method, {})
        name = a.run_name or (a.segments.stem if a.segments else f"{a.method} on {a.series.stem}")
        src = (a.segments or a.series).name
    else:
        ap.error("give --legacy, or --series with --segments or --method")
    with SessionLocal() as db:
        stats = load(db, series, segments, name, src)
    changed = stats.pop("changed_reviewed_series")
    print(", ".join(f"{k}={v}" for k, v in stats.items()))
    if changed:
        print(f"WARNING: {len(changed)} already-reviewed series changed: {changed[:10]}")


if __name__ == "__main__":
    main()
