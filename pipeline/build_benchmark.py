"""Cut accepted outbreak windows out of the full series and write the IDOBE benchmark CSV.

    python -m pipeline.build_benchmark SERIES.csv SEGMENTS.csv --out outbreaks.csv \
        [--pad 0] [--min-score 0.5]

Writes the existing wide schema (so baselines, stats and the synthetic generator run unchanged)
plus a sidecar `<out>.provenance.csv` mapping each unique_id to its series, segment and method.
`duration` is the length of the accepted outbreak in weeks; `--pad` adds context weeks on each
side of the stored values (the published v1 windows carry about 9 such weeks in total).
Windows longer than 60 weeks after padding are skipped and reported.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .formats import BENCHMARK_COLUMNS, MAX_WEEKS, read_segments, read_series


def build(series, segments, *, pad: int = 0, min_score: float | None = None):
    rows, prov, skipped = [], [], []
    for seg in sorted(segments, key=lambda s: (s.series_id, s.start_date)):
        if min_score is not None and (seg.score is None or seg.score < min_score):
            continue
        s = series.get(seg.series_id)
        if s is None:
            skipped.append((seg.segment_id, "series not found"))
            continue
        a, b = s.index_of(seg.start_date), s.index_of(seg.end_date)
        lo, hi = max(0, a - pad), min(len(s.values) - 1, b + pad)
        vals = s.values[lo:hi + 1]
        if len(vals) > MAX_WEEKS:
            skipped.append((seg.segment_id, f"{len(vals)} weeks > {MAX_WEEKS}"))
            continue
        uid = len(rows) + 1
        total = sum(v for v in vals if v is not None)
        rows.append([uid, s.disease, s.location, s.event, s.dates[lo].isoformat(), s.dates[hi].isoformat(),
                     b - a + 1, total] + [("" if v is None else v) for v in vals] + [""] * (MAX_WEEKS - len(vals)))
        prov.append([uid, s.series_id, s.source, s.geo_level, seg.segment_id, seg.method, seg.method_version,
                     "" if seg.score is None else seg.score])
    return rows, prov, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", type=Path)
    ap.add_argument("segments", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--min-score", type=float)
    a = ap.parse_args()
    rows, prov, skipped = build(read_series(a.series), read_segments(a.segments), pad=a.pad, min_score=a.min_score)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BENCHMARK_COLUMNS)
        w.writerows(rows)
    with open(a.out.with_suffix(".provenance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "series_id", "source", "geo_level", "segment_id", "method", "method_version", "score"])
        w.writerows(prov)
    print(f"{len(rows)} outbreaks -> {a.out}; skipped {len(skipped)}")
    for sid, why in skipped[:20]:
        print(f"  skipped {sid}: {why}")


if __name__ == "__main__":
    main()
