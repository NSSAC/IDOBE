"""Combine expert segment reviews into one set of accepted outbreak windows per series.

Input: the review export written by the annotation app (`annotation/scripts/export_segment_reviews.py`),
one row per reviewer per segment decision:
    series_id, run_method, reviewer, proposed_segment_id, verdict, start_date, end_date, confidence,
    series_unusable
verdict is accept | adjust | reject | added. start/end are the reviewer's final boundaries (the
proposed ones for accept, blank for reject).

Rules (deliberately simple, documented, and easy to change):
- Only series reviewed by at least `min_reviewers` experts are decided; the rest are left out.
- A series is dropped if a majority of its reviewers marked it unusable.
- A proposed segment is kept if a strict majority of the series' reviewers kept it (accept/adjust).
  Its boundaries are the median of the keepers' boundaries, rounded to whole weeks.
- Segments experts added are grouped when they overlap (IoU >= `match_iou`); a group is kept if a
  strict majority of the series' reviewers added it, with median boundaries.
`score` on the output is the fraction of reviewers supporting the segment.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from .formats import SegmentRow, write_segments

METHOD = "expert-consensus"


def _median_date(ds: list[date]) -> date:
    origin = min(ds)
    weeks = sorted((d - origin).days // 7 for d in ds)
    return origin + timedelta(weeks=int(statistics.median_low(weeks)))


def iou(a: tuple[date, date], b: tuple[date, date]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, (hi - lo).days // 7 + 1)
    union = ((a[1] - a[0]).days + (b[1] - b[0]).days) // 7 + 2 - inter
    return inter / union if union else 0.0


def consensus(rows: list[dict], *, min_reviewers: int = 1, match_iou: float = 0.5) -> tuple[list[SegmentRow], dict]:
    by_series: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_series[r["series_id"]].append(r)

    out: list[SegmentRow] = []
    stats = defaultdict(int)
    for sid, rs in sorted(by_series.items()):
        reviewers = {r["reviewer"] for r in rs}
        n = len(reviewers)
        if n < min_reviewers:
            stats["series_too_few_reviewers"] += 1
            continue
        unusable = {r["reviewer"] for r in rs if str(r.get("series_unusable", "")).lower() in ("1", "true", "yes")}
        if len(unusable) * 2 > n:
            stats["series_unusable"] += 1
            continue
        stats["series_decided"] += 1
        method = rs[0].get("run_method", "")

        proposed: dict[str, list[dict]] = defaultdict(list)
        added: list[dict] = []
        for r in rs:
            if r["reviewer"] in unusable:
                continue
            if r["verdict"] == "added":
                added.append(r)
            elif r.get("proposed_segment_id"):
                proposed[r["proposed_segment_id"]].append(r)

        k = 0
        for seg_id, votes in sorted(proposed.items()):
            keep = [v for v in votes if v["verdict"] in ("accept", "adjust")]
            if len(keep) * 2 <= n:
                stats["proposed_rejected"] += 1
                continue
            stats["proposed_kept"] += 1
            stats["proposed_adjusted"] += any(v["verdict"] == "adjust" for v in keep)
            k += 1
            out.append(SegmentRow(
                series_id=sid, segment_id=f"{sid}#c{k}", method=METHOD, method_version=method,
                start_date=_median_date([date.fromisoformat(v["start_date"]) for v in keep]),
                end_date=_median_date([date.fromisoformat(v["end_date"]) for v in keep]),
                score=round(len(keep) / n, 3),
                extra={"n_reviewers": n, "source_segment_id": seg_id, "origin": "proposed"}))

        groups: list[list[dict]] = []
        for r in added:
            span = (date.fromisoformat(r["start_date"]), date.fromisoformat(r["end_date"]))
            for g in groups:
                if r["reviewer"] not in {x["reviewer"] for x in g} and iou(span, g[0]["_span"]) >= match_iou:
                    g.append({**r, "_span": span})
                    break
            else:
                groups.append([{**r, "_span": span}])
        for g in groups:
            if len(g) * 2 <= n:
                stats["added_unconfirmed"] += 1
                continue
            stats["added_kept"] += 1
            k += 1
            out.append(SegmentRow(
                series_id=sid, segment_id=f"{sid}#c{k}", method=METHOD, method_version=method,
                start_date=_median_date([x["_span"][0] for x in g]),
                end_date=_median_date([x["_span"][1] for x in g]),
                score=round(len(g) / n, 3),
                extra={"n_reviewers": n, "source_segment_id": "", "origin": "added"}))
    out.sort(key=lambda s: (s.series_id, s.start_date))
    return out, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reviews", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-reviewers", type=int, default=1)
    ap.add_argument("--match-iou", type=float, default=0.5)
    a = ap.parse_args()
    with open(a.reviews, newline="") as f:
        rows = list(csv.DictReader(f))
    segs, stats = consensus(rows, min_reviewers=a.min_reviewers, match_iou=a.match_iou)
    write_segments(a.out, segs, ["n_reviewers", "source_segment_id", "origin"])
    print(", ".join(f"{k}={v}" for k, v in sorted(stats.items())), f"-> {len(segs)} segments in {a.out}")


if __name__ == "__main__":
    main()
