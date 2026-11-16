"""Run a segmentation method over a series file and write a segments file.

    python -m pipeline.segment SERIES.csv --method seasonal-threshold --out SEGMENTS.csv [--param k=v ...]
    python -m pipeline.segment --legacy raw_data/outbreaks_disease_location.csv \
        --series-out SERIES.csv --out SEGMENTS.csv [--diseases COVID-19,INFLUENZA,ILI,RSV]

The --legacy form writes the full series rebuilt from the published CSV plus its current
(epidemicKabu) windows, so they can be reviewed in the annotation app.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .formats import SegmentRow, read_series, write_segments, write_series
from .segmentation import METHODS, check_segments, get_segmenter
from .segmentation import legacy


def run(series: dict, method: str, params: dict) -> list[SegmentRow]:
    seg = get_segmenter(method, **params)
    rows = []
    for sid, s in series.items():
        found = check_segments(seg.segment(s.values, event=s.event, disease=s.disease), len(s.values))
        for i, x in enumerate(found, 1):
            rows.append(SegmentRow(series_id=sid, segment_id=f"{sid}#{i}", method=seg.name,
                                   method_version=seg.version, start_date=s.dates[x.start],
                                   end_date=s.dates[x.end], score=x.score,
                                   peak_date=s.dates[x.peak] if x.peak is not None else None,
                                   core_start_date=s.dates[x.core_start] if x.core_start is not None else None,
                                   core_end_date=s.dates[x.core_end] if x.core_end is not None else None))
    return rows


def _param(kv: str):
    k, v = kv.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", type=Path, nargs="?")
    ap.add_argument("--method", default="seasonal-threshold", choices=sorted(METHODS))
    ap.add_argument("--param", action="append", default=[], type=_param, help="method parameter k=v")
    ap.add_argument("--legacy", type=Path, help="rebuild series + windows from the published wide CSV")
    ap.add_argument("--diseases", help="comma-separated diseases to keep (with --legacy)")
    ap.add_argument("--series-out", type=Path, help="where to write rebuilt series (with --legacy)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.legacy:
        diseases = {d.strip().upper() for d in a.diseases.split(",")} if a.diseases else None
        series, segs, stats = legacy.stitch(a.legacy, diseases)
        if a.series_out:
            write_series(a.series_out, series)
        write_segments(a.out, segs, ["legacy_unique_id"])
        print(", ".join(f"{k}={v}" for k, v in stats.items()))
        return
    if not a.series:
        ap.error("give a series file or --legacy")
    rows = run(read_series(a.series), a.method, dict(a.param))
    write_segments(a.out, rows)
    print(f"{len(rows)} segments -> {a.out}")


if __name__ == "__main__":
    main()
