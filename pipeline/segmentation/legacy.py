"""Rebuild full series and the current (epidemicKabu) outbreak windows from the IDOBE wide CSV.

The published CSV only holds the outbreak windows, not the full series they were cut from. Windows
of the same disease/location/event are laid back onto one weekly axis; weeks not covered by any
window are missing. Each row is the core outbreak plus about 9 context weeks (`duration` counts only
the core), but the split of those weeks before/after the core is not recorded, so the whole window
is used as the proposed segment. Overlapping windows agree on >99.8% of shared weeks; where they differ the
earlier window's value is kept and the clash is counted.

Each original row becomes one proposed segment (method "epidemickabu-legacy"), so experts can
review the existing dataset's boundaries in the same UI as any new method. Because consecutive
windows often overlap by several weeks, this "method" is the one exception to the no-overlap rule.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path

from ..formats import MAX_WEEKS, SegmentRow, Series

METHOD = "epidemickabu-legacy"
VERSION = "idobe-2026"


def series_key(disease: str, location: str, event: str) -> str:
    slug = lambda s: re.sub(r"[^A-Za-z0-9]+", "-", s.strip()).strip("-").lower()  # noqa: E731
    return f"{slug(disease)}__{slug(location)}__{slug(event)}"


def stitch(csv_path: Path, diseases: set[str] | None = None) -> tuple[dict[str, Series], list[SegmentRow], dict]:
    groups: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if diseases and r["disease"].strip().upper() not in diseases:
                continue
            key = series_key(r["disease"], r["location"], r["event"])
            g = groups.setdefault(key, {"meta": r, "rows": []})
            g["rows"].append(r)

    series: dict[str, Series] = {}
    segments: list[SegmentRow] = []
    clashes = 0
    for key, g in groups.items():
        points: dict[date, float] = {}
        rows = sorted(g["rows"], key=lambda r: (r["start_date"], int(r["unique_id"])))
        for r in rows:
            start = date.fromisoformat(r["start_date"])
            vals = [r.get(str(i), "") for i in range(MAX_WEEKS)]
            while vals and vals[-1] == "":
                vals.pop()
            last = start
            for i, v in enumerate(vals):
                d = start + timedelta(weeks=i)
                if v == "":
                    continue
                last = d
                v = float(v)
                if d in points:
                    clashes += abs(points[d] - v) > 1e-9 * max(1.0, abs(v))
                else:
                    points[d] = v
            segments.append(SegmentRow(series_id=key, segment_id=f"legacy-{r['unique_id']}", method=METHOD,
                                       method_version=VERSION, start_date=start,
                                       end_date=min(date.fromisoformat(r["end_date"]), last) if vals else start,
                                       extra={"legacy_unique_id": r["unique_id"]}))
        if not points:
            continue
        days = sorted(points)
        first = days[0]
        if any((d - first).days % 7 for d in days):
            raise ValueError(f"{key}: windows are not on a common weekly grid")
        n = (days[-1] - first).days // 7 + 1
        values: list[float | None] = [None] * n
        for d, v in points.items():
            values[(d - first).days // 7] = v
        m = g["meta"]
        series[key] = Series(series_id=key, source="idobe-v1", disease=m["disease"].strip(),
                             location=m["location"].strip(), event=m["event"].strip(), start_date=first,
                             values=values,
                             value_type="rate" if "PERCENT" in m["event"].upper() else "count")
    segments = [s for s in segments if s.series_id in series]
    return series, segments, {"series": len(series), "segments": len(segments), "value_clashes": clashes}
