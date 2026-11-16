"""Rebuild full series and the current (epidemicKabu) outbreak windows from the IDOBE wide CSV.

The published CSV only holds the outbreak windows, not the full series they were cut from. Windows
of the same disease/location/event are laid back onto one weekly axis; weeks not covered by any
window are missing. Overlapping windows agree on >99.8% of shared weeks; where they differ the
earlier window's value is kept and the clash is counted.

Each original row becomes one proposed segment (method "epidemickabu-legacy"), so experts can
review the existing dataset's boundaries in the same UI as any new method. Following the IDOBE
convention (outbreak = onset to offset, 4 context weeks each side, duration = offset - onset), the
proposed window is the core: start_date + 4 weeks through + `duration` weeks. Rows clipped at a
series edge (~2%) hold fewer context weeks: if the row starts at the series' first week the
leading context is taken as the short side, otherwise the trailing one.
"""

from __future__ import annotations

import csv
import re
from datetime import date, timedelta
from pathlib import Path

from ..formats import MAX_WEEKS, SegmentRow, Series

METHOD = "epidemickabu-legacy"
VERSION = "idobe-2026"
PAD = 4  # context weeks each side of the core in the released CSV


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
        pending = []
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
            pending.append((r, start, len(vals), last))
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
        for r, start, n_vals, last in pending:
            dur = int(r["duration"])
            short = n_vals < dur + 1 + 2 * PAD
            lead = max(0, n_vals - 1 - PAD - dur) if short and start == first else PAD
            core0 = start + timedelta(weeks=min(lead, max(0, n_vals - 1)))
            core1 = min(core0 + timedelta(weeks=dur), last)
            segments.append(SegmentRow(series_id=key, segment_id=f"legacy-{r['unique_id']}", method=METHOD,
                                       method_version=VERSION, start_date=core0, end_date=core1,
                                       extra={"legacy_unique_id": r["unique_id"]}))
        m = g["meta"]
        series[key] = Series(series_id=key, source="idobe-v1", disease=m["disease"].strip(),
                             location=m["location"].strip(), event=m["event"].strip(), start_date=first,
                             values=values,
                             value_type="rate" if "PERCENT" in m["event"].upper() else "count")
    segments = [s for s in segments if s.series_id in series]
    return series, segments, {"series": len(series), "segments": len(segments), "value_clashes": clashes}
