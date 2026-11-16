"""The files that connect the pipeline stages.

series (long format, one row per series per week)
    series_id, source, disease, location, geo_level, event, date, value
    [, parent_location, population, value_type]
    `date` is the week-ending Saturday. `value` may be blank (missing week). Optional columns:
    parent_location (e.g. the state of a county), population, value_type ("count" or "rate").
    Synthetic series use the same file, with source = "synthetic:<generator>" and their true
    outbreak windows written as a segments file with method = "ground-truth".

segments (one row per proposed or accepted outbreak)
    series_id, segment_id, method, method_version, start_date, end_date, score
    [, peak_date, core_start_date, core_end_date]
    Dates are inclusive week-ending Saturdays inside the series. `score` and the bracketed
    columns are optional: start..end is the full window (e.g. trough to trough), core_start..core_end
    the outbreak proper inside it (e.g. onset to end), when a method distinguishes them.

benchmark (the existing IDOBE wide schema, unchanged)
    unique_id, disease, location, event, start_date, end_date, duration, total, 0..59
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

SERIES_COLUMNS = ["series_id", "source", "disease", "location", "geo_level", "event", "date", "value"]
OPTIONAL_SERIES_COLUMNS = ["parent_location", "population", "value_type"]
SEGMENT_COLUMNS = ["series_id", "segment_id", "method", "method_version", "start_date", "end_date", "score"]
OPTIONAL_SEGMENT_DATES = ["peak_date", "core_start_date", "core_end_date"]
MAX_WEEKS = 60
BENCHMARK_COLUMNS = ["unique_id", "disease", "location", "event", "start_date", "end_date", "duration",
                     "total"] + [str(i) for i in range(MAX_WEEKS)]


@dataclass
class Series:
    """One full weekly time series (not yet split into outbreaks)."""

    series_id: str
    disease: str
    location: str
    event: str
    start_date: date
    values: list[float | None]
    source: str = ""
    geo_level: str = ""
    parent_location: str = ""
    population: float | None = None
    value_type: str = ""

    @property
    def dates(self) -> list[date]:
        return [self.start_date + timedelta(weeks=i) for i in range(len(self.values))]

    def index_of(self, d: date) -> int:
        offset = (d - self.start_date).days
        if offset % 7:
            raise ValueError(f"{d} is not a week-ending date of series {self.series_id}")
        return offset // 7


@dataclass
class SegmentRow:
    series_id: str
    segment_id: str
    method: str
    method_version: str
    start_date: date
    end_date: date
    score: float | None = None
    peak_date: date | None = None
    core_start_date: date | None = None
    core_end_date: date | None = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> list:
        opt = [getattr(self, c) for c in OPTIONAL_SEGMENT_DATES]
        return [self.series_id, self.segment_id, self.method, self.method_version,
                self.start_date.isoformat(), self.end_date.isoformat(),
                "" if self.score is None else self.score] + [d.isoformat() if d else "" for d in opt]


def _float(v: str) -> float | None:
    v = (v or "").strip()
    if v == "":
        return None
    x = float(v)
    return None if math.isnan(x) else x


def read_series(path: Path) -> dict[str, Series]:
    """Read a long-format series file. Weeks missing from the file become None."""
    rows: dict[str, dict] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sid = r["series_id"]
            s = rows.setdefault(sid, {"meta": r, "points": {}})
            d = date.fromisoformat(r["date"])
            if d in s["points"]:
                raise ValueError(f"series {sid}: duplicate week {d}")
            s["points"][d] = _float(r.get("value", ""))
    out = {}
    for sid, s in rows.items():
        days = sorted(s["points"])
        start, end = days[0], days[-1]
        n = (end - start).days // 7 + 1
        values: list[float | None] = [None] * n
        for d, v in s["points"].items():
            if (d - start).days % 7:
                raise ValueError(f"series {sid}: {d} is not on the same weekday as {start}")
            values[(d - start).days // 7] = v
        m = s["meta"]
        out[sid] = Series(series_id=sid, source=m.get("source", ""), disease=m["disease"],
                          location=m["location"], geo_level=m.get("geo_level", ""), event=m["event"],
                          start_date=start, values=values, parent_location=m.get("parent_location", ""),
                          population=_float(m.get("population", "")), value_type=m.get("value_type", ""))
    return out


def write_series(path: Path, series: list[Series] | dict[str, Series]) -> int:
    items = series.values() if isinstance(series, dict) else series
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SERIES_COLUMNS + OPTIONAL_SERIES_COLUMNS)
        for s in items:
            opt = [s.parent_location, "" if s.population is None else s.population, s.value_type]
            for d, v in zip(s.dates, s.values):
                w.writerow([s.series_id, s.source, s.disease, s.location, s.geo_level, s.event,
                            d.isoformat(), "" if v is None else v] + opt)
                n += 1
    return n


def _date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


def read_segments(path: Path) -> list[SegmentRow]:
    known = SEGMENT_COLUMNS + OPTIONAL_SEGMENT_DATES
    with open(path, newline="") as f:
        return [
            SegmentRow(series_id=r["series_id"], segment_id=r["segment_id"], method=r["method"],
                       method_version=r.get("method_version", ""),
                       start_date=date.fromisoformat(r["start_date"]),
                       end_date=date.fromisoformat(r["end_date"]), score=_float(r.get("score", "")),
                       **{c: _date(r.get(c)) for c in OPTIONAL_SEGMENT_DATES},
                       extra={k: v for k, v in r.items() if k not in known})
            for r in csv.DictReader(f)
        ]


def write_segments(path: Path, segments: list[SegmentRow], extra_columns: list[str] = ()) -> int:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SEGMENT_COLUMNS + OPTIONAL_SEGMENT_DATES + list(extra_columns))
        for s in segments:
            w.writerow(s.as_row() + [s.extra.get(c, "") for c in extra_columns])
    return len(segments)
