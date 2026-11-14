"""Combine full series and segments into the outbreak CSV, in the original IDOBE wide schema.

Usage (from the repository root):
    python -m respiratory.build_outbreaks

Inputs : respiratory/data/series_long.csv.gz, respiratory/data/segments.csv
Outputs: respiratory/data/outbreaks_respiratory.csv        same columns as raw_data/outbreaks_disease_location.csv
         respiratory/data/outbreaks_respiratory_meta.csv   one row per unique_id with provenance
         respiratory/data/legacy_respiratory_outbreaks.csv respiratory rows copied from the original dataset

Conventions copied from the original file: each row holds the core outbreak plus up to PAD weeks of
context on each side (cut short at the ends of a series); start_date and end_date are the first and last
week in the row; `duration` is the core length in weeks without the padding; `total` sums the row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import DATA, ROOT

PAD = 4            # context weeks before and after the core outbreak
N_COLS = 60        # value columns 0..59
MAX_GAP = 2        # longest run of missing weeks that is linearly interpolated; longer gaps drop the row
FIRST_ID = 100001  # keeps unique_id clear of the original dataset's ids
HOLDOUT_SEASONS = {"2024-25", "2025-26"}
RESPIRATORY = ["COVID-19", "INFLUENZA", "ILI", "RSV"]


def season_of(date: pd.Timestamp) -> str:
    """Season label with a July 1 boundary, e.g. 2024-07-01..2025-06-30 -> '2024-25'."""
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def main() -> None:
    series = pd.read_csv(DATA / "series_long.csv.gz", dtype={"location_fips": str, "target": str})
    segments = pd.read_csv(DATA / "segments.csv", dtype=str)
    groups = {sid: g for sid, g in series.groupby("series_id", sort=False)}
    rows, meta, dropped = [], [], 0
    for seg in segments.itertuples(index=False):
        g = groups[seg.series_id]
        dates = g["date"].to_numpy()
        a, b = np.searchsorted(dates, seg.start_date), np.searchsorted(dates, seg.end_date)
        lo, hi = max(a - PAD, 0), min(b + PAD, len(dates) - 1)
        vals = pd.Series(g["value"].to_numpy(float)[lo:hi + 1])
        filled = vals.interpolate(limit=MAX_GAP, limit_area="inside")
        # Padding must not reach across a reporting gap: trim it back to the weeks joined to the core.
        nan_at = np.flatnonzero(filled.isna().to_numpy()) + lo
        if ((nan_at >= a) & (nan_at <= b)).any():
            dropped += 1
            continue
        new_lo = max([lo] + [i + 1 for i in nan_at if i < a])
        new_hi = min([hi] + [i - 1 for i in nan_at if i > b])
        vals, filled = vals[new_lo - lo:new_hi - lo + 1], filled[new_lo - lo:new_hi - lo + 1]
        lo, hi = new_lo, new_hi
        first = g.iloc[0]
        uid = FIRST_ID + len(rows)
        v = filled.round(6).tolist()
        rows.append([uid, first["disease"], first["location"], first["event"], dates[lo], dates[hi],
                     b - a, round(float(filled.sum()), 6)] + v + [np.nan] * (N_COLS - len(v)))
        peak = pd.Timestamp(seg.peak_date)
        meta.append({
            "unique_id": uid, "series_id": seg.series_id, "segment_id": seg.segment_id,
            "source": first["source"], "geo_level": first["geo_level"], "location_fips": first["location_fips"],
            "target": first["target"], "core_start_date": seg.start_date, "core_end_date": seg.end_date,
            "peak_date": seg.peak_date, "season": season_of(peak), "holdout": season_of(peak) in HOLDOUT_SEASONS,
            "truncated": bool(a == 0 or b == len(dates) - 1 or np.isnan(g["value"].to_numpy(float)[[a, b]]).any()),
            "n_interpolated": int(vals.isna().sum()), "method": seg.method, "method_version": seg.method_version,
            "as_of": first["as_of"],
        })
    cols = ["unique_id", "disease", "location", "event", "start_date", "end_date", "duration", "total"]
    out = pd.DataFrame(rows, columns=cols + [str(i) for i in range(N_COLS)])
    out.to_csv(DATA / "outbreaks_respiratory.csv", index=False)
    m = pd.DataFrame(meta)
    m.to_csv(DATA / "outbreaks_respiratory_meta.csv", index=False)

    legacy = pd.read_csv(ROOT.parent / "raw_data" / "outbreaks_disease_location.csv")
    legacy = legacy[legacy["disease"].isin(RESPIRATORY)]
    legacy.to_csv(DATA / "legacy_respiratory_outbreaks.csv", index=False)

    print(out.join(m[["source", "geo_level"]]).groupby(["source", "disease", "event", "geo_level"]).size())
    print(f"{len(out):,} outbreaks written; {dropped} segments dropped for unfillable gaps; "
          f"{len(legacy):,} respiratory rows copied from the original dataset")


if __name__ == "__main__":
    main()
