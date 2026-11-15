"""Split full weekly series into outbreak segments. This step is deliberately swappable.

Usage (from the repository root):
    python -m respiratory.segment                      # default method
    python -m respiratory.segment --method my_method   # any name registered in SEGMENTERS

Input : respiratory/data/series_long.csv.gz
Output: respiratory/data/segments.csv with one row per segment:
    series_id, segment_id, method, method_version, start_date, end_date, peak_date, score

To plug in a different splitter, write a function `f(values, count_like, covid=False) -> list[Segment]` and add it
to SEGMENTERS. `values` is the full weekly series (NaN where missing). Returning an empty list is fine.
Nothing upstream (fetch, build_series) or downstream (build_outbreaks) needs to change.

The default, `prominence`, is the segmenter chosen in the segmentation rework (package `segmentation/`).
Outbreaks run from onset to end (offset); each wave is its own outbreak; 4 weeks of context padding are
added later by build_outbreaks. `gaussian_valley` is the earlier trough-to-trough stand-in for the
EpidemicKabu splitting of the original IDOBE and is kept for comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .common import DATA

MAX_GAP = 2  # longest run of missing weeks tolerated inside a segment
MIN_WEEKS, MAX_WEEKS = 8, 51  # same bounds as the original dataset (51 + padding fits 60 columns)


@dataclass
class Segment:
    start: int                     # index of first week of the core outbreak (onset)
    end: int                       # index of last week of the core outbreak (end), inclusive
    peak: int | None = None        # index of the peak week
    score: float | None = None     # optional method-specific confidence


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="reflect"), k, mode="valid")


def gaussian_valley(values: np.ndarray, count_like: bool, sigma: float = 3.0, min_prominence: float = 0.1,
                    min_peak_count: float = 10.0) -> list[Segment]:
    """Cut at local minima of the Gaussian-smoothed series; keep segments that look like outbreaks.

    sigma           smoothing bandwidth in weeks
    min_prominence  the smoothed peak must exceed the higher of the two bounding valleys by at least
                    this fraction of the peak (drops flat stretches and shoulders)
    min_peak_count  for count data, the raw peak week must reach this value (drops noise in tiny counts)
    """
    n = len(values)
    if n < MIN_WEEKS + 1 or np.isnan(values).all():
        return []
    x = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    y = gaussian_smooth(x, sigma)
    d = np.diff(y)
    valleys = [i for i in range(1, n - 1) if d[i - 1] < 0 <= d[i]]
    cuts = sorted({0, n - 1, *valleys})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if not MIN_WEEKS <= b - a <= MAX_WEEKS:
            continue
        peak = a + int(np.argmax(y[a:b + 1]))
        prominence = y[peak] - max(y[a], y[b])
        if y[peak] <= 0 or prominence < min_prominence * y[peak]:
            continue
        raw_peak = np.nanmax(values[a:b + 1]) if not np.isnan(values[a:b + 1]).all() else 0
        if count_like and raw_peak < min_peak_count:
            continue
        out.append(Segment(a, b, a + int(np.nanargmax(values[a:b + 1])), float(prominence / y[peak])))
    return out


def contiguous_runs(values: np.ndarray, max_gap: int = MAX_GAP) -> list[tuple[int, int]]:
    """Index ranges of the series that contain no run of missing weeks longer than max_gap.

    Some sources stop reporting between seasons (e.g. FluSurv-NET in summer). Each reporting stretch is
    segmented on its own, so no outbreak ever spans a long gap."""
    obs = np.flatnonzero(~np.isnan(values))
    if len(obs) == 0:
        return []
    breaks = np.flatnonzero(np.diff(obs) > max_gap + 1)
    starts = np.r_[obs[0], obs[breaks + 1]]
    ends = np.r_[obs[breaks], obs[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def prominence(values: np.ndarray, count_like: bool, covid: bool = False) -> list[Segment]:
    """Onset-to-end outbreaks from the prominence segmenter in `segmentation/` (weekly preset).

    The core of each outbreak runs from onset to offset, so flat baseline weeks are left out of
    `duration` and only appear as context padding. Every wave, including each COVID-19 wave, is its own
    outbreak. Trough-to-trough bounds are not used here.

    With the WEEKLY preset, a large wave sets the local scale and smaller neighbouring COVID-19 waves
    are merged into it (in JHU national cases, summer 2020 with winter 2020-21, and Delta with Omicron).
    COVID-19 series therefore use the COVID_WEEKLY preset (rel_prominence 0.05)."""
    from segmentation import COVID_WEEKLY, WEEKLY, ProminenceSegmenter

    params = COVID_WEEKLY if covid else WEEKLY

    dates = pd.date_range("2000-01-01", periods=len(values), freq="7D")  # runs are already gap-free
    return [Segment(s.onset, s.offset, s.peak, s.score)
            for s in ProminenceSegmenter(params).segment(dates, values)
            if MIN_WEEKS <= s.offset - s.onset <= MAX_WEEKS]


def _prominence_version() -> str:
    from segmentation import ProminenceSegmenter
    return ProminenceSegmenter().version


SEGMENTERS: dict[str, tuple[Callable[[], str], Callable[..., list[Segment]]]] = {
    # name: (version, function)
    "prominence": (_prominence_version, prominence),
    "gaussian_valley": (lambda: "0.1", lambda v, c, covid=False: gaussian_valley(v, c)),
}
COUNT_EVENTS = {"CASES", "DEATHS", "HOSPITALIZATIONS"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="prominence", choices=sorted(SEGMENTERS))
    args = ap.parse_args()
    version_fn, fn = SEGMENTERS[args.method]
    version = version_fn()
    series = pd.read_csv(DATA / "series_long.csv.gz", usecols=["series_id", "disease", "event", "date", "value"])
    rows = []
    for sid, g in series.groupby("series_id", sort=True):
        dates, values = g["date"].to_numpy(), g["value"].to_numpy(float)
        count_like, k = g["event"].iloc[0] in COUNT_EVENTS, 0
        covid = g["disease"].iloc[0] == "COVID-19"
        for lo, hi in contiguous_runs(values):
            for s in fn(values[lo:hi + 1], count_like, covid=covid):
                rows.append({"series_id": sid, "segment_id": f"{sid}#{k:02d}", "method": args.method,
                             "method_version": version, "start_date": dates[lo + s.start],
                             "end_date": dates[lo + s.end],
                             "peak_date": dates[lo + s.peak] if s.peak is not None else "",
                             "score": "" if s.score is None else round(s.score, 4)})
                k += 1
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "segments.csv", index=False)
    print(f"{len(out):,} segments from {out['series_id'].nunique():,} of {series['series_id'].nunique():,} series")


if __name__ == "__main__":
    main()
