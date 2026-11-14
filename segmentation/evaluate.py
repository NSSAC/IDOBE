"""Compare the prominence segmenter with the epidemicKabu cut rule.

Two checks:

1. Synthetic series with known waves (``synthetic.random_waves``) at three
   count levels: peak recall, precision and timing error.
2. Real series rebuilt from ``raw_data/outbreaks_disease_location.csv``:
   agreement with the released (legacy) outbreak boundaries, and outbreaks per
   series-year.

Usage: python -m segmentation.evaluate [path/to/outbreaks_disease_location.csv]
Writes CSVs to segmentation/results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .legacy_kabu import kabu_segments
from .prominence import WEEKLY, ProminenceSegmenter
from .synthetic import make_series, random_waves

ROOT = Path(__file__).resolve().parent
TOL_DAYS = 28
LEGACY_PAD_WEEKS = 9  # each CSV row = core outbreak + 9 weeks of shared context


# --------------------------------------------------------------- methods
def run_prominence(dates, y):
    return [(s.start, s.end, s.peak) for s in ProminenceSegmenter(WEEKLY).segment(dates, y)]


def run_kabu(kernel):
    def f(dates, y):
        sm = gaussian_filter1d(np.nan_to_num(y), kernel / 2 / 7)
        out = []
        for a, b in kabu_segments(dates, np.nan_to_num(y), kernel_days=kernel):
            pk = a + int(np.argmax(sm[a : b + 1]))
            if a < pk < b:  # a segment whose maximum sits on its edge has no peak
                out.append((a, b, pk))
        return out

    return f


METHODS = {
    "epidemicKabu k=18d": run_kabu(18),
    "epidemicKabu k=28d": run_kabu(28),
    "prominence v0.1": run_prominence,
}


def match(pred_days, true_days, tol=TOL_DAYS):
    """Greedy one-to-one matching of peak days; returns matched pairs."""
    pairs, used = [], set()
    for t in true_days:
        best = None
        for j, p in enumerate(pred_days):
            if j in used or abs(p - t) > tol:
                continue
            if best is None or abs(p - t) < abs(pred_days[best] - t):
                best = j
        if best is not None:
            used.add(best)
            pairs.append((t, pred_days[best]))
    return pairs


# ------------------------------------------------------------- synthetic
def synthetic(n_series=200, n_days=5 * 365, seed=0):
    rows = []
    for level_name, height in [("~500/wk peaks", 500), ("~50/wk peaks", 50), ("~5/wk peaks", 5)]:
        rng = np.random.default_rng(seed)
        for i in range(n_series):
            waves = random_waves(rng, n_days, height)
            dates, y, _ = make_series(waves, n_days, baseline=0.02 * height, rng=rng)
            true = [w.peak_day for w in waves]
            for name, f in METHODS.items():
                segs = f(dates, y)
                pred = [(dates[p] - dates[0]).days for _, _, p in segs]
                pairs = match(pred, true)
                rows.append(
                    dict(level=level_name, method=name, series=i, n_true=len(true), n_pred=len(pred),
                         n_match=len(pairs), abs_err_days=np.mean([abs(a - b) for a, b in pairs]) if pairs else np.nan)
                )
    df = pd.DataFrame(rows)
    summ = (
        df.groupby(["level", "method"], sort=False)
        .agg(true=("n_true", "sum"), predicted=("n_pred", "sum"), matched=("n_match", "sum"),
             mean_peak_error_days=("abs_err_days", "mean"))
        .assign(recall=lambda d: d.matched / d.true, precision=lambda d: d.matched / d.predicted)
        .round(3)
        .reset_index()
    )
    return summ


# ----------------------------------------------------------------- legacy
def legacy_series(csv):
    df = pd.read_csv(csv, parse_dates=["start_date", "end_date"])
    cols = [str(i) for i in range(60)]
    for key, g in df.groupby(["disease", "location", "event"]):
        g = g.sort_values("start_date")
        vals = {}
        cuts = []  # legacy boundary windows: overlap of consecutive rows
        prev_end = None
        for _, r in g.iterrows():
            v = r[cols].dropna().values.astype(float)
            for k, x in enumerate(v):
                vals[r.start_date + pd.Timedelta(weeks=k)] = x
            if prev_end is not None and r.start_date <= prev_end:
                cuts.append((r.start_date, prev_end))
            prev_end = r.end_date
        s = pd.Series(vals).sort_index()
        s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="7D"))
        yield key, s, cuts


def legacy_agreement(csv):
    rows = []
    for key, s, cuts in legacy_series(csv):
        if s.notna().sum() < 52:
            continue
        dates, y = s.index, s.values
        years = s.notna().sum() / 52.18
        covered = s.notna().values
        for name, f in METHODS.items():
            segs = f(dates, y) if name.startswith("prominence") else f(dates, s.interpolate(limit_area="inside").fillna(0).values)
            starts = [dates[a] for a, _, _ in segs[1:]]  # interior boundaries
            hit = sum(any(lo <= c <= hi for c in starts) for lo, hi in cuts)
            inside = [c for c in starts if covered[dates.get_loc(c)]]
            in_window = sum(any(lo <= c <= hi for lo, hi in cuts) for c in inside)
            rows.append(dict(disease=key[0], event=key[2], location=key[1], method=name, years=years,
                             n_outbreaks=len(segs), legacy_cuts=len(cuts), legacy_cuts_matched=hit,
                             new_cuts=len(inside), new_cuts_in_legacy_window=in_window))
    df = pd.DataFrame(rows)
    df["group"] = df.disease.where(~df.disease.isin(["COVID-19"]), df.disease + " " + df.event.str.lower())
    df.loc[~df.disease.isin(["COVID-19", "INFLUENZA", "ILI", "RSV"]), "group"] = "historical (Tycho)"
    summ = (
        df.groupby(["group", "method"], sort=False)
        .agg(series=("location", "size"), years=("years", "sum"), outbreaks=("n_outbreaks", "sum"),
             legacy_cuts=("legacy_cuts", "sum"), legacy_cuts_matched=("legacy_cuts_matched", "sum"),
             new_cuts=("new_cuts", "sum"), new_cuts_in_legacy_window=("new_cuts_in_legacy_window", "sum"))
        .assign(outbreaks_per_year=lambda d: d.outbreaks / d.years,
                legacy_recall=lambda d: d.legacy_cuts_matched / d.legacy_cuts,
                legacy_precision=lambda d: d.new_cuts_in_legacy_window / d.new_cuts)
        .round(3)
        .reset_index()
    )
    return summ


def main(argv):
    csv = argv[1] if len(argv) > 1 else ROOT.parent / "raw_data" / "outbreaks_disease_location.csv"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    syn = synthetic()
    syn.to_csv(out / "synthetic_comparison.csv", index=False)
    print(syn.to_string(index=False))
    leg = legacy_agreement(csv)
    leg.to_csv(out / "legacy_agreement.csv", index=False)
    print(leg.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
