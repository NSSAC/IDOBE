"""Validate a synthetic outbreak table against the real one.

Four checks, each judged against how much two samples of *real* data differ
from each other (the "real-vs-real null"), not against a fixed cutoff:

1. Per-feature distributions. The repo's own statistics from
   stats/stats_compute.ipynb (Shannon entropy, permutation entropy, and the
   shape moments of the curve as a distribution over weeks) plus features that
   capture noise and count structure (lag-1 autocorrelation, fraction of zero
   weeks, fraction of integer values, peak share, week-to-week log change,
   dispersion index, series length, log total, relative peak position).
   For each feature we compute the two-sample KS statistic between real and
   synthetic, and compare it with the 95th percentile of KS between a random
   real subsample of the same size and the remaining real rows. A feature
   passes when its KS is within that null band. This is done pooled and per
   stratum (by default, per disease).

2. Classifier two-sample test (C2ST). A gradient-boosted classifier is trained
   to tell real from synthetic using the features above (5-fold CV ROC AUC).
   0.5 means indistinguishable. The same test on two halves of the real data
   gives the null AUC. Permutation importances show which features give the
   synthetic data away.

3. Memorization. Each curve is normalized to a probability curve on a common
   60-point grid; we compare the distance from every synthetic curve to its
   nearest real curve with the distance from held-out real curves to their
   nearest real (training) curve. Many synthetic curves much closer than real
   neighbors are, suggests copying.

Example:
    python synthetic/validate.py \
        --real raw_data/outbreaks_disease_location.csv \
        --synthetic synthetic/output/synthetic_outbreaks.csv \
        --report synthetic/output/validation_report.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

VALUE_COLS = [str(i) for i in range(60)]

# The original metric set from stats/stats_compute.ipynb.
CORE_METRICS = ["entropy", "PE", "mean", "median", "variance",
                "skewness", "excess_kurtosis"]
# Extra features that capture noise, count structure, length and scale.
EXTRA_METRICS = ["lag1_ac", "zero_frac", "integer_frac", "peak_frac",
                 "dlog_sd", "dispersion", "n_obs", "log_total", "peak_pos"]
ALL_METRICS = CORE_METRICS + EXTRA_METRICS


# --------------------------------------------------------------------------
# Per-curve features
# --------------------------------------------------------------------------

def _series(row: np.ndarray) -> np.ndarray:
    y = row[~np.isnan(row)]
    return np.clip(y, 0.0, None)


def shannon_entropy(row: np.ndarray) -> float:
    y = _series(row)
    if y.sum() <= 0:
        return np.nan
    p = y / y.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def permutation_entropy(row: np.ndarray) -> float:
    import ordpy
    y = _series(row)
    if y.size < 4:
        return np.nan
    return float(ordpy.permutation_entropy(y, dx=3, taux=1, normalized=True))


def shape_moments(row: np.ndarray) -> dict[str, float]:
    """Moments of the curve as a discrete PDF over week index (see notebook)."""
    y = _series(row)
    if y.sum() <= 0:
        return {k: np.nan for k in
                ("mean", "median", "variance", "skewness", "excess_kurtosis")}
    x = np.arange(y.size, dtype=float)
    p = y / y.sum()
    mu = np.sum(x * p)
    order = np.argsort(x)
    cdf = np.cumsum(p[order])
    median = x[order][np.searchsorted(cdf, 0.5)]
    var = np.sum((x - mu) ** 2 * p)
    sigma = np.sqrt(var)
    if sigma < 1e-12:
        skew = kurt = np.nan
    else:
        skew = np.sum((x - mu) ** 3 * p) / sigma ** 3
        kurt = np.sum((x - mu) ** 4 * p) / sigma ** 4 - 3
    return {"mean": mu, "median": float(median), "variance": var,
            "skewness": skew, "excess_kurtosis": kurt}


def extra_features(row: np.ndarray) -> dict[str, float]:
    """Noise, count-structure, length and scale features of one curve."""
    y = _series(row)
    n = y.size
    out = {k: np.nan for k in EXTRA_METRICS}
    out["n_obs"] = float(n)
    if n < 3 or y.sum() <= 0:
        return out
    if y[:-1].std() > 0 and y[1:].std() > 0:
        out["lag1_ac"] = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    out["zero_frac"] = float(np.mean(y == 0))
    out["integer_frac"] = float(np.mean(np.abs(y - np.round(y)) < 1e-9))
    out["peak_frac"] = float(y.max() / y.sum())
    out["dlog_sd"] = float(np.std(np.diff(np.log1p(y))))
    out["dispersion"] = float(np.log1p(y.var() / max(y.mean(), 1e-12)))
    out["log_total"] = float(np.log1p(y.sum()))
    out["peak_pos"] = float(np.argmax(y) / (n - 1))
    return out


def compute_metrics(df: pd.DataFrame, extra: bool = False) -> pd.DataFrame:
    """Per-row metrics. ``extra=False`` returns only the notebook's metrics."""
    values = df[VALUE_COLS].to_numpy(dtype=float)
    rows = []
    for row in values:
        rec = {"entropy": shannon_entropy(row), "PE": permutation_entropy(row)}
        rec.update(shape_moments(row))
        if extra:
            rec.update(extra_features(row))
        rows.append(rec)
    cols = ALL_METRICS if extra else CORE_METRICS
    return pd.DataFrame(rows, columns=cols, index=df.index)


# --------------------------------------------------------------------------
# 1. Per-feature KS against a real-vs-real null
# --------------------------------------------------------------------------

def _ks(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(stats.ks_2samp(a, b)[0])


def ks_with_null(real: pd.DataFrame, synth: pd.DataFrame, metrics: list[str],
                 reps: int, rng: np.random.Generator) -> pd.DataFrame:
    """KS(real, synth) per metric, plus the null KS distribution.

    The null draws a real subsample the size of the synthetic sample (capped at
    half the real rows) and compares it with the remaining real rows, so the
    null reflects the same sample sizes as the real-vs-synthetic comparison.
    """
    n_r, n_s = len(real), len(synth)
    k = min(n_s, n_r // 2)
    null = {m: [] for m in metrics}
    R = real[metrics].to_numpy(dtype=float)
    for _ in range(reps):
        idx = rng.permutation(n_r)
        a, b = R[idx[:k]], R[idx[k:]]
        for j, m in enumerate(metrics):
            null[m].append(_ks(a[:, j], b[:, j]))
    rows = []
    for m in metrics:
        r = real[m].to_numpy(dtype=float)
        s = synth[m].to_numpy(dtype=float)
        nm = np.asarray(null[m], dtype=float)
        ks = _ks(r, s)
        p95 = float(np.nanquantile(nm, 0.95)) if nm.size else np.nan
        rows.append({
            "metric": m,
            "real_mean": np.nanmean(r), "synth_mean": np.nanmean(s),
            "real_std": np.nanstd(r), "synth_std": np.nanstd(s),
            "ks_stat": ks,
            "null_ks_median": float(np.nanmedian(nm)) if nm.size else np.nan,
            "null_ks_p95": p95,
            "ks_over_null": ks / p95 if p95 and p95 > 0 else np.nan,
            "pass": bool(ks <= p95) if not np.isnan(ks) else False,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Classifier two-sample test
# --------------------------------------------------------------------------

def c2st_auc(a: pd.DataFrame, b: pd.DataFrame, metrics: list[str],
             seed: int) -> float:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = pd.concat([a[metrics], b[metrics]]).to_numpy(dtype=float)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def c2st_importances(a: pd.DataFrame, b: pd.DataFrame, metrics: list[str],
                     seed: int) -> pd.Series:
    """Permutation importance (drop in AUC) of each feature on a held-out split."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    X = pd.concat([a[metrics], b[metrics]]).to_numpy(dtype=float)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=seed)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    imp = permutation_importance(clf, Xte, yte, scoring="roc_auc",
                                 n_repeats=5, random_state=seed)
    return pd.Series(imp.importances_mean, index=metrics).sort_values(
        ascending=False)


def c2st(real: pd.DataFrame, synth: pd.DataFrame, metrics: list[str],
         seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real))
    h = len(real) // 2
    out = {
        "auc_core_metrics": c2st_auc(real, synth, CORE_METRICS, seed),
        "auc_all_features": c2st_auc(real, synth, metrics, seed),
        "auc_null_real_vs_real": c2st_auc(real.iloc[idx[:h]],
                                          real.iloc[idx[h:]], metrics, seed),
    }
    out["top_features"] = c2st_importances(real, synth, metrics, seed).head(5) \
        .round(4).to_dict()
    return out


# --------------------------------------------------------------------------
# 3. Memorization
# --------------------------------------------------------------------------

def _grid(row: np.ndarray, length: int = 60) -> np.ndarray | None:
    y = _series(row)
    if y.size < 2 or y.sum() <= 0:
        return None
    y = y / y.sum()
    g = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, y.size), y)
    return g / g.sum()


def _grids(df: pd.DataFrame) -> np.ndarray:
    gs = [_grid(r) for r in df[VALUE_COLS].to_numpy(dtype=float)]
    return np.vstack([g for g in gs if g is not None])


def memorization(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                 seed: int) -> dict:
    """Nearest-neighbor distances: synthetic->real vs held-out real->real."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    R = _grids(real_df)
    S = _grids(synth_df)
    idx = rng.permutation(len(R))
    h = len(R) // 5
    hold, train = R[idx[:h]], R[idx[h:]]
    d_real = NearestNeighbors(n_neighbors=1).fit(train).kneighbors(hold)[0][:, 0]
    d_syn = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(S)[0][:, 0]
    p5 = float(np.quantile(d_real, 0.05))
    return {
        "nn_dist_real_holdout_median": float(np.median(d_real)),
        "nn_dist_synth_median": float(np.median(d_syn)),
        "real_holdout_p5": p5,
        # Expect ~0.05 if synthetic curves are no closer to real ones than
        # unseen real curves are; much higher suggests copying.
        "synth_frac_below_real_p5": float(np.mean(d_syn < p5)),
        "synth_exact_copies": int(np.sum(d_syn < 1e-9)),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(real_df: pd.DataFrame, synth_df: pd.DataFrame, by: str | None,
        min_stratum: int, reps: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    real = compute_metrics(real_df, extra=True)
    synth = compute_metrics(synth_df, extra=True)

    tables = [ks_with_null(real, synth, ALL_METRICS, reps, rng)
              .assign(stratum="ALL", n_real=len(real), n_synth=len(synth))]
    skipped = []
    if by:
        for key in sorted(set(real_df[by]) | set(synth_df[by])):
            r = real[real_df[by].to_numpy() == key]
            s = synth[synth_df[by].to_numpy() == key]
            if len(r) < min_stratum or len(s) < min_stratum:
                skipped.append({"stratum": str(key), "n_real": len(r),
                                "n_synth": len(s)})
                continue
            tables.append(ks_with_null(r, s, ALL_METRICS, reps, rng)
                          .assign(stratum=str(key), n_real=len(r),
                                  n_synth=len(s)))
    report = pd.concat(tables, ignore_index=True)
    report = report[["stratum", "n_real", "n_synth"] +
                    [c for c in report.columns
                     if c not in ("stratum", "n_real", "n_synth")]]

    summary = {
        "n_real": len(real), "n_synth": len(synth),
        "ks_pass_rate_pooled":
            float(report.loc[report.stratum == "ALL", "pass"].mean()),
        "ks_pass_rate_strata":
            float(report.loc[report.stratum != "ALL", "pass"].mean())
            if (report.stratum != "ALL").any() else None,
        "strata_skipped_too_small": skipped,
        "c2st": c2st(real, synth, ALL_METRICS, seed),
        "memorization": memorization(real_df, synth_df, seed),
    }
    return report, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", default="raw_data/outbreaks_disease_location.csv")
    p.add_argument("--synthetic", default="synthetic/output/synthetic_outbreaks.csv")
    p.add_argument("--report", default=None,
                   help="Optional CSV path for the per-stratum, per-metric table.")
    p.add_argument("--summary", default=None,
                   help="Optional JSON path for the classifier and "
                        "memorization results.")
    p.add_argument("--by", default="disease",
                   help="Column to stratify by ('' for pooled only).")
    p.add_argument("--min-stratum", type=int, default=30,
                   help="Minimum synthetic rows per stratum (real needs 2x).")
    p.add_argument("--null-reps", type=int, default=20,
                   help="Random real-vs-real splits for the KS null.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    real_df = pd.read_csv(args.real)
    synth_df = pd.read_csv(args.synthetic)
    report, summary = run(real_df, synth_df, args.by or None,
                          args.min_stratum, args.null_reps, args.seed)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pooled = report[report.stratum == "ALL"].drop(columns=["stratum"])
    print("Pooled: KS(real, synthetic) vs real-vs-real null (pass = within null p95)\n")
    print(pooled[["metric", "real_mean", "synth_mean", "real_std", "synth_std",
                  "ks_stat", "null_ks_p95", "pass"]].to_string(index=False))

    strata = report[report.stratum != "ALL"]
    if len(strata):
        print(f"\nPer-{args.by} KS pass rate (share of metrics within null p95):\n")
        print(strata.groupby("stratum")
              .agg(n_real=("n_real", "first"), n_synth=("n_synth", "first"),
                   pass_rate=("pass", "mean"),
                   worst_metric=("ks_over_null",
                                 lambda s: strata.loc[s.idxmax(), "metric"]),
                   worst_ks_over_null=("ks_over_null", "max"))
              .to_string())
    if summary["strata_skipped_too_small"]:
        names = ", ".join(s["stratum"] for s in summary["strata_skipped_too_small"])
        print(f"\nStrata too small to test: {names}")

    c = summary["c2st"]
    print("\nClassifier two-sample test (ROC AUC; 0.5 = indistinguishable)")
    print(f"  real vs synthetic, notebook metrics only: {c['auc_core_metrics']:.3f}")
    print(f"  real vs synthetic, all features:          {c['auc_all_features']:.3f}")
    print(f"  null, real half vs real half:             {c['auc_null_real_vs_real']:.3f}")
    print("  features that most give synthetic away: " +
          ", ".join(f"{k} ({v:.3f})" for k, v in c["top_features"].items()))

    m = summary["memorization"]
    print("\nMemorization (nearest-neighbor distance on normalized curves)")
    print(f"  share of synthetic closer to a real curve than 95% of unseen real "
          f"curves are: {m['synth_frac_below_real_p5']:.3f} (expect ~0.05)")
    print(f"  exact copies: {m['synth_exact_copies']}")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        report.to_csv(args.report, index=False)
        print(f"\nWrote {args.report}", file=sys.stderr)
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
