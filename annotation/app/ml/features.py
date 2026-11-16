"""Numeric descriptors of each weekly series, used as model inputs.

Most features are scale-free (computed on the series divided by its peak), so cases, hospitalizations
and %ILI series are comparable. Measure type and disease are added as one-hot columns.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks

SHAPE_FEATURES = [
    "n_weeks", "log_total", "log_peak", "peak_position", "log_peak_over_start", "log_peak_over_end",
    "start_level", "end_level", "peak_over_mean", "frac_zero", "frac_above_half", "coef_variation",
    "roughness", "unimodality", "lag1_autocorr", "n_peaks", "mass_before_peak", "time_spread",
    "time_skew", "trend", "perm_entropy", "all_zero",
]


def _clean(values: Sequence[float | None]) -> np.ndarray:
    x = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.isnan(x).any():
        idx = np.arange(len(x))
        ok = ~np.isnan(x)
        x = np.interp(idx, idx[ok], x[ok]) if ok.any() else np.zeros_like(x)
    return np.clip(x, 0, None)


def _perm_entropy(x: np.ndarray) -> float:
    """Normalized permutation entropy of order 3 (0 = perfectly regular, 1 = noise-like)."""
    if len(x) < 4:
        return 0.0
    order = np.argsort(np.lib.stride_tricks.sliding_window_view(x, 3), axis=1, kind="stable")
    counts = np.bincount(order @ np.array([9, 3, 1]), minlength=27)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(6))


def series_features(values: Sequence[float | None]) -> list[float]:
    x = _clean(values)
    n = len(x)
    peak = x.max() if n else 0.0
    if n < 2 or peak <= 0:
        out = dict.fromkeys(SHAPE_FEATURES, 0.0)
        out.update(n_weeks=float(n), all_zero=1.0)
        return [out[k] for k in SHAPE_FEATURES]

    xn = x / peak
    t = np.arange(n) / (n - 1)
    total = x.sum()
    mean = x.mean()
    k = max(1, min(4, n // 5))
    start, end = np.median(xn[:k]), np.median(xn[-k:])
    eps = 0.01
    i_peak = int(np.argmax(x))
    d = np.diff(xn)
    abs_moves = np.abs(d).sum()
    wrong_way = -d[:i_peak][d[:i_peak] < 0].sum() + d[i_peak:][d[i_peak:] > 0].sum()
    w = x / total
    t_mean = (w * t).sum()
    t_std = math.sqrt(max((w * (t - t_mean) ** 2).sum(), 0.0))
    t_skew = (w * (t - t_mean) ** 3).sum() / t_std**3 if t_std > 1e-9 else 0.0
    centered = xn - xn.mean()
    denom = (centered**2).sum()
    lag1 = float((centered[:-1] * centered[1:]).sum() / denom) if denom > 1e-12 else 0.0

    out = {
        "n_weeks": float(n),
        "log_total": math.log1p(total),
        "log_peak": math.log1p(peak),
        "peak_position": i_peak / (n - 1),
        "log_peak_over_start": math.log((1 + eps) / (start + eps)),
        "log_peak_over_end": math.log((1 + eps) / (end + eps)),
        "start_level": float(start),
        "end_level": float(end),
        "peak_over_mean": peak / mean,
        "frac_zero": float((x == 0).mean()),
        "frac_above_half": float((xn >= 0.5).mean()),
        "coef_variation": float(x.std() / mean),
        "roughness": float(np.abs(d).mean()),
        "unimodality": float(1 - wrong_way / abs_moves) if abs_moves > 0 else 1.0,
        "lag1_autocorr": lag1,
        "n_peaks": float(len(find_peaks(xn, prominence=0.25)[0])),
        "mass_before_peak": float(x[:i_peak].sum() / total),
        "time_spread": t_std,
        "time_skew": float(t_skew),
        "trend": float(((t - t.mean()) * (xn - xn.mean())).sum() / ((t - t.mean()) ** 2).sum()),
        "perm_entropy": _perm_entropy(x),
        "all_zero": 0.0,
    }
    return [float(out[k]) for k in SHAPE_FEATURES]


def feature_matrix(outbreaks) -> tuple[list[int], np.ndarray, list[str]]:
    """(outbreak ids, feature matrix, column names) for Outbreak rows (or objects with the same fields)."""
    outbreaks = list(outbreaks)
    events = sorted({o.event for o in outbreaks})
    diseases = sorted({o.disease for o in outbreaks})
    names = SHAPE_FEATURES + [f"event={e}" for e in events] + [f"disease={d}" for d in diseases]
    X = np.zeros((len(outbreaks), len(names)))
    ev_col = {e: len(SHAPE_FEATURES) + i for i, e in enumerate(events)}
    dz_col = {d: len(SHAPE_FEATURES) + len(events) + i for i, d in enumerate(diseases)}
    for r, o in enumerate(outbreaks):
        X[r, : len(SHAPE_FEATURES)] = series_features(o.values)
        X[r, ev_col[o.event]] = 1.0
        X[r, dz_col[o.disease]] = 1.0
    return [o.id for o in outbreaks], X, names
