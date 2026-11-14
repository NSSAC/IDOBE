"""Re-implementation of the epidemicKabu (v0.2.7) wave cut rule, for comparison only.

Mirrors ``EpidemicKabu.waves.run``: normalise by the maximum, smooth with a
date-based Gaussian kernel (sigma = kernel / 2 days, weights renormalised at the
edges, as in ``curveSmoothing2``), take the first difference, smooth it again,
and cut where the smoothed derivative changes from negative to positive,
choosing the neighbouring date whose derivative is closer to zero. The optional
``fbi`` filter drops cuts closer than ``fbi`` days to the last kept cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def _smooth(t_days: np.ndarray, x: np.ndarray, sigma_days: float) -> np.ndarray:
    """Gaussian smoothing over dates with weights renormalised at the edges.

    Assumes regular spacing (true for IDOBE series); equivalent to the library's
    O(n^2) loop up to the truncation of the kernel at 4 sigma.
    """
    step = float(np.median(np.diff(t_days))) if len(t_days) > 1 else 1.0
    sig = sigma_days / step
    num = gaussian_filter1d(x, sig, mode="constant", cval=0.0)
    den = gaussian_filter1d(np.ones_like(x), sig, mode="constant", cval=0.0)
    return num / den


def kabu_cuts(dates, values, kernel_days: float = 18, fbi_days: float = 0) -> list[int]:
    d = pd.to_datetime(pd.Index(dates))
    t = (d - d[0]).days.values.astype(float)
    y = np.asarray(values, dtype=float)
    yn = y / np.abs(y).max() if np.abs(y).max() > 0 else y
    sigma = kernel_days / 2
    s = _smooth(t, yn, sigma)
    d1 = np.r_[np.nan, np.diff(s)]
    d1[0] = 0.0  # pandas rolling(2) gives NaN; the library's smoothing treats it as missing
    ds = _smooth(t, d1, sigma)
    cuts = []
    for i in range(1, len(ds)):
        if ds[i - 1] < 0 and ds[i] > 0:
            cuts.append(i if abs(ds[i]) < abs(ds[i - 1]) else i - 1)
    if fbi_days > 0 and cuts:
        kept = [cuts[0]]
        for c in cuts[1:]:
            if t[c] - t[kept[-1]] > fbi_days:
                kept.append(c)
        cuts = kept
    return cuts


def kabu_segments(dates, values, kernel_days: float = 18, fbi_days: float = 0):
    """Return (start, end_inclusive) index pairs between consecutive cuts (and the series edges)."""
    cuts = kabu_cuts(dates, values, kernel_days, fbi_days)
    bounds = [0] + cuts + [len(values)]
    return [(a, b - 1) for a, b in zip(bounds[:-1], bounds[1:]) if b - 1 >= a]
