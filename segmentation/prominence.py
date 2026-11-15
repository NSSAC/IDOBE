"""Prominence-based outbreak segmentation for IDOBE.

A replacement candidate for the epidemicKabu-based segmentation. It keeps the
trough-to-trough idea (outbreaks tile the series between local minima) but only
cuts between peaks that stand out from the noise, and additionally reports a
trimmed onset/offset for each outbreak so that long flat baselines can be
excluded.

Algorithm (per contiguous run of observations):

1. Variance-stabilize the counts (``sqrt`` by default, so Poisson noise has
   roughly constant variance whatever the level) and Gaussian-smooth them.
   The smoothing width is given in days, so the same parameters work for daily
   and weekly series.
2. Find peaks of the smoothed curve whose topographic prominence is at least
   ``rel_prominence`` times the local scale of the series (the maximum of the
   smoothed curve within ``local_window_days``) and at least
   ``noise_prominence`` times the estimated sd of the smoothed noise, and that
   are at least ``min_separation_days`` apart. The local scale means a small
   flu season next to a huge one is still detected; the noise floor stops
   wiggles in long quiet periods from being called outbreaks.
3. Drop peaks whose smoothed height (original units) is below ``min_peak_value``.
4. Cut at the lowest point of the smoothed curve between each pair of kept
   peaks (the middle of the lowest plateau when it is flat). The first outbreak
   starts at the lowest point before its peak, the last ends at the lowest
   point after its peak.
5. Within each outbreak, the onset is the first time the smoothed curve
   (original units) rises above ``baseline + onset_frac * (peak - baseline)``
   and the offset is the last time it is above the same level on the way down,
   with the baseline taken at the corresponding trough.

Missing values: gaps of at most ``max_gap_days`` are linearly interpolated;
longer gaps (or missing dates) split the series into runs that are segmented
independently.

All indices returned refer to positions in the arrays passed to ``segment``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks

__all__ = ["Segment", "ProminenceParams", "ProminenceSegmenter", "WEEKLY", "DAILY"]


@dataclass(frozen=True)
class Segment:
    """One outbreak. Indices are positions in the input arrays (inclusive)."""

    start: int  # trough-to-trough start
    end: int  # trough-to-trough end (inclusive)
    peak: int  # peak of the smoothed curve
    onset: int  # first point above the onset threshold
    offset: int  # last point above the offset threshold
    score: float  # peak prominence / local scale (>= rel_prominence)


@dataclass(frozen=True)
class ProminenceParams:
    smooth_sigma_days: float = 10.5
    transform: str = "sqrt"  # "sqrt", "log1p" or "none"
    rel_prominence: float = 0.15
    local_window_days: float = 365.0
    min_separation_days: float = 56.0
    noise_prominence: float = 5.0
    min_peak_value: float = 0.0
    onset_frac: float = 0.10
    max_gap_days: float = 21.0
    min_run_points: int = 8


# Starting points per cadence; to be tuned on expert labels.
WEEKLY = ProminenceParams()
DAILY = ProminenceParams(smooth_sigma_days=10.5, max_gap_days=7.0, min_run_points=56)


_TRANSFORMS = {
    "sqrt": (np.sqrt, np.square),
    "log1p": (np.log1p, np.expm1),
    "none": (lambda x: x, lambda x: x),
}


@dataclass
class ProminenceSegmenter:
    params: ProminenceParams = field(default_factory=ProminenceParams)
    name: str = "prominence"
    version: str = "0.1.0"

    # ------------------------------------------------------------------ API
    def segment(self, dates: Sequence, values: Sequence[float]) -> list[Segment]:
        """Segment one series. ``dates`` must be sorted ascending."""
        d = pd.to_datetime(pd.Index(dates))
        y = np.asarray(values, dtype=float)
        if len(d) != len(y):
            raise ValueError("dates and values must have the same length")
        if len(y) == 0:
            return []
        if not d.is_monotonic_increasing:
            raise ValueError("dates must be sorted ascending")
        step = self._step_days(d)
        out: list[Segment] = []
        for lo, hi in self._runs(d, y, step):
            run = self._interpolate(y[lo:hi])
            for s in self._segment_run(run, step):
                out.append(
                    Segment(
                        start=s.start + lo,
                        end=s.end + lo,
                        peak=s.peak + lo,
                        onset=s.onset + lo,
                        offset=s.offset + lo,
                        score=s.score,
                    )
                )
        return out

    def segment_frame(self, dates: Sequence, values: Sequence[float]) -> pd.DataFrame:
        """Same as ``segment`` but returns a DataFrame with dates attached."""
        d = pd.to_datetime(pd.Index(dates))
        rows = []
        for i, s in enumerate(self.segment(d, values)):
            r = asdict(s)
            r.update(
                segment_id=i,
                start_date=d[s.start],
                end_date=d[s.end],
                peak_date=d[s.peak],
                onset_date=d[s.onset],
                offset_date=d[s.offset],
                method=self.name,
                method_version=self.version,
            )
            rows.append(r)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _step_days(d: pd.DatetimeIndex) -> float:
        if len(d) < 2:
            return 7.0
        diffs = np.diff(d.values).astype("timedelta64[s]").astype(float) / 86400.0
        return float(np.median(diffs))

    def _runs(self, d: pd.DatetimeIndex, y: np.ndarray, step: float):
        """Split into runs at long date gaps or long stretches of NaN."""
        p = self.params
        max_missing = int(np.floor(p.max_gap_days / step))
        breaks = []
        if len(d) > 1:
            diffs = np.diff(d.values).astype("timedelta64[s]").astype(float) / 86400.0
            breaks = list(np.where(diffs > p.max_gap_days + 0.5 * step)[0] + 1)
        bounds = [0] + breaks + [len(y)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            # split further on NaN stretches longer than max_missing
            nan = np.isnan(y[a:b])
            i = 0
            start = None
            n = b - a
            while i < n:
                if nan[i]:
                    j = i
                    while j < n and nan[j]:
                        j += 1
                    if j - i > max_missing:
                        if start is not None:
                            yield from self._emit(a + start, a + i)
                        start = None
                    i = j
                else:
                    if start is None:
                        start = i
                    i += 1
            if start is not None:
                yield from self._emit(a + start, b)

    def _emit(self, lo: int, hi: int):
        if hi - lo >= self.params.min_run_points:
            yield lo, hi

    @staticmethod
    def _interpolate(run: np.ndarray) -> np.ndarray:
        run = run.copy()
        nan = np.isnan(run)
        if nan.any():
            idx = np.arange(len(run))
            run[nan] = np.interp(idx[nan], idx[~nan], run[~nan])
        return run

    def _segment_run(self, y: np.ndarray, step: float) -> list[Segment]:
        p = self.params
        fwd, _ = _TRANSFORMS[p.transform]
        y = np.clip(y, 0, None)
        sigma = max(p.smooth_sigma_days / step, 1e-9)
        z = gaussian_filter1d(fwd(y), sigma, mode="nearest")
        level = gaussian_filter1d(y, sigma, mode="nearest")  # original units
        if np.ptp(z) <= 0:
            return []

        window = max(int(round(p.local_window_days / step)) | 1, 3)
        scale = maximum_filter1d(z, size=window, mode="nearest")
        # Noise floor: robust sd of the residuals, shrunk by the smoothing.
        resid = fwd(y) - z
        noise_sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        smoothed_sd = noise_sd / np.sqrt(2 * np.sqrt(np.pi) * max(sigma, 0.5))
        required = np.maximum(p.rel_prominence * scale, p.noise_prominence * smoothed_sd)
        required = np.maximum(required, 1e-12)
        distance = max(int(round(p.min_separation_days / step)), 1)
        peaks, props = find_peaks(z, prominence=(required, None), distance=distance)
        if len(peaks) == 0:
            return []
        keep = level[peaks] >= p.min_peak_value
        peaks = peaks[keep]
        scores = props["prominences"][keep] / scale[peaks]
        if len(peaks) == 0:
            return []

        n = len(z)
        troughs = [self._trough(z, 0, peaks[0])]
        for a, b in zip(peaks[:-1], peaks[1:]):
            troughs.append(self._trough(z, a, b + 1))
        troughs.append(self._trough(z, peaks[-1], n))

        segs = []
        for k, pk in enumerate(peaks):
            start = troughs[k]
            end = troughs[k + 1] if k == len(peaks) - 1 else troughs[k + 1] - 1
            end = max(end, pk)
            onset = self._crossing(level, start, pk, p.onset_frac, rising=True)
            offset = self._crossing(level, pk, end, p.onset_frac, rising=False)
            segs.append(Segment(int(start), int(end), int(pk), int(onset), int(offset), float(scores[k])))
        return segs

    @staticmethod
    def _trough(z: np.ndarray, a: int, b: int) -> int:
        """Middle of the lowest plateau of z[a:b]."""
        seg = z[a:b]
        lo = seg.min()
        tol = 1e-9 * max(np.ptp(z), 1e-12)
        cand = np.where(seg <= lo + tol)[0]
        return int(a + cand[len(cand) // 2])

    @staticmethod
    def _crossing(level: np.ndarray, a: int, b: int, frac: float, rising: bool) -> int:
        if rising:
            base, top = level[a], level[b]
            thr = base + frac * (top - base)
            idx = np.where(level[a : b + 1] >= thr)[0]
            return int(a + idx[0]) if len(idx) else int(b)
        base, top = level[b], level[a]
        thr = base + frac * (top - base)
        idx = np.where(level[a : b + 1] >= thr)[0]
        return int(a + idx[-1]) if len(idx) else int(a)
