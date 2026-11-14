"""A small, fully documented reference segmenter ("seasonal threshold").

It is not meant to be the final IDOBE method. It exists so the pipeline and the expert review UI
work end to end on new series before a replacement for epidemicKabu is chosen, and as a transparent
baseline that expert labels can score other methods against.

Steps, on each series:
0. Split the series at runs of `max_gap` or more missing weeks; each piece is segmented on its own,
   so no window spans a reporting gap.
1. Fill missing weeks by linear interpolation, clip at 0, smooth with a centered moving average.
2. Local baseline = a low rolling quantile; local scale = a high rolling quantile (both over
   `window` weeks). A week is "active" when it exceeds baseline + `rise` * (scale - baseline)
   and the local scale is meaningfully above the baseline.
3. Active runs separated by fewer than `merge_gap` quiet weeks are merged; runs shorter than
   `min_weeks` are dropped.
4. Each run is widened to the nearest trough on either side (at most `pad` weeks), without
   touching its neighbours, then trimmed to at most `max_weeks` around its mass.
Score = log(peak / baseline), a rough measure of how pronounced the outbreak is.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .base import Segment, Segmenter


def _fill(values: Sequence[float | None]) -> np.ndarray:
    x = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    ok = ~np.isnan(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.clip(np.interp(idx, idx[ok], x[ok]), 0, None)


def _rolling(x: np.ndarray, window: int, q: float) -> np.ndarray:
    half = window // 2
    return np.array([np.quantile(x[max(0, i - half): i + half + 1], q) for i in range(len(x))])


class SeasonalThreshold(Segmenter):
    name = "seasonal-threshold"
    version = "0.1"

    @classmethod
    def default_params(cls):
        return {"max_gap": 4, "smooth": 3, "window": 104, "low_q": 0.15, "high_q": 0.95, "rise": 0.2,
                "min_ratio": 1.5, "merge_gap": 2, "min_weeks": 6, "pad": 8, "max_weeks": 60}

    def segment(self, values, *, event="", disease=""):
        out: list[Segment] = []
        for a, b in self._pieces(values, self.params["max_gap"]):
            for seg in self._segment_piece(values[a:b]):
                out.append(Segment(seg.start + a, seg.end + a, seg.score,
                                   peak=None if seg.peak is None else seg.peak + a))
        return out

    @staticmethod
    def _pieces(values, max_gap):
        """[start, stop) ranges of the series separated by >= max_gap missing weeks, trimmed of
        missing weeks at either end."""
        pieces, start, gap = [], None, 0
        for i, v in enumerate(list(values) + [None] * max_gap):
            if v is None:
                gap += 1
                if start is not None and gap >= max_gap:
                    pieces.append((start, i - gap + 1))
                    start = None
            else:
                if start is None:
                    start = i
                gap = 0
        return pieces

    def _segment_piece(self, values):
        p = self.params
        n = len(values)
        if n < p["min_weeks"]:
            return []
        raw = _fill(values)
        k = max(1, int(p["smooth"]))
        x = np.convolve(np.pad(raw, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        base = _rolling(x, p["window"], p["low_q"])
        top = _rolling(x, p["window"], p["high_q"])
        eps = 1e-9 + 1e-6 * (x.max() if x.size else 0)
        active = (x > base + p["rise"] * (top - base)) & (top > p["min_ratio"] * base + eps)

        runs, i = [], 0
        while i < n:
            if active[i]:
                j = i
                while j + 1 < n and active[j + 1]:
                    j += 1
                if runs and i - runs[-1][1] - 1 < p["merge_gap"]:
                    runs[-1][1] = j
                else:
                    runs.append([i, j])
                i = j + 1
            else:
                i += 1
        runs = [r for r in runs if r[1] - r[0] + 1 >= p["min_weeks"]]

        out: list[Segment] = []
        for idx, (a, b) in enumerate(runs):
            lo = out[-1].end + 1 if out else 0
            hi = runs[idx + 1][0] - 1 if idx + 1 < len(runs) else n - 1
            a = self._to_trough(x, a, -1, max(lo, a - p["pad"]))
            b = self._to_trough(x, b, +1, min(hi, b + p["pad"]))
            a, b = self._trim(raw, a, b, p["max_weeks"])
            peak = float(raw[a:b + 1].max())
            score = math.log((peak + eps) / (float(np.median(base[a:b + 1])) + eps))
            out.append(Segment(int(a), int(b), round(score, 4), peak=int(a + np.argmax(raw[a:b + 1]))))
        return out

    @staticmethod
    def _to_trough(x, i, step, limit):
        best = i
        j = i
        while (j + step >= limit if step < 0 else j + step <= limit):
            j += step
            if x[j] <= x[best]:
                best = j
        return best

    @staticmethod
    def _trim(raw, a, b, max_weeks):
        if b - a + 1 <= max_weeks:
            return a, b
        w = raw[a:b + 1]
        sums = np.convolve(w, np.ones(max_weeks), mode="valid")
        s = int(np.argmax(sums))
        return a + s, a + s + max_weeks - 1
