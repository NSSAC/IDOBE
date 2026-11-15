"""Synthetic multi-wave series with known outbreak peaks, for tests and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Wave:
    peak_day: float
    height: float  # expected count per period at the peak
    rise_days: float  # Gaussian sd of the rising side
    fall_days: float  # Gaussian sd of the falling side


def expected(t_days: np.ndarray, waves: list[Wave], baseline: float = 0.0) -> np.ndarray:
    lam = np.full(t_days.shape, float(baseline))
    for w in waves:
        sd = np.where(t_days < w.peak_day, w.rise_days, w.fall_days)
        lam += w.height * np.exp(-0.5 * ((t_days - w.peak_day) / sd) ** 2)
    return lam


def make_series(
    waves: list[Wave],
    n_days: int,
    step_days: int = 7,
    baseline: float = 0.0,
    rng: np.random.Generator | None = None,
    start: str = "2020-01-04",
):
    """Return (dates, observed counts, expected counts). Poisson noise if rng given."""
    t = np.arange(0, n_days, step_days, dtype=float)
    lam = expected(t, waves, baseline)
    y = rng.poisson(lam).astype(float) if rng is not None else lam.copy()
    dates = pd.date_range(start, periods=len(t), freq=f"{step_days}D")
    return dates, y, lam


def random_waves(rng: np.random.Generator, n_days: int, height: float) -> list[Wave]:
    """Random multi-wave scenario: waves 12-40 weeks apart over the whole span, heights 0.2-1x."""
    waves = []
    day = rng.uniform(60, 150)
    while day < n_days - 60:
        waves.append(
            Wave(
                peak_day=day,
                height=height * rng.uniform(0.2, 1.0),
                rise_days=rng.uniform(10, 25),
                fall_days=rng.uniform(14, 40),
            )
        )
        day += rng.uniform(84, 280)
    return waves
