"""Synthetic outbreak generator for the IDOBE dataset.

An outbreak decomposes into three loosely independent pieces, and this module
models each one separately:

  1. Metadata   -- (disease, location, event), start_date and duration.
  2. Magnitude  -- the `total` (the overall scale of the curve).
  3. Shape      -- the normalized weekly curve, i.e. how the total is spread
                   across the outbreak's weeks.

Metadata, duration and magnitude are resampled empirically (bootstrapped from
real outbreaks) so their marginal and joint distributions match the source data
by construction. The *shape* -- which is what the repo's entropy / permutation
entropy / shape-moment statistics actually measure -- is generated fresh from a
per-disease functional PCA (FPCA) model, so the synthetic curves are new rather
than copied.

See README.md for the rationale and for how to make magnitude/duration fully
parametric later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns 0..59 hold the weekly values; everything else is metadata.
MAX_WEEKS = 60
VALUE_COLS = [str(i) for i in range(MAX_WEEKS)]
META_COLS = ["unique_id", "disease", "location", "event",
             "start_date", "end_date", "duration", "total"]

# Length of the common grid every curve is resampled onto before PCA.
GRID_LEN = 60
# A shape group needs at least this many real curves to get its own PCA model;
# thinner groups fall back to a coarser model (see the fallback chain in fit()).
MIN_GROUP_FOR_OWN_MODEL = 40
# Bin edges for the observed series length (the count of non-NaN weekly values,
# which -- note -- is not the `duration` metadata field; the value columns
# always extend past `duration`). Longer series also tend to peak later, so
# shape is conditioned on observed length to preserve that correlation.
# Edges track the length quartiles: [14,26), [26,33), [33,44), [44,60].
OBS_BINS = [14, 26, 33, 44, 61]


def _obs_bin(n_obs: int) -> int:
    """Index of the observed-length bin a given series length falls into."""
    return int(np.clip(np.digitize(n_obs, OBS_BINS[1:-1]),
                       0, len(OBS_BINS) - 2))


def _n_obs(curve: np.ndarray) -> int:
    """Number of observed (non-NaN) weekly values in a raw curve."""
    return int(np.count_nonzero(~np.isnan(curve)))


def _value_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the weekly-value columns as a float array (NaN past duration)."""
    return df[VALUE_COLS].to_numpy(dtype=float)


def _to_normalized_grid(curve: np.ndarray) -> np.ndarray | None:
    """Resample one outbreak's curve onto a fixed-length probability curve.

    Drops trailing NaNs, clips negatives, normalizes to sum 1, then linearly
    interpolates from the outbreak's own week index onto ``GRID_LEN`` points in
    [0, 1]. Returns None for curves that carry no signal (all zero / empty).
    """
    y = curve[~np.isnan(curve)]
    y = np.clip(y, 0.0, None)
    if y.size < 2 or y.sum() <= 0:
        return None
    y = y / y.sum()
    src = np.linspace(0.0, 1.0, num=y.size)
    dst = np.linspace(0.0, 1.0, num=GRID_LEN)
    g = np.interp(dst, src, y)
    s = g.sum()
    if s <= 0:
        return None
    return g / s


class _ShapeModel:
    """A functional-PCA model of normalized outbreak shapes.

    Fits PCA on length-normalized probability curves and samples new scores
    from a Gaussian fit to the real scores. Sampled curves are reconstructed on
    the common grid; callers rescale them to a concrete duration and total.
    """

    def __init__(self, n_components: int = 6, rng: np.random.Generator | None = None):
        self.n_components = n_components
        self.rng = rng or np.random.default_rng()
        self.mean_curve: np.ndarray | None = None
        self.components: np.ndarray | None = None      # (k, GRID_LEN)
        self.score_mean: np.ndarray | None = None      # (k,)
        self.score_cov: np.ndarray | None = None       # (k, k)

    def fit(self, grids: np.ndarray) -> "_ShapeModel":
        # sklearn is optional at import time; only needed when a model is fit.
        from sklearn.decomposition import PCA

        k = min(self.n_components, grids.shape[0] - 1, grids.shape[1])
        k = max(k, 1)
        pca = PCA(n_components=k)
        scores = pca.fit_transform(grids)
        self.mean_curve = pca.mean_.copy()
        self.components = pca.components_.copy()
        self.score_mean = scores.mean(axis=0)
        # rowvar=False -> variables are columns (the k components).
        self.score_cov = np.cov(scores, rowvar=False)
        if self.score_cov.ndim == 0:  # single component -> scalar variance
            self.score_cov = self.score_cov.reshape(1, 1)
        return self

    def sample_grid(self) -> np.ndarray:
        """Draw one new normalized shape on the common grid."""
        z = self.rng.multivariate_normal(self.score_mean, self.score_cov)
        curve = self.mean_curve + z @ self.components
        curve = np.clip(curve, 0.0, None)
        s = curve.sum()
        if s <= 0:  # degenerate draw; fall back to the mean shape
            curve = np.clip(self.mean_curve, 0.0, None)
            s = curve.sum()
        return curve / s


def _grid_to_length(grid: np.ndarray, length: int) -> np.ndarray:
    """Resample a grid-length probability curve onto ``length`` weeks."""
    src = np.linspace(0.0, 1.0, num=grid.size)
    dst = np.linspace(0.0, 1.0, num=length)
    out = np.clip(np.interp(dst, src, grid), 0.0, None)
    s = out.sum()
    if s <= 0:
        return np.full(length, 1.0 / length)
    return out / s


class SyntheticOutbreakGenerator:
    """Fit on the real IDOBE table, then sample synthetic outbreaks from it."""

    def __init__(self, n_components: int = 10, roughness: float = 0.04,
                 seed: int | None = None):
        # `roughness` adds multiplicative log-normal week-to-week noise to the
        # otherwise-smooth FPCA curves. Real outbreaks are noisier than a
        # low-rank reconstruction, which shows up as too-low permutation
        # entropy; a small roughness restores it. 0.04 matches the real PE
        # distribution; set 0 for the pure smooth reconstruction.
        self.n_components = n_components
        self.roughness = roughness
        self.rng = np.random.default_rng(seed)
        self._templates: pd.DataFrame | None = None
        self._models: dict[tuple[str, int], _ShapeModel] = {}
        self._bin_models: dict[int, _ShapeModel] = {}
        self._global_model: _ShapeModel | None = None

    def fit(self, df: pd.DataFrame) -> "SyntheticOutbreakGenerator":
        values = _value_matrix(df)

        # Template rows carry the metadata/duration/total/dates we bootstrap
        # from, plus the observed series length that fixes each curve's support.
        self._templates = df[META_COLS].reset_index(drop=True)
        self._templates["_n_obs"] = [_n_obs(row) for row in values]

        # Collect normalized shapes keyed at three granularities so we can fall
        # back from the most specific model to coarser ones when data is thin.
        by_disease_bin: dict[tuple[str, int], list[np.ndarray]] = {}
        by_bin: dict[int, list[np.ndarray]] = {}
        all_grids: list[np.ndarray] = []
        for disease, row in zip(df["disease"].to_numpy(), values):
            g = _to_normalized_grid(row)
            if g is None:
                continue
            b = _obs_bin(_n_obs(row))
            by_disease_bin.setdefault((disease, b), []).append(g)
            by_bin.setdefault(b, []).append(g)
            all_grids.append(g)

        if not all_grids:
            raise ValueError("No usable outbreak curves found to fit the shape model.")

        self._global_model = self._fit_model(all_grids)
        self._bin_models = {b: self._fit_model(grids)
                            for b, grids in by_bin.items()}
        self._models = {key: self._fit_model(grids)
                        for key, grids in by_disease_bin.items()
                        if len(grids) >= MIN_GROUP_FOR_OWN_MODEL}
        return self

    def _fit_model(self, grids: list[np.ndarray]) -> _ShapeModel:
        return _ShapeModel(self.n_components, self.rng).fit(np.vstack(grids))

    def _model_for(self, disease: str, n_obs: int) -> _ShapeModel:
        b = _obs_bin(n_obs)
        # (disease, bin) -> (bin) -> global.
        return (self._models.get((disease, b))
                or self._bin_models.get(b)
                or self._global_model)

    def sample(self, n: int, start_uid: int = 1) -> pd.DataFrame:
        """Generate ``n`` synthetic outbreaks as a DataFrame in the raw schema."""
        if self._templates is None:
            raise RuntimeError("Call fit() before sample().")

        idx = self.rng.integers(0, len(self._templates), size=n)
        rows = []
        for i, t_idx in enumerate(idx):
            t = self._templates.iloc[t_idx]
            total = float(t["total"])
            # The curve occupies the template's observed length, not `duration`
            # (the value columns extend past duration in the source data).
            n_weeks = int(t["_n_obs"])

            grid = self._model_for(t["disease"], n_weeks).sample_grid()
            shape = _grid_to_length(grid, n_weeks)
            if self.roughness > 0:
                shape = shape * np.exp(
                    self.rng.normal(0.0, self.roughness, size=n_weeks))
                shape = shape / shape.sum()
            weekly = shape * total

            record = {
                "unique_id": start_uid + i,
                "disease": t["disease"],
                "location": t["location"],
                "event": t["event"],
                # Metadata (dates, duration) is bootstrapped verbatim from the
                # template row, so it stays internally consistent and its
                # distribution matches the source.
                "start_date": t["start_date"],
                "end_date": t["end_date"],
                "duration": int(t["duration"]),
                "total": total,
            }
            for w in range(MAX_WEEKS):
                record[str(w)] = weekly[w] if w < n_weeks else np.nan
            rows.append(record)

        return pd.DataFrame(rows, columns=META_COLS + VALUE_COLS)
