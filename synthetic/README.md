# Synthetic outbreak dataset

Tools for generating a synthetic outbreak table that is **statistically similar**
to `raw_data/outbreaks_disease_location.csv`, and for validating that similarity
using IDOBE's own statistics.

"Statistically similar" is defined here exactly as in `stats/stats_compute.ipynb`:
the synthetic curves should reproduce the distributions of

- **Shannon entropy** of the normalized curve,
- **permutation entropy** (`ordpy`, `dx=3`), and
- the **shape moments** (mean, median, variance, skewness, excess kurtosis) of the
  curve treated as a probability distribution over weeks.

## Quick start

```bash
pip install pandas numpy scikit-learn scipy ordpy

# Generate a synthetic table (same size as the real one by default)
python synthetic/generate.py --seed 0

# Validate it against the real data using the repo's own metrics
python synthetic/validate.py --report synthetic/output/validation_report.csv
```

The output CSV has the **exact same schema** as `raw_data/outbreaks_disease_location.csv`
(`unique_id, disease, location, event, start_date, end_date, duration, total, 0..59`),
so the existing `stats/` and `baselines/` pipelines consume it unchanged.

## How it works

An outbreak decomposes into three loosely independent pieces, each modeled
separately (`generator.py`):

1. **Metadata** — `(disease, location, event)`, dates and `duration`. Bootstrapped
   verbatim from real rows, so their marginal and joint distributions match by
   construction.
2. **Magnitude** — `total`, the overall scale of the curve. Also taken from the
   bootstrapped template row.
3. **Shape** — the normalized weekly curve, generated fresh from a **functional
   PCA (FPCA)** model so the curves are new rather than copied. Real curves are
   length-normalized onto a common grid, PCA is fit on them, and new curves are
   sampled from a Gaussian fit to the PCA scores, then rescaled.

Two details matter for statistical fidelity, and both came directly out of the
validation harness:

- **Shape is conditioned on the observed series length.** The number of non-NaN
  weekly values (mean ≈ 35) is *not* the `duration` field (mean ≈ 26) — the value
  columns always extend past `duration`. Curves are emitted at the template's
  observed length, and the shape model is binned by that length, because longer
  series also tend to peak later. Getting this wrong shifts the whole
  center-of-mass distribution (mean-week KS jumps from ~0.03 to ~0.29).
- **A small roughness term** (`roughness`, default 0.04) adds multiplicative
  week-to-week noise. A low-rank FPCA reconstruction is too smooth and has
  too-low permutation entropy; a little roughness restores it to the real level.

## Validation

`validate.py` runs three checks. Each is judged against how much two samples
of **real** data differ from each other (the real-vs-real null), not against a
fixed cutoff: with ~10,000 rows, even a KS of 0.03 is outside sampling noise.

1. **Per-feature KS vs null.** The notebook's seven metrics plus nine features
   for noise, count structure, length and scale (lag-1 autocorrelation, zero
   fraction, integer fraction, peak share, SD of week-to-week log change,
   dispersion index, series length, log total, relative peak position). A
   feature passes when KS(real, synthetic) is within the 95th percentile of KS
   between a same-size real subsample and the remaining real rows. Pooled and
   per stratum (`--by disease` by default).
2. **Classifier two-sample test.** A gradient-boosted classifier tries to tell
   real from synthetic (5-fold ROC AUC; 0.5 = indistinguishable), next to the
   same test on two halves of the real data. Permutation importances name the
   features that give synthetic data away.
3. **Memorization.** Nearest-neighbor distance from each synthetic curve to the
   real curves, against the same distance for held-out real curves. About 5%
   of synthetic curves should fall below the real 5th percentile; much more
   suggests copying.

```bash
python synthetic/validate.py \
    --report synthetic/output/validation_report.csv \
    --summary synthetic/output/validation_summary.json
```

Results for the current generator (`--seed 0`, full size, 9,962 rows) are in
`output/validation_report.csv` and `output/validation_summary.json`:

| check | result | real-vs-real null |
|-------|--------|-------------------|
| features within null (pooled) | 2 of 16 (`n_obs`, `log_total`, both copied from real rows) | all |
| classifier AUC, notebook metrics only | 0.855 | 0.51 |
| classifier AUC, all features | 0.997 | 0.51 |
| top giveaway features | integer fraction, zero fraction, week-to-week log change, PE | |
| synthetic below real NN 5th pct | 0.006 (no copying) | 0.05 |

The shape moments are close in mean but still outside sampling noise; entropy
and PE are under-dispersed; and the output is continuous where real counts are
integers, which alone makes it trivially separable. The earlier 100-row run
(`output/validation_report_100.csv`) used the old pooled KS-only check.

## Extending this (next fidelity levers)

- **Sharpen the entropy/PE spread** by sampling PCA scores from a KDE or by
  bootstrapping real score vectors instead of a Gaussian, and/or by drawing
  `roughness` per curve from a distribution rather than using a fixed value.
- **Fully parametric metadata/magnitude**: replace the template bootstrap with
  fitted distributions (e.g. per-disease log-normal `total`, empirical
  `(disease, location, event)` frequencies) if you need to extrapolate beyond the
  observed combinations rather than resample them.
- **Count observation layer**: round or sample counts (Poisson / negative
  binomial) for count-valued series; this is the largest single giveaway.

## Files

- `generator.py` — the `SyntheticOutbreakGenerator` model (fit / sample).
- `generate.py` — CLI that fits on the real table and writes a synthetic CSV.
- `validate.py` — per-feature KS against a real-vs-real null (pooled and per
  disease), classifier two-sample test, and memorization check.
- `output/validation_report.csv` — per-stratum, per-metric table from the latest run.
- `output/validation_summary.json` — classifier and memorization results.
