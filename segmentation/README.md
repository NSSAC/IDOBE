# Outbreak segmentation (prototype)

Prominence-based segmenter proposed to replace the epidemicKabu-based segmentation
used for the released `raw_data/outbreaks_disease_location.csv`. Status: prototype,
parameters not yet tuned on expert labels. The released CSV is unchanged.

## Usage

```python
from segmentation import ProminenceSegmenter, WEEKLY, DAILY

seg = ProminenceSegmenter(WEEKLY)          # or DAILY, COVID_WEEKLY, COVID_DAILY, or ProminenceParams(...)
segments = seg.segment(dates, values)      # list[Segment], indices into the inputs
frame = seg.segment_frame(dates, values)   # same, as a DataFrame with dates
```

Each `Segment` has:

| field | meaning |
| --- | --- |
| `start`, `end` | trough-to-trough bounds (consecutive outbreaks tile the series, like the legacy data) |
| `onset`, `offset` | trimmed bounds: first/last point above `baseline + onset_frac * (peak - baseline)` |
| `peak` | peak of the smoothed curve |
| `score` | peak prominence divided by the local scale (higher = clearer outbreak) |

## Outbreak rows (decided conventions)

`segmentation.outbreaks.segment_to_outbreaks(dates, values)` returns rows in the released
wide schema (plus `onset_date`, `peak_date`, `offset_date`, `score`, `method`,
`method_version`):

- an outbreak runs **onset to offset** (baseline weeks excluded);
- **4 weeks of context padding** before the onset and after the offset, clipped at series
  edges; `start_date`/`end_date` are the padded bounds, `duration` excludes the padding, so a
  full row has `duration + 9` values, as in the released CSV;
- **every wave is its own outbreak**, including COVID-19;
- outbreaks with `duration` outside 8 to 51 weeks are dropped (configurable).

On the series rebuilt from the released CSV this keeps about 95% of detected outbreaks; median
core duration is 15 weeks for COVID-19 and 29 for ILI (legacy trough-to-trough: 24 and 28).

## Presets

`WEEKLY`/`DAILY` suit seasonal pathogens (ILI, flu, RSV). `COVID_WEEKLY`/`COVID_DAILY` lower
`rel_prominence` to 0.05 so a big COVID-19 wave does not hide a smaller neighbour (each wave is
its own outbreak). Do not use the COVID presets on seasonal series: on ILI they split a season
into ~1.7 outbreaks per year.

## Method

Per contiguous run of observations (short gaps interpolated, long gaps split the series):

1. Variance-stabilise (`sqrt`) and Gaussian-smooth with a width given in days.
2. Keep peaks whose prominence is at least `rel_prominence` x the local scale
   (max of the smoothed curve within `local_window_days`) **and** at least
   `noise_prominence` x the estimated sd of the smoothed noise, at least
   `min_separation_days` apart, with smoothed height >= `min_peak_value`.
3. Cut at the lowest point between consecutive kept peaks.
4. Trim onset and offset at `onset_frac` of the rise above each trough.

| parameter | default | role |
| --- | --- | --- |
| `smooth_sigma_days` | 10.5 | Gaussian sd (1.5 weeks) |
| `transform` | `sqrt` | variance stabilisation (`log1p`, `none`) |
| `rel_prominence` | 0.15 | minimum prominence relative to the local scale |
| `local_window_days` | 365 | window for the local scale |
| `noise_prominence` | 5.0 | minimum prominence in units of smoothed-noise sd |
| `min_separation_days` | 56 | minimum distance between peaks |
| `min_peak_value` | 0 | minimum smoothed peak height, original units |
| `onset_frac` | 0.10 | onset/offset threshold |
| `max_gap_days` | 21 (weekly), 7 (daily) | longest gap that is interpolated |
| `min_run_points` | 8 (weekly), 56 (daily) | shorter runs produce no outbreaks |

## Comparison with epidemicKabu

`legacy_kabu.py` re-implements the epidemicKabu 0.2.7 cut rule (no dependency on the
package). `python -m segmentation.evaluate` writes:

- `results/synthetic_comparison.csv`: synthetic 5-year series with known waves at three count
  levels; peak recall/precision (match within 28 days) and timing error.
- `results/legacy_agreement.csv`: series rebuilt from the released CSV; how many legacy
  boundaries each method reproduces (`legacy_recall`), how many of its interior cuts fall in a
  legacy boundary window (`legacy_precision`), and outbreaks per series-year.

Legacy boundary windows: each CSV row holds the core outbreak plus 9 weeks of context, and
consecutive rows of a series overlap by exactly those 9 weeks, so each legacy cut lies inside
the overlap.

## Tests

```
pip install pytest
python -m pytest segmentation
```
