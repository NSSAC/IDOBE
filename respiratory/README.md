# IDOBE respiratory dataset

A separate version of IDOBE built only from recent respiratory surveillance data. The original
dataset in `raw_data/` is not modified.

## Sources

| Source | What | Levels | Event name | Dataset |
|---|---|---|---|---|
| JHU CSSE | COVID-19 weekly new cases and deaths, 2020-01 to 2023-03 | national, state, county | `CASES`, `DEATHS` | `CSSEGISandData/COVID-19` US time series |
| NHSN | Weekly new hospital admissions: COVID-19, influenza, RSV | national, HHS region, state | `HOSPITALIZATIONS` | data.cdc.gov `ua7e-t2fy` |
| NSSP | Percent of emergency department visits: COVID-19, influenza, RSV | national, state, HSA (sub-state) | `PERCENT_ED_VISITS` | data.cdc.gov `rdmq-nq56` |
| RESP-NET | Hospitalization rate per 100,000 (COVID-NET, FluSurv-NET, RSV-NET), all ages | network-wide, state catchment | `HOSPITALIZATION_RATE` | data.cdc.gov `kvib-3txy` |
| ILINet | Percent of outpatient visits for influenza-like illness | national, HHS region, state | `PERCENT_UNWEIGHTED`, `PERCENT_WEIGHTED` | Delphi Epidata `fluview` (mirror of CDC FluView) |

Notes on the sources:

- NSSP publishes sub-state values per Health Service Area (HSA). Every county in an HSA carries the same
  numbers, so there is one series per HSA (`geo_level = hsa`), and `data/nssp_hsa_county_crosswalk.csv`
  lists the counties in each one. JHU CSSE is the only true county-level source here.
- RESP-NET covers selected counties in participating states. Its "national" series is the whole network
  catchment, not the whole country.
- NHSN reporting was voluntary from May to October 2024, so counts in that window are lower than true.
- JHU cumulative counts are differenced to daily counts and summed over Sunday to Saturday weeks. Weekly
  counts that come out negative because of downward corrections are set to 0.

## Pipeline

Run from the repository root. Needs `pandas`, `numpy`, `requests`, `scipy`.

```
python -m respiratory.fetch            # 1. download raw pulls        -> data/raw/<source>/<as_of>/
python -m respiratory.build_series     # 2. clean into weekly series  -> data/series_long.csv.gz
python -m respiratory.segment          # 3. split into outbreaks      -> data/segments.csv
python -m respiratory.build_outbreaks  # 4. write the benchmark CSV   -> data/outbreaks_respiratory.csv
```

Steps 1 and 2 know nothing about segmentation. Step 3 is the only place a splitting method lives, and
step 4 only reads its `segments.csv`, so a new method can replace step 3 without touching the rest.
Each raw pull is stored under the date the source was last updated and old pulls are kept.

## Files in `data/`

- `series_long.csv.gz`: every full, unsegmented weekly series, one row per week. Columns: `series_id`,
  `source`, `disease`, `location`, `location_fips`, `geo_level` (`national`, `hhs_region`, `state`, `hsa`,
  `county`), `event`, `target`, `date` (week-ending Saturday), `value`, `as_of`. `location_fips` is `US`,
  a 2-digit state code, a 5-digit county code, `HHS<n>`, or an HSA id. `target` holds the Hubverse target
  name for NHSN series (`wk inc flu hosp`, `wk inc covid hosp`, `wk inc rsv hosp`). Missing weeks are
  left empty.
- `series_index.csv`: one row per series with its coverage.
- `segments.csv`: `series_id`, `segment_id`, `method`, `method_version`, `start_date`, `end_date`,
  `peak_date`, `score`. `start_date` and `end_date` are the onset and end of the core outbreak, without
  padding.
- `outbreaks_respiratory.csv`: the benchmark file, with exactly the columns of
  `raw_data/outbreaks_disease_location.csv` (`unique_id`, `disease`, `location`, `event`, `start_date`,
  `end_date`, `duration`, `total`, `0`..`59`). `unique_id` starts at 100001.
- `outbreaks_respiratory_meta.csv`: one row per `unique_id` with `series_id`, `segment_id`, `source`,
  `geo_level`, `location_fips`, `target`, core start, end and peak dates, `season` (July 1 boundary, by
  peak week), `holdout` (seasons 2024-25 and 2025-26), `truncated` (the context padding was cut short by
  the start or end of the series or by a reporting gap, so the outbreak may be incomplete), `n_interpolated`, method and `as_of`.
- `legacy_respiratory_outbreaks.csv`: the COVID-19, influenza, ILI and RSV rows of the original dataset,
  copied unchanged. They are not merged into the new file because the same sources are re-derived here.

## Row conventions (same as the original dataset)

Each row holds the core outbreak plus 4 weeks of context before and after, cut short at the ends of a
series or at a reporting gap. `start_date` and `end_date` are the first and last week in the row.
`duration` is the core length in weeks (8 to 51) and excludes the padding. `total` is the sum of the row.
Gaps of up to 2 weeks inside a row are linearly interpolated.

## Segmentation

`segment.py` uses the prominence segmenter from `segmentation/` (WEEKLY preset) by default, following the
conventions Srini set on 2026-11-15:

- An outbreak runs from onset to end (the segmenter's onset and offset), not trough to trough, so flat
  baseline weeks are not counted in `duration`. They appear only as the 4 weeks of context padding.
- Every wave is its own outbreak, including each COVID-19 wave. With WEEKLY's `rel_prominence = 0.15`,
  a large COVID-19 wave hides smaller neighbouring ones (in JHU national cases, summer 2020 merged into
  winter 2020-21 and Delta into Omicron), so COVID-19 series use `rel_prominence = 0.05`. That gives six
  separate waves in JHU national cases and eleven in NHSN national COVID-19 admissions.
- Outbreaks with an onset-to-end length outside 8 to 51 weeks are dropped.

The earlier trough-to-trough stand-in, `gaussian_valley`, is still available with
`python -m respiratory.segment --method gaussian_valley`.

Needs `scipy` for the prominence segmenter.
