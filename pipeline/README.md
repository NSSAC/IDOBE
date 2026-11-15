# IDOBE benchmark pipeline

Turns curated surveillance series into the IDOBE outbreak benchmark, with expert review of where
each outbreak starts and ends. Every stage reads and writes plain CSV (`formats.py`), so a stage
can be swapped (e.g. a new segmentation method) without touching the others.

```
curated series ──► segmentation ──► expert review (annotation app) ──► consensus ──► benchmark CSV
 (series.csv)      (segments.csv)     /segments page                  (consensus_   (raw_data schema,
                                                                       segments.csv)  + provenance)
```

| Stage | Command | Output |
|---|---|---|
| Rebuild full series + current windows from the published CSV | `python -m pipeline.segment --legacy raw_data/outbreaks_disease_location.csv --diseases COVID-19,INFLUENZA,ILI,RSV --series-out series.csv --out legacy_segments.csv` | series + segments |
| Segment series with a registered method | `python -m pipeline.segment series.csv --method seasonal-threshold --out segments.csv` | segments |
| Load a run for expert review | `python annotation/scripts/load_segments.py --series series.csv --segments segments.csv --run-name NAME` (or `--legacy`) | review queue at `/segments` |
| Export reviews + consensus | `python annotation/scripts/export_segment_reviews.py --min-reviewers 2` | `segment_reviews.csv`, `consensus_segments.csv` |
| Build the benchmark | `python -m pipeline.build_benchmark series.csv consensus_segments.csv --out outbreaks.csv` | wide CSV + `outbreaks.provenance.csv` |

## Outbreak conventions (decided 2026-11-15)

- A segment's start..end is the outbreak from **onset to offset**; every wave is its own outbreak,
  COVID-19 included. Experts review and adjust these windows.
- Benchmark rows store **4 context weeks** before the onset and after the offset (clipped at series
  edges); `duration` = offset - onset in weeks, without padding, as in the released CSV.
- Outbreaks with duration outside **8..51 weeks** are dropped by `build_benchmark`.
- The legacy adapter maps each released row to its core (start + 4 weeks, `duration` weeks long);
  rebuilding with these defaults reproduces the released values for 99.5% of respiratory rows
  (the rest differ only where overlapping windows disagreed on a value).

## Adding a segmentation method

Subclass `pipeline.segmentation.Segmenter`, implement `segment(values) -> list[Segment]`, and add it
to `METHODS` in `pipeline/segmentation/__init__.py`. A `Segment` is a 0-based inclusive week range
with optional `score`, `peak`, and `core_start`/`core_end` (outbreak proper inside a wider
trough-to-trough window). An optional `week_scores()` can return a per-week outbreak probability.
`seasonal-threshold` is a small documented reference method. `prominence` (the recommended
replacement for epidemicKabu, developed in the top-level `segmentation/` package) is registered
automatically when that package is importable; its onset..offset becomes the segment.

Tests: `python -m pytest pipeline/tests`.
