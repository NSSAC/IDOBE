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
| Build the benchmark | `python -m pipeline.build_benchmark series.csv consensus_segments.csv --out outbreaks.csv --pad 4` | wide CSV + `outbreaks.provenance.csv` |

## Adding a segmentation method

Subclass `pipeline.segmentation.Segmenter`, implement `segment(values) -> list[Segment]`, and add it
to `METHODS` in `pipeline/segmentation/__init__.py`. A `Segment` is a 0-based inclusive week range
with optional `score`, `peak`, and `core_start`/`core_end` (outbreak proper inside a wider
trough-to-trough window). An optional `week_scores()` can return a per-week outbreak probability.
`seasonal-threshold` is a small documented reference method, not a proposed replacement for
epidemicKabu.

Tests: `python -m pytest pipeline/tests`.
