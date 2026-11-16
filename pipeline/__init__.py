"""IDOBE benchmark pipeline: curated series -> segmentation -> expert review -> benchmark CSV.

Each stage reads and writes plain CSV files (see `formats.py`), so stages can be run, swapped or
re-run independently: the respiratory curation scripts write series, any segmenter writes segments,
the annotation app reads both and exports expert reviews, and `build_benchmark.py` turns accepted
segments into the wide outbreak CSV used by the baselines, stats and synthetic generator.
"""
