"""Adapter for the prominence segmenter from the segmentation work (top-level `segmentation/`
package, branch claude/project-thread-qkwbum). Registered only when that package is importable.

Its Segment has trough-to-trough start/end plus onset/offset. IDOBE outbreaks run onset to offset
(decided 2026-11-15) and each wave is its own outbreak, so the window experts review is onset..offset; trough bounds are kept in
`info`. Padding is added later, by build_benchmark.
"""

from __future__ import annotations

from datetime import date, timedelta

from .base import Segment, Segmenter


class Prominence(Segmenter):
    """Uses the COVID_WEEKLY preset for COVID-19 series (lower prominence threshold, so a small
    wave next to a large one stays its own outbreak) and WEEKLY for everything else."""

    name = "prominence"

    def __init__(self, **params):
        import segmentation as pkg  # noqa: PLC0415  (optional dependency)
        from dataclasses import replace
        weekly = pkg.WEEKLY
        covid = getattr(pkg, "COVID_WEEKLY", weekly)
        mk = lambda preset: pkg.ProminenceSegmenter(replace(preset, **params) if params else preset)  # noqa: E731
        self._default, self._covid = mk(weekly), mk(covid)
        self.version = self._default.version
        super().__init__(**params)

    def _for(self, disease: str):
        return self._covid if "COVID" in (disease or "").upper() else self._default

    def segment(self, values, *, event="", disease=""):
        # The prominence segmenter wants dates; any weekly grid works for index output.
        dates = [date(2000, 1, 1) + timedelta(weeks=i) for i in range(len(values))]
        vals = [float("nan") if v is None else float(v) for v in values]
        return [Segment(s.onset, s.offset, float(s.score), peak=s.peak,
                        info={"trough_start": s.start, "trough_end": s.end})
                for s in self._for(disease).segment(dates, vals)]
