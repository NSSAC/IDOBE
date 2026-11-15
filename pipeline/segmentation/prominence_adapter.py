"""Adapter for the prominence segmenter from the segmentation work (top-level `segmentation/`
package, branch claude/project-thread-qkwbum). Registered only when that package is importable.

Its Segment has trough-to-trough start/end plus onset/offset. IDOBE outbreaks run onset to offset
(decided 2026-11-15), so the window experts review is onset..offset; trough bounds are kept in
`info`. Padding is added later, by build_benchmark.
"""

from __future__ import annotations

from datetime import date, timedelta

from .base import Segment, Segmenter


class Prominence(Segmenter):
    name = "prominence"

    def __init__(self, **params):
        from segmentation import WEEKLY, ProminenceSegmenter  # noqa: PLC0415  (optional dependency)
        from dataclasses import replace
        self._seg = ProminenceSegmenter(replace(WEEKLY, **params) if params else WEEKLY)
        self.version = self._seg.version
        super().__init__(**params)

    def segment(self, values, *, event="", disease=""):
        # The prominence segmenter wants dates; any weekly grid works for index output.
        dates = [date(2000, 1, 1) + timedelta(weeks=i) for i in range(len(values))]
        vals = [float("nan") if v is None else float(v) for v in values]
        return [Segment(s.onset, s.offset, float(s.score), peak=s.peak,
                        info={"trough_start": s.start, "trough_end": s.end})
                for s in self._seg.segment(dates, vals)]
