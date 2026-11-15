"""Turn segments into IDOBE outbreak rows (onset to end, plus context padding).

Conventions (decided 2026-11-14/15):

- An outbreak runs from its onset to its offset (``Segment.onset`` ..
  ``Segment.offset``), not trough to trough, so baseline weeks are excluded.
- Each row keeps context padding around the core, as in the released CSV:
  ``pad_weeks`` (default 4) weeks before the onset and after the offset,
  clipped at the edges of the series. ``start_date``/``end_date`` are the
  first/last dates of the padded values and ``duration`` excludes the padding
  (``offset - onset`` in weeks), matching the released CSV where a full row
  holds ``duration + 9`` weekly values.
- Every wave is its own outbreak (one row per segment), including COVID-19.
- Outbreaks with ``duration`` outside [``min_weeks``, ``max_weeks``] are dropped
  (8 to 51 in the released CSV, which also caps rows at 60 values).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .prominence import ProminenceSegmenter, Segment

MAX_VALUES = 60


def outbreak_rows(
    dates: Sequence,
    values: Sequence[float],
    segments: Iterable[Segment],
    pad_weeks: int = 4,
    min_weeks: int = 8,
    max_weeks: int = 51,
) -> list[dict]:
    """Rows in the released wide schema minus identifiers (no unique_id/disease/location/event)."""
    d = pd.to_datetime(pd.Index(dates))
    y = np.asarray(values, dtype=float)
    step = float(np.median(np.diff(d.values).astype("timedelta64[D]").astype(float))) if len(d) > 1 else 7.0
    pad = int(round(pad_weeks * 7 / step))
    rows = []
    for s in segments:
        duration = (d[s.offset] - d[s.onset]).days / 7
        if not (min_weeks <= duration <= max_weeks):
            continue
        lo, hi = max(s.onset - pad, 0), min(s.offset + pad, len(y) - 1)
        vals = y[lo : hi + 1]
        row = {
            "start_date": d[lo].date().isoformat(),
            "end_date": d[hi].date().isoformat(),
            "duration": int(round(duration)),
            "total": float(np.nansum(vals)),
            "onset_date": d[s.onset].date().isoformat(),
            "peak_date": d[s.peak].date().isoformat(),
            "offset_date": d[s.offset].date().isoformat(),
            "score": round(s.score, 4),
        }
        for i in range(MAX_VALUES):
            row[str(i)] = vals[i] if i < len(vals) else np.nan
        rows.append(row)
    return rows


def segment_to_outbreaks(dates, values, segmenter: ProminenceSegmenter | None = None, **kwargs) -> pd.DataFrame:
    """Segment one series and return its outbreak rows as a DataFrame."""
    seg = segmenter or ProminenceSegmenter()
    rows = outbreak_rows(dates, values, seg.segment(dates, values), **kwargs)
    df = pd.DataFrame(rows)
    if len(df):
        df["method"] = seg.name
        df["method_version"] = seg.version
    return df
