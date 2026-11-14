import csv
import math
from datetime import date, timedelta

import pytest

from pipeline.build_benchmark import build
from pipeline.consensus import consensus, iou
from pipeline.formats import Series, SegmentRow, read_segments, read_series, write_segments, write_series
from pipeline.segment import run
from pipeline.segmentation import Segment, Segmenter, check_segments, get_segmenter, legacy

SAT = date(2020, 1, 4)


def seasonal(n_seasons=4, base=5.0, peak=200.0):
    vals = []
    for _ in range(n_seasons):
        for w in range(52):
            vals.append(base + peak * math.exp(-((w - 26) / 4) ** 2))
    return vals


def test_series_roundtrip(tmp_path):
    s = Series("a", "FLU", "VA", "HOSPITALIZATIONS", SAT, [1.0, None, 3.0], source="nhsn", geo_level="state",
               population=8.6e6, value_type="count")
    write_series(tmp_path / "s.csv", [s])
    back = read_series(tmp_path / "s.csv")["a"]
    assert back.values == [1.0, None, 3.0] and back.start_date == SAT and back.population == 8.6e6


def test_segments_roundtrip_with_optional_dates(tmp_path):
    seg = SegmentRow("a", "a#1", "m", "1", SAT, SAT + timedelta(weeks=5), 0.5, peak_date=SAT + timedelta(weeks=2))
    write_segments(tmp_path / "g.csv", [seg])
    back = read_segments(tmp_path / "g.csv")[0]
    assert back.peak_date == SAT + timedelta(weeks=2) and back.core_start_date is None and back.score == 0.5


def test_threshold_finds_each_season():
    segs = get_segmenter("seasonal-threshold").segment(seasonal())
    check_segments(segs, 208)
    assert len(segs) == 4
    for k, s in enumerate(segs):
        assert s.start <= 52 * k + 26 <= s.end and s.length <= 60 and s.peak == 52 * k + 26


def test_threshold_flat_and_short_series_give_nothing():
    seg = get_segmenter("seasonal-threshold")
    assert seg.segment([3.0] * 120) == []
    assert seg.segment([1.0, 5.0]) == []
    assert seg.segment([None] * 30) == []


def test_segment_validation():
    with pytest.raises(ValueError):
        Segment(5, 3)
    with pytest.raises(ValueError):
        Segment(0, 5, peak=9)
    with pytest.raises(ValueError):
        check_segments([Segment(0, 5), Segment(5, 8)], 10)


def test_custom_segmenter_plugs_in():
    class Fixed(Segmenter):
        name = "fixed"

        def segment(self, values, *, event="", disease=""):
            return [Segment(1, 3, core_start=2, core_end=3)]

    from pipeline.segmentation import METHODS
    METHODS["fixed"] = Fixed
    try:
        rows = run({"a": Series("a", "X", "Y", "CASES", SAT, [0, 1, 2, 1, 0])}, "fixed", {})
    finally:
        del METHODS["fixed"]
    assert rows[0].start_date == SAT + timedelta(weeks=1) and rows[0].core_start_date == SAT + timedelta(weeks=2)


def write_wide(path, rows):
    cols = ["unique_id", "disease", "location", "event", "start_date", "end_date", "duration", "total"] + [str(i) for i in range(60)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for uid, start, vals in rows:
            end = start + timedelta(weeks=len(vals) - 1)
            w.writerow([uid, "ILI", "VA", "PERCENT_UNWEIGHTED", start, end, len(vals) - 2, sum(vals)]
                       + vals + [""] * (60 - len(vals)))


def test_legacy_stitch_and_rebuild(tmp_path):
    p = tmp_path / "w.csv"
    # Two overlapping windows (2 shared weeks, same values) and an unrelated disease.
    write_wide(p, [(1, SAT, [1, 2, 3, 4]), (2, SAT + timedelta(weeks=2), [3, 4, 5, 6])])
    series, segs, stats = legacy.stitch(p)
    s = series["ili__va__percent-unweighted"]
    assert s.values == [1, 2, 3, 4, 5, 6] and stats["value_clashes"] == 0 and s.value_type == "rate"
    assert [(g.start_date, g.end_date) for g in segs] == [(SAT, SAT + timedelta(weeks=3)),
                                                         (SAT + timedelta(weeks=2), SAT + timedelta(weeks=5))]
    rows, prov, skipped = build(series, segs)
    assert not skipped and rows[1][8:12] == [3, 4, 5, 6] and rows[1][6] == 4 and prov[1][4] == "legacy-2"


def test_build_pads_and_skips_long_windows():
    s = Series("a", "X", "Y", "CASES", SAT, list(range(100)))
    segs = [SegmentRow("a", "1", "m", "", SAT + timedelta(weeks=10), SAT + timedelta(weeks=12)),
            SegmentRow("a", "2", "m", "", SAT, SAT + timedelta(weeks=70))]
    rows, _, skipped = build({"a": s}, segs, pad=2)
    assert len(rows) == 1 and rows[0][6] == 3 and rows[0][8:15] == [8, 9, 10, 11, 12, 13, 14]
    assert skipped[0][0] == "2"


def d(w):
    return (SAT + timedelta(weeks=w)).isoformat()


def rev(reviewer, verdict, pid="p1", s=0, e=10, sid="a", unusable=0):
    return {"series_id": sid, "run_method": "m", "reviewer": reviewer, "proposed_segment_id": pid if verdict != "added" else "",
            "verdict": verdict, "start_date": d(s) if verdict != "reject" else "",
            "end_date": d(e) if verdict != "reject" else "", "series_unusable": unusable}


def test_iou():
    assert iou((SAT, SAT + timedelta(weeks=9)), (SAT, SAT + timedelta(weeks=9))) == 1
    assert iou((SAT, SAT + timedelta(weeks=9)), (SAT + timedelta(weeks=5), SAT + timedelta(weeks=14))) == 5 / 15


def test_consensus_majority_and_median():
    rows = [rev("x", "accept"), rev("y", "adjust", s=2, e=12), rev("z", "adjust", s=1, e=11),
            rev("x", "reject", pid="p2"), rev("y", "reject", pid="p2"), rev("z", "accept", pid="p2", s=20, e=30),
            rev("x", "added", s=40, e=50), rev("y", "added", s=41, e=52), rev("z", "added", s=80, e=90)]
    segs, stats = consensus(rows, min_reviewers=2)
    assert [(s.start_date.isoformat(), s.end_date.isoformat(), s.extra["origin"]) for s in segs] == [
        (d(1), d(11), "proposed"), (d(40), d(50), "added")]
    assert stats["proposed_rejected"] == 1 and stats["added_unconfirmed"] == 1
    assert segs[0].score == 1.0 and segs[1].score == round(2 / 3, 3)


def test_consensus_min_reviewers_and_unusable():
    assert consensus([rev("x", "accept")], min_reviewers=2)[0] == []
    rows = [rev("x", "accept", unusable=1), rev("y", "accept", unusable=1), rev("z", "accept")]
    segs, stats = consensus(rows)
    assert segs == [] and stats["series_unusable"] == 1


def test_threshold_does_not_span_gaps():
    vals = seasonal(2)
    vals[60:110] = [None] * 50          # a long reporting gap covering the second season's start
    segs = get_segmenter("seasonal-threshold").segment(vals)
    assert segs and all(not (s.start < 60 and s.end >= 110) for s in segs)
    assert all(any(v is not None for v in vals[s.start:s.end + 1]) for s in segs)
    assert all(not (60 <= s.start and s.end < 110) for s in segs)
