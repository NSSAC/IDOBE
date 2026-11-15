import numpy as np
import pandas as pd
import pytest

from segmentation import DAILY, WEEKLY, ProminenceSegmenter
from segmentation.synthetic import Wave, make_series

THREE = [
    Wave(peak_day=150, height=200, rise_days=18, fall_days=25),
    Wave(peak_day=420, height=120, rise_days=15, fall_days=30),
    Wave(peak_day=700, height=300, rise_days=20, fall_days=20),
]


def peaks_match(segs, dates, true_days, tol_days=21):
    t0 = dates[0]
    got = [(dates[s.peak] - t0).days for s in segs]
    return len(got) == len(true_days) and all(abs(g - t) <= tol_days for g, t in zip(got, true_days))


def check_invariants(segs, n):
    for a, b in zip(segs[:-1], segs[1:]):
        assert a.end + 1 == b.start, "segments should tile trough to trough"
    for s in segs:
        assert 0 <= s.start <= s.onset <= s.peak <= s.offset <= s.end < n


def test_clean_three_waves():
    dates, y, _ = make_series(THREE, 900)
    segs = ProminenceSegmenter(WEEKLY).segment(dates, y)
    assert peaks_match(segs, dates, [150, 420, 700])
    check_invariants(segs, len(y))


# peaks ~300, ~15, ~4.5 per week; at a few cases a week the peak week itself is noisy
@pytest.mark.parametrize("scale,tol", [(1.0, 21), (0.05, 21), (0.015, 35)])
def test_noisy_three_waves(scale, tol):
    waves = [Wave(w.peak_day, w.height * scale, w.rise_days, w.fall_days) for w in THREE]
    ok = 0
    for seed in range(20):
        dates, y, _ = make_series(waves, 900, rng=np.random.default_rng(seed))
        segs = ProminenceSegmenter(WEEKLY).segment(dates, y)
        check_invariants(segs, len(y))
        ok += peaks_match(segs, dates, [150, 420, 700], tol)
    assert ok >= 17, f"only {ok}/20 seeds recovered the three waves"


def test_shoulder_is_one_outbreak():
    waves = [Wave(200, 100, 15, 15), Wave(235, 85, 15, 25)]  # bump 5 weeks after the peak
    dates, y, _ = make_series(waves, 500)
    segs = ProminenceSegmenter(WEEKLY).segment(dates, y)
    assert len(segs) == 1


def test_small_season_next_to_big_one():
    waves = [Wave(150, 1000, 15, 20), Wave(515, 120, 15, 20)]
    dates, y, _ = make_series(waves, 750)
    segs = ProminenceSegmenter(WEEKLY).segment(dates, y)
    assert peaks_match(segs, dates, [150, 515])


def test_flat_and_empty_series():
    seg = ProminenceSegmenter(WEEKLY)
    dates = pd.date_range("2020-01-04", periods=60, freq="7D")
    assert seg.segment(dates, np.zeros(60)) == []
    assert seg.segment(dates[:0], []) == []
    assert seg.segment(dates[:5], np.arange(5.0)) == []  # shorter than min_run_points


def test_short_gap_is_interpolated_long_gap_splits():
    dates, y, _ = make_series(THREE, 900)
    seg = ProminenceSegmenter(WEEKLY)
    y_short = y.copy()
    y_short[30:32] = np.nan  # 2 weeks missing near the second trough
    assert len(seg.segment(dates, y_short)) == 3

    y_long = y.copy()
    y_long[40:52] = np.nan  # 12 weeks missing between waves 2 and 3
    segs = seg.segment(dates, y_long)
    assert len(segs) == 3
    assert all(not (s.start < 40 <= s.end) for s in segs), "no segment should span the gap"


def test_daily_cadence():
    rng = np.random.default_rng(0)
    waves = [Wave(w.peak_day, w.height / 7, w.rise_days, w.fall_days) for w in THREE]
    dates, y, _ = make_series(waves, 900, step_days=1, rng=rng)
    segs = ProminenceSegmenter(DAILY).segment(dates, y)
    assert peaks_match(segs, dates, [150, 420, 700])
    check_invariants(segs, len(y))


def test_onset_excludes_long_baseline():
    waves = [Wave(400, 100, 15, 20)]
    dates, y, _ = make_series(waves, 600)
    (s,) = ProminenceSegmenter(WEEKLY).segment(dates, y)
    assert s.start < 30  # trough-to-trough reaches back to the series start
    assert (dates[s.peak] - dates[s.onset]).days <= 45  # onset is near the rise


def test_frame_output():
    dates, y, _ = make_series(THREE, 900)
    df = ProminenceSegmenter(WEEKLY).segment_frame(dates, y)
    assert list(df.segment_id) == [0, 1, 2]
    assert {"start_date", "end_date", "onset_date", "offset_date", "peak_date", "score"} <= set(df.columns)


def test_quiet_period_rarely_has_false_outbreaks():
    waves = [Wave(150, 60, 15, 20), Wave(400, 40, 15, 20)]
    extra = 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        dates, y, _ = make_series(waves, 6 * 365, baseline=1.0, rng=rng)  # ~4.8 quiet years after
        segs = ProminenceSegmenter(WEEKLY).segment(dates, y)
        assert len(segs) >= 2
        extra += len(segs) - 2
    assert extra / (30 * 4.8) < 0.15, f"{extra} false outbreaks in 144 quiet years"


def test_outbreak_rows_onset_to_end_with_padding():
    from segmentation.outbreaks import segment_to_outbreaks

    dates, y, _ = make_series(THREE, 900)
    df = segment_to_outbreaks(dates, y)
    assert len(df) == 3
    for _, r in df.iterrows():
        n_vals = r[[str(i) for i in range(60)]].notna().sum()
        span = (pd.Timestamp(r.end_date) - pd.Timestamp(r.start_date)).days // 7 + 1
        assert n_vals == span
        assert n_vals == r.duration + 9  # 4 weeks before onset + core (duration + 1) + 4 after offset
        assert (pd.Timestamp(r.onset_date) - pd.Timestamp(r.start_date)).days == 28
        assert 8 <= r.duration <= 51


def test_outbreak_rows_exclude_long_baseline():
    from segmentation.outbreaks import segment_to_outbreaks

    dates, y, _ = make_series([Wave(400, 100, 15, 20)], 600)
    (r,) = segment_to_outbreaks(dates, y).itertuples()
    assert (pd.Timestamp(r.peak_date) - pd.Timestamp(r.start_date)).days < 100  # no 50-week flat lead-in


def test_covid_preset_separates_small_wave_next_to_big_one():
    from segmentation import COVID_WEEKLY

    waves = [Wave(150, 100, 15, 25), Wave(262, 1000, 20, 25)]  # small wave 16 weeks before a 10x bigger one
    dates, y, _ = make_series(waves, 500)
    assert len(ProminenceSegmenter(WEEKLY).segment(dates, y)) == 1  # the seasonal preset merges them
    assert len(ProminenceSegmenter(COVID_WEEKLY).segment(dates, y)) == 2
