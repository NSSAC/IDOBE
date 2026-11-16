import csv
import io
import sys
from datetime import date

import pytest

from app.config import REPO_ROOT
from app.models import ProposedSegment, RunSeries, SegmentationRun, SegmentReview, Series
from tests.conftest import logged_in_client

sys.path.insert(0, str(REPO_ROOT))
from pipeline.consensus import consensus  # noqa: E402


@pytest.fixture
def seg_run(db):
    run = SegmentationRun(name="test-run", method="m", method_version="1")
    db.add(run)
    db.flush()
    rows = []
    for k in range(3):
        s = Series(series_key=f"s{k}", disease="ILI", location=f"L{k}", event="CASES",
                   start_date=date(2020, 1, 4), values=[float(i % 10) for i in range(40)])
        db.add(s)
        db.flush()
        db.add(RunSeries(run_id=run.id, series_id=s.id))
        if k < 2:  # series 3 has no proposals
            db.add_all([ProposedSegment(run_id=run.id, series_id=s.id, segment_key=f"s{k}#1", start_idx=2, end_idx=9, peak_idx=5),
                        ProposedSegment(run_id=run.id, series_id=s.id, segment_key=f"s{k}#2", start_idx=20, end_idx=29)])
        rows.append(s)
    db.commit()
    return run, rows


def proposals(client, run, sid):
    return client.get(f"/api/seg/series/{sid}?run_id={run.id}").json()["proposals"]


def test_next_serves_series_with_proposals_and_dates(db, seg_run):
    run, _ = seg_run
    c = logged_in_client(db, "ann")
    d = c.get("/api/seg/next").json()
    assert not d["done"] and d["run"]["id"] == run.id
    assert len(d["series"]["dates"]) == 40 and d["series"]["dates"][1] == "2020-01-11"
    assert all(p["start_idx"] < p["end_idx"] for p in d["proposals"])


def test_review_flow_and_export(db, seg_run):
    run, series = seg_run
    c = logged_in_client(db, "ann", admin=True)
    s0 = series[0]
    p1, p2 = proposals(c, run, s0.id)
    r = c.post("/api/seg/reviews", json={"run_id": run.id, "series_id": s0.id, "confidence": 4, "items": [
        {"proposed_id": p1["id"], "verdict": "adjust", "start_idx": 3, "end_idx": 9},
        {"proposed_id": p2["id"], "verdict": "reject"},
        {"verdict": "added", "start_idx": 32, "end_idx": 38}]})
    assert r.status_code == 201, r.text
    assert [i["verdict"] for i in r.json()["items"]] == ["adjust", "reject", "added"]

    # Adjusting back to the proposed dates is recorded as accept.
    r = c.post("/api/seg/reviews", json={"run_id": run.id, "series_id": series[1].id, "confidence": 3, "items": [
        {"proposed_id": p["id"], "verdict": "adjust", "start_idx": p["start_idx"], "end_idx": p["end_idx"]}
        for p in proposals(c, run, series[1].id)]})
    assert [i["verdict"] for i in r.json()["items"]] == ["accept", "accept"]

    out = c.get(f"/api/seg/export.csv?run_id={run.id}")
    rows = list(csv.DictReader(io.StringIO(out.text)))
    assert {r["verdict"] for r in rows} == {"adjust", "reject", "added", "accept"}
    adj = next(r for r in rows if r["verdict"] == "adjust")
    assert adj["proposed_segment_id"] == "s0#1" and adj["start_date"] == "2020-01-25" and adj["series_id"] == "s0"

    segs, stats = consensus(rows)
    assert len(segs) == 4 and stats["proposed_rejected"] == 1

    mine = c.get(f"/api/seg/my-reviews?run_id={run.id}").json()
    assert mine["total"] == 2 and mine["items"][-1]["summary"] == "1 adjust, 1 reject, 1 added"


def test_review_validation(db, seg_run):
    run, series = seg_run
    c = logged_in_client(db, "ann")
    s0 = series[0]
    p1, p2 = proposals(c, run, s0.id)
    post = lambda **kw: c.post("/api/seg/reviews", json={"run_id": run.id, "series_id": s0.id, **kw})  # noqa: E731
    assert post(confidence=3, items=[{"proposed_id": p1["id"], "verdict": "accept"}]).status_code == 422  # p2 missing
    assert post(items=[{"proposed_id": p1["id"], "verdict": "accept"},
                       {"proposed_id": p2["id"], "verdict": "accept"}]).status_code == 422             # no confidence
    assert post(confidence=3, items=[{"proposed_id": p1["id"], "verdict": "accept"}, {"proposed_id": p2["id"], "verdict": "reject"},
                                     {"verdict": "added", "start_idx": 30, "end_idx": 45}]).status_code == 422  # past end
    assert post(confidence=3, items=[{"proposed_id": 999, "verdict": "reject"}]).status_code == 422
    # Unusable series may leave proposals undecided; skip needs nothing.
    assert post(confidence=2, series_unusable=True).status_code == 201
    assert post(skipped=True).status_code == 201


def test_queue_moves_on_and_finishes(db, seg_run):
    run, series = seg_run
    c = logged_in_client(db, "ann")
    seen = set()
    for _ in range(3):
        d = c.get("/api/seg/next").json()
        sid = d["series"]["id"]
        assert sid not in seen
        seen.add(sid)
        c.post("/api/seg/reviews", json={"run_id": run.id, "series_id": sid, "skipped": True})
    assert c.get("/api/seg/next").json()["done"]
    # Series 3 (no proposals) can be reviewed with no decisions or an added outbreak.
    c2 = logged_in_client(db, "ann2")
    r = c2.post("/api/seg/reviews", json={"run_id": run.id, "series_id": series[2].id, "confidence": 5, "items": []})
    assert r.status_code == 201


def test_target_reviews_stop_serving(db, seg_run):
    run, series = seg_run  # TARGET_ANNOTATIONS=2 in tests
    for name in ("a", "b"):
        c = logged_in_client(db, name)
        c.post("/api/seg/reviews", json={"run_id": run.id, "series_id": series[2].id, "confidence": 3})
    c = logged_in_client(db, "c")
    served = {c.get("/api/seg/next").json()["series"]["id"] for _ in range(10)}
    assert series[2].id not in served


def test_export_requires_admin_and_page_served(db, seg_run):
    c = logged_in_client(db, "ann")
    assert c.get("/api/seg/export.csv").status_code == 403
    assert "segment review" in c.get("/segments").text.lower()


def test_loader_legacy_and_method(db, tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "annotation" / "scripts"))
    from load_segments import load
    from pipeline.formats import Series as PSeries
    from pipeline.segment import run as run_method
    from pipeline.segmentation import legacy
    import math

    series, segs, _ = legacy.stitch(REPO_ROOT / "raw_data" / "outbreaks_disease_location.csv", {"RSV"})
    stats = load(db, series, segs, "legacy-rsv", "x.csv")
    assert stats["series"] == len(series) and stats["segments"] == len(segs) > 0
    with pytest.raises(SystemExit):
        load(db, series, segs, "legacy-rsv", "x.csv")

    vals = [5 + 100 * math.exp(-(((w % 52) - 26) / 4) ** 2) for w in range(156)]
    synth = {"t": PSeries("t", "FLU", "VA", "CASES", date(2020, 1, 4), vals)}
    stats = load(db, synth, run_method(synth, "seasonal-threshold", {}), "thr", None)
    assert stats["segments"] == 3
    assert db.query(ProposedSegment).filter(ProposedSegment.peak_idx.isnot(None)).count() == 3
    assert db.query(SegmentReview).count() == 0
