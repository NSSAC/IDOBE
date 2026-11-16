import math
import random
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.ml.features import SHAPE_FEATURES, series_features
from app.ml.results import series_results
from app.ml.train import readiness, retrain_if_due, train_and_predict
from app.models import Annotation, Assignment, ModelPrediction, ModelRun, Outbreak
from app.queue import pick_next
from tests.conftest import PASSWORD, logged_in_client, make_user

N = 60   # even ids: clear single wave ("outbreak"); odd ids: flat noise ("not an outbreak")


def hump(rng, n=30):
    c = rng.uniform(0.35, 0.65) * n
    return [round(100 * math.exp(-((i - c) / (n / 8)) ** 2) + rng.uniform(0, 3), 2) for i in range(n)]


def flat(rng, n=30):
    return [round(20 + rng.uniform(-6, 6), 2) for _ in range(n)]


@pytest.fixture
def synthetic(db):
    rng = random.Random(0)
    db.add_all([
        Outbreak(id=i, disease="MEASLES", location=f"L{i}", event="CASES", start_date=date(1950, 1, 7),
                 end_date=date(1950, 12, 30), duration=21, total=1.0,
                 values=hump(rng) if i % 2 == 0 else flat(rng), source_file="synthetic")
        for i in range(1, N + 1)
    ])
    db.commit()


def vote(db, user, oid, label=None, served_as="random", confidence=4):
    label = label or ("outbreak" if oid % 2 == 0 else "not_outbreak")
    db.add(Annotation(outbreak_id=oid, user_id=user.id, label=label, confidence=confidence, served_as=served_as))
    db.commit()


def label_first(db, users, k):
    """Each user gives the true label to series 1..k."""
    for u in users:
        for oid in range(1, k + 1):
            vote(db, u, oid)


def test_features_are_scale_free_and_safe():
    rng = random.Random(1)
    x = hump(rng)
    a, b = series_features(x), series_features([10 * v for v in x])
    scale_dependent = {"log_total", "log_peak"}
    for name, va, vb in zip(SHAPE_FEATURES, a, b):
        if name not in scale_dependent:
            assert va == pytest.approx(vb, abs=1e-9), name
    for edge in ([0, 0, 0, 0], [5.0], [None, 3, None, 1], [1, 1, 1, 1, 1]):
        assert all(math.isfinite(v) for v in series_features(edge)), edge
    assert series_features([0, 0, 0])[SHAPE_FEATURES.index("all_zero")] == 1


def test_waits_for_enough_labels(db, synthetic):
    a = make_user(db, "a")
    label_first(db, [a], 4)
    assert readiness(db)["ready"] is False
    assert train_and_predict(db) is None
    assert db.execute(select(ModelPrediction)).first() is None


def test_learns_to_separate_shapes(db, synthetic):
    a, b = make_user(db, "a"), make_user(db, "b")
    label_first(db, [a, b], 24)
    run = train_and_predict(db)
    assert run is not None and run.n_votes == 48 and run.n_outbreaks == 24
    m = run.metrics
    assert m["cv_auc"] > 0.95 and set(m["cv"]) == {"logistic", "boosted_trees"}
    preds = {p.outbreak_id: p.p_outbreak for p in db.execute(select(ModelPrediction)).scalars()}
    assert len(preds) == N
    unlabeled = range(25, N + 1)
    correct = sum((preds[i] >= 0.5) == (i % 2 == 0) for i in unlabeled)
    assert correct / len(unlabeled) > 0.9
    assert m["evaluation"]["random_sample"]["n"] == 24 and m["evaluation"]["random_sample"]["accuracy"] > 0.9
    assert sum(m["predicted"]["histogram"]) == N


def test_random_sample_counts_only_series_first_served_at_random(db, synthetic):
    a, b = make_user(db, "a"), make_user(db, "b")
    for oid in range(1, 21):
        vote(db, a, oid, served_as="random" if oid <= 8 else "uncertain")
        vote(db, b, oid, served_as="random")   # later random draws don't change how the series was picked
    m = train_and_predict(db).metrics
    assert m["evaluation"]["random_sample"]["n"] == 8
    assert m["evaluation"]["labeled"]["n"] == 20


def test_active_queue_serves_uncertain_series(db, synthetic):
    a, c = make_user(db, "a"), make_user(db, "c")
    label_first(db, [a], 16)
    run = train_and_predict(db)
    assert run.metrics["cv_auc"] >= 0.85                # so confident, agreeing series need no second label
    rows = db.execute(select(ModelPrediction)).scalars().all()
    preds = {p.outbreak_id: p.uncertainty for p in rows}
    p_out = {p.outbreak_id: p.p_outbreak for p in rows}
    settled = {i for i in range(1, 17) if preds[i] < 0.2 and (p_out[i] >= 0.5) == (i % 2 == 0)}
    eligible = [i for i in preds if i not in settled]
    most_uncertain = max(eligible, key=preds.get)
    oid, reason, run_id = pick_next(db, c.id, 2, "active", random_fraction=0, top_k=1)
    assert (oid, reason) == (most_uncertain, "uncertain") and run_id is not None
    top5 = sorted(eligible, key=preds.get, reverse=True)[:5]
    for _ in range(20):
        assert pick_next(db, c.id, 2, "active", random_fraction=0, top_k=5)[0] in top5
    assert pick_next(db, c.id, 2, "active", random_fraction=1)[1] == "random"


def test_confident_agreement_needs_only_one_label(db, outbreaks):
    a, b, c = (make_user(db, n) for n in "abc")
    run = ModelRun(model_type="logistic", n_votes=50, n_outbreaks=40, train_seconds=0, metrics={"cv_auc": 0.95})
    db.add(run)
    db.flush()
    db.add_all([ModelPrediction(outbreak_id=i, run_id=run.id, p_outbreak=0.95 if i == 1 else 0.5,
                                uncertainty=0.1 if i == 1 else 1.0) for i in range(1, 6)])
    db.commit()
    vote(db, a, 1, "outbreak")                          # agrees with a confident model
    served = {pick_next(db, c.id, 2, "active", random_fraction=1)[0] for _ in range(60)}
    assert 1 not in served and served == {2, 3, 4, 5}

    vote(db, b, 1, "not_outbreak")                      # experts now disagree on 1
    run.metrics = {"cv_auc": 0.6}                       # and a weak model never cuts labeling short
    db.commit()
    vote(db, b, 2, "outbreak")
    served = {pick_next(db, c.id, 3, "active", random_fraction=1)[0] for _ in range(60)}
    assert {1, 2} <= served


def test_served_as_is_recorded(db, outbreaks):
    client = logged_in_client(db, "alice")
    oid = client.get("/api/next").json()["outbreak"]["id"]
    client.post("/api/annotations", json={"outbreak_id": oid, "label": "outbreak", "confidence": 3})
    client.post("/api/annotations", json={"outbreak_id": oid, "label": "unsure", "confidence": 2})
    other = next(i for i in range(1, 6) if i != oid)
    client.post("/api/annotations", json={"outbreak_id": other, "label": "outbreak", "confidence": 3})
    rows = db.execute(select(Annotation.outbreak_id, Annotation.served_as).order_by(Annotation.id)).all()
    assert rows == [(oid, "random"), (oid, "revisit"), (other, "lookup")]
    assert db.execute(select(Assignment.reason)).scalars().all() == ["random"]


def test_retrains_every_n_answers(db, synthetic):
    a, b = make_user(db, "a"), make_user(db, "b")
    label_first(db, [a], 10)
    first = retrain_if_due()
    assert first is not None and first.n_votes == 10
    vote(db, b, 1)
    assert retrain_if_due() is None                     # 1 new answer < RETRAIN_EVERY (4)
    for oid in (2, 3, 4):
        vote(db, b, oid)
    second = retrain_if_due()
    assert second is not None and second.n_votes == 14


def test_admin_model_endpoints_and_final_labels(db, synthetic):
    admin = logged_in_client(db, "admin", admin=True)
    assert admin.get("/api/admin/model").json()["run"] is None
    assert admin.post("/api/admin/retrain").status_code == 400

    a = make_user(db, "a")
    label_first(db, [a], 12)
    vote(db, make_user(db, "b"), 2, "not_outbreak")     # series 2: a tie -> falls back to the model
    r = admin.post("/api/admin/retrain")
    assert r.status_code == 200, r.text
    m = admin.get("/api/admin/model").json()
    assert m["run"]["n_votes"] == 13 and len(m["most_uncertain"]) == 15
    assert m["served"] == {"random": 13}

    rows = {r["outbreak_id"]: r for r in series_results(db)}
    assert rows[1]["label_source"] == "expert" and rows[1]["final_label"] == "not_outbreak"
    assert rows[2]["label_source"] == "model" and rows[2]["expert_label"] is None
    assert rows[40]["label_source"] == "model" and rows[40]["p_outbreak"] is not None
    csv_lines = admin.get("/api/admin/predictions").text.strip().splitlines()
    assert len(csv_lines) == N + 1 and csv_lines[0].startswith("outbreak_id,")
    assert TestClient(app).get("/api/admin/model").status_code == 401


def test_confident_agreement_rule_needs_enough_answers(db, outbreaks):
    a, c = make_user(db, "a"), make_user(db, "c")
    run = ModelRun(model_type="logistic", n_votes=5, n_outbreaks=5, train_seconds=0, metrics={"cv_auc": 0.99})
    db.add(run)
    db.flush()
    db.add(ModelPrediction(outbreak_id=1, run_id=run.id, p_outbreak=0.99, uncertainty=0.02))
    db.commit()
    vote(db, a, 1, "outbreak")
    # Trained on only 5 answers (< EARLY_STOP_MIN_LABELS = 10): series 1 still goes to a second expert.
    assert 1 in {pick_next(db, c.id, 2, "active", random_fraction=1)[0] for _ in range(60)}
