from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.models import Annotation, User
from tests.conftest import PASSWORD, logged_in_client, make_user


def test_endpoints_require_login(db, outbreaks):
    client = TestClient(app)
    for method, path in [("get", "/api/me"), ("get", "/api/next"), ("get", "/api/outbreaks/1"),
                         ("get", "/api/my-annotations"), ("get", "/api/admin/stats"), ("get", "/api/admin/export")]:
        assert getattr(client, method)(path).status_code == 401, path
    assert client.post("/api/annotations", json={"outbreak_id": 1, "label": "skip"}).status_code == 401


def test_admin_endpoints_reject_experts(db, outbreaks):
    client = logged_in_client(db, "alice")
    assert client.get("/api/admin/stats").status_code == 403
    assert client.get("/api/admin/export").status_code == 403


def test_login_failures_and_rate_limit(db):
    make_user(db, "alice")
    client = TestClient(app)
    assert client.post("/api/login", json={"username": "alice", "password": "nope"}).status_code == 401
    assert client.post("/api/login", json={"username": "nobody", "password": "nope"}).status_code == 401
    for _ in range(10):
        client.post("/api/login", json={"username": "alice", "password": "nope"})
    assert client.post("/api/login", json={"username": "alice", "password": PASSWORD}).status_code == 429


def test_logout_and_inactive_users(db, outbreaks):
    client = logged_in_client(db, "alice")
    assert client.get("/api/me").json()["username"] == "alice"
    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401

    client = logged_in_client(db, "bob")
    bob = db.execute(select(User).where(User.username == "bob")).scalar_one()
    bob.is_active = False
    db.commit()
    assert client.get("/api/me").status_code == 401


def test_annotation_validation(db, outbreaks):
    client = logged_in_client(db, "alice")
    post = lambda **kw: client.post("/api/annotations", json={"outbreak_id": 1, **kw})  # noqa: E731
    assert post(label="outbreak").status_code == 422                      # confidence required
    assert post(label="outbreak", confidence=6).status_code == 422
    assert post(label="maybe", confidence=3).status_code == 422
    assert post(label="outbreak", confidence=3, comment="x" * 5001).status_code == 422
    assert client.post("/api/annotations", json={"outbreak_id": 999, "label": "skip"}).status_code == 404

    r = post(label="skip", confidence=4)
    assert r.status_code == 201 and r.json()["confidence"] is None    # skip drops confidence
    r = post(label="unsure", confidence=2, comment="  ", time_spent_ms=1234)
    assert r.status_code == 201 and r.json()["comment"] is None       # blank comment stored as null


def test_me_does_not_report_queue_size(db, outbreaks):
    me = logged_in_client(db, "alice").get("/api/me").json()
    assert "remaining" not in me and (me["labeled"], me["skipped"]) == (0, 0)


def test_edits_are_append_only(db, outbreaks):
    client = logged_in_client(db, "alice")
    client.post("/api/annotations", json={"outbreak_id": 3, "label": "outbreak", "confidence": 2})
    client.post("/api/annotations", json={"outbreak_id": 3, "label": "not_outbreak", "confidence": 5, "comment": "on reflection"})

    rows = db.execute(select(Annotation).order_by(Annotation.id)).scalars().all()
    assert [r.label for r in rows] == ["outbreak", "not_outbreak"]
    mine = client.get("/api/my-annotations").json()
    assert mine["total"] == 1 and mine["items"][0]["label"] == "not_outbreak"
    assert client.get("/api/outbreaks/3").json()["my_annotation"]["comment"] == "on reflection"
    me = client.get("/api/me").json()
    assert (me["labeled"], me["skipped"]) == (1, 0)


def test_outbreak_payload_has_weekly_dates(db, outbreaks):
    client = logged_in_client(db, "alice")
    o = client.get("/api/outbreaks/2").json()
    assert o["dates"] == ["2020-01-04", "2020-01-11", "2020-01-18", "2020-01-25", "2020-02-01"]
    assert o["values"] == [0.0, 1.0, 5.0, 2.0, 0.0] and o["my_annotation"] is None
    assert client.get("/api/outbreaks/999").status_code == 404


def test_admin_stats_and_export(db, outbreaks):
    admin = logged_in_client(db, "admin", admin=True)
    alice = logged_in_client(db, "alice")
    for client, label in [(admin, "outbreak"), (alice, "outbreak")]:
        client.post("/api/annotations", json={"outbreak_id": 1, "label": label, "confidence": 4})
    alice.post("/api/annotations", json={"outbreak_id": 2, "label": "skip"})

    s = admin.get("/api/admin/stats").json()
    assert s["total_outbreaks"] == 5 and s["complete"] == 1
    assert s["coverage"] == {"0": 4, "2": 1}
    assert s["agreement"]["overall"]["unanimous"] == 1
    assert {a["username"] for a in s["annotators"]} == {"admin", "alice"}

    csv_text = admin.get("/api/admin/export?scope=all").text
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("annotation_id,outbreak_id") and len(lines) == 4


def test_pages_are_served(db):
    client = TestClient(app)
    for path in ["/", "/login", "/admin", "/static/app.js", "/healthz"]:
        assert client.get(path).status_code == 200, path
    assert client.get("/").headers["x-frame-options"] == "DENY"


def test_timestamps_are_utc_aware(db, outbreaks):
    client = logged_in_client(db, "alice")
    created = client.post("/api/annotations", json={"outbreak_id": 1, "label": "skip"}).json()["created_at"]
    assert created.endswith(("+00:00", "Z")), created
    assert client.get("/api/my-annotations").json()["items"][0]["created_at"].endswith(("+00:00", "Z"))
