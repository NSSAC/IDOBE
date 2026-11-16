from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import PASSWORD, logged_in_client


def login(username, password):
    client = TestClient(app)
    return client, client.post("/api/login", json={"username": username, "password": password})


def test_admin_creates_user_with_chosen_password(db):
    admin = logged_in_client(db, "admin", admin=True)
    r = admin.post("/api/admin/users", json={"username": " JDoe ", "password": "measles1960",
                                             "display_name": "J. Doe", "affiliation": "", "is_admin": False})
    assert r.status_code == 201, r.text
    assert r.json()["username"] == "jdoe" and r.json()["affiliation"] is None
    _, r = login("jdoe", "measles1960")
    assert r.status_code == 200 and r.json()["display_name"] == "J. Doe" and not r.json()["is_admin"]
    users = {u["username"]: u for u in admin.get("/api/admin/users").json()}
    assert set(users) == {"admin", "jdoe"} and users["jdoe"]["n_labels"] == 0


def test_user_creation_is_validated(db):
    admin = logged_in_client(db, "admin", admin=True)
    post = lambda **kw: admin.post("/api/admin/users", json={"username": "ok_name", "password": "longenough", **kw})  # noqa: E731
    assert post(password="short").status_code == 422
    assert post(username="x").status_code == 422
    assert post(username="has space").status_code == 422
    assert post().status_code == 201
    assert post().status_code == 409                       # duplicate


def test_admin_sets_password_and_deactivates(db):
    admin = logged_in_client(db, "admin", admin=True)
    logged_in_client(db, "alice")
    assert admin.patch("/api/admin/users/alice", json={"password": "easy2remember"}).status_code == 200
    assert login("alice", PASSWORD)[1].status_code == 401
    assert login("alice", "easy2remember")[1].status_code == 200
    assert admin.patch("/api/admin/users/alice", json={"password": "short"}).status_code == 422

    assert admin.patch("/api/admin/users/alice", json={"is_active": False}).json()["is_active"] is False
    assert login("alice", "easy2remember")[1].status_code == 401
    admin.patch("/api/admin/users/alice", json={"is_active": True})
    assert login("alice", "easy2remember")[1].status_code == 200
    assert admin.patch("/api/admin/users/nobody", json={"is_active": False}).status_code == 404


def test_admin_cannot_lock_themselves_out(db):
    admin = logged_in_client(db, "admin", admin=True)
    assert admin.patch("/api/admin/users/admin", json={"is_active": False}).status_code == 400
    assert admin.patch("/api/admin/users/admin", json={"is_admin": False}).status_code == 400
    assert admin.patch("/api/admin/users/admin", json={"password": "newadminpass"}).status_code == 200


def test_user_management_is_admin_only(db):
    expert = logged_in_client(db, "alice")
    assert expert.get("/api/admin/users").status_code == 403
    assert expert.post("/api/admin/users", json={"username": "eve", "password": "longenough"}).status_code == 403
    assert expert.patch("/api/admin/users/alice", json={"is_admin": True}).status_code == 403
    assert TestClient(app).get("/api/admin/users").status_code == 401
