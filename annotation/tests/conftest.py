import os
import sys
import tempfile
from datetime import date
from pathlib import Path

# Settings are read at import time, so point the app at a throwaway database before importing it.
_TMP = Path(tempfile.mkdtemp(prefix="idobe-annotation-tests-"))
# Set TEST_DATABASE_URL to run the suite against Postgres; its tables are dropped and recreated per test.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{_TMP / 'test.db'}"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["TARGET_ANNOTATIONS"] = "2"
os.environ["AUTO_RETRAIN"] = "false"          # tests call training explicitly
os.environ["MODEL_MIN_LABELS"] = "8"
os.environ["MODEL_MIN_PER_CLASS"] = "3"
os.environ["RETRAIN_EVERY"] = "4"
os.environ["EARLY_STOP_MIN_LABELS"] = "10"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.auth import hash_password, login_limiter  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.ml.train import _feature_cache  # noqa: E402
from app.models import Base, Outbreak, User  # noqa: E402

PASSWORD = "correct horse battery"


@pytest.fixture
def db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    login_limiter._failures.clear()
    _feature_cache.clear()
    with SessionLocal() as session:
        yield session


@pytest.fixture
def outbreaks(db):
    rows = [
        Outbreak(id=i, disease="MEASLES", location=f"LOC{i}", event="CASES", start_date=date(2020, 1, 4),
                 end_date=date(2020, 3, 28), duration=3, total=float(i), values=[0.0, 1.0, 5.0, 2.0, 0.0],
                 source_file="test")
        for i in range(1, 6)
    ]
    db.add_all(rows)
    db.commit()
    return rows


def make_user(db, username: str, admin: bool = False) -> User:
    user = User(username=username, password_hash=hash_password(PASSWORD), is_admin=admin)
    db.add(user)
    db.commit()
    return user


def logged_in_client(db, username: str, admin: bool = False) -> TestClient:
    make_user(db, username, admin)
    client = TestClient(app)
    r = client.post("/api/login", json={"username": username, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return client
