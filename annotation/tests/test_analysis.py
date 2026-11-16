import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from app.analysis import fleiss_kappa, majority_label

ANNOTATION_DIR = Path(__file__).resolve().parent.parent


def test_fleiss_kappa():
    assert fleiss_kappa([]) is None
    assert fleiss_kappa([Counter(outbreak=1)]) is None            # single rater ignored
    assert fleiss_kappa([Counter(outbreak=3), Counter(not_outbreak=3)]) == pytest.approx(1.0)
    # P_i = 1, 1, 0 -> P_bar = 2/3; p = (.5, .5) -> P_e = .5; kappa = 1/3
    items = [Counter(outbreak=2), Counter(not_outbreak=2), Counter(outbreak=1, not_outbreak=1)]
    assert fleiss_kappa(items) == pytest.approx(1 / 3)


def test_majority_label():
    assert majority_label(Counter(outbreak=2, unsure=1)) == "outbreak"
    assert majority_label(Counter(outbreak=1, not_outbreak=1)) is None
    assert majority_label(Counter()) is None


def test_alembic_migration_matches_models(tmp_path):
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'm.db'}"}
    alembic = [sys.executable, "-m", "alembic"]
    for args in (["upgrade", "head"], ["check"], ["downgrade", "base"]):
        r = subprocess.run(alembic + args, cwd=ANNOTATION_DIR, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
