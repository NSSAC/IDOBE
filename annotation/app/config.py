"""Settings read from environment variables, so the same code runs locally and when hosted."""

import os
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path

ANNOTATION_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ANNOTATION_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from annotation/.env; real environment variables take precedence."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv(ANNOTATION_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _secret_key() -> str:
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    warnings.warn(
        "SECRET_KEY is not set; using a random key. Sessions will not survive a restart. "
        "Set SECRET_KEY in production.",
        stacklevel=2,
    )
    return secrets.token_urlsafe(32)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", f"sqlite:///{ANNOTATION_DIR / 'annotation.db'}")
    # Hosting providers often hand out postgres:// URLs; SQLAlchemy needs an explicit driver.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_database_url)
    secret_key: str = field(default_factory=_secret_key)
    target_annotations: int = field(default_factory=lambda: int(os.environ.get("TARGET_ANNOTATIONS", "3")))
    # "active": random until a model exists, then mostly series the model is unsure about (see app/ml).
    # "random": any eligible series, uniformly at random.
    # "spread": least-labeled series first (coverage grows evenly across the dataset).
    # "complete": most-labeled series still under target first (consensus labels and agreement sooner).
    queue_order: str = field(default_factory=lambda: os.environ.get("QUEUE_ORDER", "active"))
    # Active learning (queue_order "active"): after warm-up, this share of series is drawn at random and
    # the rest from the `active_top_k` eligible series the model is least sure about.
    active_random_fraction: float = field(default_factory=lambda: float(os.environ.get("ACTIVE_RANDOM_FRACTION", "0.33")))
    active_top_k: int = field(default_factory=lambda: int(os.environ.get("ACTIVE_TOP_K", "200")))
    # Train once there are this many Outbreak / Not outbreak answers, with at least this many of each.
    model_min_labels: int = field(default_factory=lambda: int(os.environ.get("MODEL_MIN_LABELS", "50")))
    model_min_per_class: int = field(default_factory=lambda: int(os.environ.get("MODEL_MIN_PER_CLASS", "10")))
    retrain_every: int = field(default_factory=lambda: int(os.environ.get("RETRAIN_EVERY", "20")))
    auto_retrain: bool = field(default_factory=lambda: _bool("AUTO_RETRAIN", True))
    # Stop asking more experts about a series once one expert agrees with a confident model
    # (uncertainty below this), but only once the model was trained on at least early_stop_min_labels
    # answers and its cross-validated AUC is at least early_stop_min_auc (small-sample AUCs are noisy).
    confident_uncertainty: float = field(default_factory=lambda: float(os.environ.get("CONFIDENT_UNCERTAINTY", "0.2")))
    early_stop_min_labels: int = field(default_factory=lambda: int(os.environ.get("EARLY_STOP_MIN_LABELS", "200")))
    early_stop_min_auc: float = field(default_factory=lambda: float(os.environ.get("EARLY_STOP_MIN_AUC", "0.85")))
    cookie_secure: bool = field(default_factory=lambda: _bool("COOKIE_SECURE", False))
    session_max_age: int = field(default_factory=lambda: int(os.environ.get("SESSION_MAX_AGE", str(14 * 24 * 3600))))
    login_max_failures: int = field(default_factory=lambda: int(os.environ.get("LOGIN_MAX_FAILURES", "10")))
    login_window_seconds: int = field(default_factory=lambda: int(os.environ.get("LOGIN_WINDOW_SECONDS", "900")))


settings = Settings()
QUEUE_ORDERS = {"active", "random", "spread", "complete"}
if settings.queue_order not in QUEUE_ORDERS:
    raise ValueError(f"QUEUE_ORDER must be one of {sorted(QUEUE_ORDERS)}, not {settings.queue_order!r}")
