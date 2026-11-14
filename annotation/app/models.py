from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

LABELS = ("outbreak", "not_outbreak", "unsure", "skip")
# Why a series was shown: drawn at random, picked because the model was unsure about it, reopened by the
# expert from their own history, or opened directly by link. "random" draws give an unbiased sample for
# measuring the model.
SERVED_AS = ("random", "uncertain", "revisit", "lookup")
VOTE_LABELS = ("outbreak", "not_outbreak", "unsure")  # labels that count toward coverage


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend.

    SQLite drops the offset; Postgres returns values in the session time zone.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        return value.astimezone(timezone.utc) if value is not None else None

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(128))
    affiliation: Mapped[str | None] = mapped_column(String(255))
    expertise: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Outbreak(Base):
    __tablename__ = "outbreaks"

    # Same value as unique_id in raw_data/outbreaks_disease_location.csv.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    disease: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[str] = mapped_column(String(128), index=True)
    event: Mapped[str] = mapped_column(String(64))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    duration: Mapped[int] = mapped_column(Integer)
    total: Mapped[float | None] = mapped_column(Float)
    # Weekly values; value i is for the week ending start_date + 7*i. Trailing blanks removed.
    values: Mapped[list] = mapped_column(JSON)
    source_file: Mapped[str] = mapped_column(String(255))
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Annotation(Base):
    """Append-only: an edit adds a new row. A user's current answer is their row with the highest id."""

    __tablename__ = "annotations"
    __table_args__ = (
        CheckConstraint("label IN ('outbreak','not_outbreak','unsure','skip')", name="ck_annotation_label"),
        CheckConstraint(
            "(label = 'skip' AND confidence IS NULL) OR (label <> 'skip' AND confidence BETWEEN 1 AND 5)",
            name="ck_annotation_confidence",
        ),
        Index("ix_annotations_outbreak_user_id", "outbreak_id", "user_id", "id"),
        Index("ix_annotations_user_id", "user_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    time_spent_ms: Mapped[int | None] = mapped_column(Integer)
    client_info: Mapped[str | None] = mapped_column(String(512))
    app_version: Mapped[str | None] = mapped_column(String(32))
    served_as: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    user: Mapped[User] = relationship()
    outbreak: Mapped[Outbreak] = relationship()


class Assignment(Base):
    """Log of every series served by the queue, and why."""

    __tablename__ = "assignments"
    __table_args__ = (Index("ix_assignments_user_outbreak", "user_id", "outbreak_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"))
    reason: Mapped[str] = mapped_column(String(16))
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id", ondelete="SET NULL"))
    served_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ModelRun(Base):
    """One training run: what it was trained on and how well it did."""

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    model_type: Mapped[str] = mapped_column(String(32))
    n_votes: Mapped[int] = mapped_column(Integer)          # expert Outbreak / Not outbreak answers used
    n_outbreaks: Mapped[int] = mapped_column(Integer)      # distinct series those answers cover
    train_seconds: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict] = mapped_column(JSON)            # cross-validation and random-sample evaluation


class ModelPrediction(Base):
    """The latest model's score for every series (replaced wholesale by each training run)."""

    __tablename__ = "model_predictions"
    __table_args__ = (Index("ix_model_predictions_uncertainty", "uncertainty"),)

    outbreak_id: Mapped[int] = mapped_column(ForeignKey("outbreaks.id", ondelete="CASCADE"), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    p_outbreak: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[float] = mapped_column(Float)      # 1 at p = 0.5, 0 at p = 0 or 1


# ---------- segment review: experts check where a segmentation method split a full series ----------

SEGMENT_VERDICTS = ("accept", "adjust", "reject", "added")


class Series(Base):
    """A full weekly series, before it is split into outbreaks (pipeline "series" file)."""

    __tablename__ = "series"

    id: Mapped[int] = mapped_column(primary_key=True)
    series_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)   # series_id in the files
    source: Mapped[str | None] = mapped_column(String(128))
    disease: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[str] = mapped_column(String(128))
    geo_level: Mapped[str | None] = mapped_column(String(32))
    event: Mapped[str] = mapped_column(String(64))
    start_date: Mapped[date] = mapped_column(Date)
    # Weekly values; value i is for the week ending start_date + 7*i. None = missing week.
    values: Mapped[list] = mapped_column(JSON)
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SegmentationRun(Base):
    """One method's output loaded into the app. Experts review one run at a time."""

    __tablename__ = "segmentation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    method: Mapped[str] = mapped_column(String(128))
    method_version: Mapped[str | None] = mapped_column(String(64))
    source_file: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class RunSeries(Base):
    """Which series a run covers, including series where the method found no outbreak (experts can
    still add one there)."""

    __tablename__ = "run_series"

    run_id: Mapped[int] = mapped_column(ForeignKey("segmentation_runs.id", ondelete="CASCADE"), primary_key=True)
    series_id: Mapped[int] = mapped_column(ForeignKey("series.id", ondelete="CASCADE"), primary_key=True)


class ProposedSegment(Base):
    __tablename__ = "proposed_segments"
    __table_args__ = (Index("ix_proposed_segments_run_series", "run_id", "series_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("segmentation_runs.id", ondelete="CASCADE"))
    series_id: Mapped[int] = mapped_column(ForeignKey("series.id", ondelete="CASCADE"))
    segment_key: Mapped[str] = mapped_column(String(255))     # segment_id in the files
    start_idx: Mapped[int] = mapped_column(Integer)
    end_idx: Mapped[int] = mapped_column(Integer)              # inclusive
    score: Mapped[float | None] = mapped_column(Float)
    # Optional, for methods that separate the full window from the outbreak proper.
    peak_idx: Mapped[int | None] = mapped_column(Integer)
    core_start_idx: Mapped[int | None] = mapped_column(Integer)
    core_end_idx: Mapped[int | None] = mapped_column(Integer)


class SegmentReview(Base):
    """One expert's review of all proposed segments of one series in one run. Append-only, like
    Annotation: the current review is the user's row with the highest id.

    `items` holds one entry per decision:
      {"proposed_id": int | None, "verdict": "accept"|"adjust"|"reject"|"added",
       "start_idx": int | None, "end_idx": int | None}
    """

    __tablename__ = "segment_reviews"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR confidence BETWEEN 1 AND 5", name="ck_segment_review_confidence"),
        Index("ix_segment_reviews_run_series_user", "run_id", "series_id", "user_id", "id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("segmentation_runs.id", ondelete="CASCADE"))
    series_id: Mapped[int] = mapped_column(ForeignKey("series.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    items: Mapped[list] = mapped_column(JSON)
    series_unusable: Mapped[bool] = mapped_column(Boolean, default=False)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    time_spent_ms: Mapped[int | None] = mapped_column(Integer)
    app_version: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
