"""The interface every outbreak segmentation method implements.

A segmenter looks at one full weekly series and proposes outbreak windows. It knows nothing about
files, the annotation app or the benchmark: `run_segmenter` does that wiring. To add a method,
subclass `Segmenter`, implement `segment`, and register it in `segmentation/__init__.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class Segment:
    """A proposed outbreak: weeks start..end of the series, both inclusive (0-based indices).

    start..end is the full window (e.g. trough to trough). Methods that also locate the outbreak
    proper can give core_start..core_end inside it (e.g. onset to end after trimming) and the peak
    week. All optional.
    """

    start: int
    end: int
    score: float | None = None
    peak: int | None = None
    core_start: int | None = None
    core_end: int | None = None
    info: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid segment {self.start}..{self.end}")
        for name in ("peak", "core_start", "core_end"):
            v = getattr(self, name)
            if v is not None and not self.start <= v <= self.end:
                raise ValueError(f"{name}={v} is outside the segment {self.start}..{self.end}")
        if self.core_start is not None and self.core_end is not None and self.core_end < self.core_start:
            raise ValueError("core_end is before core_start")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class Segmenter(ABC):
    name: str = "segmenter"
    version: str = "0"

    def __init__(self, **params: Any):
        self.params = {**self.default_params(), **params}

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @abstractmethod
    def segment(self, values: Sequence[float | None], *, event: str = "", disease: str = "") -> list[Segment]:
        """Return non-overlapping segments sorted by start (an empty list is fine for flat or short
        series). `values` may contain None (missing weeks)."""

    def week_scores(self, values: Sequence[float | None], *, event: str = "", disease: str = "") -> list[float] | None:
        """Optional: per-week probability of being inside an outbreak (e.g. from an HMM). None if
        the method has no such notion. Shown to experts as a faint band; not required."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"method": self.name, "version": self.version, "params": self.params}


def check_segments(segments: list[Segment], n: int) -> list[Segment]:
    """Validate a segmenter's output against a series of length n."""
    segments = sorted(segments, key=lambda s: s.start)
    for a, b in zip(segments, segments[1:]):
        if b.start <= a.end:
            raise ValueError(f"overlapping segments {a.start}..{a.end} and {b.start}..{b.end}")
    if segments and segments[-1].end >= n:
        raise ValueError(f"segment ends at {segments[-1].end}, series has {n} weeks")
    return segments
