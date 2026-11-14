"""Segmentation methods. Add a new method to METHODS to make it available everywhere."""

from .base import Segment, Segmenter, check_segments
from .threshold import SeasonalThreshold

METHODS: dict[str, type[Segmenter]] = {
    SeasonalThreshold.name: SeasonalThreshold,
}


def get_segmenter(name: str, **params) -> Segmenter:
    try:
        return METHODS[name](**params)
    except KeyError:
        raise ValueError(f"unknown segmentation method {name!r}; known: {sorted(METHODS)}") from None


__all__ = ["METHODS", "Segment", "Segmenter", "check_segments", "get_segmenter"]
