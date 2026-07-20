"""
src.matching — centralised vehicle matching decision logic.

This package is the integration seam for the Phase-1 parallel workstreams
(color, type, OCR, fast-ReID). Every match decision the
``VehicleRegistry`` makes passes through ``MatchDecision`` so the threshold
constants live in one place and plugins can be swapped without touching the
identity mixin.

Public surface:

    MatchDecision        - main decision orchestrator
    Decision             - NamedTuple {verdict, reason, scores}
    ColorCheck           - result of MatchDecision.color_check(...)
    ColorClassifier      - ABC for color modality plugins
    TypeClassifier       - ABC for body-type modality plugins
    PlateOCR             - ABC for plate-OCR cross-check plugins
    NoopColorClassifier  - Phase-0 default that wraps the legacy color predicate
    NoopTypeClassifier   - Phase-0 default; returns ("unknown", 0.0)
    NoopPlateOCR         - Phase-0 default; returns ("", 0.0)
    PlateRegionDetector  - ABC for plate region detection plugins
    NoopPlateRegionDetector - default no-op detector that returns empty list
    GalleryIndex         - Phase 3 / T3.2 FAISS-CPU IVF gallery index
"""

from .gallery_index import GalleryIndex
from .match_decision import ColorCheck, Decision, MatchDecision
from .match_voter import MatchVoter
from .plugins import (
    ColorClassifier,
    NoopColorClassifier,
    NoopPlateOCR,
    NoopPlateRegionDetector,
    NoopTypeClassifier,
    PlateOCR,
    PlateRegionDetector,
    TypeClassifier,
)

__all__ = [
    "ColorCheck",
    "Decision",
    "MatchDecision",
    "MatchVoter",
    "ColorClassifier",
    "TypeClassifier",
    "PlateOCR",
    "PlateRegionDetector",
    "NoopColorClassifier",
    "NoopTypeClassifier",
    "NoopPlateOCR",
    "NoopPlateRegionDetector",
    "GalleryIndex",
]
