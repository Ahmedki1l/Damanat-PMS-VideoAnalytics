"""
src.ocr — plate-OCR plugins for the matching cascade.

Phase 1 / Wave 1 (WS-D) ships ``PaddlePlateOCR``: a thin, lazy-init wrapper
around PaddleOCR-mobile that implements the ``PlateOCR`` ABC from
``src.matching.plugins``. Phase 2 will gate its execution to the marginal
ReID score band defined by ``MatchingConfig.ocr_marginal_{low,high}`` — this
plugin is agnostic to that gating; it only answers "what does the plate read?"
when called.

Public surface:

    PaddlePlateOCR       - PlateOCR implementation backed by PaddleOCR-mobile
    plates_match         - Levenshtein-based plate equivalence helper
    normalise_plate_text - Uppercase + strip non-alphanumerics
"""

from .plate_ocr import (
    PaddlePlateOCR,
    levenshtein_distance,
    normalise_plate_text,
    plates_match,
)

__all__ = [
    "PaddlePlateOCR",
    "levenshtein_distance",
    "normalise_plate_text",
    "plates_match",
]
