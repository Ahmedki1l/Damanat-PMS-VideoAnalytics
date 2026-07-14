"""
src.matching.plugins — Plugin ABCs for color / type classifiers and plate OCR.

The matching cascade (see ``MatchDecision``) treats every secondary modality as
a pluggable component: color, vehicle type, and plate-OCR each conform to a
small ABC. Phase 0 ships only Noop implementations so behaviour is unchanged
from the previous single-score, single-threshold flow. Phase 1 workstreams
(WS-B color, WS-C type, WS-D plate OCR) replace these with real OpenVINO /
PaddleOCR backed plugins.

Public types:

    ColorClassifier        - ABC; predict() -> (label, confidence)
    TypeClassifier         - ABC; predict() -> (label, confidence)
    PlateOCR               - ABC; read() -> (text, confidence)
    NoopColorClassifier    - wraps the legacy HSV / LAB-k-means color heuristic
    NoopTypeClassifier     - always returns ("unknown", 0.0)
    NoopPlateOCR           - always returns ("", 0.0)

The Noop color classifier intentionally preserves the existing decision logic
from ``vehicle_registry_identity.py`` (``_compare_dominant_colors`` predicate
plus the optional ``color_compatible`` HSV gate). Wave 1 / Phase 1 will swap it
for a learned 11-class head without touching the identity mixin.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# ABCs
# --------------------------------------------------------------------------- #


class ColorClassifier(ABC):
    """
    Predicts a coarse color label for a vehicle crop.

    Phase 1 will ship a learned 11-class head (black/white/grey/silver/red/
    blue/green/yellow/brown/beige/other). Phase 0 ships only ``NoopColorClassifier``
    which wraps the legacy HSV heuristic so behaviour is preserved.
    """

    @abstractmethod
    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Return ``(label, confidence)`` for the given BGR crop."""

    def colors_match(self, a: Tuple[str, float], b: Tuple[str, float]) -> bool:
        """Default policy: two predictions match iff their labels agree.

        Real classifiers may override this to e.g. treat
        ``(silver, grey, white)`` as compatible.
        """
        if not a or not b:
            return False
        label_a, _ = a
        label_b, _ = b
        if not label_a or not label_b:
            return False
        return label_a == label_b


class TypeClassifier(ABC):
    """
    Predicts a coarse body type for a vehicle crop.

    Phase 1 will ship a learned 6-class head (sedan/SUV/hatchback/pickup/van/
    bus). Phase 0 ships only ``NoopTypeClassifier``.
    """

    @abstractmethod
    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Return ``(label, confidence)`` for the given BGR crop."""


class PlateOCR(ABC):
    """
    Reads a license plate from a vehicle crop.

    Phase 1 (WS-D) ships a PaddleOCR-mobile wrapper. Phase 0 ships only
    ``NoopPlateOCR``. The OCR is NOT a primary ANPR replacement; it runs only
    when ReID cosine lands in the marginal band defined by
    ``MatchingConfig.ocr_marginal_{low,high}``.
    """

    @abstractmethod
    def read(self, crop_bgr: np.ndarray, **kwargs) -> Tuple[str, float]:
        """Return ``(text, confidence)`` where text is normalised alphanumerics.

        Implementations accept optional keyword flags (``allow_retry``,
        ``apply_plate_roi``); the base contract ignores any it does not use, so
        callers can pass them uniformly without probing the concrete type.
        """


# --------------------------------------------------------------------------- #
# Noop implementations (Phase 0 defaults)
# --------------------------------------------------------------------------- #


class NoopColorClassifier(ColorClassifier):
    """
    Legacy-equivalent color predicate. Wraps:

    1. ``VehicleImageMatcher._compare_dominant_colors`` — LAB k-means similarity
       in ``[0.0, 1.0]``. Acts as a hard-reject when below the configured
       ``color_dominant_filter`` floor.
    2. ``reid_attributes.color_compatible`` — optional HSV tolerance check,
       gated by ``MatchingConfig.use_color_filter`` (defaults to off).

    The classifier does NOT produce a discrete color label — Phase 0 wants
    *zero behaviour change*, and the legacy logic is purely score-based. The
    ``predict`` method therefore returns ``("unknown", 0.0)``. The real
    cross-check happens via ``MatchDecision.color_check(...)``, which calls
    the helpers below.

    Constructed by ``MatchDecision`` with shared references to the legacy
    matchers so we do not re-instantiate them.
    """

    def __init__(
        self,
        image_matcher=None,
        hsv_h_tol: float = 25.0,
        hsv_s_tol: float = 80.0,
        hsv_v_tol: float = 80.0,
    ):
        # ``image_matcher`` may be either a ``VehicleImageMatcher`` instance
        # (or anything exposing ``_compare_dominant_colors(img_a, img_b)``)
        # OR a zero-arg callable that lazily resolves one — needed because
        # ``VehicleRegistry.matcher`` is a lazy property and we must not
        # force its construction at __init__ time (it loads OpenCV-heavy
        # state on demand).
        self._image_matcher = image_matcher
        self._hsv_h_tol = float(hsv_h_tol)
        self._hsv_s_tol = float(hsv_s_tol)
        self._hsv_v_tol = float(hsv_v_tol)

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return ("unknown", 0.0)

    def colors_match(self, a: Tuple[str, float], b: Tuple[str, float]) -> bool:
        # The Noop classifier never claims to know the color, so it cannot
        # answer this question — defer to ``compare_pair`` below.
        return False

    # --- Helpers used by MatchDecision.color_check ------------------------- #

    def _resolve_image_matcher(self):
        """Resolve ``self._image_matcher`` whether it is an instance or a
        zero-arg factory. Returns None if no matcher is available."""
        matcher = self._image_matcher
        if matcher is None:
            return None
        if callable(matcher) and not hasattr(matcher, "_compare_dominant_colors"):
            try:
                return matcher()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "[NoopColorClassifier] image-matcher factory failed: %r", exc
                )
                return None
        return matcher

    def compare_dominant(
        self,
        query_image: np.ndarray,
        candidate_image: np.ndarray,
    ) -> float:
        """
        Returns the LAB k-means dominant-color similarity in ``[0.0, 1.0]``.

        Matches the historical call at
        ``vehicle_registry_identity.py:389`` (``self.matcher._compare_dominant_colors``).
        """
        matcher = self._resolve_image_matcher()
        if matcher is None:
            return 1.0  # fail-open — same as missing legacy matcher
        try:
            return float(
                matcher._compare_dominant_colors(  # noqa: SLF001 - intentional
                    query_image,
                    candidate_image,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[NoopColorClassifier] dominant compare failed: %r", exc)
            return 0.0

    def hsv_compatible(
        self,
        query_hsv: Optional[Tuple[float, float, float]],
        candidate_hsvs: Optional[list],
    ) -> bool:
        """
        Returns True iff at least one candidate HSV signature is compatible
        with the query under the configured (h, s, v) tolerances.

        Replicates the gate at ``vehicle_registry_identity.py:398-413``. The
        caller is responsible for only invoking this when the
        ``use_color_filter`` flag is enabled.
        """
        if query_hsv is None:
            return True
        if not candidate_hsvs:
            # No HSV signature recorded — historical code falls through and
            # does not reject the candidate.
            return True

        from src.reid_matcher import color_compatible

        return any(
            color_compatible(
                query_hsv,
                cand_hsv,
                h_tol=self._hsv_h_tol,
                s_tol=self._hsv_s_tol,
                v_tol=self._hsv_v_tol,
            )
            for cand_hsv in candidate_hsvs
            if cand_hsv is not None
        )


class NoopTypeClassifier(TypeClassifier):
    """Returns ``("unknown", 0.0)`` — Phase 0 has no type modality."""

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return ("unknown", 0.0)


class NoopPlateOCR(PlateOCR):
    """Returns ``("", 0.0)`` — Phase 0 has no OCR cross-check."""

    def read(self, crop_bgr: np.ndarray, **kwargs) -> Tuple[str, float]:
        return ("", 0.0)
