"""
src.matching.match_decision — Centralised decision chokepoint for vehicle ReID.

Before Phase 0 the decision logic was spread across three callsites in
``vehicle_registry_identity.py``:

  * Lines 429-480 (B1_Entrance confirmation fork)
  * Lines 826-834 (match_global_session plate / cross-camera fork)
  * Lines 985-991 (reattach_track_to_confirmed_session fork)

Each callsite had its own constants (0.47 / 0.55 / 0.46 / 0.43 / 0.52) and an
implicit "single-candidate fallback" policy. Phase 0 consolidates all of them
into ``MatchDecision`` so the five Phase-1 workstreams (color, type, OCR,
fast-ReID, test-infra) can plug in without re-editing the identity mixin.

``MatchDecision`` is constructed once with a ``MatchingConfig`` and (optionally)
plugin instances. It is held on ``VehicleRegistry`` and called from the
identity mixin.

Public surface:

    MatchDecision(config, *, color_classifier=None, type_classifier=None,
                  plate_ocr=None, image_matcher=None)
    .decide_b1(score_reid, *, is_anpr_candidate, candidate_count) -> Decision
    .decide_global(score_reid, *, has_plate, cross_camera, similarity_threshold) -> Decision
    .decide_reattach(score_reid, *, cross_camera, similarity_threshold) -> Decision
    .color_check(query_image, candidate_image, *, query_hsv, candidate_hsvs) -> ColorCheck

Each ``decide_*`` returns a ``Decision`` namedtuple:

    Decision(verdict='confirm'|'reject'|'tentative', reason=str, scores=dict)

Phase 0 implementation is a literal port of the historical thresholds — see
inline comments tagging each branch back to its original source location.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

from src.config import MatchingConfig

from .plugins import (
    ColorClassifier,
    NoopColorClassifier,
    NoopPlateOCR,
    NoopTypeClassifier,
    PlateOCR,
    TypeClassifier,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


class Decision(NamedTuple):
    """
    Outcome of a single matching decision.

    Attributes:
        verdict: 'confirm' | 'reject' | 'tentative'.
            * 'confirm'   — the candidate is the same vehicle.
            * 'reject'    — the candidate is NOT the same vehicle.
            * 'tentative' — neither confirmed nor rejected outright. Phase 2
              ensemble / voting will resolve. Phase 0 callsites currently treat
              this the same as 'reject' (i.e. "do not promote") to preserve
              behaviour.
        reason: short tag identifying which branch fired (used by audit logs).
        scores: dict carrying the input/output scores for downstream logging.
    """

    verdict: str
    reason: str
    scores: dict


@dataclass
class ColorCheck:
    """Outcome of ``MatchDecision.color_check`` (T0.3 callsite refactor).

    Attributes:
        passes_dominant: True iff dominant-color similarity met the
            ``color_dominant_filter`` floor.
        dominant_score: raw [0.0, 1.0] dominant-color similarity.
        passes_hsv: True iff the HSV compatibility check passed. This is
            False ONLY when ``use_color_filter`` is enabled AND the candidate
            HSV signature is incompatible with the query.
        hard_reject: True iff the candidate must be removed from the pool
            (separate from "skip but maybe fall back later"). Mirrors the
            historical distinction between the dominant-color filter (skip)
            and the HSV color filter (hard reject so the single-candidate
            fallback also discards it).
    """

    passes_dominant: bool
    dominant_score: float
    passes_hsv: bool
    hard_reject: bool


# --------------------------------------------------------------------------- #
# Main decision class
# --------------------------------------------------------------------------- #


class MatchDecision:
    """Centralised threshold logic + plugin orchestration for ReID matching."""

    def __init__(
        self,
        config: MatchingConfig,
        *,
        color_classifier: Optional[ColorClassifier] = None,
        type_classifier: Optional[TypeClassifier] = None,
        plate_ocr: Optional[PlateOCR] = None,
        image_matcher=None,
    ):
        self._config = config
        # Default to Noop implementations so Phase 0 has zero behaviour change.
        self._color_classifier = color_classifier or NoopColorClassifier(
            image_matcher=image_matcher,
            hsv_h_tol=config.hsv_h_tol,
            hsv_s_tol=config.hsv_s_tol,
            hsv_v_tol=config.hsv_v_tol,
        )
        self._type_classifier = type_classifier or NoopTypeClassifier()
        self._plate_ocr = plate_ocr or NoopPlateOCR()

    # ------------------------------------------------------------------ #
    # Properties for callers that need to inspect / swap plugins later.
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> MatchingConfig:
        return self._config

    @property
    def color_classifier(self) -> ColorClassifier:
        return self._color_classifier

    @property
    def type_classifier(self) -> TypeClassifier:
        return self._type_classifier

    @property
    def plate_ocr(self) -> PlateOCR:
        return self._plate_ocr

    # ------------------------------------------------------------------ #
    # B1_Entrance confirmation (vehicle_registry_identity.py:429-480)
    # ------------------------------------------------------------------ #

    def decide_b1(
        self,
        score_reid: float,
        *,
        is_anpr_candidate: bool,
        candidate_count: int = 0,
        color_check: Optional[ColorCheck] = None,
        type_check=None,
        ocr_check=None,
        cross_camera: bool = False,
    ) -> Decision:
        """
        Decide whether a B1_Entrance frame confirms a Park_Entry candidate.

        Replicates the inline threshold fork at
        ``vehicle_registry_identity.py:429-440`` and the "single-candidate
        fallback" at lines 464-480. Phase 0 leaves color / type / ocr / cross_camera
        parameters available but inert — Phase 2 wires them up.
        """
        cfg = self._config
        # ANPR-image candidates were captured by the dedicated ANPR camera and
        # are a higher-quality reference — use a lower acceptance threshold
        # than opportunistic zone crops.
        if is_anpr_candidate:
            threshold = cfg.b1_anpr
            threshold_reason = "b1_anpr"
        else:
            threshold = cfg.b1_zone
            threshold_reason = "b1_zone"

        scores = {
            "reid": float(score_reid),
            "threshold": float(threshold),
            "is_anpr_candidate": bool(is_anpr_candidate),
            "candidate_count": int(candidate_count),
        }

        if score_reid >= threshold:
            return Decision(
                verdict="confirm",
                reason=threshold_reason,
                scores=scores,
            )

        # Strict visual threshold was not met. Apply the historical
        # single-candidate fallback policy: when exactly one provisional
        # candidate remains (after hard-rejects), confirm it anyway with a
        # warning. The callsite emits the warning log so the wording remains
        # identical to the pre-refactor message.
        if candidate_count == 1:
            return Decision(
                verdict="confirm",
                reason="single_candidate_fallback",
                scores=scores,
            )

        return Decision(verdict="reject", reason="below_threshold", scores=scores)

    # ------------------------------------------------------------------ #
    # match_global_session (vehicle_registry_identity.py:826-834)
    # ------------------------------------------------------------------ #

    def decide_global(
        self,
        score_reid: float,
        *,
        has_plate: bool,
        cross_camera: bool,
        similarity_threshold: Optional[float] = None,
        color_check: Optional[ColorCheck] = None,
        type_check=None,
    ) -> Decision:
        """
        Decide whether a query feature vector matches a globally confirmed
        session. Mirrors the threshold fork at
        ``vehicle_registry_identity.py:826-834``.

        The legacy caller passes its own ``similarity_threshold`` (default 0.55),
        which we honour as the "no plate" baseline. When the session has a
        plate, the threshold drops to ``global_with_plate``; on cross-camera,
        it drops further to ``global_cross_camera``.
        """
        cfg = self._config
        base = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else cfg.global_default
        )

        if has_plate:
            effective = cfg.global_with_plate
            reason = "global_with_plate"
            if cross_camera:
                effective = cfg.global_cross_camera
                reason = "global_cross_camera"
        else:
            effective = base
            reason = "global_default"

        scores = {
            "reid": float(score_reid),
            "threshold": float(effective),
            "has_plate": bool(has_plate),
            "cross_camera": bool(cross_camera),
        }

        if score_reid >= effective:
            return Decision(verdict="confirm", reason=reason, scores=scores)
        return Decision(verdict="reject", reason="below_threshold", scores=scores)

    # ------------------------------------------------------------------ #
    # reattach_track_to_confirmed_session (vehicle_registry_identity.py:985-991)
    # ------------------------------------------------------------------ #

    def decide_reattach(
        self,
        score_reid: float,
        *,
        cross_camera: bool,
        similarity_threshold: Optional[float] = None,
        color_check: Optional[ColorCheck] = None,
        type_check=None,
    ) -> Decision:
        """
        Decide whether to reattach a currently-anonymous track to an existing
        confirmed session. Replicates the
        ``effective_threshold = min(similarity_threshold, 0.43)`` cross-camera
        rule at ``vehicle_registry_identity.py:985-991``.
        """
        cfg = self._config
        base = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else cfg.reattach_default
        )

        if cross_camera:
            effective = min(base, cfg.reattach_cross_camera)
            reason = "reattach_cross_camera"
        else:
            effective = base
            reason = "reattach_default"

        scores = {
            "reid": float(score_reid),
            "threshold": float(effective),
            "cross_camera": bool(cross_camera),
        }

        if score_reid >= effective:
            return Decision(verdict="confirm", reason=reason, scores=scores)
        return Decision(verdict="reject", reason="below_threshold", scores=scores)

    # ------------------------------------------------------------------ #
    # Color predicate (replaces the inline _compare_dominant_colors +
    # color_compatible block at vehicle_registry_identity.py:389-413).
    # ------------------------------------------------------------------ #

    def color_check(
        self,
        query_image: np.ndarray,
        candidate_image: np.ndarray,
        *,
        query_hsv: Optional[Tuple[float, float, float]] = None,
        candidate_hsvs: Optional[list] = None,
        use_color_filter: Optional[bool] = None,
    ) -> ColorCheck:
        """
        Run the legacy two-step color predicate:

          1. Dominant-color similarity (LAB k-means) — must clear
             ``color_dominant_filter`` (default 0.45). A failure here causes
             the caller to ``continue`` without hard-rejecting the candidate
             from the single-candidate fallback pool — matching the historical
             behaviour at vehicle_registry_identity.py:389-396.
          2. HSV ``color_compatible`` gate — only consulted when
             ``use_color_filter`` is enabled. Failure here HARD-REJECTS the
             candidate (i.e. it is removed from the fallback pool too),
             matching vehicle_registry_identity.py:398-413.

        Args:
            use_color_filter: optional override for the
                ``MatchingConfig.use_color_filter`` flag. The B1 callsite uses
                this to honour module-level patches against the legacy
                ``REID_USE_COLOR_FILTER`` global without rebuilding the
                config.
        """
        cfg = self._config
        cc = self._color_classifier
        effective_use_filter = (
            cfg.use_color_filter if use_color_filter is None else bool(use_color_filter)
        )

        # --- Step 1: dominant color predicate -----------------------------
        if hasattr(cc, "compare_dominant"):
            dominant_score = cc.compare_dominant(query_image, candidate_image)
        else:
            dominant_score = 1.0

        passes_dominant = dominant_score >= cfg.color_dominant_filter

        # --- Step 2: HSV gate (only when use_color_filter) ----------------
        if effective_use_filter and hasattr(cc, "hsv_compatible"):
            passes_hsv = cc.hsv_compatible(query_hsv, candidate_hsvs)
        else:
            passes_hsv = True

        # In the historical code, only an HSV failure adds the candidate to
        # ``hard_rejected_candidate_ids``. A dominant-color failure causes a
        # plain ``continue`` and the candidate remains eligible for the
        # single-candidate fallback. Preserve that distinction.
        hard_reject = effective_use_filter and not passes_hsv

        return ColorCheck(
            passes_dominant=passes_dominant,
            dominant_score=float(dominant_score),
            passes_hsv=passes_hsv,
            hard_reject=hard_reject,
        )
