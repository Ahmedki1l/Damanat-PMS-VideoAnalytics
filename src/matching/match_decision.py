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

Phase 2 (T2.1) extends the decide_* methods with an ensemble rule that calls
the real color / type / OCR plugins. The ensemble is feature-flagged by
``MatchingConfig.ensemble_min_modalities_agree`` and the solo-confirm threshold
``MatchingConfig.reid_solo_confirm`` — both default to values that preserve
Phase-0 behaviour when no real plugins are loaded.

``MatchDecision`` is constructed once with a ``MatchingConfig`` and (optionally)
plugin instances. It is held on ``VehicleRegistry`` and called from the
identity mixin.

Public surface:

    MatchDecision(config, *, color_classifier=None, type_classifier=None,
                  plate_ocr=None, image_matcher=None)
    .decide_b1(score_reid, *, is_anpr_candidate, candidate_count,
               modalities=None) -> Decision
    .decide_global(score_reid, *, has_plate, cross_camera,
                   similarity_threshold, modalities=None) -> Decision
    .decide_reattach(score_reid, *, cross_camera, similarity_threshold,
                     modalities=None) -> Decision
    .check_modalities(query_crop, candidate_crop, candidate,
                      pending_plate=None) -> dict
    .color_check(query_image, candidate_image, *, query_hsv, candidate_hsvs) -> ColorCheck

Each ``decide_*`` returns a ``Decision`` namedtuple:

    Decision(verdict='confirm'|'reject'|'tentative', reason=str, scores=dict)

The ensemble rule (T2.1):

    confirm iff (score_reid >= reid_solo_confirm)
             OR (>= ensemble_min_modalities_agree of
                 {ReID>=threshold, color_match, type_match, ocr_match})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

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
        modalities: Optional[Mapping[str, Any]] = None,
    ) -> Decision:
        """
        Decide whether a B1_Entrance frame confirms a Park_Entry candidate.

        Replicates the inline threshold fork at
        ``vehicle_registry_identity.py:429-440`` and the "single-candidate
        fallback" at lines 464-480.

        Phase 2 (T2.1) ensemble extension:
        ``modalities`` is the dict returned by ``check_modalities(...)``. When
        provided, the ensemble rule is applied: confirm if (ReID >=
        ``reid_solo_confirm``) OR (>= ``ensemble_min_modalities_agree`` of
        {ReID >= threshold, color_match, type_match, ocr_match}) agree.
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

        scores: Dict[str, Any] = {
            "reid": float(score_reid),
            "threshold": float(threshold),
            "is_anpr_candidate": bool(is_anpr_candidate),
            "candidate_count": int(candidate_count),
        }

        # Apply the ensemble rule first when modality signals are supplied.
        ensemble = self._apply_ensemble(score_reid, threshold, modalities, scores)
        if ensemble is not None:
            return ensemble

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
            # Open-set appearance floor: a lone provisional candidate is confirmed
            # only when its ReID clears ``single_candidate_min_reid``. The historical
            # policy confirmed it at ANY score (the caller even passed 0.0), which
            # let the NEXT car adopt a PREVIOUS car's stale pending plate at ~0
            # appearance agreement — the gate identity-swap incident. The caller now
            # passes the real survivor score so this floor can reject a blind bind.
            floor = float(getattr(cfg, "single_candidate_min_reid", 0.0))
            scores["single_candidate_floor"] = floor
            if score_reid >= floor:
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
        modalities: Optional[Mapping[str, Any]] = None,
    ) -> Decision:
        """
        Decide whether a query feature vector matches a globally confirmed
        session. Mirrors the threshold fork at
        ``vehicle_registry_identity.py:826-834``.

        The legacy caller passes its own ``similarity_threshold`` (default 0.55),
        which we honour as the "no plate" baseline. When the session has a
        plate, the threshold drops to ``global_with_plate``; on cross-camera,
        it drops further to ``global_cross_camera``.

        Phase 2 (T2.1): ``modalities`` enables the ensemble rule — see
        :meth:`decide_b1`.
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

        scores: Dict[str, Any] = {
            "reid": float(score_reid),
            "threshold": float(effective),
            "has_plate": bool(has_plate),
            "cross_camera": bool(cross_camera),
        }

        ensemble = self._apply_ensemble(score_reid, effective, modalities, scores)
        if ensemble is not None:
            return ensemble

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
        modalities: Optional[Mapping[str, Any]] = None,
    ) -> Decision:
        """
        Decide whether to reattach a currently-anonymous track to an existing
        confirmed session. Replicates the
        ``effective_threshold = min(similarity_threshold, 0.43)`` cross-camera
        rule at ``vehicle_registry_identity.py:985-991``.

        Phase 2 (T2.1): ``modalities`` enables the ensemble rule.
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

        scores: Dict[str, Any] = {
            "reid": float(score_reid),
            "threshold": float(effective),
            "cross_camera": bool(cross_camera),
        }

        ensemble = self._apply_ensemble(score_reid, effective, modalities, scores)
        if ensemble is not None:
            return ensemble

        if score_reid >= effective:
            return Decision(verdict="confirm", reason=reason, scores=scores)
        return Decision(verdict="reject", reason="below_threshold", scores=scores)

    # ------------------------------------------------------------------ #
    # Ensemble helper (Phase 2 T2.1)
    # ------------------------------------------------------------------ #

    def _apply_ensemble(
        self,
        score_reid: float,
        threshold: float,
        modalities: Optional[Mapping[str, Any]],
        scores: Dict[str, Any],
    ) -> Optional[Decision]:
        """Apply the ensemble rule when ``modalities`` is provided.

        Returns a ``Decision`` if the ensemble rule fires (either solo confirm
        or hard reject because of an OCR contradiction), or None to let the
        caller's legacy threshold fork run unchanged.

        Rule:
          1. ReID solo confirm — if ``score_reid >= reid_solo_confirm`` always
             confirm (reason='ensemble_reid_solo').
          2. OCR hard reject — if OCR was attempted, returned a high-confidence
             reading, and disagrees with the pending plate, hard reject.
          3. OCR hard confirm — if OCR agrees with the pending plate at high
             confidence, confirm even at marginal ReID.
          4. K-of-N agreement — count {ReID >= threshold, color_match,
             type_match, ocr_match}; confirm iff >=
             ``ensemble_min_modalities_agree``.

        When the score is in the marginal band (>= ocr_marginal_low,
        < threshold), we still defer to the legacy single-candidate fallback
        at decide_b1's caller — ensemble may decline to fire (return None).
        """
        if not modalities:
            return None

        cfg = self._config
        color_match = bool(modalities.get("color_match", False))
        type_match = bool(modalities.get("type_match", False))
        ocr_match = bool(modalities.get("ocr_match", False))
        ocr_contradiction = bool(modalities.get("ocr_contradiction", False))
        color_conf = float(modalities.get("color_conf", 0.0))
        type_conf = float(modalities.get("type_conf", 0.0))
        ocr_conf = float(modalities.get("ocr_conf", 0.0))

        # Carry modality signals into the decision scores dict so the audit
        # log captures what the cascade saw.
        scores.update(
            {
                "color_match": color_match,
                "type_match": type_match,
                "ocr_match": ocr_match,
                "ocr_contradiction": ocr_contradiction,
                "color_conf": color_conf,
                "type_conf": type_conf,
                "ocr_conf": ocr_conf,
            }
        )

        # Rule 1 — ReID solo confirm above the no-arguments-needed bar.
        if score_reid >= cfg.reid_solo_confirm:
            scores["ensemble_reason"] = "reid_solo_confirm"
            return Decision(
                verdict="confirm",
                reason="ensemble_reid_solo",
                scores=scores,
            )

        # Rule 2 — OCR contradiction is a hard reject regardless of ReID.
        if ocr_contradiction:
            scores["ensemble_reason"] = "ocr_contradiction"
            return Decision(
                verdict="reject",
                reason="ensemble_ocr_contradiction",
                scores=scores,
            )

        # Rule 3 — OCR agreement at high confidence is a hard confirm.
        # Threshold (0.55) matches the OCR rec_score floor; we require the
        # plate match boolean rather than just per-line confidence.
        if ocr_match and ocr_conf >= cfg.ocr_marginal_high:
            scores["ensemble_reason"] = "ocr_match"
            return Decision(
                verdict="confirm",
                reason="ensemble_ocr_match",
                scores=scores,
            )

        # Rule 4 — K-of-N agreement, gated by a ReID floor and a high-entropy
        # signal. Colour and body-type are LOW-entropy (many cars share them),
        # so they must never confirm an identity on their own. A K-of-N confirm
        # now requires BOTH:
        #   (a) ReID above a floor (``ocr_marginal_low``) — a near-zero ReID is
        #       a different car no matter what else agrees; and
        #   (b) at least one high-entropy modality in the agreeing set — ReID
        #       clearing its threshold, or an OCR plate match.
        # This closes the "two white sedans confirmed at ReID 0.30" hole while
        # still letting a marginal ReID be rescued by an OCR agreement (OCR at
        # high confidence is already a hard confirm via Rule 3 above).
        reid_pass = score_reid >= threshold
        reid_floor = float(getattr(cfg, "ocr_marginal_low", threshold))
        reid_floor_pass = score_reid >= reid_floor
        high_entropy_agree = reid_pass or ocr_match
        agree_count = sum([reid_pass, color_match, type_match, ocr_match])
        scores["agree_count"] = int(agree_count)
        scores["min_modalities_agree"] = int(cfg.ensemble_min_modalities_agree)
        scores["reid_floor"] = round(reid_floor, 4)
        scores["reid_floor_pass"] = bool(reid_floor_pass)
        scores["high_entropy_agree"] = bool(high_entropy_agree)

        if (
            reid_floor_pass
            and high_entropy_agree
            and agree_count >= cfg.ensemble_min_modalities_agree
        ):
            scores["ensemble_reason"] = f"k_of_n_{agree_count}"
            return Decision(
                verdict="confirm",
                reason="ensemble_agree",
                scores=scores,
            )

        # Ensemble did not fire — fall back to the caller's legacy
        # threshold-based reject/fallback path. We return None so the caller
        # can decide (e.g. single-candidate fallback at B1).
        return None

    # ------------------------------------------------------------------ #
    # Modality runner (Phase 2 T2.1)
    # ------------------------------------------------------------------ #

    def check_modalities(
        self,
        query_crop: Optional[np.ndarray],
        candidate_crop: Optional[np.ndarray],
        candidate: Optional[Any] = None,
        *,
        pending_plate: Optional[str] = None,
        score_reid: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run color / type / OCR predictions and return a modality summary.

        Args:
            query_crop: BGR crop of the live (B1 / global / reattach) frame.
            candidate_crop: BGR crop of the gallery / candidate. Used to
                cross-check colour / type predictions against the live frame.
            candidate: optional dataclass (ParkEntryCandidate, VehicleSession)
                whose ``color_class`` / ``type_class`` / ``ocr_plate`` fields
                are reused when present so we do not re-run inference for a
                gallery image we already classified.
            pending_plate: ANPR plate (when available) the OCR reading is
                compared against. OCR is skipped when this is None.
            score_reid: current ReID cosine score for this pair. The OCR
                plugin is only invoked when this score sits inside the
                ``ocr_marginal_*`` band — outside the band, ReID alone is
                already decisive and the OCR cost is wasted.

        Returns:
            ``{
                color_match: bool, color_conf: float,
                type_match: bool, type_conf: float,
                ocr_match: bool, ocr_conf: float, ocr_text: str,
                ocr_contradiction: bool,
            }``

            All booleans default to False (no signal) so the ensemble rule
            treats missing modalities as silence rather than disagreement.
        """
        cfg = self._config
        result: Dict[str, Any] = {
            "color_match": False,
            "color_conf": 0.0,
            "color_query_label": "",
            "color_candidate_label": "",
            "type_match": False,
            "type_conf": 0.0,
            "type_query_label": "",
            "type_candidate_label": "",
            "ocr_match": False,
            "ocr_conf": 0.0,
            "ocr_text": "",
            "ocr_contradiction": False,
        }

        # --- Color ------------------------------------------------------ #
        try:
            color_query = self._predict_or_noop(self._color_classifier, query_crop)
            color_candidate = self._predict_or_noop(
                self._color_classifier,
                candidate_crop,
                fallback_pair=self._extract_label_conf(
                    candidate, "color_class", "color_class_conf"
                ),
            )
            result["color_query_label"] = color_query[0]
            result["color_candidate_label"] = color_candidate[0]
            result["color_conf"] = float(min(color_query[1], color_candidate[1]))
            if hasattr(self._color_classifier, "colors_match"):
                try:
                    result["color_match"] = bool(
                        self._color_classifier.colors_match(
                            color_query, color_candidate
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("[MatchDecision] color comparison failed: %r", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[MatchDecision] color modality failed: %r", exc)

        # --- Type ------------------------------------------------------- #
        try:
            type_query = self._predict_or_noop(self._type_classifier, query_crop)
            type_candidate = self._predict_or_noop(
                self._type_classifier,
                candidate_crop,
                fallback_pair=self._extract_label_conf(
                    candidate, "type_class", "type_class_conf"
                ),
            )
            result["type_query_label"] = type_query[0]
            result["type_candidate_label"] = type_candidate[0]
            result["type_conf"] = float(min(type_query[1], type_candidate[1]))
            if hasattr(self._type_classifier, "types_match"):
                try:
                    result["type_match"] = bool(
                        self._type_classifier.types_match(
                            type_query, type_candidate
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("[MatchDecision] type comparison failed: %r", exc)
            else:
                # Default ABC equality (NoopTypeClassifier)
                a_label = type_query[0] or ""
                b_label = type_candidate[0] or ""
                result["type_match"] = bool(
                    a_label
                    and b_label
                    and a_label != "unknown"
                    and a_label == b_label
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[MatchDecision] type modality failed: %r", exc)

        # --- OCR -------------------------------------------------------- #
        # Only run OCR when we have a pending plate to compare against AND
        # the ReID score sits inside the marginal band. Outside the band,
        # ReID is already decisive and the OCR latency cost is wasted.
        ocr_in_band = True
        if score_reid is not None:
            ocr_in_band = (
                cfg.ocr_marginal_low <= float(score_reid) <= cfg.ocr_marginal_high
            )

        if pending_plate and ocr_in_band:
            try:
                ocr_text, ocr_conf = ("", 0.0)
                if query_crop is not None and hasattr(self._plate_ocr, "read"):
                    ocr_text, ocr_conf = self._plate_ocr.read(query_crop)
                ocr_text = (ocr_text or "").strip()
                result["ocr_text"] = ocr_text
                result["ocr_conf"] = float(ocr_conf or 0.0)
                if ocr_text:
                    matches = self._plates_match(ocr_text, pending_plate)
                    result["ocr_match"] = bool(matches)
                    # Contradiction: high-conf reading with >=4 chars that
                    # does NOT match the pending plate. Treat as a hard
                    # reject signal in _apply_ensemble().
                    if (
                        not matches
                        and len(ocr_text) >= 4
                        and float(ocr_conf or 0.0) >= cfg.ocr_marginal_high
                    ):
                        result["ocr_contradiction"] = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("[MatchDecision] OCR modality failed: %r", exc)

        return result

    @staticmethod
    def _plates_match(ocr_text: str, expected_plate: str) -> bool:
        """Levenshtein <= 1 plate-match shim that does not require importing
        the WS-D ocr module at decide-time."""
        try:
            from src.ocr.plate_ocr import plates_match  # type: ignore
        except Exception:
            # Plate ocr module missing — fall back to a simple uppercase
            # comparison so the ensemble still works without paddleocr.
            return ocr_text.upper() == expected_plate.upper()
        return bool(plates_match(ocr_text, expected_plate))

    def _predict_or_noop(
        self,
        classifier,
        crop: Optional[np.ndarray],
        *,
        fallback_pair: Optional[Tuple[str, float]] = None,
    ) -> Tuple[str, float]:
        """Wrap a classifier predict() in defensive error handling.

        When the crop is missing or predict() raises, returns ``fallback_pair``
        if supplied (e.g. a precomputed label cached on the candidate), else
        ``("unknown", 0.0)`` so the ensemble treats this as no-signal.
        """
        if crop is None:
            if fallback_pair and fallback_pair[0]:
                return fallback_pair
            return ("unknown", 0.0)
        if not hasattr(classifier, "predict"):
            return fallback_pair or ("unknown", 0.0)
        try:
            label, conf = classifier.predict(crop)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[MatchDecision] classifier predict raised: %r", exc
            )
            return fallback_pair or ("unknown", 0.0)
        # An "unknown" prediction at zero confidence is the Noop sentinel —
        # prefer a fallback pair if one is available (e.g. session.color_class
        # was already filled in by the cascade earlier).
        if (
            (not label or label == "unknown")
            and fallback_pair
            and fallback_pair[0]
        ):
            return fallback_pair
        return (str(label or "unknown"), float(conf or 0.0))

    @staticmethod
    def _extract_label_conf(
        obj: Any,
        label_attr: str,
        conf_attr: str,
    ) -> Optional[Tuple[str, float]]:
        """Pull a (label, confidence) tuple off ``obj`` when present.

        Used to reuse a cached color_class / type_class from a candidate or
        session dataclass so we do not re-run inference for a gallery image
        we already classified.
        """
        if obj is None:
            return None
        try:
            label = getattr(obj, label_attr, None)
            conf = getattr(obj, conf_attr, 0.0)
        except Exception:  # pragma: no cover - defensive
            return None
        if not label:
            return None
        return (str(label), float(conf or 0.0))

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
        # The HSV gate is config-driven and plugin-independent: even when the
        # real OpenVINO color classifier replaces NoopColorClassifier, the
        # legacy HSV signature on a candidate is still a useful hard-reject
        # signal. We therefore prefer the classifier's hsv_compatible() when
        # it exposes one (Noop path), and fall back to a direct call to the
        # module-level helper when the classifier doesn't (OpenVINO path).
        if effective_use_filter:
            if hasattr(cc, "hsv_compatible"):
                passes_hsv = cc.hsv_compatible(query_hsv, candidate_hsvs)
            else:
                passes_hsv = self._hsv_compatible_fallback(
                    query_hsv,
                    candidate_hsvs,
                )
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

    def _hsv_compatible_fallback(
        self,
        query_hsv: Optional[Tuple[float, float, float]],
        candidate_hsvs: Optional[list],
    ) -> bool:
        """Plugin-independent HSV compatibility check.

        Used by :meth:`color_check` when the active color classifier does not
        expose its own ``hsv_compatible`` (e.g. the OpenVINOColorClassifier
        path). Mirrors the predicate in ``src/reid_matcher/reid_attributes.py``
        but with thresholds drawn from ``MatchingConfig``.
        """
        if query_hsv is None:
            return True
        if not candidate_hsvs:
            return True
        try:
            from src.reid_matcher import color_compatible
        except Exception:  # pragma: no cover - defensive (reid not installed)
            return True
        cfg = self._config
        for cand_hsv in candidate_hsvs:
            if cand_hsv is None:
                continue
            if color_compatible(
                query_hsv,
                cand_hsv,
                h_tol=cfg.hsv_h_tol,
                s_tol=cfg.hsv_s_tol,
                v_tol=cfg.hsv_v_tol,
            ):
                return True
        return False
