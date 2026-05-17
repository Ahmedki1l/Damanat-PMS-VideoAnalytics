"""
tests/test_matching_e2e_with_plugins.py — Phase 2 / T2.4 plugin cascade smoke.

Drives the full Phase-2 cascade end-to-end on synthetic crops to verify that
:class:`MatchDecision.check_modalities` and the ensemble rule in
:meth:`MatchDecision.decide_b1` / ``decide_global`` / ``decide_reattach``
integrate with the Wave-1 plugins (color, type, OCR) without crashing.

The OCR plugin is heavy and not required to be installed for CI, so the
plate-OCR scenario uses a tiny stub plugin that mimics the real ABC. The
color / type classifiers are loaded from the real IRs when present; when
absent the Noop fallbacks are used and the asserts still hold (we only
assert "no crash" for those modalities).

This complements ``tests/test_matching_e2e.py`` (which exercises the
non-plugin failure modes) by covering the cascade *with* the plugins
plumbed in.
"""

from __future__ import annotations

import unittest
from typing import Tuple

import numpy as np

from src.config import MatchingConfig
from src.matching import MatchDecision, MatchVoter
from src.matching.plugins import ColorClassifier, PlateOCR, TypeClassifier


# --------------------------------------------------------------------------- #
# Stub plugins — drive deterministic decisions without loading heavy weights.
# --------------------------------------------------------------------------- #


class _StubColorClassifier(ColorClassifier):
    """Returns a fixed label / confidence for every crop. Used to drive
    the ensemble rule into known states without touching the OpenVINO IR.
    """

    def __init__(self, label: str = "red", confidence: float = 0.95):
        self._label = label
        self._conf = confidence

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return (self._label, self._conf)


class _StubTypeClassifier(TypeClassifier):
    def __init__(self, label: str = "sedan", confidence: float = 0.92):
        self._label = label
        self._conf = confidence

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return (self._label, self._conf)

    def types_match(self, a, b) -> bool:
        # Mirror the real types_match shape — accept tuples or strings.
        def _label(p):
            if isinstance(p, tuple):
                return str(p[0])
            return str(p)

        return _label(a) == _label(b)


class _StubPlateOCR(PlateOCR):
    """OCR stub that returns a configurable plate string."""

    def __init__(self, text: str = "ABC123", confidence: float = 0.9):
        self._text = text
        self._conf = confidence

    def read(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        return (self._text, self._conf)


def _crop(bgr=(64, 200, 32), size=(48, 48)) -> np.ndarray:
    h, w = size
    return np.full((h, w, 3), bgr, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Cascade scenarios
# --------------------------------------------------------------------------- #


class TestEnsembleConfirmsWithAllModalities(unittest.TestCase):
    """When color + type + OCR all agree, the ensemble confirms even at
    marginal ReID scores."""

    def test_confirm_with_three_agreements_below_reid_threshold(self) -> None:
        decision = MatchDecision(
            MatchingConfig(),
            color_classifier=_StubColorClassifier("red", 0.95),
            type_classifier=_StubTypeClassifier("sedan", 0.92),
            plate_ocr=_StubPlateOCR("ABC123", 0.95),
        )
        modalities = decision.check_modalities(
            query_crop=_crop(),
            candidate_crop=_crop(),
            candidate=None,
            pending_plate="ABC123",
            score_reid=0.42,  # marginal: below b1_zone=0.55, above ocr_marginal_low=0.40
        )
        self.assertTrue(modalities["color_match"])
        self.assertTrue(modalities["type_match"])
        self.assertTrue(modalities["ocr_match"])

        verdict = decision.decide_b1(
            0.42,
            is_anpr_candidate=False,
            candidate_count=3,
            modalities=modalities,
        )
        self.assertEqual(verdict.verdict, "confirm")
        # OCR-match short-circuit gets credit before K-of-N counting.
        self.assertIn(verdict.reason, ("ensemble_ocr_match", "ensemble_agree"))


class TestOCRContradictionRejects(unittest.TestCase):
    """A high-confidence OCR reading that disagrees with the pending plate
    is a hard reject even when ReID is decent."""

    def test_ocr_contradiction_rejects(self) -> None:
        decision = MatchDecision(
            MatchingConfig(),
            plate_ocr=_StubPlateOCR("ZZZ999", 0.97),
        )
        modalities = decision.check_modalities(
            query_crop=_crop(),
            candidate_crop=_crop(),
            candidate=None,
            pending_plate="ABC123",
            score_reid=0.52,  # in the ocr marginal band [0.40, 0.55]
        )
        self.assertTrue(modalities["ocr_contradiction"])
        verdict = decision.decide_b1(
            0.52,
            is_anpr_candidate=False,
            candidate_count=3,
            modalities=modalities,
        )
        self.assertEqual(verdict.verdict, "reject")
        self.assertEqual(verdict.reason, "ensemble_ocr_contradiction")


class TestReIDSoloConfirmAboveThreshold(unittest.TestCase):
    """Score >= reid_solo_confirm (0.70) confirms regardless of modalities."""

    def test_reid_solo_confirms_above_0_70(self) -> None:
        decision = MatchDecision(
            MatchingConfig(),
            color_classifier=_StubColorClassifier("red", 0.95),
            type_classifier=_StubTypeClassifier("sedan", 0.92),
        )
        # All modalities silent; high ReID still confirms.
        modalities = decision.check_modalities(
            query_crop=_crop(),
            candidate_crop=_crop(),
            candidate=None,
            pending_plate=None,
            score_reid=0.75,
        )
        verdict = decision.decide_b1(
            0.75,
            is_anpr_candidate=False,
            candidate_count=3,
            modalities=modalities,
        )
        self.assertEqual(verdict.verdict, "confirm")
        self.assertEqual(verdict.reason, "ensemble_reid_solo")


class TestNoModalityFallback(unittest.TestCase):
    """When no modalities dict is passed, the decision falls back to the
    legacy threshold-only behaviour. This is the path most existing call
    sites take when they only have a score."""

    def test_no_modalities_uses_legacy_threshold(self) -> None:
        decision = MatchDecision(MatchingConfig())
        # Score below threshold → reject (no modalities supplied).
        verdict = decision.decide_b1(
            0.30, is_anpr_candidate=False, candidate_count=3
        )
        self.assertEqual(verdict.verdict, "reject")

        # Score above threshold → confirm via b1_zone.
        verdict = decision.decide_b1(
            0.60, is_anpr_candidate=False, candidate_count=3
        )
        self.assertEqual(verdict.verdict, "confirm")
        self.assertEqual(verdict.reason, "b1_zone")


class TestCascadeWithVoter(unittest.TestCase):
    """Smoke test: the cascade + voter pipeline does not crash when invoked
    repeatedly on the same key. The voter is the integration seam at the
    slot-link callsite (vehicle_registry_identity.py:try_link_to_slot)."""

    def test_voter_eventually_commits_unanimous_confirms(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=3,
            voting_min_agree=2,
        )
        decision = MatchDecision(
            cfg,
            color_classifier=_StubColorClassifier("red", 0.95),
            type_classifier=_StubTypeClassifier("sedan", 0.92),
        )
        voter = MatchVoter(cfg)
        confirms = 0
        for i in range(4):
            modalities = decision.check_modalities(
                query_crop=_crop(),
                candidate_crop=_crop(),
                candidate=None,
                pending_plate=None,
                score_reid=0.80,
            )
            verdict = decision.decide_global(
                0.80,
                has_plate=True,
                cross_camera=False,
                modalities=modalities,
            )
            # Inject a plate into the verdict scores so the voter has a
            # vote key — mirrors how try_link_to_slot wraps the commit.
            verdict = type(verdict)(
                verdict=verdict.verdict,
                reason=verdict.reason,
                scores={**verdict.scores, "plate": "ABC-123"},
            )
            committed = voter.submit("CAM-01", 1, verdict)
            if committed is not None:
                confirms += 1
        # After 4 unanimous confirms in a 3-frame window with 2-of-3
        # agreement, at least 3 of the 4 submits should commit (the first
        # one defers until the window has reached min_agree).
        self.assertGreaterEqual(confirms, 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
