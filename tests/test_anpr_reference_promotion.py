"""ANPR image joins ReID matching — but only AFTER CAM-03 confirmation.

The wide ANPR gate image is held out of matching at the entry event
(gate_reference_only) so it can't false-lock onto a car already parked in the
garage. Its car-cropped embedding is stashed as ``pending_anpr_vector`` and
promoted to a matchable reference only when CAM-03 (B1_Entrence) confirms the
plate — at which point the session is anchored to this car, so the extra frontal
view enriches the appearance profile without the entry-time swap risk.
"""

import unittest

import numpy as np

from tests.fixtures.match_fixtures import make_test_registry, make_color_crop


class TestAnprReferencePromotion(unittest.TestCase):
    def test_anpr_ref_held_then_promoted_at_b1_confirmation(self):
        registry = make_test_registry()

        # Distinct crops so the stub ReID yields different vectors (the promoted
        # ANPR ref must survive the dedup-against-primary check).
        anpr_crop = make_color_crop(bgr=(30, 30, 200))   # reddish
        cam03_crop = make_color_crop(bgr=(200, 30, 30))  # bluish
        anpr_vec = registry._reid_matcher.extract_feature(anpr_crop)

        # --- Gate entry: ANPR direct session -----------------------------
        sid = registry.confirm_anpr_session_directly(
            plate="CAR-X",
            image=anpr_crop,
            event_id="ev1",
            candidate_id="cand1",
            gate_snapshot_paths=[],
        )
        session = registry._sessions[sid]

        # Held out of matching (gate_reference_only excludes the whole session
        # from match_global_session) with the embedding stashed for promotion.
        self.assertTrue(session.gate_reference_only)
        self.assertIsNotNone(session.pending_anpr_vector)

        # --- CAM-03 B1 confirmation --------------------------------------
        registry.confirm_b1_entrance_by_plate("CAR-X", cam03_crop)
        cam03_vec = registry._reid_matcher.extract_feature(cam03_crop)

        # Now matchable: gate flag cleared, pending drained. CAM-03 confirmation
        # overwrites reference_feature_vectors with its own primary, and the
        # promotion re-adds the ANPR crop — so BOTH views are matchable.
        self.assertFalse(session.gate_reference_only)
        self.assertIsNone(session.pending_anpr_vector)
        self.assertTrue(
            any(np.array_equal(anpr_vec, r) for r in session.reference_feature_vectors),
            "ANPR vector must be a matchable reference after CAM-03 confirmation",
        )
        self.assertTrue(
            any(np.array_equal(cam03_vec, r) for r in session.reference_feature_vectors),
            "CAM-03 primary must remain a reference",
        )

    def test_duplicate_anpr_ref_not_appended_when_same_as_primary(self):
        """If the ANPR and CAM-03 views embed near-identically, the dedup guard
        keeps the reference list from accumulating a redundant copy."""
        registry = make_test_registry()
        same = make_color_crop(bgr=(40, 120, 40))

        sid = registry.confirm_anpr_session_directly(
            plate="CAR-Y",
            image=same,
            event_id="ev2",
            candidate_id="cand2",
            gate_snapshot_paths=[],
        )
        registry.confirm_b1_entrance_by_plate("CAR-Y", same)

        session = registry._sessions[sid]
        self.assertIsNone(session.pending_anpr_vector)
        # Only the CAM-03 primary remains; the near-identical ANPR ref is deduped
        # rather than accumulating a redundant copy.
        self.assertEqual(len(session.reference_feature_vectors), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
