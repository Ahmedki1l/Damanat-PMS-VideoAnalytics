"""Regression: the B1 single-candidate fallback must not confirm an
appearance-blind bind. This is the mechanism by which the NEXT car adopted a
PREVIOUS car's stale pending plate in the night gate identity-swap incident.
"""
import unittest

from src.config import MatchingConfig
from src.matching import MatchDecision


class TestSingleCandidateFloor(unittest.TestCase):
    @staticmethod
    def _decision(floor):
        cfg = MatchingConfig()  # b1_zone default 0.55
        cfg.single_candidate_min_reid = floor
        return MatchDecision(cfg)

    def test_legacy_zero_floor_preserves_old_behaviour(self):
        # floor=0.0 (dataclass default) => lone candidate confirmed at any score.
        dec = self._decision(0.0)
        v = dec.decide_b1(0.0, is_anpr_candidate=False, candidate_count=1)
        self.assertEqual(v.verdict, "confirm")
        self.assertEqual(v.reason, "single_candidate_fallback")

    def test_floor_rejects_appearance_blind_lone_candidate(self):
        # A different car (low ReID vs the stale candidate) below the floor must
        # NOT be confirmed — no more inheriting the previous car's plate.
        dec = self._decision(0.35)
        v = dec.decide_b1(0.10, is_anpr_candidate=False, candidate_count=1)
        self.assertEqual(v.verdict, "reject")
        self.assertNotEqual(v.reason, "single_candidate_fallback")

    def test_floor_admits_marginal_same_car_lone_candidate(self):
        # A genuine lone car whose cross-view ReID is marginal (below b1_zone but
        # above the floor) still confirms — the fallback's legitimate purpose.
        dec = self._decision(0.35)
        v = dec.decide_b1(0.50, is_anpr_candidate=False, candidate_count=1)
        self.assertEqual(v.verdict, "confirm")
        self.assertEqual(v.reason, "single_candidate_fallback")

    def test_floor_does_not_affect_multi_candidate_reject(self):
        # With >1 candidate and a sub-threshold score, still a plain reject.
        dec = self._decision(0.35)
        v = dec.decide_b1(0.50, is_anpr_candidate=False, candidate_count=3)
        self.assertEqual(v.verdict, "reject")


if __name__ == "__main__":
    unittest.main()
