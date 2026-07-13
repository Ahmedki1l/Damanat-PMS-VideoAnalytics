"""
A car must not compete with itself.

Observed live (2026-07-11, DJS-7842 driving gate -> B13 COO):

    16:11:09  [gallery] reloaded plate=DJS-7842 -> reload_7b10dd4a3b67
    16:11:45  cam=CAM-24 matched session sess_f3aed8571b66 (score=0.725)   <- ANONYMOUS
    16:11:48  cam=CAM-24 abstain: ambiguous (best=0.716 runner_up=0.699)
    16:12:11  cam=CAM-24 matched session sess_f3aed8571b66 (score=0.728)   <- ANONYMOUS

The car is plated at the gate, but the worker owning the slot cameras also builds its
own anonymous session for the same car from ITS viewpoint. The anonymous copy scores
HIGHER (same viewpoint) than the real plated session (gate views only -> cross-view),
so the two land inside global_match_margin and the matcher abstains. The car parks with
plate=(none) forever.

The fix collapses the nameless session into the named one and carries its
slot-viewpoint vectors across — which is also what closes the cross-view gap that no
threshold could.
"""

import unittest

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry


def _vec(*head) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    for i, x in enumerate(head):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


class TestSelfCompetitionMerge(unittest.TestCase):
    def setUp(self):
        self.reg = VehicleRegistry(matching_config=MatchingConfig())

    def _mk(self, sid, plate, vec, cam="CAM-24", tid=7):
        from src.vehicle_registry.vehicle_registry_models import VehicleSession

        now = self.reg._clock()
        s = VehicleSession(
            session_id=sid,
            plate=plate,
            feature_vector=vec,
            first_seen_at=now,
            last_seen_at=now,
            last_seen_camera=cam,
            status="confirmed",
        )
        s.observing_tracks[cam] = tid
        s.observing_scores[cam] = 0.7
        self.reg._sessions[sid] = s
        self.reg._track_session_map[(cam, tid)] = sid
        return s

    def test_anonymous_session_is_absorbed_into_the_plated_one(self):
        """The nameless copy dies; the plate survives and INHERITS its vectors."""
        plated = self._mk("reload_1", "DJS-7842", _vec(1.0), tid=1)   # gate views
        anon_vec = _vec(0.0, 1.0)                                     # slot viewpoint
        anon = self._mk("sess_1", None, anon_vec, tid=2)              # anonymous
        anon.reference_feature_vectors = [_vec(0.0, 0.0, 1.0)]

        ok = self.reg._absorb_anonymous_session("sess_1", "reload_1")

        self.assertTrue(ok)
        # The anonymous session is gone...
        self.assertNotIn("sess_1", self.reg._sessions)
        self.assertEqual(anon.status, "merged")
        # ...the plated one survives, keeps its plate...
        self.assertIn("reload_1", self.reg._sessions)
        self.assertEqual(plated.plate, "DJS-7842")
        # ...and has INHERITED the slot-viewpoint vectors it was missing. This is the
        # part that actually closes the cross-view gap.
        self.assertEqual(len(plated.reference_feature_vectors), 2)
        # Every track that pointed at the nameless session now points at the plate.
        self.assertEqual(self.reg._track_session_map[("CAM-24", 2)], "reload_1")
        self.assertEqual(plated.observing_tracks["CAM-24"], 2)

    def test_two_plated_sessions_are_NEVER_merged(self):
        """The guardrail. Merging two named cars would be a real identity swap —
        precisely the failure the abstain margin exists to prevent."""
        self._mk("reload_a", "DJS-7842", _vec(1.0), tid=1)
        self._mk("reload_b", "HGD-2926", _vec(0.0, 1.0), tid=2)

        self.assertFalse(self.reg._absorb_anonymous_session("reload_b", "reload_a"))
        self.assertIn("reload_b", self.reg._sessions)
        self.assertEqual(self.reg._sessions["reload_b"].plate, "HGD-2926")

    def test_never_collapses_a_plated_session_into_an_anonymous_one(self):
        """Direction matters: the identity must always be the survivor."""
        self._mk("sess_1", None, _vec(1.0), tid=1)
        self._mk("reload_1", "DJS-7842", _vec(0.0, 1.0), tid=2)

        # anon=reload_1 (plated) / plated=sess_1 (anonymous) — an inverted call.
        self.assertFalse(self.reg._absorb_anonymous_session("reload_1", "sess_1"))
        self.assertIn("reload_1", self.reg._sessions)
        self.assertEqual(self.reg._sessions["reload_1"].plate, "DJS-7842")

    def test_reference_vectors_respect_the_gallery_cap(self):
        plated = self._mk("reload_1", "DJS-7842", _vec(1.0), tid=1)
        cap = MatchingConfig().gallery_max_refs_per_car
        plated.reference_feature_vectors = [_vec(1.0) for _ in range(cap)]

        anon = self._mk("sess_1", None, _vec(0.0, 1.0), tid=2)
        anon.reference_feature_vectors = [_vec(0.0, 0.0, 1.0) for _ in range(5)]

        self.assertTrue(self.reg._absorb_anonymous_session("sess_1", "reload_1"))
        self.assertEqual(len(plated.reference_feature_vectors), cap)


if __name__ == "__main__":
    unittest.main()
