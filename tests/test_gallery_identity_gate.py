"""Regression: a foreign car's crop must not be merged into an identity's
gallery (the night gate contamination — "the other car's images ended up in my
gallery"). Exercises the open-set identity floor in _apply_session_gallery.
"""
import unittest

import numpy as np

from tests.fixtures.match_fixtures import make_test_registry, make_vehicle_session


class _DotMatcher:
    @staticmethod
    def compute_similarity(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class TestGalleryIdentityGate(unittest.TestCase):
    def _registry(self, floor):
        reg = make_test_registry()
        reg._reid_matcher = _DotMatcher()
        reg._matching_config.gallery_min_identity_similarity = floor
        return reg

    @staticmethod
    def _prepared(vecs):
        return ([None] * len(vecs), vecs, [f"p{i}" for i in range(len(vecs))])

    def _session_with(self, reg, vec):
        s = make_vehicle_session("CAR-A")
        reg._sessions[s.session_id] = s
        reg._apply_session_gallery(s, self._prepared([vec]), 0)
        return s

    def test_foreign_crop_rejected(self):
        reg = self._registry(0.35)
        vA = _norm(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        s = self._session_with(reg, vA)
        self.assertEqual(len(s.reference_feature_vectors), 1)
        # A foreign car (orthogonal → similarity 0 < floor) must be refused.
        vB = _norm(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        reg._apply_session_gallery(s, self._prepared([vB]), 0)
        self.assertEqual(len(s.reference_feature_vectors), 1)
        self.assertTrue(all(np.allclose(v, vA) for v in s.reference_feature_vectors))

    def test_same_car_view_admitted(self):
        reg = self._registry(0.35)
        vA = _norm(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        s = self._session_with(reg, vA)
        # Same car, a different view (similarity ~0.71 > floor) is admitted.
        vA2 = _norm(np.array([0.7, 0.7, 0.0], dtype=np.float32))
        reg._apply_session_gallery(s, self._prepared([vA2]), 0)
        self.assertEqual(len(s.reference_feature_vectors), 2)

    def test_gate_off_preserves_legacy(self):
        reg = self._registry(0.0)  # disabled
        vA = _norm(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        s = self._session_with(reg, vA)
        vB = _norm(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        reg._apply_session_gallery(s, self._prepared([vB]), 0)
        self.assertEqual(len(s.reference_feature_vectors), 2)


if __name__ == "__main__":
    unittest.main()
