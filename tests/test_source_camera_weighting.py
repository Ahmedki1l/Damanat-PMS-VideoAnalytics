"""Regression: ReID gallery references are weighted by their SOURCE camera.

The parking has three GROUND-TRUTH cameras (ANPR front, CAM-23 top, CAM-03
front+back). A reference captured by one of them scores at full weight; a
reference from any other camera is multiplied by ``secondary_camera_weight`` so
an oblique side-view can only win a match when it is substantially stronger than
every ground-truth view. Covers:

  * the per-camera weight (_reference_weight)
  * the weighted best score (_best_weighted_score): a query identical to a
    secondary ref scores below one identical to a ground-truth ref
  * source-camera tagging stays index-aligned with the vectors through
    _apply_session_gallery and accumulate_reference (incl. the diversity cap)
  * the conservative default: an untagged ref is treated as secondary
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

    @staticmethod
    def extract_feature(image):
        return None


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class TestSourceCameraWeighting(unittest.TestCase):
    def _registry(self, weight=0.6):
        reg = make_test_registry()
        reg._reid_matcher = _DotMatcher()
        reg._matching_config.ground_truth_cameras = ("ANPR", "CAM-23", "CAM-03")
        reg._matching_config.secondary_camera_weight = weight
        return reg

    def test_reference_weight_by_camera(self):
        reg = self._registry(0.6)
        self.assertEqual(reg._reference_weight("CAM-03"), 1.0)
        self.assertEqual(reg._reference_weight("CAM-23"), 1.0)
        self.assertEqual(reg._reference_weight("ANPR"), 1.0)
        self.assertEqual(reg._reference_weight("CAM-09"), 0.6)
        self.assertEqual(reg._reference_weight(""), 0.6)  # unknown -> secondary

    def test_ground_truth_beats_equal_secondary_match(self):
        reg = self._registry(0.6)
        s = make_vehicle_session("CAR-A", feature_vector=_norm([1.0, 0.0, 0.0]))
        # A ground-truth (CAM-03) ref and a secondary (CAM-09) ref, orthogonal.
        gt = _norm([0.0, 1.0, 0.0])
        sec = _norm([0.0, 0.0, 1.0])
        s.reference_feature_vectors = [gt, sec]
        s.reference_source_cameras = ["CAM-03", "CAM-09"]

        # Query identical to the ground-truth ref -> full weight.
        self.assertAlmostEqual(reg._best_weighted_score(gt, s), 1.0, places=5)
        # Query identical to the secondary ref -> down-weighted to 0.6.
        self.assertAlmostEqual(reg._best_weighted_score(sec, s), 0.6, places=5)

    def test_primary_is_always_full_weight(self):
        reg = self._registry(0.6)
        primary = _norm([1.0, 0.0, 0.0])
        s = make_vehicle_session("CAR-B", feature_vector=primary)
        s.reference_feature_vectors = []
        s.reference_source_cameras = []
        # A query matching the primary is full weight regardless of camera set.
        self.assertAlmostEqual(reg._best_weighted_score(primary, s), 1.0, places=5)

    def test_apply_session_gallery_tags_source_camera(self):
        reg = self._registry(0.6)
        s = make_vehicle_session("CAR-C", feature_vector=None)
        s.feature_vector = None
        s.reference_feature_vectors = []
        s.reference_source_cameras = []
        reg._sessions[s.session_id] = s
        vec = _norm([1.0, 0.0, 0.0])
        prepared = ([None], [vec], ["p0"])
        reg._apply_session_gallery(s, prepared, 0, source_camera="CAM-23")
        self.assertEqual(len(s.reference_feature_vectors), 1)
        self.assertEqual(
            len(s.reference_source_cameras), len(s.reference_feature_vectors)
        )
        self.assertEqual(s.reference_source_cameras[0], "CAM-23")

    def test_untagged_reference_defaults_to_secondary(self):
        reg = self._registry(0.6)
        s = make_vehicle_session("CAR-D", feature_vector=_norm([1.0, 0.0, 0.0]))
        ref = _norm([0.0, 1.0, 0.0])
        s.reference_feature_vectors = [ref]
        s.reference_source_cameras = []  # no tag at all
        # Missing tag -> secondary weight (conservative), so a query identical to
        # the untagged ref scores 0.6, never full weight.
        self.assertAlmostEqual(reg._best_weighted_score(ref, s), 0.6, places=5)


if __name__ == "__main__":
    unittest.main()
