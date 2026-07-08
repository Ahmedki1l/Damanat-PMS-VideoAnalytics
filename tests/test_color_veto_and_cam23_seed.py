"""Regression for the white-car gallery-poisoning fixes (#1–#4).

A dark car (DJS-7842) had its gallery poisoned by a WHITE car that drove past it
on a floor camera: the floor-camera match path ran the colour check with no
crops (silence, not veto) and `accumulate_reference` had no appearance gate, so
the wrong car's crops were written as references and then reinforced more wrong
matches. These tests lock in:

  * Fix 1 — `accumulate_reference` rejects a colour-incompatible crop against the
    session's ground-truth colour, and accepts a compatible one.
  * Fix 2 — `match_global_session` vetoes a candidate whose ground-truth colour
    is incompatible with the live query crop (a white query cannot bind a dark
    session), while an un-cropped query still matches.
  * Fix 3 — a new reference must resemble the GROUND-TRUTH refs (not the possibly
    poisoned secondary refs) to be admitted; `_best_ground_truth_similarity`
    excludes secondary-camera refs.
  * Fix 4 — the CAM-23 top-view seed is decoupled from the one-shot bind: a
    candidate's plate resolves after the bind frame, the seed only marks a
    candidate done on SUCCESS (so a poor first crop doesn't block the retry), and
    `latest_park_entry_candidate_for_plate` finds the candidate for the CAM-03
    fallback.
"""
import unittest

import numpy as np

from src.config import MatchingConfig
from src.core.engine.engine_tracking import ParkingEngineTrackingMixin
from src.reid_matcher import body_colour_compatible, dominant_color_hsv
from tests.fixtures.match_fixtures import (
    make_color_crop,
    make_test_registry,
    make_vehicle_session,
)

DARK = (30, 30, 30)      # near-black BGR (our car)
WHITE = (235, 235, 235)  # the passing car


def _gallery_config():
    cfg = MatchingConfig()
    cfg.gallery_persist_enabled = True
    cfg.reid_openvino_model_dir = ""  # model tag -> "…:default"
    # Open the quality/throttle gates so the colour/similarity gates are what we
    # are actually exercising (solid crops have zero sharpness).
    cfg.gallery_min_view_quality = 0.0
    cfg.gallery_min_sharpness = 0.0
    cfg.gallery_accumulate_interval_s = 0.0
    cfg.gallery_min_crop_area = 0.0  # isolate colour/similarity gates from size
    return cfg


class TestBodyColourCompatible(unittest.TestCase):
    """The saturation/value-aware comparator: hue is ignored for achromatic
    (grey/black/white) crops, tolerated for lighting shifts on coloured cars,
    and only a gross mismatch is rejected."""

    def test_white_vs_black_incompatible(self):
        self.assertFalse(body_colour_compatible((0, 5, 235), (0, 5, 35)))

    def test_grey_vs_grey_compatible_despite_noisy_hue(self):
        # Two greys whose (meaningless) hue differs wildly but brightness agrees.
        self.assertTrue(body_colour_compatible((10, 20, 120), (170, 25, 130)))

    def test_dark_vs_dark_compatible(self):
        # Low value -> achromatic; hue noise must not split the same dark car.
        self.assertTrue(body_colour_compatible((76, 41, 64), (20, 38, 57)))

    def test_lighting_hue_shift_compatible(self):
        # Same coloured car, moderate hue shift between gate and garage light.
        self.assertTrue(body_colour_compatible((57, 60, 90), (88, 55, 98)))

    def test_teal_vs_orange_incompatible(self):
        # HSR-8327 real case: teal-ish ground truth vs a gold/orange other car.
        self.assertFalse(body_colour_compatible((86, 57, 82), (33, 85, 95)))

    def test_missing_colour_fails_open(self):
        self.assertTrue(body_colour_compatible(None, (0, 0, 0)))


class TestAccumulateColourVeto(unittest.TestCase):
    def _registry_with_dark_session(self, cfg=None):
        reg = make_test_registry(matching_config=cfg or _gallery_config())
        dark = make_color_crop(DARK)
        s = make_vehicle_session(
            "DARK-1",
            feature_vector=reg.reid_matcher.extract_feature(dark),
        )
        s.ground_truth_hsv = dominant_color_hsv(dark)
        reg._sessions[s.session_id] = s
        return reg, s

    def test_white_crop_is_rejected_against_dark_ground_truth(self):
        reg, s = self._registry_with_dark_session()
        before = len(s.reference_feature_vectors)
        added = reg.accumulate_reference(s, make_color_crop(WHITE), "CAM-20", 1.0)
        self.assertFalse(added, "a white crop must not join a dark car's gallery")
        self.assertEqual(len(s.reference_feature_vectors), before)

    def test_same_colour_crop_is_accepted(self):
        reg, s = self._registry_with_dark_session()
        # A slightly different dark shade — colour-compatible. Pin its ReID sim
        # to a mid value so it clears the ground-truth floor yet isn't a dedup
        # near-duplicate (solid-grey stub vectors are otherwise parallel).
        crop = make_color_crop((40, 40, 40))
        reg.reid_matcher.pin_similarity(
            reg.reid_matcher.extract_feature(crop), s.feature_vector, 0.8
        )
        added = reg.accumulate_reference(s, crop, "CAM-20", 1.0)
        self.assertTrue(added, "a same-colour crop is a legitimate reference")
        self.assertEqual(s.reference_source_cameras[-1], "CAM-20")

    def test_no_veto_when_ground_truth_colour_unset(self):
        # Fail-open: legacy sessions with no anchored colour are not colour-gated.
        reg, s = self._registry_with_dark_session()
        s.ground_truth_hsv = None
        crop = make_color_crop(WHITE)
        reg.reid_matcher.pin_similarity(
            reg.reid_matcher.extract_feature(crop), s.feature_vector, 0.8
        )
        added = reg.accumulate_reference(s, crop, "CAM-20", 1.0)
        self.assertTrue(added, "no colour anchor -> colour veto is inert")


class TestViewQualitySizeAware(unittest.TestCase):
    """`_bbox_view_quality` folds apparent car SIZE in, so a far camera whose
    tiny car is fully framed no longer scores as a clean full view."""

    class _Harness(ParkingEngineTrackingMixin):
        pass

    class _Det:
        def __init__(self, bbox):
            self.bbox = bbox

    def setUp(self):
        self.h = self._Harness()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_near_large_car_scores_high(self):
        # ~300px tall, comfortably inside the frame.
        q = self.h._bbox_view_quality(self.frame, self._Det((500, 200, 760, 520)))
        self.assertGreaterEqual(q, 0.9)

    def test_far_tiny_car_scores_low(self):
        # 80x53 — the CAM-22 HNR-8001 case: fully framed but distant.
        q = self.h._bbox_view_quality(self.frame, self._Det((600, 300, 680, 353)))
        self.assertLess(q, 0.5)

    def test_far_tiny_car_loses_to_near_car(self):
        near = self.h._bbox_view_quality(self.frame, self._Det((500, 200, 760, 520)))
        far = self.h._bbox_view_quality(self.frame, self._Det((600, 300, 680, 353)))
        self.assertGreater(near, far)


class TestReferenceMinSizeGate(unittest.TestCase):
    def _registry_with_dark_session(self, min_area):
        cfg = _gallery_config()
        cfg.gallery_min_crop_area = min_area
        reg = make_test_registry(matching_config=cfg)
        dark = make_color_crop(DARK, size=(150, 200))
        s = make_vehicle_session(
            "SIZE-1", feature_vector=reg.reid_matcher.extract_feature(dark)
        )
        s.ground_truth_hsv = dominant_color_hsv(dark)
        reg._sessions[s.session_id] = s
        return reg, s

    def test_tiny_crop_is_rejected(self):
        reg, s = self._registry_with_dark_session(12000.0)
        # 80x53 ~= 4.2k px, well under the 12k floor — the CAM-22 case.
        tiny = make_color_crop(DARK, size=(53, 80))
        self.assertFalse(reg.accumulate_reference(s, tiny, "CAM-22", 1.0))

    def test_large_crop_passes_size_gate(self):
        reg, s = self._registry_with_dark_session(12000.0)
        big = make_color_crop((40, 40, 40), size=(150, 200))  # 30k px
        reg.reid_matcher.pin_similarity(
            reg.reid_matcher.extract_feature(big), s.feature_vector, 0.8
        )
        self.assertTrue(reg.accumulate_reference(s, big, "CAM-22", 1.0))


class TestGroundTruthSimilarityGate(unittest.TestCase):
    def test_best_ground_truth_similarity_excludes_secondary(self):
        reg = make_test_registry()
        primary = np.array([1.0, 0.0], dtype=np.float32)
        gt_ref = np.array([0.0, 1.0], dtype=np.float32)
        sec_ref = np.array([0.7, 0.7], dtype=np.float32)
        s = make_vehicle_session("GT-1", feature_vector=primary)
        s.reference_feature_vectors = [gt_ref, sec_ref]
        s.reference_source_cameras = ["CAM-03", "CAM-20"]  # gt, secondary
        # A query identical to the SECONDARY ref must not count — only the
        # primary + CAM-03 ground-truth ref are anchors. Were the secondary ref
        # included, self-similarity would make this 1.0.
        sim = reg._best_ground_truth_similarity(sec_ref, s)
        expected = max(
            reg.reid_matcher.compute_similarity(sec_ref, primary),
            reg.reid_matcher.compute_similarity(sec_ref, gt_ref),
        )
        self.assertAlmostEqual(sim, expected, places=5)
        self.assertLess(sim, 0.99, "secondary ref must be excluded from anchors")

    def test_low_ground_truth_similarity_crop_is_rejected(self):
        cfg = _gallery_config()
        cfg.gallery_accumulate_min_gt_similarity = 0.5
        reg = make_test_registry(matching_config=cfg)
        dark = make_color_crop(DARK)
        s = make_vehicle_session(
            "GT-2", feature_vector=reg.reid_matcher.extract_feature(dark)
        )
        s.ground_truth_hsv = dominant_color_hsv(dark)
        reg._sessions[s.session_id] = s
        # A colour-compatible crop whose ReID vector is pinned far from the
        # ground-truth anchor is rejected by the feedback-loop guard.
        crop = make_color_crop((35, 35, 35))
        crop_vec = reg.reid_matcher.extract_feature(crop)
        reg.reid_matcher.pin_similarity(crop_vec, s.feature_vector, 0.10)
        self.assertFalse(reg.accumulate_reference(s, crop, "CAM-20", 1.0))


class TestMatchColourVeto(unittest.TestCase):
    def _confirmed_dark_session(self, reg):
        dark = make_color_crop(DARK)
        s = make_vehicle_session(
            "DARK-M", feature_vector=reg.reid_matcher.extract_feature(dark)
        )
        s.ground_truth_hsv = dominant_color_hsv(dark)
        s.gate_reference_only = False
        reg._sessions[s.session_id] = s
        return s

    def test_white_query_crop_vetoes_dark_session(self):
        reg = make_test_registry()
        s = self._confirmed_dark_session(reg)
        query = np.array([0.5, 0.5], dtype=np.float32)
        reg.reid_matcher.pin_similarity(query, s.feature_vector, 0.9)
        # Baseline: no crop -> the strong ReID score binds the session.
        self.assertEqual(
            reg.match_global_session(query, camera_id="CAM-20", track_id=1),
            s.session_id,
        )
        # With a WHITE live crop, the colour veto drops the dark candidate.
        self.assertIsNone(
            reg.match_global_session(
                query,
                camera_id="CAM-20",
                track_id=2,
                query_crop=make_color_crop(WHITE),
            )
        )


class TestCam23SeedDecoupling(unittest.TestCase):
    def _registry(self):
        return make_test_registry(matching_config=_gallery_config())

    def test_plate_resolves_after_bind_frame(self):
        reg = self._registry()
        reg.register_anpr_event("SEED-4", "entry", timestamp=reg._clock())
        cand = reg.open_park_entry_candidate("CAM-23", 5)
        reg.update_park_entry_candidate_snapshot(
            cand.candidate_id, make_color_crop(DARK), quality_score=5.0
        )
        self.assertEqual(
            reg.bind_next_pending_anpr_to_candidate(cand.candidate_id), "SEED-4"
        )
        # The one-shot bind returns None on later frames, but the plate is still
        # resolvable from the candidate for the cross-frame seed retry.
        self.assertIsNone(reg.bind_next_pending_anpr_to_candidate(cand.candidate_id))
        self.assertEqual(
            reg.plate_for_park_entry_candidate(cand.candidate_id), "SEED-4"
        )
        self.assertEqual(
            reg.latest_park_entry_candidate_for_plate("SEED-4"), cand.candidate_id
        )

    def test_poor_first_crop_does_not_block_retry(self):
        reg = self._registry()
        reg.register_anpr_event("SEED-5", "entry", timestamp=reg._clock())
        cand = reg.open_park_entry_candidate("CAM-23", 6)
        reg.bind_next_pending_anpr_to_candidate(cand.candidate_id)
        # First seed attempt with no usable crop fails and must NOT mark the
        # candidate seeded (else the retry is permanently blocked).
        self.assertFalse(reg.seed_gallery_from_park_entry(cand.candidate_id, "SEED-5"))
        # A later good crop then seeds successfully.
        reg.update_park_entry_candidate_snapshot(
            cand.candidate_id, make_color_crop(DARK), quality_score=5.0
        )
        self.assertTrue(reg.seed_gallery_from_park_entry(cand.candidate_id, "SEED-5"))
        vectors, _, cams = reg.gallery_store.load_vectors("SEED-5")
        self.assertTrue(vectors)
        self.assertIn("CAM-23", cams)


if __name__ == "__main__":
    unittest.main()
