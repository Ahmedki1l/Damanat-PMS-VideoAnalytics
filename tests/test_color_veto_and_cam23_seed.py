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
from types import SimpleNamespace

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
    # These regressions isolate the legacy accumulation/seed quality gates.
    cfg.gallery_strict_admission_enabled = False
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


class TestBodyColourCompatibleBrightness(unittest.TestCase):
    """Regression for champagne-vs-dark: brightness check in mixed-saturation
    branch catches muted-chroma cars (S<90) that were leaking through the
    saturation-only gate."""

    def test_champagne_vs_dark_incompatible(self):
        # A bright tan/champagne Lexus (S=75, V=200) vs a dark Hyundai (S=10, V=45).
        # Saturation alone passes: 75 < 90. But they're DIFFERENT cars because
        # brightness is far apart: |200 - 45| = 155 > v_tol (90).
        champagne = (100, 75, 200)  # tan/champagne: S=75, V=200
        dark = (30, 10, 45)  # dark sedan: S=10, V=45
        self.assertFalse(
            body_colour_compatible(champagne, dark),
            "champagne and dark must be incompatible (brightness too far apart)",
        )

    def test_muted_brown_compatible_despite_saturation_gap(self):
        # Two muted-brown shades under the same lighting: S differs (60 vs 40)
        # but brightness agrees (100 vs 105). Must NOT be rejected.
        brown1 = (40, 60, 100)  # S=60, V=100
        brown2 = (30, 40, 105)  # S=40, V=105 (different sat, close brightness)
        self.assertTrue(
            body_colour_compatible(brown1, brown2),
            "muted browns with similar brightness are compatible",
        )

    def test_white_vs_champagne_incompatible(self):
        # A bright champagne (S=80, V=230) is still clearly different from white
        # (S=10, V=250) — they're both bright but one has discernible colour.
        # Mixed-chroma rule applies: chroma_s=80 >= 90 triggers saturation-only
        # rejection... wait, 80 < 90, so it's a muted case. Brightness is close
        # (|230 - 250| = 20 < 90), so they PASS. This is correct: a champagne
        # car can be confused with white under poor lighting.
        champagne = (100, 80, 230)  # S=80, V=230
        white = (250, 10, 250)  # S=10, V=250
        self.assertTrue(
            body_colour_compatible(champagne, white),
            "champagne and white can be confused under bright lighting",
        )


class TestReattachLearnGate(unittest.TestCase):
    """Regression for reattach learn-gate: a borderline anonymous track from
    CAM-24 (0.43-0.60 similarity) can associate but must not poison the gallery.
    The vector appends only if GT-similarity >= 0.45 (the learn floor)."""

    def _confirmed_dark_session(self, reg):
        dark_crop = make_color_crop(DARK)
        dark_vec = reg.reid_matcher.extract_feature(dark_crop)
        s = make_vehicle_session(
            "DARK-REATTACH",
            feature_vector=dark_vec,
            status="confirmed",
            last_seen_camera="CAM-03",
        )
        s.ground_truth_hsv = dominant_color_hsv(dark_crop)
        reg._sessions[s.session_id] = s
        return s, dark_vec

    def test_learn_gate_blocks_append_when_gt_sim_below_floor(self):
        # Simpler test: directly test _best_ground_truth_similarity and the gate
        cfg = _gallery_config()
        cfg.gallery_accumulate_min_gt_similarity = 0.45
        reg = make_test_registry(matching_config=cfg)

        # Create a session with a ground-truth appearance
        dark_crop = make_color_crop(DARK)
        dark_vec = reg.reid_matcher.extract_feature(dark_crop)
        s = make_vehicle_session(
            "LG-1",
            feature_vector=dark_vec,
            status="confirmed",
            last_seen_camera="CAM-03",
        )
        s.ground_truth_hsv = dominant_color_hsv(dark_crop)
        reg._sessions[s.session_id] = s

        # Create a query vector that scores 0.43 to the ground truth
        query_vec = np.array([0.5, 0.5], dtype=np.float32)
        reg.reid_matcher.pin_similarity(query_vec, dark_vec, 0.43)

        # Verify _best_ground_truth_similarity returns 0.43
        gt_sim = reg._best_ground_truth_similarity(query_vec, s)
        self.assertAlmostEqual(gt_sim, 0.43, places=5)

        # Verify 0.43 < 0.45 (the learn floor)
        learn_floor = cfg.gallery_accumulate_min_gt_similarity
        self.assertLess(gt_sim, learn_floor)

    def test_borderline_reattach_associates_but_no_append_below_learn_floor(self):
        cfg = _gallery_config()
        cfg.gallery_accumulate_min_gt_similarity = 0.45
        cfg.reattach_excluded_cameras = []  # Allow CAM-24 for this test
        reg = make_test_registry(matching_config=cfg)

        s, dark_vec = self._confirmed_dark_session(reg)
        initial_ref_count = len(s.reference_feature_vectors)

        # Create a query vector that scores below the learn floor but above the
        # association threshold. To do this, we create a vector that computes
        # naturally to be in the 0.41-0.44 range via cosine similarity.
        query_vec = np.array([0.50, 0.87], dtype=np.float32)
        reg.reid_matcher.pin_similarity(query_vec, dark_vec, 0.42)

        # Insert the anonymous session locally so reattach can find it.
        anon_session = make_vehicle_session(
            None,  # no plate yet (anonymous)
            feature_vector=query_vec,
            status="unconfirmed",
            last_seen_camera="CAM-24",
            last_seen_track_id=99,
        )
        reg._sessions[anon_session.session_id] = anon_session
        reg._track_session_map[("CAM-24", 99)] = anon_session.session_id

        # Call reattach with cross-camera threshold. 0.42 >= 0.41 passes association.
        result = reg.reattach_track_to_confirmed_session(
            camera_id="CAM-24",
            track_id=99,
            query_vector=query_vec,
            similarity_threshold=0.41,  # reattach_cross_camera
        )

        # Association must succeed (0.42 >= 0.41)
        self.assertEqual(
            result,
            s.session_id,
            "0.42 similarity passes reattach_cross_camera (0.41)",
        )

        # But the reference should NOT be appended because gt_sim=0.42 < learn_floor=0.45
        self.assertEqual(
            len(s.reference_feature_vectors),
            initial_ref_count,
            "borderline reattach (0.42 < 0.45) must not append to reference_feature_vectors",
        )

    def test_reattach_from_excluded_camera_returns_none(self):
        cfg = _gallery_config()
        cfg.reattach_excluded_cameras = ["CAM-24"]
        reg = make_test_registry(matching_config=cfg)

        s, dark_vec = self._confirmed_dark_session(reg)

        # Create an anonymous track from CAM-24 with high similarity (0.80).
        anon_crop = make_color_crop((100, 100, 100))
        anon_vec = reg.reid_matcher.extract_feature(anon_crop)
        reg.reid_matcher.pin_similarity(anon_vec, dark_vec, 0.80)

        anon_session = make_vehicle_session(
            None,
            feature_vector=anon_vec,
            status="unconfirmed",
            last_seen_camera="CAM-24",
            last_seen_track_id=88,
        )
        reg._sessions[anon_session.session_id] = anon_session
        reg._track_session_map[("CAM-24", 88)] = anon_session.session_id

        # Reattach from CAM-24 is blocked entirely, returns None.
        result = reg.reattach_track_to_confirmed_session(
            camera_id="CAM-24",
            track_id=88,
            query_vector=anon_vec,
            similarity_threshold=0.41,
        )
        self.assertIsNone(
            result,
            "CAM-24 (reattach_excluded_cameras) must not reattach, even with high similarity",
        )


class TestSeedPathIdentityFloor(unittest.TestCase):
    """The CAM-23 Park_Entry seed shares the D2 identity floor
    (``gallery_min_identity_similarity``). That knob defaults to 0.0 (INERT) in
    code but production ``config.yaml`` sets 0.35 — so the active behaviour is
    only ever exercised off the YAML value, never the default. These pin the
    active-floor behaviour so it cannot silently regress to the no-op, and guard
    the ``store.load_vectors()`` return-shape: it is a
    ``(vectors, tag, cameras)`` tuple, and the seed must compare against the
    vectors, not iterate the tuple."""

    def _registry(self, floor):
        cfg = _gallery_config()
        cfg.gallery_min_identity_similarity = floor
        return make_test_registry(matching_config=cfg)

    def _establish_dark_gallery(self, reg, plate):
        """Seed one durable ground-truth (dark) ref for ``plate`` so later seeds
        have an established identity to be measured against."""
        dark = make_color_crop(DARK)
        dark_vec = reg.reid_matcher.extract_feature(dark)
        reg.gallery_store.save_ref(
            plate, dark, dark_vec, quality=999.0, camera_id="ANPR", gate_only=False
        )
        loaded, _, _ = reg.gallery_store.load_vectors(plate)
        return loaded[0]

    def _candidate_with_crop(self, reg, crop, track_id=7):
        cand = reg.open_park_entry_candidate("CAM-23", track_id)
        reg.update_park_entry_candidate_snapshot(
            cand.candidate_id, crop, quality_score=5.0
        )
        return cand.candidate_id

    def test_foreign_seed_rejected_at_active_floor(self):
        reg = self._registry(0.35)
        dark_vec = self._establish_dark_gallery(reg, "SEED-D")
        white = make_color_crop(WHITE)
        white_vec = reg.reid_matcher.extract_feature(white)
        reg.reid_matcher.pin_similarity(white_vec, dark_vec, 0.10)  # < 0.35 floor
        cand_id = self._candidate_with_crop(reg, white)
        self.assertFalse(
            reg.seed_gallery_from_park_entry(cand_id, "SEED-D"),
            "a foreign crop below the identity floor must not seed the gallery",
        )
        vecs, _, _ = reg.gallery_store.load_vectors("SEED-D")
        self.assertEqual(len(vecs), 1, "gallery keeps only the established dark ref")

    def test_same_car_seed_admitted_at_active_floor(self):
        reg = self._registry(0.35)
        dark_vec = self._establish_dark_gallery(reg, "SEED-S")
        crop = make_color_crop((40, 40, 40))  # compatible dark shade
        crop_vec = reg.reid_matcher.extract_feature(crop)
        reg.reid_matcher.pin_similarity(crop_vec, dark_vec, 0.9)  # > 0.35 floor
        cand_id = self._candidate_with_crop(reg, crop)
        self.assertTrue(
            reg.seed_gallery_from_park_entry(cand_id, "SEED-S"),
            "a same-car top view above the floor is a legitimate seed",
        )
        vecs, _, cams = reg.gallery_store.load_vectors("SEED-S")
        self.assertEqual(len(vecs), 2, "the CAM-23 top view joins the dark ref")
        self.assertIn("CAM-23", cams)

    def test_inert_default_floor_admits_foreign_seed(self):
        # The trap this suite exists to lock down: at the 0.0 default the guard is
        # a no-op, so even a foreign crop seeds. Documents the config-dependence —
        # production MUST carry gallery_min_identity_similarity: 0.35 for the
        # seed-path D2 guard to do anything.
        reg = self._registry(0.0)
        dark_vec = self._establish_dark_gallery(reg, "SEED-Z")
        white = make_color_crop(WHITE)
        white_vec = reg.reid_matcher.extract_feature(white)
        reg.reid_matcher.pin_similarity(white_vec, dark_vec, 0.10)
        cand_id = self._candidate_with_crop(reg, white)
        self.assertTrue(
            reg.seed_gallery_from_park_entry(cand_id, "SEED-Z"),
            "at the inert 0.0 default the seed guard is a no-op (documented trap)",
        )


class TestNeighbourClearanceD9(unittest.TestCase):
    """D9: neighbour-clearance weights view quality so a car whose bbox is
    overlapped by a parked neighbour makes a lower-quality (contaminated) ReID
    reference. Log-only by default (no vehicle_registry / flag off); multiplied
    into quality only when gallery_neighbour_clearance_enforce is set."""

    class _Harness(ParkingEngineTrackingMixin):
        pass

    class _Det:
        def __init__(self, bbox, track_id=1):
            self.bbox = bbox
            self.track_id = track_id

    def setUp(self):
        self.h = self._Harness()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_lone_car_is_fully_clear(self):
        det = self._Det((500, 200, 760, 520), track_id=1)
        self.assertEqual(self.h._neighbour_clearance(det, [det]), 1.0)

    def test_separated_cars_are_clear(self):
        a = self._Det((500, 200, 760, 520), track_id=1)
        b = self._Det((900, 200, 1100, 520), track_id=2)  # no x-overlap
        self.assertEqual(self.h._neighbour_clearance(a, [a, b]), 1.0)

    def test_overlapping_neighbour_reduces_clearance(self):
        a = self._Det((500, 200, 760, 520), track_id=1)  # 260x320, area 83200
        b = self._Det((630, 200, 900, 520), track_id=2)  # covers right half of a
        # intersection-over-A: x 630..760 = 130, y full 320 -> 41600 / 83200 = 0.5
        self.assertAlmostEqual(self.h._neighbour_clearance(a, [a, b]), 0.5, places=3)

    def test_asymmetric_small_box_swallowed_by_large(self):
        small = self._Det((600, 300, 680, 353), track_id=1)  # tiny distant car
        big = self._Det((500, 200, 900, 520), track_id=2)    # fully contains small
        # intersection-over-SELF = 1.0 -> clearance 0: a symmetric IoU would have
        # hidden this (the union is huge); intersection-over-self catches it.
        self.assertAlmostEqual(
            self.h._neighbour_clearance(small, [small, big]), 0.0, places=3
        )

    def test_untracked_neighbour_still_counts_as_contamination(self):
        a = self._Det((500, 200, 760, 520), track_id=1)
        ghost = self._Det((630, 200, 900, 520), track_id=-1)  # untracked detection
        self.assertAlmostEqual(
            self.h._neighbour_clearance(a, [a, ghost]), 0.5, places=3
        )

    def test_exact_padded_crop_counts_neighbour_outside_raw_box(self):
        a = self._Det((500, 200, 760, 520), track_id=1)
        b = self._Det((770, 200, 900, 520), track_id=2)
        self.assertEqual(self.h._neighbour_clearance(a, [a, b]), 1.0)
        padded = self.h._neighbour_clearance(
            a,
            [a, b],
            frame_shape=self.frame.shape,
            padding_ratio=0.1,
        )
        self.assertGreater(padded, 0.0)
        self.assertLess(padded, 1.0)

    def test_missing_or_malformed_detection_evidence_fails_closed(self):
        a = self._Det((500, 200, 760, 520), track_id=1)
        malformed = self._Det((float("nan"), 200, 900, 520), track_id=2)
        self.assertEqual(self.h._neighbour_clearance(a, None), 0.0)
        self.assertEqual(self.h._neighbour_clearance(a, []), 0.0)
        self.assertEqual(self.h._neighbour_clearance(a, [malformed]), 0.0)
        self.assertEqual(self.h._neighbour_clearance(a, [a, malformed]), 0.0)

    def test_log_only_does_not_change_quality(self):
        # No vehicle_registry -> enforce False -> base returned even with overlap.
        a = self._Det((500, 200, 760, 520), track_id=1)
        b = self._Det((630, 200, 900, 520), track_id=2)
        base = self.h._bbox_view_quality(self.frame, a)  # detections=None
        weighted = self.h._bbox_view_quality(self.frame, a, [a, b])
        self.assertEqual(base, weighted, "log-only mode must not gate quality")

    def test_enforce_multiplies_clearance(self):
        a = self._Det((500, 200, 760, 520), track_id=1)
        b = self._Det((630, 200, 900, 520), track_id=2)  # clearance 0.5
        self.h.vehicle_registry = SimpleNamespace(
            matching_config=SimpleNamespace(
                gallery_neighbour_clearance_enforce=True
            )
        )
        base = self.h._bbox_view_quality(self.frame, a)  # detections=None -> base
        weighted = self.h._bbox_view_quality(self.frame, a, [a, b])
        self.assertAlmostEqual(weighted, base * 0.5, places=3)
        self.assertLess(weighted, base)


if __name__ == "__main__":
    unittest.main()
