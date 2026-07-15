"""Regression: the raw ANPR gate FRAME must never become a matchable reference.

src/api.py parks the uncropped 2688x1552 gate frame on a Park_Entry candidate on
purpose — it is the durable gate photo. The CAM-03 confirmation's "Fix 4"
fallback asked for "the latest Park_Entry candidate for this plate" WITHOUT
filtering by camera, so whenever CAM-23 had not seeded (the common case) it got
the ANPR candidate and filed that frame as a matchable ref. Found 2026-07-15 in
the live gallery: 62 such refs, in all 38 plate folders.

A full frame embeds the SCENE, not the car: measured 0.41 cosine against its own
car's crops (below the match bar, so it never helps) and 0.33 against OTHER cars'
gate frames (they share a background — similarity with no identity in it). Worse,
multishot scores it ~1900 on sharpness against a real crop's 999, so at prune
time it OUTRANKS and evicts the good crop.

Three independent guards, one test class each — any one of them alone stops the
leak, which is the point: this cost us the whole gallery once.
"""
import unittest

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry_identity import is_plausible_car_crop
from tests.fixtures.match_fixtures import make_test_registry

# The gate stream's real frame size, and the largest GENUINE car crop measured in
# the live gallery (a car close to the lens). The bar must separate these two.
FULL_FRAME = (1552, 2688)
BIGGEST_REAL_CROP = (1309, 2012)


def _frame(hw):
    # Noise, not flat colour: a flat array is trivially rejected by any future
    # appearance check, which would make this test pass for the wrong reason.
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (hw[0], hw[1], 3), dtype=np.uint8)


class TestIsPlausibleCarCropRejectsFullFrames(unittest.TestCase):
    """Guard 1 — geometry. Aspect alone could not see this: a 2688x1552 frame is
    aspect 1.73, well inside the 2.2 cap, so it passed as 'one whole car'."""

    def test_rejects_full_sensor_frame(self):
        self.assertFalse(is_plausible_car_crop(_frame(FULL_FRAME)))

    def test_accepts_largest_real_car_crop(self):
        # The bar must not eat genuine close-up crops — all 245 real refs in the
        # live gallery pass, and this is the largest of them.
        self.assertTrue(is_plausible_car_crop(_frame(BIGGEST_REAL_CROP)))

    def test_accepts_ordinary_car_crop(self):
        self.assertTrue(is_plausible_car_crop(_frame((300, 400))))

    def test_still_rejects_merged_cars_by_aspect(self):
        # The original reason this function exists (a 1527x519 CAM-03 box, aspect
        # 2.94, that landed in EEB-80's gallery twice on 2026-07-12).
        self.assertFalse(is_plausible_car_crop(_frame((519, 1527))))


class TestParkEntryCandidateCameraFilter(unittest.TestCase):
    """Guard 2 — provenance. The CAM-03 fallback wants the CAM-23 TOP VIEW; it
    must not silently accept the ANPR gate candidate standing in for it."""

    def _registry(self):
        cfg = MatchingConfig()
        cfg.gallery_persist_enabled = True
        cfg.reid_openvino_model_dir = ""
        return make_test_registry(matching_config=cfg)

    def _bound_candidate(self, reg, plate, camera_id, track_id):
        reg.register_anpr_event(plate, "entry", timestamp=reg._clock())
        cand = reg.open_park_entry_candidate(camera_id, track_id)
        reg.update_park_entry_candidate_snapshot(
            cand.candidate_id, _frame((300, 400)), quality_score=5.0
        )
        reg.bind_next_pending_anpr_to_candidate(cand.candidate_id)
        return cand

    def test_anpr_candidate_is_not_returned_when_cam23_is_asked_for(self):
        reg = self._registry()
        self._bound_candidate(reg, "LEAK-1", "ANPR", 11)
        # This is the bug: unfiltered, the ANPR gate candidate answered a request
        # meant for the CAM-23 top view, and its frame was seeded as a ref.
        self.assertIsNone(
            reg.latest_park_entry_candidate_for_plate("LEAK-1", camera_id="CAM-23")
        )

    def test_cam23_candidate_is_still_found(self):
        reg = self._registry()
        cand = self._bound_candidate(reg, "LEAK-2", "CAM-23", 12)
        self.assertEqual(
            reg.latest_park_entry_candidate_for_plate("LEAK-2", camera_id="CAM-23"),
            cand.candidate_id,
        )

    def test_unfiltered_call_still_returns_any_candidate(self):
        # The filter is opt-in; existing callers keep their behaviour.
        reg = self._registry()
        cand = self._bound_candidate(reg, "LEAK-3", "ANPR", 13)
        self.assertEqual(
            reg.latest_park_entry_candidate_for_plate("LEAK-3"), cand.candidate_id
        )


class TestSeedGalleryRefusesFullFrame(unittest.TestCase):
    """Guard 3 — the write itself. seed_gallery_from_park_entry was the only
    save_ref caller with no shape check at all, whatever camera fed it."""

    def _registry(self, tmpdir):
        cfg = MatchingConfig()
        cfg.gallery_persist_enabled = True
        cfg.reid_openvino_model_dir = ""
        cfg.gallery_min_identity_similarity = 0.0
        return make_test_registry(matching_config=cfg, image_dir=tmpdir)

    def test_full_frame_candidate_is_not_seeded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            reg.register_anpr_event("LEAK-4", "entry", timestamp=reg._clock())
            cand = reg.open_park_entry_candidate("CAM-23", 14)
            reg.update_park_entry_candidate_snapshot(
                cand.candidate_id, _frame(FULL_FRAME), quality_score=999.0
            )
            reg.bind_next_pending_anpr_to_candidate(cand.candidate_id)
            self.assertFalse(
                reg.seed_gallery_from_park_entry(cand.candidate_id, "LEAK-4")
            )
            store = reg.gallery_store
            self.assertEqual(store.load_vectors("LEAK-4")[0], [])

    def test_real_crop_candidate_is_still_seeded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reg = self._registry(tmp)
            reg.register_anpr_event("LEAK-5", "entry", timestamp=reg._clock())
            cand = reg.open_park_entry_candidate("CAM-23", 15)
            reg.update_park_entry_candidate_snapshot(
                cand.candidate_id, _frame((300, 400)), quality_score=999.0
            )
            reg.bind_next_pending_anpr_to_candidate(cand.candidate_id)
            self.assertTrue(
                reg.seed_gallery_from_park_entry(cand.candidate_id, "LEAK-5")
            )


if __name__ == "__main__":
    unittest.main()
