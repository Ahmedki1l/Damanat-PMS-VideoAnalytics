"""Tests for single-camera ownership of a cross-camera car.

When several overlapping cameras see the same physical car at once, exactly
one camera should "own" the identity — the live observer with the highest ReID
score (or the slot's camera once the car is parked). Non-owning cameras get the
``"0"`` display fallback (box without the duplicated plate) and ``None`` for the
plate (so plate-driven data is attributed to one camera).

These tests drive the registry's internal maps directly — no torch / OpenVINO
model load — exactly like ``test_plate_keyed_guard_smoke``.
"""
import unittest
from datetime import datetime, timedelta

from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_identity import (
    OWNER_STALENESS_SECONDS,
    OWNER_SWITCH_MARGIN,
)
from src.vehicle_registry.vehicle_registry_models import VehicleSession


class _FakeClock:
    def __init__(self):
        self.now = datetime(2024, 1, 1, 12, 0, 0)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class TestSingleCameraOwnership(unittest.TestCase):
    PLATE = "ABC-123"

    def setUp(self):
        self.clock = _FakeClock()
        self.registry = VehicleRegistry(image_dir="tests/test_images", clock=self.clock)
        self.session = VehicleSession(
            session_id="s1",
            plate=self.PLATE,
            first_seen_at=self.clock(),
            last_seen_at=self.clock(),
            last_seen_camera="CAM-09",
            status="confirmed",
        )
        self.registry._sessions["s1"] = self.session

    def _observe(self, camera_id, track_id, score):
        """Register a live observing track with a given ReID score."""
        self.session.observing_tracks[camera_id] = track_id
        self.session.observing_scores[camera_id] = score
        self.registry._track_session_map[(camera_id, track_id)] = "s1"
        self.registry._track_last_seen[(camera_id, track_id)] = self.clock()

    def test_highest_score_camera_owns_identity(self):
        self._observe("CAM-09", 1, 0.70)
        self._observe("CAM-11", 2, 0.90)

        # CAM-11 has the higher score → it shows the plate.
        self.assertEqual(self.registry.get_display_id_for_track("CAM-11", 2), self.PLATE)
        self.assertEqual(self.registry.get_plate_for_track("CAM-11", 2), self.PLATE)

        # CAM-09 is a non-owning live observer → identity suppressed, box stays.
        self.assertEqual(self.registry.get_display_id_for_track("CAM-09", 1), "0")
        self.assertIsNone(self.registry.get_plate_for_track("CAM-09", 1))

    def test_single_observer_is_always_owner(self):
        self._observe("CAM-09", 1, 0.10)  # low score, but the only observer
        self.assertEqual(self.registry.get_display_id_for_track("CAM-09", 1), self.PLATE)
        self.assertEqual(self.registry.get_plate_for_track("CAM-09", 1), self.PLATE)

    def test_ownership_transfers_when_owner_track_goes_stale(self):
        self._observe("CAM-09", 1, 0.70)
        self._observe("CAM-11", 2, 0.90)
        self.assertEqual(self.session.owner_camera, None)
        # Resolve once so CAM-11 becomes the incumbent owner.
        self.assertEqual(self.registry.get_display_id_for_track("CAM-11", 2), self.PLATE)
        self.assertEqual(self.session.owner_camera, "CAM-11")

        # CAM-11's track goes stale; only CAM-09 stays live.
        self.clock.advance(OWNER_STALENESS_SECONDS + 2)
        self.registry._track_last_seen[("CAM-09", 1)] = self.clock()

        self.assertEqual(self.registry.get_display_id_for_track("CAM-09", 1), self.PLATE)
        self.assertEqual(self.session.owner_camera, "CAM-09")

    def test_hysteresis_keeps_incumbent_within_margin(self):
        self._observe("CAM-09", 1, 0.70)
        self._observe("CAM-11", 2, 0.72)
        # Make CAM-09 the incumbent owner first.
        self.session.owner_camera = "CAM-09"

        # Challenger CAM-11 only leads by 0.02 (< margin) → incumbent keeps it.
        self.assertLess(0.72 - 0.70, OWNER_SWITCH_MARGIN)
        self.assertEqual(self.registry.get_display_id_for_track("CAM-09", 1), self.PLATE)
        self.assertEqual(self.registry.get_display_id_for_track("CAM-11", 2), "0")

        # Now CAM-11 clears the margin → ownership switches.
        self.session.observing_scores["CAM-11"] = 0.90
        self.assertEqual(self.registry.get_display_id_for_track("CAM-11", 2), self.PLATE)
        self.assertEqual(self.registry.get_display_id_for_track("CAM-09", 1), "0")

    def test_slot_linked_camera_owns_regardless_of_reid_score(self):
        # Car is parked in CAM-14's slot, but CAM-11 has a higher live ReID score.
        self.session.linked_slot = "B25"
        self.session.linked_camera = "CAM-14"
        self._observe("CAM-11", 2, 0.99)

        owner = self.registry._resolve_owner_camera(self.session, self.clock())
        self.assertEqual(owner, "CAM-14")
        # The high-score but non-owning camera is suppressed.
        self.assertEqual(self.registry.get_display_id_for_track("CAM-11", 2), "0")

    def test_no_suppression_with_a_single_appearance_session(self):
        # An ID-only (no plate) session on one camera should still display its ID.
        self.session.plate = None
        self._observe("CAM-09", 1, 0.5)
        display = self.registry.get_display_id_for_track("CAM-09", 1)
        self.assertTrue(display.startswith("ID-"))


if __name__ == "__main__":
    unittest.main()
