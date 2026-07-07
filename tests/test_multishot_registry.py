import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np

from src.vehicle_registry import vehicle_registry_core
from src.vehicle_registry import vehicle_registry_identity
from src.vehicle_registry.vehicle_registry import VehicleRegistry


class FakeMatcher:
    def __init__(self):
        self.calls = []

    def extract_feature(self, image):
        value = float(np.mean(image) / 255.0)
        return np.array([value, 1.0 - value], dtype=np.float32)

    def extract_features_batch(self, images):
        self.calls.append(len(images))
        return [self.extract_feature(image) for image in images]

    @staticmethod
    def compute_similarity(feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))


class FakeImageMatcher:
    @staticmethod
    def _compare_dominant_colors(_image_a, _image_b):
        return 1.0

    @staticmethod
    def compare(_image_a, _image_b):
        return 0.0


class TestMultishotRegistry(unittest.TestCase):
    def test_duplicate_entry_within_short_window_is_deduped(self):
        registry = self._make_registry()
        now = datetime.now()

        first = registry.register_anpr_event("DUP-001", "entry", timestamp=now)
        second = registry.register_anpr_event("DUP-001", "entry", timestamp=now)

        self.assertEqual(first.event_id, second.event_id)
        self.assertEqual(len(registry._pending_event_order), 1)
        self.assertEqual(len(registry.get_pending_entries()), 1)

    def test_single_snapshot_behavior_preserved_when_flag_off(self):
        registry = self._make_registry()
        candidate = registry.open_park_entry_candidate("CAM_01", 7)
        first = self._image(40)
        second = self._image(160)

        with patch.object(vehicle_registry_core, "REID_USE_MULTISHOT", False):
            registry.update_park_entry_candidate_snapshot(candidate.candidate_id, first, 10.0)
            registry.update_park_entry_candidate_snapshot(candidate.candidate_id, second, 5.0)

        self.assertEqual(candidate.quality_score, 10.0)
        np.testing.assert_array_equal(candidate.snapshot_image, first)
        self.assertEqual(candidate.snapshot_images, [])
        self.assertEqual(candidate.feature_vectors, [])

    def test_multishot_stores_at_most_three_references_and_syncs_primary(self):
        registry = self._make_registry()
        candidate = registry.open_park_entry_candidate("CAM_01", 7)
        images = [
            self._image(30),
            self._checkerboard(0, 100),
            self._checkerboard(0, 180),
            self._checkerboard(0, 220),
        ]

        with patch.object(vehicle_registry_core, "REID_USE_MULTISHOT", True):
            for image in images:
                registry.update_park_entry_candidate_snapshot(candidate.candidate_id, image, 1.0)

        self.assertLessEqual(len(candidate.snapshot_images), 3)
        self.assertEqual(len(candidate.feature_vectors), len(candidate.snapshot_images))
        np.testing.assert_array_equal(candidate.snapshot_image, candidate.snapshot_images[0])
        np.testing.assert_array_equal(candidate.feature_vector, candidate.feature_vectors[0])
        self.assertEqual(candidate.snapshot_path, candidate.snapshot_paths[0])

    def test_confirmation_matches_any_multishot_vector(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("ABC-123", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 7)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._image(80)
        candidate.feature_vectors = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]

        plate = registry.confirm_at_b1_entrance(
            "CAM_03",
            11,
            self._image(255),
            reference_images=[],
        )

        self.assertEqual(plate, "ABC-123")

    def test_confirmation_supports_single_legacy_vector(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("LEG-001", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 8)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._image(80)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        plate = registry.confirm_at_b1_entrance(
            "CAM_03",
            12,
            self._image(255),
            reference_images=[],
        )

        self.assertEqual(plate, "LEG-001")

    def test_color_filter_off_does_not_change_confirmation(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("OFF-001", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 1)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._bgr(255, 0, 0)
        candidate.color_hsv = (120.0, 255.0, 255.0)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        with patch.object(vehicle_registry_identity, "REID_USE_COLOR_FILTER", False):
            plate = registry.confirm_at_b1_entrance(
                "CAM_03",
                21,
                self._image(255),
                reference_images=[],
            )

        self.assertEqual(plate, "OFF-001")

    def test_color_filter_rejects_obvious_mismatch_before_reid(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("CLR-001", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 2)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._bgr(255, 0, 0)
        candidate.color_hsv = (120.0, 255.0, 255.0)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        with patch.object(vehicle_registry_identity, "REID_USE_COLOR_FILTER", True):
            plate = registry.confirm_at_b1_entrance(
                "CAM_03",
                22,
                self._bgr(0, 0, 255),
                reference_images=[],
            )

        self.assertIsNone(plate)

    def test_color_filter_allows_compatible_color(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("CLR-002", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 3)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._bgr(0, 0, 200)
        candidate.color_hsv = (0.0, 255.0, 200.0)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        with patch.object(vehicle_registry_identity, "REID_USE_COLOR_FILTER", True):
            plate = registry.confirm_at_b1_entrance(
                "CAM_03",
                23,
                self._bgr(0, 0, 255),
                reference_images=[],
            )

        self.assertEqual(plate, "CLR-002")

    def test_color_filter_multishot_passes_if_any_hsv_is_compatible(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("CLR-003", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 4)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._bgr(0, 0, 200)
        candidate.color_hsv_values = [
            (120.0, 255.0, 255.0),
            (0.0, 255.0, 200.0),
        ]
        candidate.feature_vectors = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        ]

        with patch.object(vehicle_registry_identity, "REID_USE_COLOR_FILTER", True):
            plate = registry.confirm_at_b1_entrance(
                "CAM_03",
                24,
                self._bgr(0, 0, 255),
                reference_images=[],
            )

        self.assertEqual(plate, "CLR-003")

    def test_color_filter_missing_hsv_does_not_reject(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("CLR-004", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 5)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._bgr(255, 0, 0)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        with patch.object(vehicle_registry_identity, "REID_USE_COLOR_FILTER", True):
            plate = registry.confirm_at_b1_entrance(
                "CAM_03",
                25,
                self._image(255),
                reference_images=[],
            )

        self.assertEqual(plate, "CLR-004")

    def test_confirmation_keeps_ordered_cam03_gallery_and_uses_primary_index(self):
        registry = self._make_registry()
        event = registry.register_anpr_event("TIM-001", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 9)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._image(80)
        candidate.snapshot_path = "gate.jpg"
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        entry = self._image(70)
        deep = self._image(255)
        exit_view = self._image(120)

        plate = registry.confirm_at_b1_entrance(
            "CAM_03",
            30,
            deep,
            ordered_images=[entry, deep, exit_view],
            primary_snapshot_index=1,
        )

        self.assertEqual(plate, "TIM-001")
        session_id = registry.get_session_id_for_track("CAM_03", 30)
        session = registry._sessions[session_id]
        self.assertEqual(len(session.reference_snapshot_paths), 3)
        self.assertEqual(len(session.reference_feature_vectors), 3)
        self.assertEqual(session.gate_snapshot_paths, ["gate.jpg"])
        np.testing.assert_array_equal(
            session.feature_vector,
            registry._reid_matcher.extract_feature(deep),
        )

    def test_update_confirmed_session_gallery_merges_cam03_references(self):
        """The confirmed-session gallery ACCUMULATES views across crossings
        (front + rear, across cameras) instead of replacing them with the latest
        camera's timeline. Distinct earlier refs are retained and new distinct
        views appended; the primary tracks the newest crossing's chosen index."""
        registry = self._make_registry()
        event = registry.register_anpr_event("TIM-002", "entry", timestamp=datetime.now())
        candidate = registry.open_park_entry_candidate("CAM_01", 10)
        candidate.status = "provisional"
        candidate.bound_event_id = event.event_id
        event.status = "provisional"
        event.candidate_id = candidate.candidate_id
        candidate.snapshot_image = self._image(80)
        candidate.feature_vector = np.array([1.0, 0.0], dtype=np.float32)

        plate = registry.confirm_at_b1_entrance(
            "CAM_03",
            31,
            self._image(255),
            ordered_images=[self._image(255), self._image(120)],
            primary_snapshot_index=0,
        )
        self.assertEqual(plate, "TIM-002")
        session_id = registry.get_session_id_for_track("CAM_03", 31)
        session = registry._sessions[session_id]
        refs_after_confirm = len(session.reference_feature_vectors)

        entry = self._image(60)
        deep = self._image(180)
        exit_view = self._image(90)
        updated = registry.update_confirmed_session_gallery(
            "CAM_03",
            31,
            [entry, deep, exit_view],
            primary_snapshot_index=1,
        )

        self.assertTrue(updated)
        # Merge: the two distinct confirm refs are retained AND the three new
        # distinct views are appended (2 + 3 = 5), not replaced down to 3.
        self.assertEqual(len(session.reference_snapshot_paths), 5)
        self.assertEqual(len(session.reference_feature_vectors), 5)
        self.assertGreater(len(session.reference_feature_vectors), refs_after_confirm)
        # An earlier view (img255) survives the update — accumulation, not replace.
        img255_vec = registry._reid_matcher.extract_feature(self._image(255))
        self.assertTrue(
            any(np.allclose(v, img255_vec) for v in session.reference_feature_vectors)
        )
        # Primary tracks the newest crossing's chosen index (deep).
        np.testing.assert_array_equal(
            session.feature_vector,
            registry._reid_matcher.extract_feature(deep),
        )

    def test_get_plate_location_exposes_gate_and_gallery_snapshot_urls(self):
        registry = self._make_registry()
        session = registry.create_appearance_session(
            "CAM_03",
            50,
            np.array([1.0, 0.0], dtype=np.float32),
        )
        with registry._lock:
            vehicle_session = registry._sessions[session]
            vehicle_session.plate = "URL-001"
            vehicle_session.linked_slot = "SLOT-1"
            vehicle_session.linked_slot_name = "Slot 1"
            vehicle_session.linked_camera = "CAM_05"
            vehicle_session.linked_floor = "B1"
            vehicle_session.linked_zone_id = "ZONE-A"
            vehicle_session.linked_zone_name = "Zone A"
            vehicle_session.linked_at = datetime.now()
            vehicle_session.snapshot_path = "session_primary.jpg"
            vehicle_session.gate_snapshot_paths = ["gate_1.jpg"]
            vehicle_session.reference_snapshot_paths = [
                "cam3_entry.jpg",
                "cam3_deep.jpg",
                "cam3_exit.jpg",
            ]
            registry._parked["SLOT-1"] = vehicle_session

        location = registry.get_plate_location("URL-001")

        self.assertEqual(location["snapshot_url"], "/pms-video-analytics/snapshots/session_primary.jpg")
        self.assertEqual(location["gate_snapshot_urls"], ["/pms-video-analytics/snapshots/gate_1.jpg"])
        self.assertEqual(
            location["gallery_snapshot_urls"],
            [
                "/pms-video-analytics/snapshots/cam3_entry.jpg",
                "/pms-video-analytics/snapshots/cam3_deep.jpg",
                "/pms-video-analytics/snapshots/cam3_exit.jpg",
            ],
        )

    @staticmethod
    def _make_registry():
        image_dir = tempfile.mkdtemp(prefix="pms_reid_test_")
        registry = VehicleRegistry(image_dir=image_dir)
        registry._reid_matcher = FakeMatcher()
        registry._matcher = FakeImageMatcher()
        return registry

    @staticmethod
    def _image(value):
        return np.full((24, 24, 3), value, dtype=np.uint8)

    @staticmethod
    def _checkerboard(low, high):
        gray = np.indices((32, 32)).sum(axis=0) % 2
        gray = np.where(gray == 0, low, high).astype(np.uint8)
        return np.dstack([gray, gray, gray])

    @staticmethod
    def _bgr(blue, green, red):
        return np.full((24, 24, 3), (blue, green, red), dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
