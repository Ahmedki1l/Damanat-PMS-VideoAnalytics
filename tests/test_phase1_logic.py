"""
test_phase1_logic.py — Standalone logic verification for Phase 1 (Virtual Line & Burst Collection).

Mocks detections and registry to verify:
1. Burst collection of crops while in the confirmation zone.
2. Quality-based best-frame selection.
3. Virtual Line trigger upon zone exit.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from shapely.geometry import Polygon, Point
from datetime import datetime

# Add src to path
sys.path.append('.')

from src.core.engine import ParkingEngine
from src.models.slot import ParkingSlot
from src.config import AppConfig

class MockDetection:
    def __init__(self, track_id, bbox, bottom_center):
        self.track_id = track_id
        self.bbox = bbox # [x1, y1, x2, y2]
        self.bottom_center = bottom_center

class TestPhase1Logic(unittest.TestCase):
    def setUp(self):
        # 1. Create a dummy config
        self.config = MagicMock(spec=AppConfig)
        self.config.detector = MagicMock()
        self.config.tracker = MagicMock()
        self.config.output = MagicMock()
        self.config.output.log_file = "test.log"
        self.config.cameras = []
        # Newer AppConfig fields are declared via default_factory, so
        # MagicMock(spec=...) does not auto-provide them — set them
        # explicitly: preprocessing (per-camera TrackedDetector build) and
        # areas (zoning AreaRegistry; empty list = zoning disabled).
        self.config.preprocessing = MagicMock()
        self.config.areas = []

        # 2. Mock dependencies
        self.mock_registry = MagicMock()
        # [FIX] Registry must return None for unknown tracks so burst starts
        self.mock_registry.get_plate_for_track.return_value = None
        # _is_consistent_confirmation_crop compares this against a float —
        # a bare MagicMock return would TypeError inside the burst loop.
        self.mock_registry.matcher._compare_dominant_colors.return_value = 1.0

        self.mock_detector = MagicMock()
        
        # Patch TrackedDetector to avoid loading YOLO
        with patch('src.core.engine.engine.TrackedDetector', return_value=self.mock_detector), \
             patch('src.core.engine.engine.EventBus', return_value=MagicMock()):
            self.engine = ParkingEngine(self.config, vehicle_registry=self.mock_registry)

        # 3. Create a mock confirmation zone (B1_Entrence)
        poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        self.zone = ParkingSlot(id="B1_Entrence", polygon=poly, label="B1 Entrance Zone")
        self.cam_id = "CAM_03"
        self.engine.special_zones[self.cam_id] = {"B1_Entrence": self.zone}

    def test_confirmation_burst_and_virtual_line(self):
        """Best frame is kept; early in-zone confirmation attempts fire once
        the car is deep in the zone (post-Phase-1 behaviour), and the final
        confirmation triggers on zone exit with the timeline gallery."""
        track_id = 42
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Early attempts must not "confirm" here — force the exit-time path.
        self.mock_registry.confirm_at_b1_entrance.return_value = None

        # --- Frame 1: Enter zone with small crop ---
        det1 = MockDetection(track_id, [10, 10, 30, 30], (20, 20)) # Size 20x20
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [det1], self.zone)

        # Check: Burst started
        burst_key = (self.cam_id, track_id)
        self.assertIn(burst_key, self.engine._confirmation_bursts)
        # Quality is a bounded score, not raw bbox area. It used to be exactly
        # `area * 1.5` (the sharpness term was pinned at 1.0 on every real
        # frame), so a closer car outranked every other signal by pixel count
        # alone. Area now saturates, which means the absolute number is not
        # meaningful on its own — what this test cares about is that a better
        # frame replaces the first one.
        first_quality = self.engine._confirmation_bursts[burst_key]['best_quality']
        self.assertGreater(first_quality, 0.0)
        # First frame only seeds the burst — no confirmation attempt yet.
        self.mock_registry.confirm_at_b1_entrance.assert_not_called()

        # --- Frame 2: Still in zone with LARGER crop ---
        det2 = MockDetection(track_id, [10, 10, 60, 60], (40, 40)) # Size 50x50
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [det2], self.zone)

        # Check: the larger, closer view replaced the entry frame as best.
        self.assertGreater(
            self.engine._confirmation_bursts[burst_key]['best_quality'], first_quality
        )
        self.assertEqual(self.engine._confirmation_bursts[burst_key]['frames_collected'], 2)
        # The car is deep in the zone now, so an EARLY confirmation attempt is
        # expected (this replaced the old exit-only Virtual Line rule). The
        # mock returned None, so the burst stays unconfirmed.
        early_calls = self.mock_registry.confirm_at_b1_entrance.call_count
        self.assertGreaterEqual(early_calls, 1)
        self.assertFalse(self.engine._confirmation_bursts[burst_key]['confirmed'])

        # --- Frame 3: Leave zone ---
        # No detections in this frame
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [], self.zone)

        # Check: exit triggered the final confirmation attempt
        self.assertGreater(
            self.mock_registry.confirm_at_b1_entrance.call_count, early_calls
        )
        args, kwargs = self.mock_registry.confirm_at_b1_entrance.call_args
        # args[0]=cam_id, args[1]=tid, args[2]=primary crop
        self.assertEqual(args[0], self.cam_id)
        self.assertEqual(args[1], track_id)
        # The primary crop passed is the deep (largest) view, cropped to the
        # ZONE and masked to its polygon — not a plain bbox crop.
        #
        # This assertion used to call _crop_detection (plain bbox + padding) and
        # broke on fbb60c2 (2026-07-09, "mask zone vehicle snapshots to ROI
        # polygon"), which switched _process_confirmation_zone to
        # _crop_detection_to_zone so a car outside the entrance polygon can no
        # longer leak into the confirmation crop. The two differ in more than
        # masking: _crop_detection enforces MINIMUM padding (max(16,...) /
        # max(12,...)), so on this 50x50 box it padded to 72x76, while the zone
        # path uses max(0,...) and clips to the polygon bounds, giving 62x62.
        expected_crop = self.engine._crop_detection_to_zone(
            dummy_frame,
            det2,
            self.zone,
            padding_ratio=0.12,
            mask_outside_zone=True,
        )
        self.assertEqual(args[2].shape, expected_crop.shape)

        # Check: Burst state CLEARED
        self.assertNotIn(burst_key, self.engine._confirmation_bursts)

    def test_already_confirmed_ignore(self):
        """Verify that already confirmed tracks don't restart bursts."""
        track_id = 99
        self.mock_registry.get_plate_for_track.return_value = "ABC-123"
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det = MockDetection(track_id, [10, 10, 30, 30], (20, 20))
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [det], self.zone)

        burst_key = (self.cam_id, track_id)
        self.assertNotIn(burst_key, self.engine._confirmation_bursts)
        self.mock_registry.confirm_at_b1_entrance.assert_not_called()

if __name__ == "__main__":
    unittest.main()
