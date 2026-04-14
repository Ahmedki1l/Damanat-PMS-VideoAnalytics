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

        # 2. Mock dependencies
        self.mock_registry = MagicMock()
        # [FIX] Registry must return None for unknown tracks so burst starts
        self.mock_registry.get_plate_for_track.return_value = None

        self.mock_detector = MagicMock()
        
        # Patch TrackedDetector to avoid loading YOLO
        with patch('src.core.engine.TrackedDetector', return_value=self.mock_detector), \
             patch('src.core.engine.EventBus', return_value=MagicMock()):
            self.engine = ParkingEngine(self.config, vehicle_registry=self.mock_registry)

        # 3. Create a mock confirmation zone (B1_Entrence)
        poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        self.zone = ParkingSlot(id="B1_Entrence", polygon=poly, label="B1 Entrance Zone")
        self.cam_id = "CAM_03"
        self.engine.special_zones[self.cam_id] = {"B1_Entrence": self.zone}

    def test_confirmation_burst_and_virtual_line(self):
        """Verify that best frame is kept and event triggers only on exit."""
        track_id = 42
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # --- Frame 1: Enter zone with small crop ---
        det1 = MockDetection(track_id, [10, 10, 30, 30], (20, 20)) # Size 20x20
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [det1], self.zone)

        # Check: Burst started
        burst_key = (self.cam_id, track_id)
        self.assertIn(burst_key, self.engine._confirmation_bursts)
        self.assertEqual(self.engine._confirmation_bursts[burst_key]['best_quality'], 400) # 20*20
        # Check: NO confirmation call yet (Virtual Line exit required)
        self.mock_registry.confirm_at_b1_entrance.assert_not_called()

        # --- Frame 2: Still in zone with LARGER crop ---
        det2 = MockDetection(track_id, [10, 10, 60, 60], (40, 40)) # Size 50x50
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [det2], self.zone)

        # Check: Best quality updated (2500)
        self.assertEqual(self.engine._confirmation_bursts[burst_key]['best_quality'], 2500)
        self.assertEqual(self.engine._confirmation_bursts[burst_key]['frames_collected'], 2)
        self.mock_registry.confirm_at_b1_entrance.assert_not_called()

        # --- Frame 3: Leave zone ---
        # No detections in this frame
        self.engine._process_confirmation_zone(self.cam_id, dummy_frame, [], self.zone)

        # Check: confirm_at_b1_entrance TRIIGGERED with best crop
        self.mock_registry.confirm_at_b1_entrance.assert_called_once()
        args, kwargs = self.mock_registry.confirm_at_b1_entrance.call_args
        # args might be pos/keyword depending on implementation
        # args[0]=cam_id, args[1]=tid, args[2]=crop
        self.assertEqual(args[0], self.cam_id)
        self.assertEqual(args[1], track_id)
        # Verify the crop passed was indeed the best one (50x50)
        self.assertEqual(args[2].shape[0], 50)
        
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
