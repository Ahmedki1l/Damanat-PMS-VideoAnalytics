"""
test_cross_camera.py — Verification of Phase 3 (Global Tracking).

Simulates a vehicle confirmed on CAM_03 appearing later on CAM_04.
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock

import types
import logging

# Configure logging to see Registry output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.vehicle_registry import VehicleRegistry
from src.core.engine import ParkingEngine

def test_global_tracking():
    # 1. Setup Registry and Engine
    registry = VehicleRegistry(image_dir="test_images_p3")
    
    # Mock engine to avoid full YOLO initialization
    engine = MagicMock(spec=ParkingEngine)
    engine.vehicle_registry = registry
    engine._reid_check_timer = {}
    
    # Use real crop logic from engine if possible, but for test we mock it
    # to return our known car image.
    engine._crop_detection = lambda f, d: f

    # 2. Use a real image for matching (e.g. NDD-4141)
    img_path = "vehicle_images/NDD-4141_CAM01_bound_20260413_080606.jpg"
    if not os.path.exists(img_path):
        print("Sample image not found. Please ensure vehicle_images/ contains NDD-4141...")
        return
    
    img = cv2.imread(img_path)

    print("\n" + "="*60)
    print("  PHASE 3: GLOBAL TRACKING TEST")
    print("="*60)

    # Confirming vehicle 'TEST-PLATE' on CAM_03...
    cand = registry.open_park_entry_candidate("CAM_01", 100)
    registry.update_park_entry_candidate_snapshot(cand.candidate_id, img, 1000.0)
    event = registry.register_anpr_event("TEST-PLATE", "entry")
    registry.bind_next_pending_anpr_to_candidate(cand.candidate_id) # FIX: pass cand.candidate_id
    
    # Confirm it at B1 Entrance (CAM_03)
    registry.confirm_at_b1_entrance("CAM_03", 200, img)
    
    plate_c3 = registry.get_plate_for_track("CAM_03", 200)
    print(f"  -> CAM_03 Identity: {plate_c3}")

    # --- STEP 2: Appearance on CAM_04 ---
    print("\n[STEP 2] Vehicle appears on CAM_04 (New Track ID 300)...")
    
    class MockDet:
        def __init__(self, tid, bbox):
            self.track_id = tid
            self.bbox = bbox

    detections = [MockDet(300, [10, 10, 100, 100])]
    
    # Bind the real method to our mock engine for testing
    import types
    engine._process_global_tracking = types.MethodType(ParkingEngine._process_global_tracking, engine)
    
    # Run the tracking logic
    engine._process_global_tracking("CAM_04", img, detections)

    # --- STEP 3: Verification ---
    plate_c4 = registry.get_plate_for_track("CAM_04", 300)
    
    print(f"\n[RESULT]")
    print(f"  -> CAM_04 Identity: {plate_c4}")
    
    if plate_c4 == "TEST-PLATE":
        print("\n  -> SUCCESS: Global Tracking linked CAM_04 to TEST-PLATE via ReID!")
    else:
        print("\n  -> FAILURE: CAM_04 track remains unidentified.")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_global_tracking()
