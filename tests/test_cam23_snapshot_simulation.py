"""
Simulation test for CAM-23 Park_Entry snapshot capture pipeline.

This test exercises the full pipeline without requiring real vehicles or cameras:
  1. Create a pending ANPR entry event (simulates gate read)
  2. Simulate vehicle detection in the Park_Entry zone on CAM-23
  3. Exercise the snapshot capture, quality scoring, and FIFO-bind flow
  4. Verify the snapshot lands in the on-disk gallery

Run with: python -m pytest tests/test_cam23_snapshot_simulation.py -v -s
"""

import logging
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.config import load_config
from src.database import init_db
from src.detection.detector import Detection
from src.model.parkingslot import ParkingSlot
from src.models.state_machine import SlotState
from src.vehicle_registry.vehicle_registry import VehicleRegistry

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_test_crop() -> np.ndarray:
    """Create a synthetic vehicle crop (200x100 RGB image)."""
    crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    # Add some structure so Laplacian variance is > MIN_SHARPNESS
    crop[10:90, 10:190] = np.random.randint(100, 200, (80, 180, 3), dtype=np.uint8)
    return crop


def create_mock_detection(track_id: int = 1, bbox: tuple = (50, 50, 150, 200)) -> Detection:
    """Create a mock Detection object."""
    det = Detection(
        bbox=bbox,
        class_id=2,  # car
        confidence=0.95,
        track_id=track_id,
    )
    return det


def create_mock_park_entry_zone() -> ParkingSlot:
    """Create a mock Park_Entry zone (a simple trapezoid covering a ramp)."""
    from shapely.geometry import Polygon

    # Simple rectangle covering the detection area
    polygon = Polygon([(0, 50), (400, 50), (400, 250), (0, 250)])

    zone = MagicMock(spec=ParkingSlot)
    zone.id = "Park_Entry"
    zone.polygon = polygon
    zone.label = "B1 Entry Ramp - Park_Entry"
    return zone


class TestCAM23SnapshotSimulation(unittest.TestCase):
    """Simulate the full CAM-23 Park_Entry snapshot capture pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up database and VehicleRegistry once for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.gallery_dir = os.path.join(cls.temp_dir, "gallery")
        os.makedirs(cls.gallery_dir, exist_ok=True)

        # Load config and init DB
        cls.config = load_config("config.yaml")
        cls.config.output.snapshot_base_dir = cls.gallery_dir
        cls.db_manager = init_db(cls.config.database.DATABASE_URL)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up a fresh VehicleRegistry for each test."""
        self.registry = VehicleRegistry(
            matching_config=self.config.matching,
            gallery_store_base_dir=self.gallery_dir,
            db_manager=self.db_manager,
        )

    def test_01_simulate_gate_read_creates_pending_event(self):
        """Step 1: Simulate ANPR gate read, which creates a pending plate binding."""
        logger.info("=" * 70)
        logger.info("TEST 01: Gate read creates pending ANPR event")
        logger.info("=" * 70)

        plate = "TEST-0001"
        event_id = self.registry.register_anpr_event(
            plate=plate,
            direction="entry",
            camera_id="ANPR-Gate",
            timestamp=datetime.utcnow(),
        )

        self.assertIsNotNone(event_id)
        logger.info(f"✓ ANPR event created: event_id={event_id}, plate={plate}")
        self.assertTrue(hasattr(self.registry, "_pending_events"))
        self.assertEqual(len(self.registry._pending_events), 1)
        logger.info(f"✓ Pending events queue has 1 entry")

    def test_02_cam23_detection_opens_park_entry_candidate(self):
        """Step 2: CAM-23 detects vehicle in Park_Entry zone, opens candidate."""
        logger.info("=" * 70)
        logger.info("TEST 02: CAM-23 detection opens Park_Entry candidate")
        logger.info("=" * 70)

        # First, create the pending ANPR event
        plate = "TEST-0002"
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=datetime.utcnow()
        )

        # Now simulate CAM-23 detection
        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.camera_id, "CAM-23")
        self.assertEqual(candidate.status, "open")
        logger.info(f"✓ Park_Entry candidate opened: candidate_id={candidate.candidate_id}, track_id={track_id}")

    def test_03_snapshot_quality_scoring(self):
        """Step 3: CAM-23 captures crop and scores quality."""
        logger.info("=" * 70)
        logger.info("TEST 03: Snapshot quality scoring")
        logger.info("=" * 70)

        plate = "TEST-0003"
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=datetime.utcnow()
        )

        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)
        candidate_id = candidate.candidate_id

        # Simulate multiple frames with increasing quality
        crop1 = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop1, quality_score=30.0
        )

        with self.registry._lock:
            cand1 = self.registry._park_entry_candidates[candidate_id]
            quality1 = cand1.quality_score
        logger.info(f"✓ Frame 1: quality={quality1:.1f} (accepted, first frame)")
        self.assertEqual(quality1, 30.0)

        # Second frame with better quality
        crop2 = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop2, quality_score=50.0
        )

        with self.registry._lock:
            cand2 = self.registry._park_entry_candidates[candidate_id]
            quality2 = cand2.quality_score
        logger.info(f"✓ Frame 2: quality={quality2:.1f} (rejected, 50 > 30)")
        self.assertEqual(quality2, 50.0)

        # Third frame with worse quality (should be rejected)
        crop3 = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop3, quality_score=40.0
        )

        with self.registry._lock:
            cand3 = self.registry._park_entry_candidates[candidate_id]
            quality3 = cand3.quality_score
        logger.info(f"✓ Frame 3: quality={quality3:.1f} (rejected, 40 < 50)")
        self.assertEqual(quality3, 50.0)  # Should still be 50 from frame 2

    def test_04_fifo_bind_success(self):
        """Step 4: FIFO bind succeeds and resolves pending plate."""
        logger.info("=" * 70)
        logger.info("TEST 04: FIFO bind resolves plate within TTL")
        logger.info("=" * 70)

        plate = "TEST-0004"
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=datetime.utcnow()
        )

        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)
        candidate_id = candidate.candidate_id

        # Add a snapshot to the candidate
        crop = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop, quality_score=50.0
        )

        # Attempt FIFO bind
        bound_plate = self.registry.bind_next_pending_anpr_to_candidate(candidate_id)

        self.assertEqual(bound_plate, plate)
        logger.info(f"✓ FIFO bind succeeded: bound_plate={bound_plate} to candidate={candidate_id}")

    def test_05_fifo_bind_failure_on_ttl_expiry(self):
        """Step 5: FIFO bind fails if pending event is older than TTL."""
        logger.info("=" * 70)
        logger.info("TEST 05: FIFO bind fails when pending event > TTL")
        logger.info("=" * 70)

        plate = "TEST-0005"
        old_timestamp = datetime.utcnow()
        # Manually set the event timestamp to be older than TTL
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=old_timestamp
        )

        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)
        candidate_id = candidate.candidate_id

        crop = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop, quality_score=50.0
        )

        # Mock the clock to simulate time passing
        original_clock = self.registry._clock
        future_timestamp = old_timestamp.replace(microsecond=0) + \
            self.registry._timedelta_seconds(
                self.registry.PENDING_ANPR_BIND_TTL_SECONDS + 5  # 15 seconds later
            )
        self.registry._clock = lambda: future_timestamp

        try:
            # Attempt FIFO bind with a "stale" pending event
            bound_plate = self.registry.bind_next_pending_anpr_to_candidate(candidate_id)

            # Should fail because the event is older than TTL
            self.assertIsNone(bound_plate)
            logger.info(
                f"✓ FIFO bind failed as expected: pending event too old "
                f"(> {self.registry.PENDING_ANPR_BIND_TTL_SECONDS}s TTL)"
            )
        finally:
            self.registry._clock = original_clock

    def test_06_full_pipeline_snapshot_saved_to_disk(self):
        """Step 6: Full pipeline — snapshot is captured and saved to gallery."""
        logger.info("=" * 70)
        logger.info("TEST 06: Full pipeline — snapshot saved to disk")
        logger.info("=" * 70)

        plate = "TEST-FULL-06"
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=datetime.utcnow()
        )
        logger.info(f"✓ Step 1: ANPR gate read -> pending event for {plate}")

        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)
        candidate_id = candidate.candidate_id
        logger.info(f"✓ Step 2: CAM-23 detected vehicle -> opened candidate {candidate_id}")

        crop = create_test_crop()
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop, quality_score=60.0
        )
        logger.info(f"✓ Step 3: Captured and scored snapshot (quality=60.0)")

        bound_plate = self.registry.bind_next_pending_anpr_to_candidate(candidate_id)
        self.assertEqual(bound_plate, plate)
        logger.info(f"✓ Step 4: FIFO bind succeeded -> plate={bound_plate}")

        seeded_ok = self.registry.seed_gallery_from_park_entry(candidate_id, bound_plate)
        self.assertTrue(seeded_ok)
        logger.info(f"✓ Step 5: seed_gallery_from_park_entry succeeded")

        # Verify file was written to disk
        gallery_dir = os.path.join(self.gallery_dir, plate)
        self.assertTrue(os.path.isdir(gallery_dir), f"Gallery directory not created: {gallery_dir}")
        logger.info(f"✓ Gallery directory created: {gallery_dir}")

        # Check for crop image files
        crop_files = [f for f in os.listdir(gallery_dir) if f.startswith("crop_") and f.endswith(".jpg")]
        self.assertGreater(len(crop_files), 0, f"No crop images found in {gallery_dir}")
        logger.info(f"✓ Crop image saved: {crop_files[0]}")

        # Check for feature vector file
        feature_files = [f for f in os.listdir(gallery_dir) if f.startswith("crop_") and f.endswith(".npy")]
        self.assertGreater(len(feature_files), 0, f"No feature vectors found in {gallery_dir}")
        logger.info(f"✓ Feature vector saved: {feature_files[0]}")

        logger.info(f"✓✓✓ FULL PIPELINE SUCCESS: snapshot for {plate} captured and saved to gallery")

    def test_07_cam23_snapshot_includes_correct_metadata(self):
        """Step 7: Verify CAM-23 snapshot has correct metadata (camera_id, quality)."""
        logger.info("=" * 70)
        logger.info("TEST 07: Verify CAM-23 metadata in gallery")
        logger.info("=" * 70)

        plate = "TEST-META-07"
        self.registry.register_anpr_event(
            plate=plate, direction="entry", camera_id="ANPR-Gate", timestamp=datetime.utcnow()
        )

        track_id = 1
        candidate = self.registry.open_park_entry_candidate("CAM-23", track_id)
        candidate_id = candidate.candidate_id

        crop = create_test_crop()
        quality = 55.0
        self.registry.update_park_entry_candidate_snapshot(
            candidate_id, crop, quality_score=quality
        )

        bound_plate = self.registry.bind_next_pending_anpr_to_candidate(candidate_id)
        seeded_ok = self.registry.seed_gallery_from_park_entry(candidate_id, bound_plate)
        self.assertTrue(seeded_ok)

        # Load the meta.json to verify camera_id is CAM-23
        import json
        gallery_dir = os.path.join(self.gallery_dir, plate)
        meta_file = os.path.join(gallery_dir, "meta.json")

        self.assertTrue(os.path.exists(meta_file), f"meta.json not found in {gallery_dir}")
        with open(meta_file, "r") as f:
            meta = json.load(f)

        logger.info(f"✓ meta.json loaded: {meta}")

        # Check that references list has our CAM-23 entry
        refs = meta.get("references", [])
        cam23_refs = [r for r in refs if r.get("camera") == "CAM-23"]
        self.assertGreater(len(cam23_refs), 0, "CAM-23 reference not found in meta.json")
        logger.info(f"✓ CAM-23 reference found in metadata: {cam23_refs[0]}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_cam23_snapshot_simulation.py -v -s
    unittest.main(verbosity=2)
