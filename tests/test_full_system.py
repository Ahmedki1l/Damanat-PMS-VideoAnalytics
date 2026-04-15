"""
test_full_system.py — Comprehensive End-to-End Verification.

Vehicle Journey:
1. ANPR Entry (Plate 'TEST-E2E')
2. Park_Entry detection (Mock CAM_01)
3. B1_Entrance confirmation (Mock CAM_03) -> session confirmed via ReID
4. Parking sighting (Mock CAM_05, Slot-03)
5. DB Verification (Check if Slot-03 is occupied by TEST-E2E in SQL Server)
"""

import sys
import os
import cv2
import time
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('.')

from src.config import load_config
from src.database import init_db
from src.core.engine import ParkingEngine
from src.vehicle_registry import VehicleRegistry
from src.repositories.slot_status_repository import SlotStatusRepository
from src.services.parking_service import sync_slots_from_config
from src.models.state_machine import SlotEvent

def test_full_process():
    print("\n" + "="*60)
    print("  PHASE 4: FULL END-TO-END SYSTEM TEST")
    print("="*60)

    # 1. Initialization
    config = load_config("config.yaml")
    db_manager = init_db(config.database.url)
    db_manager.create_tables()
    
    registry = VehicleRegistry(image_dir="vehicle_images")
    engine = ParkingEngine(config, vehicle_registry=registry, db_manager=db_manager)

    # --- Step 0: Sync slots so Foreign Keys pass ---
    test_slot_id = "SLOT_TEST_E2E"
    with db_manager.SessionLocal() as db_session:
        from src.models.slot import ParkingSlot as PipelineSlot
        from shapely.geometry import Polygon
        # Mock a slot for the test
        mock_pipeline_slot = PipelineSlot(
            id=test_slot_id,
            label="E2E-TEST-SLOT",
            polygon=Polygon([(0,0), (1,0), (1,1), (0,1)]),
            zone_id="TEST-ZONE",
            zone_name="TEST-ZONE-NAME"
        )
        sync_slots_from_config(db_session, [mock_pipeline_slot], floor=1)
        print(f"[TEST] Mock slot {test_slot_id} created in DB.")

    # --- Step 0.1: Register DB Subscriber (The Bridge) ---
    def db_subscriber(event):
        from src.services.slot_status_service import log_vehicle_event
        from src.database import get_db
        if event.event_type in ("vehicle_parked", "slot_vacant"):
            try:
                db_gen = get_db()
                db_session = next(db_gen)
                log_vehicle_event(
                    db=db_session,
                    slot_id=event.slot_id,
                    plate=getattr(event, "plate_number", ""),
                    is_parked=(event.event_type == "vehicle_parked"),
                    camera_id=event.camera_id
                )
                db_session.close()
                print(f"[TEST] Successfully persisted {event.event_type} to DB.")
            except StopIteration: pass
            except Exception as e:
                print(f"[TEST ERROR] DB Persistence failed: {e}")

    engine.event_bus.subscribe(db_subscriber)

    # 2. Prepare Sample Image for ReID
    img_path = "vehicle_images/NDD-4141_CAM01_bound_20260413_080606.jpg"
    if not os.path.exists(img_path):
        print(f"[SKIP] Sample image '{img_path}' not found.")
        return
    img = cv2.imread(img_path)

    # --- STEP 1: ANPR Ingestion ---
    print("\n[STEP 1] ANPR: Vehicle 'TEST-E2E' enters the facility.")
    registry.register_anpr_event("TEST-E2E", "entry")

    # --- STEP 2: Gate Sighting (CAM_01) ---
    print("[STEP 2] CAM_01: Vehicle seen at Park_Entry.")
    cand = registry.open_park_entry_candidate("CAM_01", 500)
    registry.update_park_entry_candidate_snapshot(cand.candidate_id, img, 1000.0)
    registry.bind_next_pending_anpr_to_candidate(cand.candidate_id)

    # --- STEP 3: Entrance Confirmation (CAM_03) ---
    print("[STEP 3] CAM_03: Crossing virtual line at B1 Entrance.")
    registry.confirm_at_b1_entrance("CAM_03", 600, img)
    
    plate_confirmed = registry.get_plate_for_track("CAM_03", 600)
    print(f"  -> Identity confirmed as: {plate_confirmed}")

    # --- STEP 4: Parking sighting ---
    print(f"[STEP 4] CAM_05: Vehicle enters {test_slot_id}.")
    
    # Global ReID (Phase 3)
    class MockDet:
        def __init__(self, tid, bbox):
            self.track_id = tid
            self.bbox = bbox
    
    engine._crop_detection = lambda f, d: img
    engine._process_global_tracking("CAM_05", img, [MockDet(700, [0,0,10,10])])
    plate_at_slot = registry.get_plate_for_track("CAM_05", 700)

    print(f"  -> Identified plate at slot: {plate_at_slot}")

    # Emit the "vehicle_parked" event
    mock_event = SlotEvent(
        event_type="vehicle_parked",
        slot_id=test_slot_id,
        track_id=700,
        timestamp=datetime.now().isoformat(),
        plate_number=plate_at_slot or ""
    )
    mock_event.camera_id = "CAM_05"
    engine.event_bus.emit(mock_event)

    # --- STEP 5: Database Verification ---
    print("\n[STEP 5] Database Verification (SQL Server)...")
    time.sleep(1) 
    
    with db_manager.SessionLocal() as db_session:
        latest = SlotStatusRepository.get_latest_by_plate(db_session, "TEST-E2E")
        
        if latest and latest.slot_id == test_slot_id and latest.status == "occupied":
            print(f"  -> SUCCESS: DB shows TEST-E2E is OCCUPYING {test_slot_id}")
        else:
            print(f"  -> FAILURE: DB record mismatch. Got: {latest}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_full_process()
