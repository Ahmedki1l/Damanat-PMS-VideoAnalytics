import unittest
from datetime import datetime
import numpy as np
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import PendingANPREvent, ParkEntryCandidate

class TestDuplicateSlotFix(unittest.TestCase):
    def setUp(self):
        self.registry = VehicleRegistry(image_dir="tests/test_images")
        self.camera_id = "CAM_03"
        self.plate = "94-LNV"
        
    def test_one_slot_per_vehicle_enforcement(self):
        # 1. Register ANPR entry
        event = self.registry.register_anpr_event(self.plate, "entry")
        
        # 2. Simulate B1 Entrance confirmation
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        track_id = 10
        
        # Seed a candidate first
        candidate = self.registry.open_park_entry_candidate(self.camera_id, track_id)
        self.registry.update_park_entry_candidate_snapshot(candidate.candidate_id, dummy_img, 1.0)
        self.registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        
        # Confirm at B1
        confirmed_plate = self.registry.confirm_at_b1_entrance(
            self.camera_id, track_id, dummy_img, similarity_threshold=0.1
        )
        self.assertEqual(confirmed_plate, self.plate)
        
        # 3. Link to Slot_A
        self.registry.try_link_to_slot(
            slot_id="Slot_A",
            slot_name="Slot A",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id=self.camera_id,
            floor="B1",
            track_id=track_id,
            timestamp=datetime.now()
        )
        
        # Verify Slot_A is occupied
        self.assertEqual(self.registry.get_slot_plate("Slot_A"), self.plate)
        self.assertEqual(len(self.registry.get_all_parked()), 1)
        
        # 4. Link same session to Slot_B (simulating a move or ReID update)
        # Note: In real engine, this might be a new track_id, but here we reuse same for simplicity
        # or use a new track that matches same session.
        
        self.registry.try_link_to_slot(
            slot_id="Slot_B",
            slot_name="Slot B",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id=self.camera_id,
            floor="B1",
            track_id=track_id,
            timestamp=datetime.now()
        )
        
        # 5. VERIFY: Slot_A should be empty, Slot_B should have the plate
        self.assertIsNone(self.registry.get_slot_plate("Slot_A"), "Slot_A should have been unlinked")
        self.assertEqual(self.registry.get_slot_plate("Slot_B"), self.plate)
        self.assertEqual(len(self.registry.get_all_parked()), 1, "There should only be ONE parked entry")

if __name__ == "__main__":
    unittest.main()
