import unittest
import uuid
from datetime import datetime, timedelta
import numpy as np
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import (
    PendingANPREvent,
    ParkEntryCandidate,
    VehicleSession,
)

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

    def test_one_slot_per_plate_across_sessions(self):
        """If two DIFFERENT VehicleSessions exist for the same plate (e.g. the
        120 s duplicate-session guard missed because the prior session was
        created longer ago, or _parked was rebuilt from observation after a
        restart), linking the newer session to a slot must release the older
        session's slot in-memory."""
        # 1. Inject a stale, parked session for the plate into Slot_A.
        # The session is intentionally old (>120 s) so it doesn't match the
        # duplicate-session reuse path; it represents an orphaned binding
        # that the new plate-keyed guard must clean up.
        stale_session_id = f"stale-{uuid.uuid4().hex[:8]}"
        stale = VehicleSession(
            session_id=stale_session_id,
            plate=self.plate,
            first_seen_at=datetime.now() - timedelta(minutes=10),
            last_seen_at=datetime.now() - timedelta(minutes=10),
            last_seen_camera=self.camera_id,
            status="parked",
            linked_slot="Slot_A",
            linked_slot_name="Slot A",
            linked_camera=self.camera_id,
            linked_floor="B1",
            linked_at=datetime.now() - timedelta(minutes=10),
        )
        self.registry._sessions[stale_session_id] = stale
        self.registry._parked["Slot_A"] = stale

        # 2. Drive the normal flow for a fresh session with the same plate.
        self.registry.register_anpr_event(self.plate, "entry")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        track_id = 11
        candidate = self.registry.open_park_entry_candidate(self.camera_id, track_id)
        self.registry.update_park_entry_candidate_snapshot(
            candidate.candidate_id, dummy_img, 1.0
        )
        self.registry.bind_next_pending_anpr_to_candidate(candidate.candidate_id)
        confirmed_plate = self.registry.confirm_at_b1_entrance(
            self.camera_id, track_id, dummy_img, similarity_threshold=0.1
        )
        self.assertEqual(confirmed_plate, self.plate)

        # Sanity: the duplicate-session guard at confirm_anpr_session_directly
        # only fires within 120 s, and our stale was injected at -10 min, so
        # confirm_at_b1_entrance is free to either create a new session or
        # reuse an existing one. The plate-keyed guard in try_link_to_slot
        # must defend against BOTH outcomes; this test cares only about the
        # case where the new linked session has a session_id distinct from
        # the stale one.
        fresh_session_id = self.registry._track_session_map.get(
            (self.camera_id, track_id)
        )
        self.assertIsNotNone(fresh_session_id)

        # 3. Link the fresh session to Slot_B. The plate-keyed guard must
        # release Slot_A even though the fresh session never previously
        # owned it.
        self.registry.try_link_to_slot(
            slot_id="Slot_B",
            slot_name="Slot B",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id=self.camera_id,
            floor="B1",
            track_id=track_id,
            timestamp=datetime.now(),
        )

        # 4. VERIFY: Slot_A is empty, Slot_B holds the plate, only one
        # parked entry total, and the stale session's linked_slot was
        # cleared so it can't ghost-bind on a later frame.
        if fresh_session_id == stale_session_id:
            # The duplicate-session guard reused the stale; in this case the
            # existing one-slot-per-vehicle code at vehicle_registry_identity
            # already covers the release. Skip the plate-keyed assertions —
            # this is the test_one_slot_per_vehicle_enforcement scenario.
            self.skipTest(
                "duplicate-session guard reused the stale session; covered "
                "by test_one_slot_per_vehicle_enforcement"
            )
        self.assertIsNone(
            self.registry.get_slot_plate("Slot_A"),
            "Slot_A should have been released by the plate-keyed guard",
        )
        self.assertEqual(self.registry.get_slot_plate("Slot_B"), self.plate)
        self.assertEqual(
            len(self.registry.get_all_parked()),
            1,
            "There should only be ONE parked entry for the plate",
        )
        self.assertIsNone(stale.linked_slot, "Stale session's linked_slot must be cleared")
        self.assertEqual(stale.status, "confirmed")


if __name__ == "__main__":
    unittest.main()
