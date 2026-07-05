"""Smoke test for the plate-keyed in-memory guard in try_link_to_slot.

Pre-existing tests (test_duplicate_slot_fix.py) drive try_link_to_slot
through confirm_at_b1_entrance, which triggers the ReID matcher load
(torchreid). That model init segfaults on some local environments
(Python 3.12 + torch + Mac ARM), so neither the existing test nor the
plate-keyed test added in the main file can run in a fresh dev env.

This smoke test bypasses the matcher entirely: it manually constructs
two VehicleSessions for the same plate, injects them into the
registry's internal dicts, and calls try_link_to_slot directly.
That's exactly the code path my plate-keyed guard guards, with zero
dependency on torch / ultralytics / OpenVINO.
"""
import unittest
import uuid
from datetime import datetime, timedelta

from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import VehicleSession


class TestPlateKeyedGuardSmoke(unittest.TestCase):
    PLATE = "94-LNV"
    CAMERA_ID = "CAM_03"  # not in the identity-disabled set

    def setUp(self):
        self.registry = VehicleRegistry(image_dir="tests/test_images")

    def _make_session(self, *, parked_slot=None, age_minutes=0):
        sid = f"sess-{uuid.uuid4().hex[:8]}"
        now = datetime.now()
        session = VehicleSession(
            session_id=sid,
            plate=self.PLATE,
            first_seen_at=now - timedelta(minutes=age_minutes),
            last_seen_at=now - timedelta(minutes=age_minutes),
            last_seen_camera=self.CAMERA_ID,
            status="parked" if parked_slot else "confirmed",
            linked_slot=parked_slot,
            linked_slot_name=parked_slot,
            linked_camera=self.CAMERA_ID if parked_slot else None,
            linked_floor="B1" if parked_slot else None,
            linked_at=now - timedelta(minutes=age_minutes) if parked_slot else None,
        )
        self.registry._sessions[sid] = session
        if parked_slot:
            self.registry._parked[parked_slot] = session
        return session

    def test_plate_keyed_release_when_fresh_session_links_to_new_slot(self):
        # 1. Stale session for the plate is already parked at Slot_A.
        stale = self._make_session(parked_slot="Slot_A", age_minutes=10)
        self.assertIn("Slot_A", self.registry._parked)

        # 2. Fresh session for the SAME plate (distinct session_id) is
        #    mapped to a live track and asked to bind Slot_B.
        fresh = self._make_session(age_minutes=0)
        track_id = 9999
        self.registry._track_session_map[(self.CAMERA_ID, track_id)] = fresh.session_id

        result = self.registry.try_link_to_slot(
            slot_id="Slot_B",
            slot_name="Slot B",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id=self.CAMERA_ID,
            floor="B1",
            track_id=track_id,
            timestamp=datetime.now(),
        )

        # 3. The plate-keyed guard should have released Slot_A and bound Slot_B.
        self.assertIsNone(
            self.registry._parked.get("Slot_A"),
            "Slot_A should have been released by the plate-keyed guard",
        )
        self.assertIs(
            self.registry._parked.get("Slot_B"),
            fresh,
            "Slot_B should be bound to the fresh session",
        )
        self.assertEqual(
            len(self.registry._parked), 1,
            "Only one slot should hold the plate after the guard runs",
        )
        # The stale session's link fields must be cleared so it can't
        # ghost-bind on a subsequent frame.
        self.assertIsNone(stale.linked_slot)
        self.assertEqual(stale.status, "confirmed")
        self.assertEqual(fresh.linked_slot, "Slot_B")
        self.assertEqual(fresh.status, "parked")
        # try_link_to_slot returns the plate on success.
        self.assertEqual(result, self.PLATE)

    def test_same_session_move_still_releases_prior_slot(self):
        # Regression: the pre-existing one-slot-per-session rule
        # (vehicle_registry_identity.py:1326-1337) keeps working when the
        # same session moves between slots — the new plate-keyed block
        # must not break this.
        session = self._make_session(parked_slot="Slot_A", age_minutes=0)
        track_id = 8888
        self.registry._track_session_map[(self.CAMERA_ID, track_id)] = session.session_id

        result = self.registry.try_link_to_slot(
            slot_id="Slot_B",
            slot_name="Slot B",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id=self.CAMERA_ID,
            floor="B1",
            track_id=track_id,
            timestamp=datetime.now(),
        )

        self.assertIsNone(self.registry._parked.get("Slot_A"))
        self.assertIs(self.registry._parked.get("Slot_B"), session)
        self.assertEqual(len(self.registry._parked), 1)
        self.assertEqual(result, self.PLATE)

    def test_identity_disabled_camera_short_circuits(self):
        # Regression: ground-floor cameras bypass try_link_to_slot entirely
        # (gated by floor via is_reid_disabled_floor), so the new guard cannot
        # fire from those cameras.
        stale = self._make_session(parked_slot="Slot_A", age_minutes=10)
        fresh = self._make_session(age_minutes=0)
        track_id = 7777
        self.registry._track_session_map[("CAM-01", track_id)] = fresh.session_id

        result = self.registry.try_link_to_slot(
            slot_id="Slot_B",
            slot_name="Slot B",
            zone_id="Z1",
            zone_name="Zone 1",
            camera_id="CAM-01",
            floor="Ground",
            track_id=track_id,
            timestamp=datetime.now(),
        )

        self.assertIsNone(result)
        # Stale binding untouched; nothing new bound.
        self.assertIs(self.registry._parked.get("Slot_A"), stale)
        self.assertNotIn("Slot_B", self.registry._parked)


if __name__ == "__main__":
    unittest.main()
