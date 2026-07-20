"""Occupancy events may not erase a slot's identity.

`log_vehicle_event` mirrors the occupancy event onto both identity twins
(`parking_slots.current_plate` and `slot_status.plate_number`). The trap it fell into:
an occupancy event carries NO plate by design — identity (OCR/ReID) runs *after*
occupancy is published, so `plate` is empty on essentially every `vehicle_parked`
(see engine_runtime._get_slot_alert_type). Mirroring that empty value turned every
occupancy write into a plate ERASER, and because `_resolve_locked_plate` freezes on
`is_plate_locked()` the slot never rewrote it — it read NULL for the rest of the
occupancy. Models the real B7_CHRO / "B12 CCO" / "B13 COO" case.

Only VACATING clears identity — the one transition that proves the old plate is gone.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

from src.services import slot_status_service


def _slot(**kw):
    base = dict(
        slot_id="B7_CHRO",
        slot_name="B7 CHRO",
        zone_id="CAM-22",
        zone_name="CAM-22",
        floor="B1",
        is_available=True,
        current_plate=None,
        plate_confidence=0.0,
        plate_locked=False,
        plate_locked_at=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _log(slot, plate, is_parked):
    """Drive log_vehicle_event against `slot` with every collaborator stubbed out."""
    created = types.SimpleNamespace(time=None)
    with patch.object(
        slot_status_service.ParkingSlotRepository, "get_by_id", return_value=slot
    ), patch.object(
        slot_status_service.SlotStatusRepository, "get_latest_by_slot", return_value=None
    ), patch.object(
        slot_status_service.SlotStatusRepository, "create", return_value=created
    ) as create_mock, patch.object(
        slot_status_service.alert_service, "report_alert", return_value=None
    ), patch.object(
        slot_status_service.alert_service, "resolve_alert", return_value=None
    ), patch.object(
        slot_status_service.alert_service,
        "auto_resolve_slot_violation_alerts",
        return_value=[],
    ), patch.object(
        slot_status_service.pms_api_client, "bind_slot_session"
    ), patch.object(
        slot_status_service.pms_api_client, "unbind_slot_session"
    ):
        slot_status_service.log_vehicle_event(
            MagicMock(), slot.slot_id, plate, is_parked, camera_id="CAM-22"
        )
    return create_mock.call_args[0][1]  # the SlotStatus row that was written


class TestOccupancyDoesNotEraseIdentity(unittest.TestCase):
    def test_plateless_occupied_event_keeps_a_bound_plate(self):
        """The regression: an identified, locked slot re-emitting `occupied`."""
        slot = _slot(
            is_available=False,
            current_plate="ZZR-1372",
            plate_confidence=1.0,
            plate_locked=True,
        )

        row = _log(slot, plate="", is_parked=True)

        self.assertEqual(slot.current_plate, "ZZR-1372")
        self.assertTrue(slot.plate_locked)
        # The twins agree: the status row carries the surviving identity too.
        self.assertEqual(row.plate_number, "ZZR-1372")
        self.assertEqual(row.status, "occupied")

    def test_occupied_event_with_a_plate_still_writes_it(self):
        slot = _slot(is_available=False)

        row = _log(slot, plate="RGR-6466", is_parked=True)

        self.assertEqual(slot.current_plate, "RGR-6466")
        self.assertEqual(row.plate_number, "RGR-6466")

    def test_occupied_event_upgrades_an_existing_plate(self):
        """A later event that DOES carry identity overrides the older one."""
        slot = _slot(is_available=False, current_plate="OLD-0001")

        row = _log(slot, plate="NEW-0002", is_parked=True)

        self.assertEqual(slot.current_plate, "NEW-0002")
        self.assertEqual(row.plate_number, "NEW-0002")

    def test_vacate_clears_identity_and_the_lock(self):
        """The one transition allowed to erase — and it must clear the lock too,
        or the row survives as `current_plate=NULL, plate_locked=True`."""
        slot = _slot(
            is_available=False,
            current_plate="ZZR-1372",
            plate_confidence=1.0,
            plate_locked=True,
            plate_locked_at="2026-07-20T03:00:00",
        )

        row = _log(slot, plate=None, is_parked=False)

        self.assertTrue(slot.is_available)
        self.assertIsNone(slot.current_plate)
        self.assertFalse(slot.plate_locked)
        self.assertIsNone(slot.plate_locked_at)
        self.assertEqual(slot.plate_confidence, 0.0)
        self.assertIsNone(row.plate_number)
        self.assertEqual(row.status, "available")


if __name__ == "__main__":
    unittest.main()
