"""A named slot's intrusion verdict belongs to the engine, not to occupancy.

`report_alert` with no explicit alert_type derives "vehicle_intrusion" from
`reservation_type == "EMPLOYEE"` alone (alert_service.py:66) — it never looks at
WHO parked. So calling it from the occupancy path raised an intrusion for every
car on a named slot, before identity existed, with an empty plate.

Measured 2026-07-20: 80 alerts on named slots, ALL vehicle_intrusion, ALL
plateless — every executive flagged as an intruder in his own slot — while the
deferred ownership path fired 0 times.

The rule: defer ONLY when there is nothing to judge with. A plate present means
the engine already ruled (engine_runtime.py:2570 returns vehicle_intrusion only
for a proven non-owner), so that alert must still be raised.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services import alert_service, slot_status_service


def _slot(reservation_type, reserved_for=None):
    return SimpleNamespace(
        slot_id="B12 CCO", slot_name="B12 CCO", zone_id="CAM-24", zone_name="CAM-24",
        floor="B1", is_available=True, current_plate=None, plate_confidence=0.0,
        plate_locked=False, plate_locked_at=None,
        reservation_type=reservation_type, reserved_for=reserved_for,
    )


def _log(slot, plate, is_parked=True):
    created = SimpleNamespace(time=None)
    with patch.object(slot_status_service.ParkingSlotRepository, "get_by_id",
                      return_value=slot), \
         patch.object(slot_status_service.SlotStatusRepository,
                      "get_latest_by_slot", return_value=None), \
         patch.object(slot_status_service.SlotStatusRepository, "create",
                      return_value=created), \
         patch.object(alert_service, "report_alert",
                      return_value=SimpleNamespace(id=1)) as report, \
         patch.object(alert_service, "resolve_alert", return_value=None), \
         patch.object(alert_service, "auto_resolve_slot_violation_alerts",
                      return_value=[]), \
         patch.object(slot_status_service.pms_api_client, "bind_slot_session"), \
         patch.object(slot_status_service.pms_api_client, "unbind_slot_session"):
        slot_status_service.log_vehicle_event(
            MagicMock(), slot.slot_id, plate, is_parked, camera_id="CAM-24"
        )
    return report


class TestNamedSlotAlertDeferral(unittest.TestCase):
    def test_named_slot_without_plate_defers(self):
        """THE regression: no plate yet -> the engine decides, we stay silent."""
        report = _log(_slot("EMPLOYEE", "CCO"), plate="")
        report.assert_not_called()

    def test_named_slot_with_plate_still_alerts(self):
        """Plate present -> the engine already proved a non-owner. Keep the alert,
        and keep its plate."""
        report = _log(_slot("EMPLOYEE", "CCO"), plate="RGR-6466")
        report.assert_called_once()
        self.assertEqual(report.call_args.args[2], "RGR-6466")

    def test_violation_zone_still_alerts_immediately(self):
        """A violation zone is a violation whoever parked — unchanged."""
        report = _log(_slot("GENERAL"), plate="")
        report.assert_called_once()

    def test_special_needs_still_alerts_immediately(self):
        """Special-needs is a violation regardless of identity — unchanged."""
        report = _log(_slot("SPECIAL"), plate="")
        report.assert_called_once()

    def test_vacate_still_resolves_on_a_named_slot(self):
        """Deferring the raise must not stop the alert being resolved on exit."""
        slot = _slot("EMPLOYEE", "CCO")
        created = SimpleNamespace(time=None)
        with patch.object(slot_status_service.ParkingSlotRepository, "get_by_id",
                          return_value=slot), \
             patch.object(slot_status_service.SlotStatusRepository,
                          "get_latest_by_slot", return_value=None), \
             patch.object(slot_status_service.SlotStatusRepository, "create",
                          return_value=created), \
             patch.object(alert_service, "resolve_alert",
                          return_value=SimpleNamespace(id=7)) as resolve, \
             patch.object(alert_service, "auto_resolve_slot_violation_alerts",
                          return_value=[]), \
             patch.object(slot_status_service.pms_api_client, "unbind_slot_session"):
            slot_status_service.log_vehicle_event(
                MagicMock(), slot.slot_id, None, False, camera_id="CAM-24"
            )
        resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
