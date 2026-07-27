"""`reserved_slot_unidentified` is recorded but not pushed to the alert stream.

That alert is the "nobody could name this car" fallback, not a proven intrusion
(see AlertsConfig.reserved_slot_identity_timeout_s). On a slot that can never be
identified — B1_CRO has OCR disabled outright via matching.slot_no_plate_view —
it fires on essentially every occupancy, and an alert panel that is permanently
red gets ignored, including for the proven intrusions next to it.

So the gate is on the STREAM, not on report_alert: the row, the snapshot and the
REST history are all untouched. These tests pin that split, because "just stop
raising it" would look equivalent and quietly delete the evidence.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.services import alert_service


class TestNotificationSuppression(unittest.TestCase):
    def setUp(self):
        self._original = alert_service._SUPPRESSED_NOTIFICATION_TYPES
        self.addCleanup(
            setattr, alert_service, "_SUPPRESSED_NOTIFICATION_TYPES", self._original
        )

    def test_reserved_slot_unidentified_suppressed_by_default(self):
        self.assertTrue(
            alert_service.notification_suppressed("reserved_slot_unidentified")
        )

    def test_proven_intrusion_still_notifies(self):
        for alert_type in (
            "vehicle_intrusion",
            "special_needs_violation",
            "named_slot_violation",
            "vehicle_violation",
        ):
            with self.subTest(alert_type=alert_type):
                self.assertFalse(alert_service.notification_suppressed(alert_type))

    def test_empty_alert_type_is_never_suppressed(self):
        self.assertFalse(alert_service.notification_suppressed(None))
        self.assertFalse(alert_service.notification_suppressed(""))

    def test_configure_alerts_replaces_the_set(self):
        alert_service.configure_alerts(
            enable_restricted_zone_alerts=True,
            suppressed_notification_types=("vehicle_violation", "  ", ""),
        )
        self.assertTrue(alert_service.notification_suppressed("vehicle_violation"))
        self.assertFalse(
            alert_service.notification_suppressed("reserved_slot_unidentified")
        )

    def test_configure_alerts_can_disable_suppression_entirely(self):
        alert_service.configure_alerts(
            enable_restricted_zone_alerts=True, suppressed_notification_types=()
        )
        self.assertFalse(
            alert_service.notification_suppressed("reserved_slot_unidentified")
        )

    def test_omitting_the_kwarg_leaves_the_set_alone(self):
        """Legacy single-kwarg callers must not silently unmute everything."""
        alert_service.configure_alerts(enable_restricted_zone_alerts=False)
        self.assertTrue(
            alert_service.notification_suppressed("reserved_slot_unidentified")
        )


class TestSuppressedAlertIsStillRecorded(unittest.TestCase):
    """Suppression must not reach report_alert — the row is the audit trail."""

    def test_report_alert_still_writes_the_row(self):
        db = MagicMock()
        created = MagicMock()

        with patch.object(alert_service, "check_slot_restricted", return_value=True), \
             patch.object(alert_service, "_restricted_zone_alerts_enabled", return_value=True), \
             patch.object(alert_service.ParkingSlotRepository, "get_by_id", return_value=None), \
             patch.object(alert_service.AlertRepository, "get_active_by_slot", return_value=None), \
             patch.object(alert_service.AlertRepository, "create", return_value=created) as create:
            result = alert_service.report_alert(
                db,
                "B1_CRO",
                None,
                camera_id="CAM-24",
                severity="warning",
                snapshot_path="snap.jpg",
                alert_type="reserved_slot_unidentified",
            )

        self.assertIs(result, created)
        self.assertTrue(create.called)
        written = create.call_args[0][1]
        self.assertEqual(written.alert_type, "reserved_slot_unidentified")
        self.assertEqual(written.snapshot_path, "snap.jpg")
        self.assertFalse(written.is_resolved)


if __name__ == "__main__":
    unittest.main()
