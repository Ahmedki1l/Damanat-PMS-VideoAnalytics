"""Tests for ANPR burst coalescing (last read wins within 3s).

When an ANPR camera emits several entry reads in quick succession for the same
passing car, they are coalesced into one pending event and the LATEST plate
wins — even if the plate text differs (an OCR refinement). Grouping is by the
same ANPR camera within a sliding ``ANPR_COALESCE_SECONDS`` window; the first
read binds immediately and later reads inside the window overwrite it.

Driven through ``register_anpr_event`` only (entry direction needs no model
load), with an injected clock.
"""
import unittest
from datetime import datetime, timedelta

from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_core import ANPR_COALESCE_SECONDS
from src.vehicle_registry.vehicle_registry_models import VehicleSession


class _FakeClock:
    def __init__(self):
        self.now = datetime(2024, 1, 1, 12, 0, 0)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class TestAnprBurstCoalescing(unittest.TestCase):
    CAM = "ANPR-1"

    def setUp(self):
        self.clock = _FakeClock()
        self.registry = VehicleRegistry(image_dir="tests/test_images", clock=self.clock)

    def _entry(self, plate, camera_id=None):
        return self.registry.register_anpr_event(
            plate, "entry", camera_id=camera_id if camera_id is not None else self.CAM
        )

    def test_burst_coalesces_to_one_event_last_plate_wins(self):
        e1 = self._entry("ABC1")
        self.clock.advance(1.5)
        e2 = self._entry("ABC2")
        self.clock.advance(1.0)  # 1.0s after e2 — still in the sliding window
        e3 = self._entry("ABC3")

        # All three reads collapse onto the same pending event.
        self.assertEqual(e2.event_id, e1.event_id)
        self.assertEqual(e3.event_id, e1.event_id)
        self.assertEqual(e3.plate, "ABC3")
        self.assertEqual(len(self.registry._pending_event_order), 1)

    def test_overwrite_propagates_to_session_created_from_first_read(self):
        e1 = self._entry("ABC1")
        # A session was already confirmed from the first read (bind-now).
        session = VehicleSession(
            session_id="s1",
            plate="ABC1",
            event_id=e1.event_id,
            status="confirmed",
        )
        self.registry._sessions["s1"] = session
        e1.session_id = "s1"

        self.clock.advance(1.0)
        self._entry("ABC2")  # corrected read inside the window

        self.assertEqual(session.plate, "ABC2")
        self.assertEqual(self.registry._pending_events[e1.event_id].plate, "ABC2")

    def test_corrected_read_after_confirmation_rebinds_session(self):
        # bind-now: the first read is confirmed into a session before the
        # corrected read arrives inside the 3s window. The corrected plate must
        # rebind that session rather than spawn a separate pending event.
        e1 = self._entry("ABC1")
        e1.status = "confirmed"  # already bound to a session
        session = VehicleSession(
            session_id="s1",
            plate="ABC1",
            event_id=e1.event_id,
            status="confirmed",
        )
        self.registry._sessions["s1"] = session
        e1.session_id = "s1"

        self.clock.advance(1.0)
        result = self._entry("ABC2")

        self.assertEqual(result.event_id, e1.event_id)
        self.assertEqual(len(self.registry._pending_event_order), 1)
        self.assertEqual(session.plate, "ABC2")

    def test_read_outside_window_starts_new_event(self):
        e1 = self._entry("ABC1")
        self.clock.advance(ANPR_COALESCE_SECONDS + 3)
        e_new = self._entry("XYZ9")

        self.assertNotEqual(e_new.event_id, e1.event_id)
        self.assertEqual(len(self.registry._pending_event_order), 2)

    def test_same_plate_refire_still_deduped_outside_window(self):
        e1 = self._entry("ABC1")
        self.clock.advance(ANPR_COALESCE_SECONDS + 2)  # past burst window
        again = self._entry("ABC1")  # identical plate, within 600s

        # Existing same-plate/600s dedup still holds: no second event.
        self.assertEqual(again.event_id, e1.event_id)
        self.assertEqual(len(self.registry._pending_event_order), 1)

    def test_different_camera_does_not_coalesce(self):
        e1 = self._entry("ABC1", camera_id="ANPR-1")
        self.clock.advance(1.0)
        e2 = self._entry("ABC2", camera_id="ANPR-2")

        # Same time window but different camera → distinct cars, distinct events.
        self.assertNotEqual(e2.event_id, e1.event_id)
        self.assertEqual(len(self.registry._pending_event_order), 2)


if __name__ == "__main__":
    unittest.main()
