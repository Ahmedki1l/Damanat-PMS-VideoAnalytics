"""
The exit janitor must not delete a car that is demonstrably inside.

What happened on 2026-07-11 — this single bug invalidated every drive test of the day:

    17:48:28  ANPR Exit:  DJS-7842        (VA sees it leave)
    17:49:06  ANPR Entry: DJS-7842        (VA sees it come back)
    17:50:23  [exit_janitor] purged in-memory state for plate=DJS-7842
              (parking_sessions.status=closed)

The car was still driving to its slot. Its identity — session, plate, gallery links —
was deleted out from under it 77 seconds after it drove through the gate. Every
downstream mechanism (the self-competition merge, acquisition by elimination, the
parked-pose capture) was then operating on a car that no longer existed, which is why
they all looked broken.

TWO things were wrong:

1. The guard was TIME-based: "did this plate re-enter within REENTRY_DB_GRACE_SECONDS
   (60s)?" The car took 77s to reach its slot. A timer cannot express the question we
   actually care about.

2. Worse, PMS-AI had stopped inserting rows entirely. The newest row for the plate was
   a *closed* one from 17:00 — so NO grace value, however large, could ever have saved
   this car. The janitor would have purged it forever.

The rule is ORDERING, not elapsed time: if VA's own ANPR saw the car enter AFTER the
row's entry_time, that row describes an OLDER visit and says nothing about the car
currently inside. VA's gate reads are authoritative for "is it in"; the DB is
authoritative only for a visit it actually recorded.
"""

import unittest
from datetime import datetime, timedelta

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry


class TestStaleClosedRow(unittest.TestCase):
    def setUp(self):
        self.reg = VehicleRegistry(matching_config=MatchingConfig())
        self.now = self.reg._clock()

    def _saw(self, plate, *, exit_s=None, entry_s=None):
        if exit_s is not None:
            self.reg._last_anpr_exit_at[plate] = self.now - timedelta(seconds=exit_s)
        if entry_s is not None:
            self.reg._last_anpr_entry_at[plate] = self.now - timedelta(seconds=entry_s)

    # ---- the accessor the janitor relies on ------------------------------------

    def test_last_anpr_entry_at_reports_vas_own_gate_observation(self):
        self._saw("DJS-7842", exit_s=115, entry_s=77)
        got = self.reg.last_anpr_entry_at("DJS-7842")
        self.assertIsNotNone(got)
        self.assertAlmostEqual((self.now - got).total_seconds(), 77, delta=1)

    def test_unknown_plate_has_no_entry(self):
        self.assertIsNone(self.reg.last_anpr_entry_at("NOPE-1"))
        self.assertIsNone(self.reg.last_anpr_entry_at(None))

    # ---- the decision the janitor now makes ------------------------------------
    # purge  <=>  NOT (va_entry_at > row_entry_time)

    def _would_purge(self, plate, row_entry_time):
        va = self.reg.last_anpr_entry_at(plate)
        return not (va is not None and row_entry_time is not None and va > row_entry_time)

    def test_the_real_failure_a_row_closed_BEFORE_the_car_drove_back_in(self):
        """THE REGRESSION. Row's visit began at 17:00; VA watched the car re-enter at
        17:49. The row is stale — it cannot possibly describe the car now inside."""
        self._saw("DJS-7842", exit_s=115, entry_s=77)          # VA: entered 77s ago
        row_entry_time = self.now - timedelta(minutes=49)      # DB: an OLD visit
        self.assertFalse(self._would_purge("DJS-7842", row_entry_time))

    def test_it_survives_even_when_the_grace_window_has_long_expired(self):
        """The old 60s grace is irrelevant now: a car can take as long as it likes to
        reach its slot, and PMS-AI can lag indefinitely."""
        self._saw("DJS-7842", exit_s=3700, entry_s=3600)       # entered an HOUR ago
        row_entry_time = self.now - timedelta(hours=3)
        self.assertFalse(self._would_purge("DJS-7842", row_entry_time))

    def test_a_car_that_genuinely_left_after_its_last_entry_IS_still_purged(self):
        """The janitor must keep working. The row's visit STARTED after VA's last gate
        entry, so it describes a later visit that has since closed — the car is out."""
        self._saw("DJS-7842", entry_s=7200)                    # entered 2h ago
        row_entry_time = self.now - timedelta(minutes=30)      # a LATER visit, now closed
        self.assertTrue(self._would_purge("DJS-7842", row_entry_time))

    def test_a_plate_va_never_saw_enter_is_purged(self):
        """No gate read on record => nothing to contradict the DB. Trust the DB."""
        row_entry_time = self.now - timedelta(minutes=5)
        self.assertTrue(self._would_purge("GHOST-1", row_entry_time))


if __name__ == "__main__":
    unittest.main()
