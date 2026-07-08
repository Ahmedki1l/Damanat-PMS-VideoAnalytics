"""Regression: a STALE unconsumed pending ANPR plate (residue from a lingering /
mis-read previous car) must stop being FIFO-bindable to a new live-track
candidate quickly, so the NEXT car cannot inherit it (the night gate swap).
The event itself still lives out the full expiry for re-entry / specific binds.
"""
import unittest
from datetime import datetime, timedelta

from tests.fixtures.match_fixtures import make_test_registry


class TestPendingBindTtl(unittest.TestCase):
    def setUp(self):
        self.reg = make_test_registry()
        self.t0 = datetime(2026, 1, 1, 12, 0, 0)

    def test_fresh_pending_binds(self):
        # A car reaching the gate a few seconds after its read binds normally.
        self.reg.register_anpr_event("FRESH-1", "entry", timestamp=self.t0)
        cand = self.reg.open_park_entry_candidate("CAM_03", 5)
        plate = self.reg.bind_next_pending_anpr_to_candidate(
            cand.candidate_id, timestamp=self.t0 + timedelta(seconds=5)
        )
        self.assertEqual(plate, "FRESH-1")

    def test_stale_pending_not_bound_but_still_alive(self):
        # A leftover plate older than the bind TTL (10s) but younger than the
        # 30s expiry must NOT bind to a new candidate...
        self.reg.register_anpr_event("STALE-1", "entry", timestamp=self.t0)
        cand = self.reg.open_park_entry_candidate("CAM_03", 6)
        plate = self.reg.bind_next_pending_anpr_to_candidate(
            cand.candidate_id, timestamp=self.t0 + timedelta(seconds=15)
        )
        self.assertIsNone(plate)
        # ...and must still exist (not expired) for re-entry / specific-event binds.
        ev = next(e for e in self.reg._pending_events.values() if e.plate == "STALE-1")
        self.assertEqual(ev.status, "pending")

    def test_expired_pending_marked_expired(self):
        # Past the full expiry, the FIFO walk marks it expired.
        self.reg.register_anpr_event("OLD-1", "entry", timestamp=self.t0)
        cand = self.reg.open_park_entry_candidate("CAM_03", 7)
        plate = self.reg.bind_next_pending_anpr_to_candidate(
            cand.candidate_id, timestamp=self.t0 + timedelta(seconds=35)
        )
        self.assertIsNone(plate)
        ev = next(e for e in self.reg._pending_events.values() if e.plate == "OLD-1")
        self.assertEqual(ev.status, "expired")


if __name__ == "__main__":
    unittest.main()
