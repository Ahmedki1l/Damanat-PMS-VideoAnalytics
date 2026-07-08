"""Regression: cross-session identity reconciliation collapses two confirmed
identities that are the SAME physical car (ANPR misread one car as two plates),
while never merging two genuinely-different (even similar) cars, never touching a
parked identity, and staying OFF by default.
"""
import unittest
from datetime import datetime, timedelta

import numpy as np

from tests.fixtures.match_fixtures import make_test_registry, make_vehicle_session


class _DotMatcher:
    @staticmethod
    def compute_similarity(a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class TestIdentityReconcile(unittest.TestCase):
    def _registry(self, floor, window=60.0):
        reg = make_test_registry()
        reg._reid_matcher = _DotMatcher()
        reg._matching_config.identity_reconcile_min_similarity = floor
        reg._matching_config.identity_reconcile_window_seconds = window
        return reg

    def _add(self, reg, plate, vec, start, status="confirmed", slot=None):
        s = make_vehicle_session(plate, feature_vector=vec, status=status)
        s.plate = plate
        s.first_seen_at = start
        if slot is not None:
            s.linked_slot = slot
        reg._sessions[s.session_id] = s
        return s

    def test_same_car_two_plates_collapsed(self):
        reg = self._registry(0.75)
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        v = _norm(np.array([1.0, 0.05, 0.0], dtype=np.float32))
        older = self._add(reg, "MISREAD-1", v, t0)                      # car A as plate 1
        newer = self._add(reg, "MISREAD-2", _norm(v + 0.01), t0 + timedelta(seconds=8))
        released = reg._reconcile_duplicate_identity(newer, t0 + timedelta(seconds=8))
        # The older duplicate is closed; the newer identity survives.
        self.assertNotIn(older.session_id, reg._sessions)
        self.assertIn(newer.session_id, reg._sessions)

    def test_different_cars_not_merged(self):
        reg = self._registry(0.75)
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        a = self._add(reg, "CAR-A", _norm(np.array([1.0, 0.0, 0.0], dtype=np.float32)), t0)
        # A different (even if same-ish colour) car: similarity ~0.5 < 0.75 floor.
        b = self._add(reg, "CAR-B", _norm(np.array([0.5, 0.87, 0.0], dtype=np.float32)),
                      t0 + timedelta(seconds=5))
        reg._reconcile_duplicate_identity(b, t0 + timedelta(seconds=5))
        self.assertIn(a.session_id, reg._sessions)
        self.assertIn(b.session_id, reg._sessions)

    def test_outside_window_not_merged(self):
        reg = self._registry(0.75, window=60.0)
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        v = _norm(np.array([1.0, 0.05, 0.0], dtype=np.float32))
        a = self._add(reg, "P-1", v, t0)
        b = self._add(reg, "P-2", _norm(v + 0.01), t0 + timedelta(seconds=120))  # >60s apart
        reg._reconcile_duplicate_identity(b, t0 + timedelta(seconds=120))
        self.assertIn(a.session_id, reg._sessions)
        self.assertIn(b.session_id, reg._sessions)

    def test_parked_identity_never_touched(self):
        reg = self._registry(0.75)
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        v = _norm(np.array([1.0, 0.05, 0.0], dtype=np.float32))
        parked = self._add(reg, "PARKED-1", v, t0, status="parked", slot="SLOT-1")
        newer = self._add(reg, "MISREAD-2", _norm(v + 0.01), t0 + timedelta(seconds=8))
        reg._reconcile_duplicate_identity(newer, t0 + timedelta(seconds=8))
        self.assertIn(parked.session_id, reg._sessions)

    def test_off_by_default(self):
        reg = self._registry(0.0)  # disabled
        t0 = datetime(2026, 1, 1, 12, 0, 0)
        v = _norm(np.array([1.0, 0.05, 0.0], dtype=np.float32))
        a = self._add(reg, "P-1", v, t0)
        b = self._add(reg, "P-2", _norm(v + 0.01), t0 + timedelta(seconds=8))
        released = reg._reconcile_duplicate_identity(b, t0 + timedelta(seconds=8))
        self.assertEqual(released, [])
        self.assertIn(a.session_id, reg._sessions)


if __name__ == "__main__":
    unittest.main()
