"""
test_entry_temporal_gate.py — the "entry anchor" temporal gate on identity
matching.

A newly-entering car (an unidentified live track hunting for its identity) may
only be matched to:
  * cars that are still NOT parked, or
  * cars that parked (their slot went vacant -> occupied) AT OR AFTER this car
    entered.

A car already sitting in a slot BEFORE this track first appeared cannot be this
track, so it is excluded from the match pool. Symmetrically, an already-parked
(old) car is never re-labelled onto a newer track.

Plain unittest (no pytest, no live cameras/DB):
    PYTHONPATH=. python tests/test_entry_temporal_gate.py
"""

import os
import sys
import unittest
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.fixtures.fake_clock import FakeClock  # noqa: E402
from tests.fixtures.match_fixtures import (  # noqa: E402
    make_test_registry,
    make_vehicle_session,
)


def _vec():
    v = np.random.rand(2).astype("float32")
    return v / np.linalg.norm(v)


class TestEntryTemporalGate(unittest.TestCase):
    def test_new_track_excludes_car_parked_before_entry(self):
        """A car parked before this fresh track appeared is NOT a match, even
        with an identical embedding."""
        clock = FakeClock()
        r = make_test_registry(clock=clock)
        now = clock.now()
        v = _vec()
        old = make_vehicle_session(
            "OLD-1",
            feature_vector=v.copy(),
            status="parked",
            session_id="old",
            timestamp=now,
        )
        old.linked_slot = "A1"
        old.linked_at = now - timedelta(minutes=10)  # parked BEFORE the track
        r._sessions["old"] = old
        # Fresh track -> entry anchor stamped = now -> "old" parked before -> excluded.
        self.assertIsNone(
            r.match_global_session(
                v, camera_id="CAM-05", track_id=1, similarity_threshold=0.4
            )
        )

    def test_new_track_matches_not_yet_parked_car(self):
        """A car that is still driving (not parked) stays eligible."""
        clock = FakeClock()
        r = make_test_registry(clock=clock)
        now = clock.now()
        v = _vec()
        conf = make_vehicle_session(
            "CONF-1",
            feature_vector=v.copy(),
            status="confirmed",
            session_id="conf",
            timestamp=now,
        )
        r._sessions["conf"] = conf
        self.assertEqual(
            r.match_global_session(
                v, camera_id="CAM-05", track_id=2, similarity_threshold=0.4
            ),
            "conf",
        )

    def test_new_track_matches_car_parked_after_entry(self):
        """A car that parked AFTER this track first appeared stays eligible —
        e.g. the entrant's own session once it reaches and takes a slot."""
        clock = FakeClock()
        r = make_test_registry(clock=clock)
        now = clock.now()
        v = _vec()
        # This track was first seen 5 min ago (pre-stamp the anchor).
        r._track_first_seen[("CAM-05", 3)] = now - timedelta(minutes=5)
        parked_after = make_vehicle_session(
            "AFTER-1",
            feature_vector=v.copy(),
            status="parked",
            session_id="after",
            timestamp=now,
        )
        parked_after.linked_slot = "B2"
        parked_after.linked_at = now - timedelta(minutes=2)  # parked AFTER the anchor
        r._sessions["after"] = parked_after
        self.assertEqual(
            r.match_global_session(
                v, camera_id="CAM-05", track_id=3, similarity_threshold=0.4
            ),
            "after",
        )

    def test_orphan_first_seen_entry_is_garbage_collected(self):
        """A track that queries match_global_session but never binds to a
        session (no _mark_track_seen -> no _track_last_seen twin) must not leave
        an immortal _track_first_seen entry: _cleanup_stale_data ages it out by
        its own timestamp, so a later reused track id gets a fresh anchor."""
        clock = FakeClock()
        r = make_test_registry(clock=clock)
        now = clock.now()
        v = _vec()
        # Query with no candidate sessions -> no match, no binding, but the
        # anchor is stamped by match_global_session's setdefault.
        self.assertIsNone(
            r.match_global_session(
                v, camera_id="CAM-05", track_id=99, similarity_threshold=0.4
            )
        )
        key = ("CAM-05", 99)
        self.assertIn(key, r._track_first_seen)
        self.assertNotIn(key, r._track_last_seen)  # the orphan condition
        # Age past the expiry window and run the GC.
        later = now + timedelta(seconds=r.TRACK_MAPPING_EXPIRY_SECONDS + 1)
        clock.set_now(later)
        r._cleanup_stale_data(later)
        self.assertNotIn(key, r._track_first_seen)  # swept, not immortal


if __name__ == "__main__":
    unittest.main()
