"""
A car cannot recognise ITSELF when it arrives at its slot — so geography must.

Measured live on 2026-07-11 (RDJ-9640, parked in B10_CTO, seen by CAM-20):

    scored against its OWN gate references : 0.583   <- below the 0.62 bar
    scored against a DIFFERENT car's        : 0.634   <- ranks FIRST

The slot view is a ~136px oblique crop; the gallery holds only front-gate photos.
Appearance is not merely weak here, it is INVERTED — so lowering the bar does not
recover the plate, it binds the WRONG one.

Elimination decides it instead: if exactly ONE plated car is still in flight (a recent
ANPR gate read, and not yet parked in any slot), it is the only car this can be.
Appearance is then only asked not to object.

The search deliberately spans ALL sessions rather than the area-scoped pool: a car in
transit is by definition leaving the area it was last seen in. DJS-7842 entered at
CAM-03 (B1-A) and parked at CAM-04 (B1-C), and B1-C is not adjacent to B1-A — so the
area pool excluded the car from its own destination and never even scored it.

These tests pin the REFUSALS as hard as the successes: this is the one path that binds
a plate on weak appearance evidence, so every constraint must be load-bearing.
"""

import unittest
from datetime import timedelta

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import VehicleSession


def _vec(*head) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    for i, x in enumerate(head):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


class TestInflightCandidates(unittest.TestCase):
    def setUp(self):
        self.reg = VehicleRegistry(matching_config=MatchingConfig())
        self.now = self.reg._clock()

    def _sess(self, sid, plate, vec, *, linked_slot=None, gate_read_s=None):
        """gate_read_s = seconds ago that ANPR read this plate at the gate.
        None => no gate read on record (e.g. a session restored from disk)."""
        s = VehicleSession(
            session_id=sid,
            plate=plate,
            feature_vector=vec,
            first_seen_at=self.now,          # reload always stamps this as "now"
            last_seen_at=self.now,
            last_seen_camera="CAM-20",
            status="confirmed",
        )
        s.linked_slot = linked_slot
        self.reg._sessions[sid] = s
        if plate and gate_read_s is not None:
            self.reg._last_anpr_entry_at[plate] = self.now - timedelta(
                seconds=gate_read_s
            )
        return s

    def _plates(self, query):
        return [s.plate for s, _ in self.reg._inflight_plated_candidates(query, self.now)]

    def test_the_arriving_car_is_in_flight(self):
        q = _vec(1.0)
        self._sess("reload_1", "RDJ-9640", q, gate_read_s=60)
        self.assertEqual(self._plates(q), ["RDJ-9640"])

    def test_an_already_parked_car_is_NOT_in_flight(self):
        """THE LOAD-BEARING EXCLUSION. A car sitting in a slot is accounted for — it
        cannot also be the car arriving at a different one. Without this, the
        long-parked DJS-7842 — which from the slot viewpoint OUT-SCORES the real
        arrival (0.634 vs 0.583) — would steal the acquisition."""
        q = _vec(1.0)
        self._sess("reload_parked", "DJS-7842", q, linked_slot="B13 COO", gate_read_s=60)
        self._sess("reload_new", "RDJ-9640", _vec(0.9, 0.4), gate_read_s=60)

        self.assertEqual(self._plates(q), ["RDJ-9640"])

    def test_a_restored_session_is_NOT_in_flight_until_it_drives_the_gate_again(self):
        """REGRESSION. _restore_vehicle_galleries rebuilds sessions with
        first_seen_at=now, so anchoring on the SESSION's timestamps would make every
        car already inside look freshly-entered after a restart — bogus 'cars in
        flight', and an acquisition made on a car that is actually parked. The anchor
        must be the real ANPR gate read."""
        q = _vec(1.0)
        restored = self._sess("reload_1", "DJS-7842", q, gate_read_s=None)
        self.assertEqual(restored.first_seen_at, self.now)  # looks brand new...
        self.assertEqual(self._plates(q), [])               # ...but never drove the gate

    def test_a_car_that_entered_too_long_ago_is_NOT_in_flight(self):
        q = _vec(1.0)
        self._sess("reload_1", "OLD-1", q, gate_read_s=10_000)
        self.assertEqual(self._plates(q), [])

    def test_anonymous_sessions_are_never_acquisition_candidates(self):
        q = _vec(1.0)
        self._sess("sess_1", None, q, gate_read_s=10)
        self.assertEqual(self._plates(q), [])

    def test_two_cars_in_flight_are_both_returned_so_the_caller_can_refuse(self):
        """Elimination is only valid when the answer is UNIQUE. Two cars in flight
        must surface as two, so the matcher declines rather than guesses."""
        q = _vec(1.0)
        self._sess("reload_a", "AAA-1", q, gate_read_s=30)
        self._sess("reload_b", "BBB-2", _vec(0.9, 0.4), gate_read_s=30)
        self.assertEqual(len(self._plates(q)), 2)

    def test_area_does_NOT_gate_the_search(self):
        """The bug this fix exists for: DJS-7842 entered at CAM-03 (area B1-A) and
        parked at CAM-04 (area B1-C). B1-C neighbours B1-B and RAMP-UP, NOT B1-A — so
        the area-scoped pool excluded the car from its own destination and it was never
        even scored. A car in transit is by definition leaving its last area, so the
        in-flight search must span all sessions."""
        q = _vec(1.0)
        s = self._sess("reload_1", "DJS-7842", q, gate_read_s=45)
        s.last_seen_camera = "CAM-03"  # area B1-A; the query comes from CAM-04 (B1-C)
        self.assertEqual(self._plates(q), ["DJS-7842"])


if __name__ == "__main__":
    unittest.main()


class TestArrivingCarPlausibility(unittest.TestCase):
    """"Only one car in flight" says WHICH plate is unaccounted for. It says nothing
    about whether the car in front of THIS camera is that car.

    On 2026-07-11 that gap bound DJS-7842 to a Nissan Sunny parked in B25 for hours:
    the B2 worker saw an anonymous car, could not name it, found exactly one car in
    flight, and claimed it. A wrong bind is worse than no bind.

    A car already being tracked BEFORE the plate was read at the gate cannot be the car
    that was read.
    """

    def setUp(self):
        self.reg = VehicleRegistry(matching_config=MatchingConfig())
        self.now = self.reg._clock()

    def _sess(self, sid, plate, first_seen_s):
        s = VehicleSession(
            session_id=sid,
            plate=plate,
            feature_vector=_vec(1.0),
            first_seen_at=self.now - timedelta(seconds=first_seen_s),
            last_seen_at=self.now,
            last_seen_camera="CAM-17",
            status="confirmed",
        )
        self.reg._sessions[sid] = s
        return s

    def _seen_before_gate(self, incumbent_sid, plate):
        """Mirrors the guard in match_global_session."""
        gate_read_at = self.reg._last_anpr_entry_at.get(plate)
        inc = self.reg._sessions.get(incumbent_sid)
        return (
            inc is not None
            and gate_read_at is not None
            and inc.first_seen_at is not None
            and inc.first_seen_at < gate_read_at
        )

    def test_the_nissan_sunny_that_stole_a_plate(self):
        """THE REGRESSION. The Sunny had been tracked for an hour; DJS-7842's gate read
        was 90s ago. It was here first, so it cannot be the arriving car."""
        self.reg._last_anpr_entry_at["DJS-7842"] = self.now - timedelta(seconds=90)
        self._sess("sess_sunny", None, first_seen_s=3600)   # parked long before the read
        self.assertTrue(self._seen_before_gate("sess_sunny", "DJS-7842"))

    def test_a_genuinely_arriving_car_is_NOT_refused(self):
        """The car that drove in appears AFTER its own gate read — it must still bind,
        or the whole acquisition path is dead."""
        self.reg._last_anpr_entry_at["DJS-7842"] = self.now - timedelta(seconds=90)
        self._sess("sess_new", None, first_seen_s=40)       # first seen AFTER the read
        self.assertFalse(self._seen_before_gate("sess_new", "DJS-7842"))
