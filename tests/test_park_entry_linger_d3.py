"""
D3 — a car lingering in the Park_Entry zone must not take the ARRIVING car's plate.

The rule under test: a candidate may only bind a plate that was read at-or-after it
entered the zone (within PARK_ENTRY_LINGER_GRACE_SECONDS). The ANPR read happens at
the gate, upstream of the CAM-23 polygon, so the car that triggered a read reaches the
zone AFTER it; a car already sitting in the zone when the read landed did not trigger
it.

Note this is deliberately NOT an absolute cap on candidate age. An absolute cap is what
force-expired legitimately-dwelling cars and had to be reverted (b3ef313), so the
dwelling cases below are regression guards for that revert as much as they are tests of
the new rule.

Run: python -m pytest tests/test_park_entry_linger_d3.py -v
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from src.config import load_config
from src.core.engine.engine_tracking import ParkingEngineTrackingMixin

from src.vehicle_registry.vehicle_registry import VehicleRegistry


def _crop() -> np.ndarray:
    crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    crop[10:90, 10:190] = np.random.randint(100, 200, (80, 180, 3), dtype=np.uint8)
    return crop


class _Det:
    """Minimal stand-in for a Detection (geometry is stubbed on the harness)."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.bbox = (50, 50, 150, 200)
        self.bottom_center = (100, 200)


class _RegistryTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.gallery_dir = os.path.join(cls.temp_dir, "gallery")
        os.makedirs(cls.gallery_dir, exist_ok=True)
        cls.config = load_config("config.yaml")

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.T0 = datetime(2026, 7, 11, 12, 0, 0)
        self.now = self.T0
        # db_manager omitted on purpose: the configured database is a live MSSQL
        # server, and the whole bind path under test is in-memory.
        self.registry = VehicleRegistry(
            image_dir=self.gallery_dir,
            matching_config=self.config.matching,
            clock=lambda: self.now,
        )
        self.grace = self.registry.PARK_ENTRY_LINGER_GRACE_SECONDS


class TestLingerGuardD3(_RegistryTestBase):
    """The registry-side rule, with every timestamp pinned explicitly."""

    def test_lingerer_cannot_take_a_plate_read_after_it_arrived(self):
        """THE DEFECT: car sits in the zone, next car's plate is read, lingerer grabs it."""
        lingerer = self.registry.open_park_entry_candidate(
            "CAM-23", track_id=1, timestamp=self.T0
        )
        # The ARRIVING car's plate is read a full minute after the lingerer parked itself
        # in the zone. The lingerer demonstrably did not trigger this read.
        self.registry.register_anpr_event(
            plate="ARRIVING-1",
            direction="entry",
            camera_id="ANPR-Gate",
            timestamp=self.T0 + timedelta(seconds=60),
        )

        bound = self.registry.bind_next_pending_anpr_to_candidate(
            lingerer.candidate_id, timestamp=self.T0 + timedelta(seconds=61)
        )
        self.assertIsNone(bound, "lingerer stole the arriving car's plate")

    def test_arriving_car_binds_the_plate_the_lingerer_was_refused(self):
        """And the plate must still reach the car that actually earned it."""
        lingerer = self.registry.open_park_entry_candidate(
            "CAM-23", track_id=1, timestamp=self.T0
        )
        read_at = self.T0 + timedelta(seconds=60)
        self.registry.register_anpr_event(
            plate="ARRIVING-1", direction="entry", camera_id="ANPR-Gate", timestamp=read_at
        )
        # The car that was read now drives into the zone.
        arriving = self.registry.open_park_entry_candidate(
            "CAM-23", track_id=2, timestamp=read_at + timedelta(seconds=2)
        )

        now = read_at + timedelta(seconds=3)
        self.assertIsNone(
            self.registry.bind_next_pending_anpr_to_candidate(
                lingerer.candidate_id, timestamp=now
            )
        )
        self.assertEqual(
            self.registry.bind_next_pending_anpr_to_candidate(
                arriving.candidate_id, timestamp=now
            ),
            "ARRIVING-1",
        )

    def test_dwelling_car_still_binds_its_own_plate(self):
        """REGRESSION (b3ef313): a car may dwell at the barrier. Only its entry time
        RELATIVE TO THE READ matters, never its absolute age."""
        read_at = self.T0
        self.registry.register_anpr_event(
            plate="DWELL-1", direction="entry", camera_id="ANPR-Gate", timestamp=read_at
        )
        # Enters the zone 2s after its own read, then sits there.
        candidate = self.registry.open_park_entry_candidate(
            "CAM-23", track_id=1, timestamp=read_at + timedelta(seconds=2)
        )
        # Becomes snapshot-ready / primary only 8s after the read (still inside the
        # 10s PENDING_ANPR_BIND_TTL_SECONDS window).
        bound = self.registry.bind_next_pending_anpr_to_candidate(
            candidate.candidate_id, timestamp=read_at + timedelta(seconds=8)
        )
        self.assertEqual(bound, "DWELL-1")

    def test_late_anpr_post_within_grace_still_binds(self):
        """event.timestamp is when the event was RECEIVED, not when the plate was read.
        A slow integrator POST can land AFTER the car is already in the zone; the grace
        exists to absorb exactly that."""
        entered_at = self.T0
        candidate = self.registry.open_park_entry_candidate(
            "CAM-23", track_id=1, timestamp=entered_at
        )
        # Plate physically read at the gate before the car reached the zone, but the
        # POST only lands (grace - 2)s after the car entered.
        received_at = entered_at + timedelta(seconds=self.grace - 2)
        self.registry.register_anpr_event(
            plate="LATE-POST",
            direction="entry",
            camera_id="ANPR-Gate",
            timestamp=received_at,
        )

        bound = self.registry.bind_next_pending_anpr_to_candidate(
            candidate.candidate_id, timestamp=received_at + timedelta(seconds=1)
        )
        self.assertEqual(
            bound, "LATE-POST", "grace failed to absorb ANPR POST latency"
        )


class _Harness(ParkingEngineTrackingMixin):
    """Drives the real _process_park_entry_zone against a real registry.

    Only the frame/zone geometry is stubbed — the bind ordering and the registry's
    linger guard are the real ones under test.
    """

    def __init__(self, registry, in_zone, ranked):
        self.vehicle_registry = registry
        self._park_entry_track_to_candidate = {}
        self._tracks_inside_zones = {}
        self._in_zone = set(in_zone)
        self._ranked = list(ranked)

    def _detection_in_zone(self, detection, zone):
        return detection.track_id in self._in_zone

    def _rank_zone_detections(self, frame, detections, zone):
        order = {tid: i for i, tid in enumerate(self._ranked)}
        return sorted(
            [d for d in detections if d.track_id in order],
            key=lambda d: order[d.track_id],
        )

    def _crop_detection(self, frame, detection):
        return _crop()

    def _score_snapshot_quality(self, detection, crop):
        return 50.0


class TestLingeringPrimaryDoesNotBlockTheGate(_RegistryTestBase):
    """The engine half of D3.

    The bind requires status == "open" and (before this fix) only the single
    best-ranked car ever attempted it. A stationary lingerer scores HIGH on
    overlap/depth/area, so it wins the primary slot — and would then either take the
    arriving car's plate or, once refused, block every car behind it forever.
    """

    def setUp(self):
        super().setUp()
        # Gallery seeding is exercised elsewhere; keep this test on the bind path.
        self.registry.seed_gallery_from_park_entry = lambda *a, **k: True
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.zone = SimpleNamespace(id="Park_Entry")

    def _run_frame(self, harness, track_ids, now):
        self.now = now  # the registry's injected clock reads this
        harness._process_park_entry_zone(
            "CAM-23", self.frame, [_Det(t) for t in track_ids], self.zone
        )

    def test_lingering_primary_does_not_block_the_arriving_car(self):
        # Frame 1 (T0): the lingerer alone in the zone. No plate pending, so it opens a
        # candidate and binds nothing.
        harness = _Harness(self.registry, in_zone=[1], ranked=[1])
        self._run_frame(harness, [1], self.T0)
        lingerer_id = harness._park_entry_track_to_candidate[1]
        self.assertIsNone(self.registry.plate_for_park_entry_candidate(lingerer_id))

        # A minute later the NEXT car is read at the gate and drives in. The lingerer is
        # still there and still ranks first (stationary, deep in the zone, big bbox).
        read_at = self.T0 + timedelta(seconds=60)
        self.registry.register_anpr_event(
            plate="ARRIVING-1", direction="entry", camera_id="ANPR-Gate", timestamp=read_at
        )
        harness._in_zone = {1, 2}
        harness._ranked = [1, 2]  # lingerer is PRIMARY
        self._run_frame(harness, [1, 2], read_at + timedelta(seconds=2))

        arriving_id = harness._park_entry_track_to_candidate[2]
        self.assertIsNone(
            self.registry.plate_for_park_entry_candidate(lingerer_id),
            "lingering primary stole the arriving car's plate",
        )
        self.assertEqual(
            self.registry.plate_for_park_entry_candidate(arriving_id),
            "ARRIVING-1",
            "lingering primary blocked the arriving car from binding",
        )

    def test_primary_still_wins_between_two_fresh_cars(self):
        """D1 must survive: with two legitimately-arriving cars, the plate goes to the
        primary, not to whichever the tracker listed first."""
        read_at = self.T0
        self.registry.register_anpr_event(
            plate="TAILGATE-1", direction="entry", camera_id="ANPR-Gate", timestamp=read_at
        )
        # Track 2 is the primary even though track 1 is listed first.
        harness = _Harness(self.registry, in_zone=[1, 2], ranked=[2, 1])
        self._run_frame(harness, [1, 2], read_at + timedelta(seconds=1))

        first_listed = harness._park_entry_track_to_candidate[1]
        primary = harness._park_entry_track_to_candidate[2]
        self.assertEqual(
            self.registry.plate_for_park_entry_candidate(primary), "TAILGATE-1"
        )
        self.assertIsNone(
            self.registry.plate_for_park_entry_candidate(first_listed),
            "plate went to the first-listed car, not the primary (D1 regression)",
        )

    def test_solo_car_binds_even_when_ranking_abstains(self):
        """REGRESSION (b3ef313): _rank_zone_detections' snapshot-ready test is stricter
        than _detection_in_zone, so a lone car must bind even when the ranking is empty."""
        read_at = self.T0
        self.registry.register_anpr_event(
            plate="SOLO-1", direction="entry", camera_id="ANPR-Gate", timestamp=read_at
        )
        harness = _Harness(self.registry, in_zone=[1], ranked=[])  # ranking abstains
        self._run_frame(harness, [1], read_at + timedelta(seconds=1))

        candidate_id = harness._park_entry_track_to_candidate[1]
        self.assertEqual(
            self.registry.plate_for_park_entry_candidate(candidate_id), "SOLO-1"
        )


if __name__ == "__main__":
    unittest.main()
