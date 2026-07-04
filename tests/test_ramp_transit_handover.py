"""Cross-floor plate handover through an inter-floor ramp (down-ramp regression).

The same-floor guard (``519b2ea``) closed the B1<->B2 ReID leak by refusing ALL
cross-floor reattach/handoff. But on this facility every plate is confirmed at
the B1 gate (CAM-03), and a car parking on B2 must drive the DOWN ramp
(B1-A -> RAMP-DN -> B2-A). The blanket guard therefore stranded the plate on B1:
the car surfaced on B2/CAM-09 as an anonymous track that could never adopt its
own plate.

The DOWN ramp has no interior camera (CAM-09 owns B2-A slots, so it is area
B2-A, not RAMP-DN), so a descending car is unobserved mid-ramp. These tests pin
the three-part fix:

  1. ``on_boundary_cross`` records the RAMP as the departure area when a car
     crosses INTO it (so the emerging aisle, adjacent to the ramp, can match it).
  2. ``CrossAreaHandoffMatcher`` admits a floorless-ramp source as the one
     legitimate cross-floor transit, still bounded by the transit window.
  3. ``reattach_track_to_confirmed_session`` admits such an in-transit car
     across floors, while still refusing an unrelated cross-floor car.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.zoning.area_registry import AreaRegistry
from src.zoning.area_state_machine import AreaStateMachine
from src.zoning.handoff_matcher import CrossAreaHandoffMatcher
from tests.fixtures.fake_clock import FakeClock
from tests.fixtures.match_fixtures import make_test_registry, make_vehicle_session


def _down_ramp_registry():
    """AreaRegistry for the down-ramp topology: B1-A -> RAMP-DN -> B2-A.

    CAM-07 -> B1-A (down-ramp entrance), CAM-09 -> B2-A (down-ramp exit). No
    camera is assigned to RAMP-DN (floorless) — it is watched only at its ends.
    """
    areas = [
        SimpleNamespace(area_id="B1-A", floor="B1", capacity=30,
                        adjacency={"B1-B": 15, "RAMP-DN": 20}),
        SimpleNamespace(area_id="B2-A", floor="B2", capacity=30,
                        adjacency={"B2-B": 15, "RAMP-DN": 20}),
        SimpleNamespace(area_id="RAMP-DN", floor="", capacity=6,
                        adjacency={"B1-A": 20, "B2-A": 20}),
    ]
    cams = [
        SimpleNamespace(id="CAM-07", area="B1-A"),
        SimpleNamespace(id="CAM-09", area="B2-A"),
    ]
    return AreaRegistry(SimpleNamespace(areas=areas, cameras=cams))


def _stub_registry():
    """Minimal stand-in for the VehicleRegistry the state machine writes to.
    Mirrors the real ``set_session_area`` contract: it stamps ``current_area``
    on the session (the empty string when a car goes in-transit)."""
    return SimpleNamespace(
        set_session_area=lambda session, area: setattr(session, "current_area", area)
    )


class TestRampEntryBridge(unittest.TestCase):
    """Crossing INTO a floorless ramp records the ramp as the departure area."""

    def test_entering_ramp_sets_departed_to_ramp(self):
        areas = _down_ramp_registry()
        clock = FakeClock()
        sm = AreaStateMachine(
            _stub_registry(),
            areas,
            clock=clock,
        )
        car = make_vehicle_session("CAR-DOWN")
        car.current_area = "B1-A"
        car.area_state = "IN_AREA"

        # The entrance band label reads B1-C (a known real-world quirk — the
        # camera's area follows its slots, not the band). The bridge must key off
        # the DESTINATION being a ramp, so the source-aisle label is irrelevant.
        sm.on_boundary_cross(car, area_from="B1-C", area_to="RAMP-DN")

        self.assertEqual(car.area_state, "IN_TRANSIT")
        self.assertEqual(car.departed_from_area, "RAMP-DN")
        self.assertEqual(car.current_area, "")

    def test_aisle_to_aisle_keeps_source_area(self):
        areas = _down_ramp_registry()
        sm = AreaStateMachine(
            _stub_registry(),
            areas,
            clock=FakeClock(),
        )
        car = make_vehicle_session("CAR-AISLE")
        car.current_area = "B1-A"
        sm.on_boundary_cross(car, area_from="B1-A", area_to="B1-B")
        # Normal within-floor handoff is unchanged — departed is the source aisle.
        self.assertEqual(car.departed_from_area, "B1-A")


class TestRampHandoffEligibility(unittest.TestCase):
    """The handoff matcher admits a ramp-sourced car at the emerging aisle."""

    def _in_transit_car(self, clock):
        car = make_vehicle_session("CAR-DOWN", last_seen_camera="CAM-07")
        car.area_state = "IN_TRANSIT"
        car.departed_from_area = "RAMP-DN"
        car.current_area = ""
        car.area_entered_at = clock()
        return car

    def test_ramp_source_eligible_at_destination_aisle(self):
        clock = FakeClock()
        hm = CrossAreaHandoffMatcher(_down_ramp_registry(), clock=clock)
        car = self._in_transit_car(clock)
        ids = hm.candidate_session_ids("B2-A", [car])
        self.assertEqual(ids, {car.session_id})

    def test_ramp_source_expires_after_transit_window(self):
        clock = FakeClock()
        hm = CrossAreaHandoffMatcher(_down_ramp_registry(), clock=clock)
        car = self._in_transit_car(clock)
        # transit(RAMP-DN@B2-A)=20 * DEFAULT_TRANSIT_SLACK(2.0) = 40s window.
        clock.advance(41)
        ids = hm.candidate_session_ids("B2-A", [car])
        self.assertEqual(ids, set())

    def test_settled_car_not_eligible(self):
        clock = FakeClock()
        hm = CrossAreaHandoffMatcher(_down_ramp_registry(), clock=clock)
        parked = make_vehicle_session("CAR-PARKED", last_seen_camera="CAM-07")
        parked.area_state = "IN_AREA"
        parked.current_area = "B1-A"
        ids = hm.candidate_session_ids("B2-A", [parked])
        self.assertEqual(ids, set())


class TestReattachRampTransit(unittest.TestCase):
    """The reattach path stamps the plate onto the B2 track for a genuine
    ramp descent, but still refuses an unrelated cross-floor car."""

    def _zoned_registry(self):
        clock = FakeClock()
        registry = make_test_registry(clock=clock)
        areas = _down_ramp_registry()
        registry._area_registry = areas
        registry._handoff_matcher = CrossAreaHandoffMatcher(areas, clock=clock)
        return registry, clock

    def test_descending_car_reattaches_across_floor(self):
        registry, clock = self._zoned_registry()
        vec = np.array([0.7, 0.3], dtype=np.float32)

        # Plated B1 car mid-descent: IN_TRANSIT off RAMP-DN, still in window.
        b1 = make_vehicle_session(
            "CAR-DOWN", feature_vector=vec, last_seen_camera="CAM-07",
            last_seen_track_id=99, status="confirmed",
        )
        b1.area_state = "IN_TRANSIT"
        b1.departed_from_area = "RAMP-DN"
        b1.current_area = ""
        b1.area_entered_at = clock()
        registry._sessions[b1.session_id] = b1

        # Anonymous appearance on CAM-09 (B2-A) with the same feature.
        registry.create_appearance_session(
            "CAM-09", track_id=12, feature_vector=vec.copy(), timestamp=clock()
        )

        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-09", track_id=12, query_vector=vec.copy(),
            similarity_threshold=0.52,
        )

        self.assertEqual(result, b1.session_id)
        self.assertEqual(registry._track_session_map.get(("CAM-09", 12)), b1.session_id)

    def test_parked_b1_car_not_reattached_from_b2(self):
        """Leak guard intact: a B1 car sitting parked (not transiting) must not
        be adopted by a B2 track even at identical ReID similarity."""
        registry, clock = self._zoned_registry()
        vec = np.array([0.7, 0.3], dtype=np.float32)

        b1 = make_vehicle_session(
            "CAR-PARKED", feature_vector=vec, last_seen_camera="CAM-07",
            last_seen_track_id=99, status="confirmed",
        )
        b1.area_state = "IN_AREA"
        b1.current_area = "B1-A"
        registry._sessions[b1.session_id] = b1

        orphan_id = registry.create_appearance_session(
            "CAM-09", track_id=12, feature_vector=vec.copy(), timestamp=clock()
        )

        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-09", track_id=12, query_vector=vec.copy(),
            similarity_threshold=0.52,
        )

        self.assertNotEqual(result, b1.session_id)
        self.assertEqual(registry._track_session_map.get(("CAM-09", 12)), orphan_id)
        self.assertIsNone(registry._sessions[orphan_id].plate)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
