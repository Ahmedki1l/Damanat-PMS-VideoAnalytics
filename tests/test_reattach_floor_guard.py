"""Reattach same-floor guard (regression for the B1<->B2 plate leak).

`aca072f` bounded `match_global_session` to the querying camera's area but left
`reattach_track_to_confirmed_session` scanning ALL confirmed+plated sessions on
every floor. That let an anonymous track on a B2 camera reattach to a B1 car's
plated session on a cross-camera ReID score as low as 0.43 — stamping the B1
plate onto a B2 car (observed live as LLJ-9005 showing on both B1/CAM-24 and
B2/CAM-17 at conf ~0.51 while the real car stayed in B1).

These tests pin ReID similarity so the ONLY thing that can block a reattach is
the floor guard: a cross-floor reattach must be refused, a same-floor one must
still succeed.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.zoning.area_registry import AreaRegistry
from tests.fixtures.fake_clock import FakeClock
from tests.fixtures.match_fixtures import make_test_registry, make_vehicle_session


def _zoned_registry():
    """Registry wired with a minimal B1/B2 zoning topology.

    CAM-24 -> B1-B, CAM-20 -> B1-A (both floor B1); CAM-17 -> B2-B (floor B2).
    """
    clock = FakeClock()
    registry = make_test_registry(clock=clock)
    areas = [
        SimpleNamespace(area_id="B1-A", floor="B1", capacity=30, adjacency={}),
        SimpleNamespace(area_id="B1-B", floor="B1", capacity=30, adjacency={}),
        SimpleNamespace(area_id="B2-B", floor="B2", capacity=30, adjacency={}),
    ]
    cams = [
        SimpleNamespace(id="CAM-24", area="B1-B"),
        SimpleNamespace(id="CAM-20", area="B1-A"),
        SimpleNamespace(id="CAM-17", area="B2-B"),
    ]
    registry._area_registry = AreaRegistry(SimpleNamespace(areas=areas, cameras=cams))
    return registry, clock


def _seed_b1_plated_session(registry):
    """Insert a confirmed, plated B1 session parked under CAM-24 (area B1-B)."""
    vec = np.array([0.7, 0.3], dtype=np.float32)
    b1 = make_vehicle_session(
        "CAR-B1",
        feature_vector=vec,
        last_seen_camera="CAM-24",
        last_seen_track_id=99,
        status="confirmed",
    )
    b1.current_area = "B1-B"
    registry._sessions[b1.session_id] = b1
    return b1, vec


class TestReattachFloorGuard(unittest.TestCase):
    def test_cross_floor_reattach_is_refused(self):
        """A B2/CAM-17 anonymous track must NOT adopt the B1 car's plate even at
        perfect ReID similarity — the leak this guard closes."""
        registry, clock = _zoned_registry()
        b1, vec = _seed_b1_plated_session(registry)

        # Anonymous appearance track on CAM-17 (B2) with the SAME feature, so
        # ReID similarity is 1.0 and only the floor guard can block the match.
        orphan_id = registry.create_appearance_session(
            "CAM-17", track_id=12, feature_vector=vec.copy(), timestamp=clock()
        )

        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-17",
            track_id=12,
            query_vector=vec.copy(),
            similarity_threshold=0.52,
        )

        # No reattach: the B2 track keeps its own anonymous session and no plate.
        self.assertNotEqual(result, b1.session_id)
        self.assertEqual(registry._track_session_map.get(("CAM-17", 12)), orphan_id)
        self.assertIsNone(registry._sessions[orphan_id].plate)

    def test_same_floor_reattach_still_succeeds(self):
        """Positive control: a B1/CAM-20 track at the same similarity DOES
        reattach — the guard only blocks cross-floor, not same-floor handoff."""
        registry, clock = _zoned_registry()
        b1, vec = _seed_b1_plated_session(registry)

        registry.create_appearance_session(
            "CAM-20", track_id=5, feature_vector=vec.copy(), timestamp=clock()
        )

        result = registry.reattach_track_to_confirmed_session(
            camera_id="CAM-20",
            track_id=5,
            query_vector=vec.copy(),
            similarity_threshold=0.52,
        )

        self.assertEqual(result, b1.session_id)
        self.assertEqual(registry._track_session_map.get(("CAM-20", 5)), b1.session_id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
