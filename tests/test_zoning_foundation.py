"""
test_zoning_foundation.py — foundation tests for the parking-area (zoning) layer.

Plain unittest (no pytest, no live cameras/DB). Run with:
    PYTHONPATH=. python tests/test_zoning_foundation.py

Covers: config parsing + helpers, AreaRegistry topology, VehicleSession area
fields, the registry per-area index, area-bounded match_global_session
(in-area + adjacent handoff, plus legacy backward-compat), the handoff matcher,
boundary-crossing geometry, the area state machine, the parking_areas
seed/sync round-trip, and the boundary slot save/load round-trip.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig, AreaEntry, CameraEntry, load_config  # noqa: E402
from src.database import Base  # noqa: E402
import src.model  # noqa: E402,F401  (registers ORMs incl. ParkingArea/ParkingSlot)
from src.zoning import (  # noqa: E402
    AreaRegistry,
    AreaStateMachine,
    BoundaryCrossingDetector,
    BoundaryZone,
    CrossAreaHandoffMatcher,
)
from src.vehicle_registry.vehicle_registry import VehicleRegistry  # noqa: E402
from src.vehicle_registry.vehicle_registry_models import VehicleSession  # noqa: E402


def _make_config():
    """A small zoned config: 3 areas, a couple of cameras, an adjacency chain."""
    cfg = AppConfig()
    cfg.cameras = [
        CameraEntry(id="CAM-01", floor="Ground", area=""),       # un-zoned
        CameraEntry(id="CAM-05", floor="B1", area="B1-E"),
        CameraEntry(id="CAM-07", floor="B1", area="B1-F"),
        CameraEntry(id="CAM-13", floor="B2", area="B2-C"),
    ]
    cfg.areas = [
        AreaEntry("B1-E", "B1 Center", "B1", 30, {"B1-F": 15.0}),
        AreaEntry("B1-F", "B1 South", "B1", 30, {"B1-E": 15.0, "B1-RAMP": 20.0}),
        AreaEntry("B1-RAMP", "B1 Ramp", "B1", 6, {"B1-F": 20.0}),
    ]
    return cfg


class _StubMatcher:
    """Cosine similarity without loading the ReID model."""

    @staticmethod
    def compute_similarity(a, b):
        a = np.asarray(a, dtype="float32")
        b = np.asarray(b, dtype="float32")
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def _zoned_registry(cfg):
    r = VehicleRegistry(image_dir="tests/test_images", area_registry=AreaRegistry(cfg))
    r._reid_matcher = _StubMatcher()  # avoid torchreid model load
    return r


class TestConfigParsing(unittest.TestCase):
    def test_helpers(self):
        cfg = _make_config()
        self.assertEqual(cfg.area_for_camera("CAM-05"), "B1-E")
        self.assertEqual(cfg.area_for_camera("CAM-01"), "")  # un-zoned ground cam
        self.assertEqual(set(cfg.cameras_in_area("B1-E")), {"CAM-05"})
        self.assertIsNotNone(cfg.area_by_id("B1-F"))
        self.assertEqual(cfg.adjacency_for("B1-F"), {"B1-E": 15.0, "B1-RAMP": 20.0})
        self.assertEqual(cfg.adjacency_for("nope"), {})

    def test_yaml_block_parses(self):
        raw = {
            "cameras": [{"id": "CAM-05", "floor": "B1", "area": "B1-E"}],
            "areas": [
                {"area_id": "B1-E", "floor": "B1", "capacity": 30,
                 "adjacency": {"B1-F": 15}},
            ],
        }
        path = os.path.join(os.path.dirname(__file__), "_tmp_zoning_cfg.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        try:
            cfg = load_config(path)
            self.assertEqual(len(cfg.areas), 1)
            self.assertEqual(cfg.areas[0].adjacency, {"B1-F": 15.0})
            self.assertEqual(cfg.area_for_camera("CAM-05"), "B1-E")
        finally:
            os.remove(path)


class TestAreaRegistry(unittest.TestCase):
    def setUp(self):
        self.ar = AreaRegistry(_make_config())

    def test_topology(self):
        self.assertTrue(self.ar.enabled)
        self.assertEqual(self.ar.area_for_camera("CAM-13"), "B2-C")
        self.assertEqual(self.ar.cameras("B1-E"), ["CAM-05"])
        self.assertEqual(self.ar.capacity("B1-RAMP"), 6)
        self.assertEqual(self.ar.floor("B1-E"), "B1")
        self.assertTrue(self.ar.is_adjacent("B1-F", "B1-RAMP"))
        self.assertEqual(self.ar.transit_seconds("B1-F", "B1-RAMP"), 20.0)
        self.assertIsNone(self.ar.transit_seconds("B1-E", "B1-RAMP"))  # not adjacent

    def test_unzoned(self):
        empty = AreaRegistry(AppConfig())
        self.assertFalse(empty.enabled)
        self.assertEqual(empty.area_for_camera("CAM-05"), "")
        self.assertEqual(empty.adjacent("anything"), {})


class TestSessionFields(unittest.TestCase):
    def test_defaults(self):
        s = VehicleSession(session_id="x")
        self.assertEqual(s.current_area, "")
        self.assertEqual(s.area_state, "IN_AREA")
        self.assertIsNone(s.area_entered_at)
        self.assertEqual(s.departed_from_area, "")


class TestAreaIndex(unittest.TestCase):
    def test_set_move_drop(self):
        r = _zoned_registry(_make_config())
        s = VehicleSession(session_id="s1", status="confirmed")
        r._sessions["s1"] = s
        r.set_session_area(s, "B1-E")
        self.assertEqual([x.session_id for x in r.sessions_in_area("B1-E")], ["s1"])
        self.assertIsNotNone(s.area_entered_at)
        r.set_session_area(s, "B1-F")
        self.assertEqual(r.sessions_in_area("B1-E"), [])
        self.assertEqual([x.session_id for x in r.sessions_in_area("B1-F")], ["s1"])
        r._drop_session_from_area_index(s)
        self.assertEqual(r.sessions_in_area("B1-F"), [])


class TestBoundedMatching(unittest.TestCase):
    def _vec(self):
        v = np.random.rand(512).astype("float32")
        return v / np.linalg.norm(v)

    def test_in_area_only_and_legacy(self):
        r = _zoned_registry(_make_config())
        v = self._vec()
        for sid, area in [("in", "B1-E"), ("out", "B2-C")]:
            s = VehicleSession(session_id=sid, status="confirmed",
                               feature_vector=v.copy())
            r._sessions[sid] = s
            r.set_session_area(s, area)
        # Bounded to B1-E -> only the in-area car is eligible.
        self.assertEqual(
            r.match_global_session(v, camera_id="CAM-05", area_id="B1-E",
                                   similarity_threshold=0.4),
            "in",
        )
        # Bounded to an area with no members (B1-F) -> no match.
        self.assertIsNone(
            r.match_global_session(v, camera_id="CAM-07", area_id="B1-F",
                                   similarity_threshold=0.4)
        )
        # Legacy (area_id=None) -> all-sessions, matches one of them.
        self.assertIn(
            r.match_global_session(v, camera_id="CAM-05", area_id=None,
                                   similarity_threshold=0.4),
            {"in", "out"},
        )

    def test_adjacent_handoff_included(self):
        r = _zoned_registry(_make_config())
        v = self._vec()
        # A car that just departed B1-F (adjacent to B1-RAMP), in transit.
        s = VehicleSession(session_id="dep", status="confirmed",
                           feature_vector=v.copy())
        s.area_state = "IN_TRANSIT"
        s.departed_from_area = "B1-F"
        s.area_entered_at = datetime.now()
        r._sessions["dep"] = s  # NOT in any area bucket (it's in transit)
        # Querying area B1-RAMP should reach the departed car via the handoff pool.
        self.assertEqual(
            r.match_global_session(v, camera_id="CAM-07", area_id="B1-RAMP",
                                   similarity_threshold=0.4),
            "dep",
        )


class TestHandoffMatcher(unittest.TestCase):
    def setUp(self):
        self.hm = CrossAreaHandoffMatcher(AreaRegistry(_make_config()))

    def _session(self, sid, state, src, entered):
        s = VehicleSession(session_id=sid)
        s.area_state = state
        s.departed_from_area = src
        s.area_entered_at = entered
        return s

    def test_window_and_adjacency(self):
        now = datetime.now()
        within = self._session("a", "IN_TRANSIT", "B1-F", now)
        expired = self._session("b", "IN_TRANSIT", "B1-F",
                                now - timedelta(seconds=999))
        not_adj = self._session("c", "IN_TRANSIT", "B1-E", now)  # B1-E !adj B1-RAMP
        settled = self._session("d", "IN_AREA", "B1-F", now)
        ids = self.hm.candidate_session_ids("B1-RAMP",
                                            [within, expired, not_adj, settled])
        self.assertEqual(ids, {"a"})


class TestBoundaryDetector(unittest.TestCase):
    def test_crossing_edge(self):
        from shapely.geometry import Polygon
        b = BoundaryZone(id="ramp", polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                         area_from="B2-C", area_to="B2-RAMP")
        det = BoundaryCrossingDetector()
        # bbox bottom-centre at (5, 9) -> inside; first frame fires a crossing.
        crossings = det.detect("CAM-13", [(1, (0, 0, 10, 9))], [b])
        self.assertEqual(len(crossings), 1)
        self.assertEqual((crossings[0].area_from, crossings[0].area_to),
                         ("B2-C", "B2-RAMP"))
        # Still inside next frame -> no repeat crossing.
        self.assertEqual(det.detect("CAM-13", [(1, (0, 0, 10, 9))], [b]), [])
        # No boundaries -> nothing.
        self.assertEqual(det.detect("CAM-13", [(1, (0, 0, 10, 9))], []), [])


class TestAreaStateMachine(unittest.TestCase):
    def test_transitions_update_index(self):
        cfg = _make_config()
        r = _zoned_registry(cfg)
        sm = AreaStateMachine(r, AreaRegistry(cfg))
        s = VehicleSession(session_id="s1", status="confirmed")
        r._sessions["s1"] = s

        sm.on_area_observed(s, "B1-F")
        self.assertEqual(s.current_area, "B1-F")
        self.assertEqual(s.area_state, "IN_AREA")
        self.assertIn("s1", {x.session_id for x in r.sessions_in_area("B1-F")})

        sm.on_boundary_cross(s, "B1-F", "B1-RAMP")
        self.assertEqual(s.area_state, "IN_TRANSIT")
        self.assertEqual(s.departed_from_area, "B1-F")
        # Departed: dropped from the old area bucket, current_area cleared.
        self.assertEqual(r.sessions_in_area("B1-F"), [])
        self.assertEqual(s.current_area, "")

        sm.on_area_observed(s, "B1-RAMP")
        self.assertEqual(s.area_state, "IN_AREA")
        self.assertEqual(s.current_area, "B1-RAMP")
        self.assertEqual(s.departed_from_area, "")

    def test_reverse_boundary_cross(self):
        # The same boundary (stored B1-F -> B1-RAMP) crossed the OTHER way:
        # a car sitting in B1-RAMP departs B1-RAMP, not B1-F.
        cfg = _make_config()
        r = _zoned_registry(cfg)
        sm = AreaStateMachine(r, AreaRegistry(cfg))
        s = VehicleSession(session_id="s2", status="confirmed")
        r._sessions["s2"] = s
        sm.on_area_observed(s, "B1-RAMP")        # car is in B1-RAMP
        sm.on_boundary_cross(s, "B1-F", "B1-RAMP")
        self.assertEqual(s.departed_from_area, "B1-RAMP")   # reversed
        self.assertEqual(s.area_state, "IN_TRANSIT")
        self.assertEqual(r.sessions_in_area("B1-RAMP"), [])  # dropped from bucket


class TestAreasDbRoundTrip(unittest.TestCase):
    def setUp(self):
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()

    def tearDown(self):
        self.db.close()

    def test_seed_and_sync(self):
        from src.services.config_service import (
            ensure_areas_initialized, sync_areas_from_db,
        )
        cfg = _make_config()
        ensure_areas_initialized(self.db, cfg)
        cfg.areas = []  # wipe runtime copy; prove the DB is the source
        sync_areas_from_db(self.db, cfg)
        self.assertEqual(len(cfg.areas), 3)
        self.assertEqual(cfg.adjacency_for("B1-F"), {"B1-E": 15.0, "B1-RAMP": 20.0})

    def test_boundary_in_own_table_round_trip(self):
        from src.services.parking_service import (
            sync_camera_slot_definitions, sync_camera_boundaries,
            load_camera_slots, SLOT_TYPE_PARKING, SLOT_TYPE_SPECIAL_ZONE,
        )
        from src.model import Boundary, ParkingSlot

        entries = [
            {"id": "A1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"id": "ramp", "type": "boundary", "area_from": "B2-C",
             "area_to": "B2-RAMP",
             "polygon": [[20, 20], [40, 20], [40, 40], [20, 40]]},
        ]
        # Slots and boundaries go to *different* tables.
        slot_entries = [e for e in entries if e.get("type") != "boundary"]
        boundary_entries = [e for e in entries if e.get("type") == "boundary"]
        sync_camera_slot_definitions(
            self.db, camera_id="CAM-13", floor="B2", slot_entries=slot_entries,
            managed_slot_types=(SLOT_TYPE_PARKING, SLOT_TYPE_SPECIAL_ZONE),
            default_zone_id="CAM-13", default_zone_name="CAM-13",
        )
        sync_camera_boundaries(self.db, camera_id="CAM-13", floor="B2",
                               boundary_entries=boundary_entries)

        # The boundary is NOT in parking_slots; the slot is.
        self.assertEqual(self.db.query(ParkingSlot).count(), 1)
        self.assertEqual(self.db.query(Boundary).count(), 1)

        parking, special, roi, boundaries = load_camera_slots(self.db,
                                                              camera_id="CAM-13")
        self.assertEqual([s.id for s in parking], ["A1"])
        self.assertEqual(len(boundaries), 1)
        self.assertEqual((boundaries[0].area_from, boundaries[0].area_to),
                         ("B2-C", "B2-RAMP"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
