"""Regression: reference snapshots are only captured from the genuine moving
entrant — never from a car parked in an already-occupied slot, and never while
the car is still inside the B1_Entrance (CAM-03) zone.

Covers the two guards in `_process_global_tracking`'s accumulation path:
`_detection_in_occupied_slot` and `_track_in_entrance_zone`.
"""
import unittest

from shapely.geometry import Polygon

from src.core.engine.engine_tracking import ParkingEngineTrackingMixin
from src.models.state_machine import SlotState


class _Slot:
    def __init__(self, slot_id, polygon):
        self.id = slot_id
        self.polygon = polygon


class _SM:
    def __init__(self, state):
        self.state = state


class _Pipeline:
    def __init__(self, slots, state_machines):
        self.slots = slots
        self.state_machines = state_machines


class _Det:
    def __init__(self, bottom_center, bbox=(0, 0, 10, 10), track_id=1):
        self.bottom_center = bottom_center
        self.bbox = bbox
        self.track_id = track_id


class _Harness(ParkingEngineTrackingMixin):
    def __init__(self, pipeline=None):
        self.pipelines = {"CAM-09": pipeline} if pipeline is not None else {}
        self._tracks_inside_zones = {}


def _pipeline_with_state(state):
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return _Pipeline([_Slot("S1", poly)], {"S1": _SM(state)})


class TestOccupiedSlotGuard(unittest.TestCase):
    def test_car_inside_occupied_slot_is_flagged(self):
        h = _Harness(_pipeline_with_state(SlotState.OCCUPIED))
        self.assertTrue(
            h._detection_in_occupied_slot("CAM-09", _Det(bottom_center=(50, 50)))
        )

    def test_leaving_slot_is_also_flagged(self):
        h = _Harness(_pipeline_with_state(SlotState.LEAVING))
        self.assertTrue(
            h._detection_in_occupied_slot("CAM-09", _Det(bottom_center=(50, 50)))
        )

    def test_vacant_slot_is_not_flagged(self):
        h = _Harness(_pipeline_with_state(SlotState.VACANT))
        self.assertFalse(
            h._detection_in_occupied_slot("CAM-09", _Det(bottom_center=(50, 50)))
        )

    def test_car_outside_the_slot_is_not_flagged(self):
        h = _Harness(_pipeline_with_state(SlotState.OCCUPIED))
        self.assertFalse(
            h._detection_in_occupied_slot("CAM-09", _Det(bottom_center=(500, 500)))
        )

    def test_camera_without_pipeline_returns_false(self):
        h = _Harness()  # no pipelines registered
        self.assertFalse(
            h._detection_in_occupied_slot("CAM-09", _Det(bottom_center=(50, 50)))
        )


class TestEntranceZoneGuard(unittest.TestCase):
    def test_track_inside_entrance_zone_is_flagged(self):
        h = _Harness()
        h._tracks_inside_zones = {("CAM-03", "B1_Entrence"): {7, 8}}
        self.assertTrue(h._track_in_entrance_zone("CAM-03", 7))

    def test_track_not_in_zone_is_not_flagged(self):
        h = _Harness()
        h._tracks_inside_zones = {("CAM-03", "B1_Entrence"): {7, 8}}
        self.assertFalse(h._track_in_entrance_zone("CAM-03", 9))

    def test_other_camera_is_not_flagged(self):
        h = _Harness()
        h._tracks_inside_zones = {("CAM-03", "B1_Entrence"): {7}}
        # Same track id, different camera — not in THAT camera's entrance zone.
        self.assertFalse(h._track_in_entrance_zone("CAM-09", 7))

    def test_non_entrance_zone_is_ignored(self):
        h = _Harness()
        h._tracks_inside_zones = {("CAM-03", "Park_Entry"): {7}}
        self.assertFalse(h._track_in_entrance_zone("CAM-03", 7))


if __name__ == "__main__":
    unittest.main()
