"""Occupancy is decided and PUBLISHED before any identity work runs.

The product requirement: a slot must flip OCCUPIED/VACANT — and the UI must be able
to see it flip — without waiting on OCR or ReID. Identity is secondary and may land a
beat later.

Until 2026-07-18 `_process_detections_and_events` ran `_process_special_zones`
(identity: ReID + approach-OCR) BEFORE `assign`/`_update_slot_state` (occupancy), on
the same thread. Measured on prod that put up to 517ms of identity work in front of an
occupancy path that costs at most ~37ms. The delay was pure statement order.

These tests pin the ordering so it cannot silently regress, and pin the one hazard the
reorder introduces: `SlotAssigner.assign` stamps synthetic negative track_ids in place,
so identity guards must test `is_untracked()` rather than `== -1`.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from src.core.slot_assigner import SlotAssigner, AssignerConfig
from src.detection.detector import Detection, is_untracked
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin
from src.models.state_machine import SlotState


# --------------------------------------------------------------------------- #
# is_untracked — the guard that makes the new order safe
# --------------------------------------------------------------------------- #
class TestIsUntracked(unittest.TestCase):
    def test_recognises_both_untracked_encodings(self):
        self.assertTrue(is_untracked(-1))    # tracker assigned nothing
        self.assertTrue(is_untracked(-100))  # assigner's synthetic id
        self.assertTrue(is_untracked(-101))
        self.assertTrue(is_untracked(None))

    def test_real_track_ids_are_tracked(self):
        for tid in (0, 1, 7, 9999):
            self.assertFalse(is_untracked(tid))

    def test_assigner_stamps_synthetic_ids_in_place(self):
        """This is WHY the guards had to change: after assign(), an untracked
        detection no longer reads -1, so `== -1` guards stop firing."""
        dets = [Detection(bbox=(0, 0, 10, 10), class_id=2, confidence=0.9)]
        self.assertEqual(dets[0].track_id, -1)
        SlotAssigner(slots=[], config=AssignerConfig(overlap_threshold=0.3)).assign(dets)
        self.assertNotEqual(dets[0].track_id, -1, "assigner mutates track_id in place")
        self.assertTrue(
            is_untracked(dets[0].track_id),
            "the synthetic id must still read as untracked, or identity work would "
            "run on blobs with an id that is re-minted every frame",
        )


# --------------------------------------------------------------------------- #
# Ordering: occupancy is emitted before identity runs
# --------------------------------------------------------------------------- #
class _SM:
    """Minimal state machine that flips on the first update."""

    def __init__(self):
        self.state = SlotState.VACANT
        self.plate_number = None
        self.latest_detection_bbox = None
        self._fired = False

    def update(self, vehicle_present, track_id=None):
        if vehicle_present and not self._fired:
            self._fired = True
            self.state = SlotState.OCCUPIED
            return [SimpleNamespace(
                event_type="vehicle_parked", slot_id="B1", plate_number=None,
                snapshot_url="", is_alert=False,
            )]
        return []

    def is_plate_locked(self):
        return False

    def bind_identity(self, plate, url, confidence=0.0, lock=False):
        self.plate_number = plate


class _Harness(ParkingEngineRuntimeMixin):
    """Records the order in which the two halves execute."""

    def __init__(self, slot, sm, slow_identity=False):
        self.calls = []
        self.vehicle_registry = None       # identity short-circuits; ordering is the point
        self.db_manager = None
        self._slow = slow_identity
        self.pipelines = {}
        self._sm = sm
        self._slot = slot

    # -- occupancy side ---------------------------------------------------- #
    def _arm_ocr_for_slot(self, slot_id):
        self.calls.append("arm")

    def _save_slot_snapshot(self, frame, slot, detection=None, bbox=None):
        self.calls.append("snapshot")
        return None

    def _filter_violation_events(self, frame, assignment, cam_id, events):
        self.calls.append("filter")
        return events

    def _persist_final_events(self, events):
        self.calls.append("EMIT")

    # -- identity side ----------------------------------------------------- #
    def _process_special_zones(self, cam_id, frame, detections):
        self.calls.append("zones")

    def _build_slot_snapshot_url(self, slot_id):
        return ""


def _pipeline(slot, sm):
    dets_slot = SimpleNamespace(
        id="B1", label="B1", zone_id="z", zone_name="Z",
        polygon=slot, centroid_x=5.0, centroid_y=5.0,
    )
    return dets_slot


class TestOrdering(unittest.TestCase):
    def _run(self):
        from shapely.geometry import Polygon

        poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        slot = SimpleNamespace(
            id="B1", label="B1", zone_id="z", zone_name="Z",
            polygon=poly, centroid_x=50.0, centroid_y=50.0,
        )
        sm = _SM()
        h = _Harness(slot, sm)
        assigner = SlotAssigner(slots=[slot], config=AssignerConfig(overlap_threshold=0.3))
        pipeline = SimpleNamespace(
            slots=[slot],
            state_machines={"B1": sm},
            floor="B1",
            assigner=assigner,
            filter_detections_to_roi=lambda d: d,
        )
        h.pipelines = {"CAM-21": pipeline}
        det = Detection(bbox=(10, 10, 60, 60), class_id=2, confidence=0.9, track_id=3)
        h._process_detections_and_events(
            "CAM-21", np.zeros((128, 128, 3), np.uint8), pipeline, [det]
        )
        return h, sm

    def test_occupancy_is_emitted_before_identity_runs(self):
        h, sm = self._run()
        self.assertIn("EMIT", h.calls, "occupancy must be published")
        self.assertIn("zones", h.calls, "identity must still run")
        self.assertLess(
            h.calls.index("EMIT"), h.calls.index("zones"),
            f"occupancy must be emitted BEFORE identity work; got {h.calls}",
        )

    def test_slot_flips_occupied_in_the_same_pass(self):
        h, sm = self._run()
        self.assertEqual(sm.state, SlotState.OCCUPIED)

    def test_identity_still_runs_after_the_emit(self):
        h, _ = self._run()
        self.assertEqual(h.calls[-1], "zones",
                         f"identity should be last, got {h.calls}")


if __name__ == "__main__":
    unittest.main()
