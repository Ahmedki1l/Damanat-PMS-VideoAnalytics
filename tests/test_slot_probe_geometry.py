"""Pin the slot-membership probe DECISION: ``bottom_center`` is ``(y1+y2)/1.5``.

This is a decision test, not a geometry test. The /1.5 probe is deliberately NOT
the bbox bottom: the production slot polygons were drawn and operator-calibrated
against it, so their shapes already compensate. Changing the formula without
re-drawing every slot polygon shifts membership by up to ~180px on near cameras.
The change to ``y2`` has been made and reverted THREE times (4668002 → db1c55e →
an AI session on 2026-07-18 → operator revert the same day). If it flips again,
these tests fail and point here.

The boundary detector's probe (``y2``) intentionally differs — each probe matches
its own calibration. See Detection.bottom_center's docstring for the full record.
"""
from shapely.geometry import Polygon

from src.config import AssignerConfig
from src.core.engine.camera_pipeline import CameraPipeline
from src.core.slot_assigner import SlotAssigner
from src.detection.detector import Detection
from src.models.slot import ParkingSlot
from src.zoning.boundary_detector import _bottom_center


def _detection(bbox):
    return Detection(bbox=bbox, class_id=2, confidence=0.9, track_id=7)


def test_bottom_center_is_the_calibrated_slot_probe_not_bbox_bottom():
    # bbox (10, 20, 30, 80): cy = (20 + 80) / 1.5 = 66.67 — NOT y2=80.
    detection = _detection((10.0, 20.0, 30.0, 80.0))
    cx, cy = detection.bottom_center
    assert cx == 20.0
    assert abs(cy - (20.0 + 80.0) / 1.5) < 1e-9
    assert cy != 80.0, (
        "bottom_center flipped to bbox-bottom y2 AGAIN. The slot polygons are "
        "calibrated to /1.5 — do not change this without re-drawing them. "
        "See Detection.bottom_center's docstring."
    )


def test_slot_and_boundary_probes_intentionally_differ():
    """Slots are calibrated to /1.5, boundaries to y2. Matching probes would be
    tidier, but each polygon set was authored against its own probe — this pins
    the operating reality so it can only change deliberately."""
    detection = _detection((100.0, 500.0, 300.0, 700.0))
    boundary_probe = _bottom_center(detection.bbox)

    assert boundary_probe.y == 700.0                       # boundary: y2
    assert abs(detection.bottom_center[1] - 800.0) < 1e-9  # slot: (500+700)/1.5


def test_roi_keeps_vehicle_when_its_probe_is_inside():
    pipeline = object.__new__(CameraPipeline)
    pipeline.roi_polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    # probe cy = (30 + 90) / 1.5 = 80 — inside the 100x100 ROI.
    detection = _detection((40.0, 30.0, 60.0, 90.0))

    assert pipeline.filter_detections_to_roi([detection]) == [detection]


def test_roi_drops_vehicle_when_its_probe_is_outside():
    pipeline = object.__new__(CameraPipeline)
    pipeline.roi_polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    # probe cy = (60 + 100) / 1.5 = 106.7 — BELOW the ROI even though the bbox
    # overlaps it. That is the /1.5 probe's character; polygons account for it.
    detection = _detection((40.0, 60.0, 60.0, 100.0))

    assert pipeline.filter_detections_to_roi([detection]) == []


def test_slot_assigner_uses_the_probe_before_overlap_fallback():
    slot = ParkingSlot(
        id="B8_CSBDO",
        polygon=Polygon([(30, 70), (70, 70), (70, 100), (30, 100)]),
    )
    # probe cy = (40 + 80) / 1.5 = 80 — inside the slot polygon (y 70..100).
    detection = _detection((40.0, 40.0, 60.0, 80.0))
    assignment = SlotAssigner(
        slots=[slot], config=AssignerConfig(overlap_threshold=0.3)
    ).assign([detection])

    assert assignment.slot_vehicle_map[slot.id][1] is detection
    assert assignment.evidence[slot.id]["method"] == "point"
