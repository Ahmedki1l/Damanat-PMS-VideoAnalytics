"""Pin the point-mode sanity floor that stops one car occupying two slots.

THE BUG THIS PREVENTS (production, 2026-08-10, B1 group)
--------------------------------------------------------
B5 and B9 read OCCUPIED continuously while empty, because a car parked in the
ADJACENT bay claimed them from a different camera:

    CAM-06 / B6  (the real car)   probe OUTSIDE its own polygon, overlap 0.340
    CAM-08 / B5  (the ghost)      probe INSIDE,  overlap 0.000
    CAM-20 / B10 (the real car)   probe OUTSIDE its own polygon, overlap 0.395
    CAM-07 / B9  (the ghost)      probe INSIDE,  overlap 0.393

One physical car, detected by two cameras, in two worker processes, committed
twice. Nothing arbitrates across cameras — slot polygons are in per-camera pixel
coordinates and are not comparable (engine_tracking._detection_in_own_slot spells
this out), so the assigner's "one vehicle can only be in one slot" break is
per-camera only and cannot see the conflict.

The B5 case is the one a rule can decide on its own: the box does not touch the
polygon AT ALL (overlap 0.000) while the (y1+y2)/1.5 probe, which sits ~27px
BELOW the box's own bottom edge, lands inside. Because the point rule assigns and
breaks, overlap_threshold is never consulted — no threshold could ever catch it.
``point_min_overlap`` is the floor that does.

The B9 case (overlap 0.393) is NOT catchable this way and is deliberately not
tested as if it were: a polygon drawn over a neighbouring bay has to be redrawn.
These tests exist so a future polygon edit or probe change cannot quietly bring
the B5 shape of fault back.

Coordinates below are the real logged values, not invented ones.
"""
import pytest
from shapely.geometry import Polygon

from src.config import AssignerConfig
from src.core.slot_assigner import SlotAssigner
from src.detection.detector import Detection
from src.models.slot import ParkingSlot

# --- Real production geometry, 2026-08-10 -----------------------------------
B5_CAM08 = [(723, 423), (1014, 428), (906, 294), (725, 297)]
B9_CAM07 = [(792, 342), (555, 321), (413, 456), (709, 489)]
B6_CAM06 = [(170, 381), (413, 366), (506, 494), (207, 509)]

# The car parked in B6, as CAM-08 sees it. Its box stops at y=292; B5's polygon
# starts at y=294. Eight consecutive observations, all identical to ~1px.
GHOST_BOX_ON_B5 = (725.0, 186.3, 832.0, 291.7)

# The same car as its OWN camera sees it — this one must keep working.
REAL_BOX_ON_B6 = (265.7, 258.8, 572.6, 498.6)


def _slot(slot_id, points):
    return ParkingSlot(id=slot_id, polygon=Polygon(points))


def _detection(bbox, track_id=1):
    return Detection(bbox=bbox, class_id=2, confidence=0.9, track_id=track_id)


def _assign(slots, detections, **overrides):
    config = AssignerConfig(**{"overlap_threshold": 0.2, **overrides})
    return SlotAssigner(slots=slots, config=config).assign(detections)


def test_the_ghost_probe_really_is_inside_b5_with_a_box_that_never_touches_it():
    """Guard the premise. If this fails the geometry moved and the rest is moot."""
    slot = _slot("B5", B5_CAM08)
    detection = _detection(GHOST_BOX_ON_B5)

    from shapely.geometry import Point

    assert slot.polygon.contains(Point(*detection.bottom_center)), (
        "the probe no longer lands inside B5 — polygons were redrawn; re-measure "
        "before trusting this module"
    )
    assert not slot.polygon.intersects(
        Polygon([
            (GHOST_BOX_ON_B5[0], GHOST_BOX_ON_B5[1]),
            (GHOST_BOX_ON_B5[2], GHOST_BOX_ON_B5[1]),
            (GHOST_BOX_ON_B5[2], GHOST_BOX_ON_B5[3]),
            (GHOST_BOX_ON_B5[0], GHOST_BOX_ON_B5[3]),
        ])
    ), "the box now touches B5 — this is no longer a zero-overlap phantom"


def test_neighbour_car_cannot_claim_a_slot_its_box_does_not_touch():
    """THE regression. B5 must stay empty while a car sits in the next bay."""
    assignment = _assign([_slot("B5", B5_CAM08)], [_detection(GHOST_BOX_ON_B5)])

    assert "B5" not in assignment.slot_vehicle_map, (
        "B5 was claimed by a car whose bounding box does not touch B5's polygon. "
        "This is the 2026-08-10 double-occupancy bug: one car, two cameras, two "
        "slots. Check assigner.point_min_overlap."
    )
    assert len(assignment.unassigned) == 1
    assert assignment.refused == [("B5", 0.0)]


def test_the_floor_is_what_refuses_it_not_something_else():
    """With the floor disabled the old behaviour returns — proving the floor is
    load-bearing and this test is not passing for an unrelated reason."""
    assignment = _assign(
        [_slot("B5", B5_CAM08)], [_detection(GHOST_BOX_ON_B5)], point_min_overlap=0.0
    )

    assert assignment.slot_vehicle_map["B5"][1].track_id == 1
    assert assignment.evidence["B5"]["method"] == "point"
    assert assignment.evidence["B5"]["overlap"] == 0.0
    assert assignment.refused == []


def test_real_car_on_its_own_camera_is_unaffected():
    """CAM-06 assigns B6 through the overlap FALLBACK (its probe falls outside
    its own polygon — 0.340 overlap). The floor must not disturb that path."""
    assignment = _assign([_slot("B6", B6_CAM06)], [_detection(REAL_BOX_ON_B6)])

    assert assignment.slot_vehicle_map["B6"][1].track_id == 1
    assert assignment.evidence["B6"]["method"] == "overlap"
    assert assignment.evidence["B6"]["overlap"] == pytest.approx(0.34, abs=0.02)


@pytest.mark.parametrize("overlap_seen", [0.310, 0.360, 0.393, 0.560])
def test_floor_sits_far_below_every_genuinely_occupied_slot_ever_measured(
    overlap_seen,
):
    """526 [SLOTHOLD] lines put real occupancy at min 0.310. The floor is 0.05 —
    if someone raises it toward the coverage range it starts deleting live
    occupancy, which is what 0.50 did on 2026-08-10. 0.393 is the B9 ghost and is
    included deliberately: it is ABOVE the floor, i.e. the floor does not pretend
    to fix a polygon drawn on the wrong bay."""
    assert AssignerConfig().point_min_overlap < 0.310
    assert AssignerConfig().point_min_overlap < overlap_seen


def test_floor_is_clamped_when_configured_above_the_fallback_gate():
    """An incoherent config must warn and degrade, not raise — deployed config
    cannot be allowed to take the engine down."""
    assigner = SlotAssigner(
        slots=[_slot("B5", B5_CAM08)],
        config=AssignerConfig(overlap_threshold=0.2, point_min_overlap=0.9),
    )

    assert assigner.point_min_overlap == 0.2


def test_coverage_mode_ignores_the_point_floor_entirely():
    """Coverage mode has no point probe, so the floor must not apply there. The
    ghost is refused anyway, by coverage_threshold — for a different reason."""
    assignment = _assign(
        [_slot("B5", B5_CAM08)],
        [_detection(GHOST_BOX_ON_B5)],
        assignment_mode="coverage",
        coverage_threshold=0.25,
    )

    assert "B5" not in assignment.slot_vehicle_map
    assert assignment.refused == []
