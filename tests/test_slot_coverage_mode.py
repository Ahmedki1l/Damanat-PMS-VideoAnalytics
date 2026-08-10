"""Coverage-mode slot assignment: membership decided by box coverage, not a probe point.

``assignment_mode="coverage"`` replaces the bottom-center point-in-polygon rule
with a single question: does the vehicle's box cover at least
``coverage_threshold`` of itself inside the slot polygon?

These tests pin the two things that make the mode worth having — it refuses a
car whose probe clips a polygon it is barely inside, and it picks the slot a
straddling box is *mostly* in — plus the guarantee that switching mode is opt-in
and a bad mode string cannot take the engine down.

Companion to test_slot_probe_geometry.py, which pins point mode.
"""
import pytest
from shapely.geometry import Polygon

from src.config import AssignerConfig, _parse_assigner_overrides
from src.core.slot_assigner import SlotAssigner
from src.detection.detector import Detection
from src.models.slot import ParkingSlot

# Unit square-ish slot, 100x100 at the origin.
LEFT = ParkingSlot(id="L1", polygon=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
RIGHT = ParkingSlot(
    id="R1", polygon=Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])
)


def _detection(bbox, track_id=7):
    return Detection(bbox=bbox, class_id=2, confidence=0.9, track_id=track_id)


def _assign(slots, detections, **cfg):
    cfg.setdefault("overlap_threshold", 0.3)
    return SlotAssigner(slots=slots, config=AssignerConfig(**cfg)).assign(detections)


def test_point_mode_is_the_default_so_switching_is_opt_in():
    """No config change = no behaviour change. The production polygons are
    calibrated to the point probe; coverage mode must never arrive by accident."""
    assigner = SlotAssigner(slots=[LEFT], config=AssignerConfig())
    assert assigner.assignment_mode == "point"


def test_unknown_mode_degrades_to_point_instead_of_raising():
    """A typo in deployed config must not crash-loop the engine."""
    assigner = SlotAssigner(
        slots=[LEFT], config=AssignerConfig(assignment_mode="coverge")
    )
    assert assigner.assignment_mode == "point"


def test_coverage_mode_refuses_a_car_whose_probe_is_inside_but_box_is_mostly_out():
    """The whole point of the change.

    bbox (20, -100, 80, 140): the probe sits at ((20+80)/2, (-100+140)/1.5) =
    (50, 26.7) — comfortably inside the polygon, so POINT mode calls this slot
    occupied. But only 100 of the box's 240 rows of height are inside the slot,
    so it covers just 41.7% of itself — coverage mode calls it a miss.
    """
    bbox = (20.0, -100.0, 80.0, 140.0)

    point_mode = _assign([LEFT], [_detection(bbox)])
    assert LEFT.id in point_mode.slot_vehicle_map
    assert point_mode.evidence[LEFT.id]["method"] == "point"

    coverage_mode = _assign(
        [LEFT], [_detection(bbox)], assignment_mode="coverage", coverage_threshold=0.5
    )
    assert coverage_mode.slot_vehicle_map == {}
    assert len(coverage_mode.unassigned) == 1


def test_coverage_mode_claims_a_car_fully_inside_whose_probe_fell_below_the_polygon():
    """bbox (10, 60, 90, 99) is entirely within the slot, but its probe lands at
    y=106 — below the polygon. Point mode only rescues this via the overlap
    fallback; coverage mode assigns it directly, and the evidence says so."""
    bbox = (10.0, 60.0, 90.0, 99.0)

    point_mode = _assign([LEFT], [_detection(bbox)])
    assert point_mode.evidence[LEFT.id]["method"] == "overlap"

    coverage_mode = _assign(
        [LEFT], [_detection(bbox)], assignment_mode="coverage", coverage_threshold=0.5
    )
    assert coverage_mode.slot_vehicle_map[LEFT.id][0] == 7
    evidence = coverage_mode.evidence[LEFT.id]
    assert evidence["method"] == "coverage"
    assert evidence["overlap"] == pytest.approx(1.0, abs=1e-3)


def test_straddling_box_goes_to_the_slot_it_covers_most_not_the_first_over_the_bar():
    """bbox x 70..190 covers 25% of itself in L1 and 75% in R1. With the gate at
    0.20 BOTH slots qualify, and L1 is scanned first — so a first-match rule
    would park the car in the wrong slot. Coverage mode takes the maximum."""
    detection = _detection((70.0, 20.0, 190.0, 60.0))

    assignment = _assign(
        [LEFT, RIGHT],
        [detection],
        assignment_mode="coverage",
        coverage_threshold=0.20,
    )

    assert RIGHT.id in assignment.slot_vehicle_map
    assert LEFT.id not in assignment.slot_vehicle_map
    evidence = assignment.evidence[RIGHT.id]
    assert evidence["overlap"] == pytest.approx(0.75, abs=1e-3)
    # The slot it lost is still logged, so a boundary dispute is visible.
    assert evidence["rivals"] == [(LEFT.id, pytest.approx(0.25, abs=1e-3))]


def test_contested_slot_goes_to_the_better_covering_vehicle_not_the_nearer_one():
    """Two cars claim one slot. POINT mode breaks the tie on centroid distance,
    which hands the slot to a huge box whose probe happens to land dead centre.
    Coverage mode gives it to the car actually sitting in the slot.

    ``sprawling``: probe lands exactly on the centroid (distance 0) but only 40%
    of the box is inside. ``parked``: entirely inside (100%), probe 68px away.
    """
    slot = ParkingSlot(
        id="C1", polygon=Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    )
    sprawling = _detection((-100.0, -50.0, 300.0, 200.0), track_id=1)
    parked = _detection((10.0, 10.0, 60.0, 110.0), track_id=2)

    point_mode = _assign([slot], [sprawling, parked])
    assert point_mode.slot_vehicle_map[slot.id][0] == 1  # nearest probe wins

    coverage_mode = _assign(
        [slot],
        [sprawling, parked],
        assignment_mode="coverage",
        coverage_threshold=0.30,
    )
    assert coverage_mode.slot_vehicle_map[slot.id][0] == 2  # best coverage wins
    assert coverage_mode.evidence[slot.id]["overlap"] == pytest.approx(1.0, abs=1e-3)
    # The loser is reported unassigned, not silently dropped.
    assert [d.track_id for d in coverage_mode.unassigned] == [1]


def test_coverage_threshold_is_inclusive():
    """``coverage_threshold`` reads as "at least this much", so a box sitting
    exactly on the bar is IN. Half of this box's height is inside the slot."""
    detection = _detection((10.0, 0.0, 90.0, 200.0))  # 100 of 200 rows inside

    assignment = _assign(
        [LEFT], [detection], assignment_mode="coverage", coverage_threshold=0.5
    )

    assert assignment.slot_vehicle_map[LEFT.id][1] is detection
    assert assignment.evidence[LEFT.id]["overlap"] == pytest.approx(0.5, abs=1e-3)


def test_default_threshold_clears_the_perspective_ceiling_of_a_sharp_slot():
    """Coverage has a ceiling that is invisible from the config file.

    Slot polygons are perspective trapezoids; detection boxes are axis-aligned
    rectangles. A car FILLING its slot still hangs its box corners outside the
    polygon, so it cannot score 1.0 — and on a sharply-angled slot it cannot
    even score 0.5. Set the gate above a slot's ceiling and that slot reads
    permanently VACANT with no error and no log.

    These are the REAL vertices of G2, the sharpest slot in the authored B1
    geometry — a diagonal bay, so its quad sits at ~30 degrees to the image axes
    and fills only 40.8% of its own bounding box. A car filling it must still be
    found at the shipped default.
    """
    sharp = ParkingSlot(
        id="G2",
        polygon=Polygon([(440, 298), (582, 423), (1000, 202), (907, 119)]),
    )
    # A car filling the slot: the detection box is the polygon's own bounds.
    x1, y1, x2, y2 = sharp.polygon.bounds
    filling_the_slot = _detection((x1, y1, x2, y2))

    default_threshold = AssignerConfig().coverage_threshold
    assignment = _assign(
        [sharp],
        [filling_the_slot],
        assignment_mode="coverage",
        coverage_threshold=default_threshold,
    )

    ceiling = assignment.evidence.get(sharp.id, {}).get("overlap")
    assert sharp.id in assignment.slot_vehicle_map, (
        f"A car filling this slot scores only {ceiling}, below the shipped "
        f"coverage_threshold of {default_threshold} — the slot would sit "
        f"permanently VACANT. Re-run tools/calibrate_coverage_threshold.py "
        f"before raising the default."
    )
    # Pin the ceiling itself. 0.41 — well under the intuitive 0.5 gate, which is
    # exactly why the shipped default is 0.30.
    assert ceiling == pytest.approx(0.408, abs=0.01)


class TestPerCameraRollout:
    """Coverage mode has to be stageable onto one camera at a time.

    Every camera's polygons were drawn against the point probe and need their
    own footage validation, so an all-or-nothing switch is not shippable.
    """

    def test_two_cameras_can_run_different_modes_at_once(self):
        config = AssignerConfig(
            assignment_mode="point",
            camera_overrides=_parse_assigner_overrides(
                {"CAM-05": {"assignment_mode": "coverage", "coverage_threshold": 0.25}}
            ),
        )

        staged = config.resolve_for_camera("CAM-05")
        untouched = config.resolve_for_camera("CAM-07")

        assert SlotAssigner([LEFT], staged).assignment_mode == "coverage"
        assert SlotAssigner([LEFT], staged).coverage_threshold == 0.25
        assert SlotAssigner([LEFT], untouched).assignment_mode == "point"

    def test_override_matches_however_the_camera_id_was_spelled(self):
        """Camera ids reach the engine as CAM-05, cam_05 and CAM05 depending on
        the source. The override must not miss because of punctuation."""
        config = AssignerConfig(
            camera_overrides=_parse_assigner_overrides(
                {"cam_05": {"assignment_mode": "coverage"}}
            ),
        )

        for spelling in ("CAM-05", "cam-05", "CAM_05", "CAM05"):
            assert config.resolve_for_camera(spelling).assignment_mode == "coverage"

    def test_a_camera_without_an_override_shares_the_global_object(self):
        """The common path must not allocate a copy per camera per rebuild."""
        config = AssignerConfig(
            camera_overrides=_parse_assigner_overrides(
                {"CAM-05": {"assignment_mode": "coverage"}}
            )
        )
        assert config.resolve_for_camera("CAM-07") is config

    def test_override_keeps_unspecified_fields_from_the_global_config(self):
        config = AssignerConfig(
            overlap_threshold=0.42,
            coverage_threshold=0.30,
            camera_overrides=_parse_assigner_overrides(
                {"CAM-05": {"assignment_mode": "coverage"}}
            ),
        )

        staged = config.resolve_for_camera("CAM-05")

        assert staged.assignment_mode == "coverage"
        assert staged.coverage_threshold == 0.30
        assert staged.overlap_threshold == 0.42

    def test_a_typo_in_an_override_fails_loudly_instead_of_silently_no_opping(self):
        """A misspelled key would leave the camera on the global rule with no
        signal — the exact failure the override exists to make visible."""
        with pytest.raises(ValueError, match="unknown key"):
            _parse_assigner_overrides({"CAM-05": {"assignment_mod": "coverage"}})

        with pytest.raises(ValueError, match="must be one of"):
            _parse_assigner_overrides({"CAM-05": {"assignment_mode": "covrage"}})

        with pytest.raises(ValueError, match="between 0 and 1"):
            _parse_assigner_overrides({"CAM-05": {"coverage_threshold": 30}})


def test_coverage_mode_never_falls_back_to_the_point_probe():
    """A vehicle that touches nothing is unassigned even if its probe would have
    landed in a slot under point mode — coverage is the only rule in this mode."""
    detection = _detection((20.0, -100.0, 80.0, 140.0))

    assignment = _assign(
        [LEFT], [detection], assignment_mode="coverage", coverage_threshold=0.99
    )

    assert assignment.slot_vehicle_map == {}
    assert assignment.unassigned == [detection]
