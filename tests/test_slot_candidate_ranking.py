"""Slot-path candidate ranking: mutual exclusion + slot-pose scoring.

Guards the two Phase-0 rules that decide WHICH cars OCR is allowed to confirm at a
parked slot. Both are cheap deterministic signals that sit in front of the expensive
ones, so a regression here silently degrades every identity the system writes.
"""
import datetime as dt
import uuid

import numpy as np
import pytest

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import VehicleSession

GATE_CAM = "CAM-03"     # a ground-truth camera (full weight)
SLOT_CAM = "CAM-08"     # an ordinary slot camera (secondary weight)
OTHER_CAM = "CAM-17"


def _vec(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def registry():
    cfg = MatchingConfig()
    cfg.ground_truth_cameras = (GATE_CAM,)
    cfg.secondary_camera_weight = 0.6
    cfg.slot_camera_ref_weight = 0.8
    return VehicleRegistry(matching_config=cfg)


def _add(reg, plate, refs, linked_slot=None):
    """refs: list of (camera, vector). The first is also the full-weight primary."""
    s = VehicleSession(session_id=str(uuid.uuid4()), plate=plate)
    s.feature_vector = refs[0][1]
    s.reference_feature_vectors = [v for _c, v in refs]
    s.reference_source_cameras = [c for c, _v in refs]
    s.linked_slot = linked_slot
    reg._sessions[s.session_id] = s
    return s


class TestMutualExclusion:
    """A car cannot occupy two slots, so a car locked into another one is not this car."""

    def test_locked_elsewhere_is_dropped_and_rank6_promotes(self, registry):
        # The confuser out-scores the true car — exactly the live failure mode.
        query = _vec(1, 0, 0)
        _add(registry, "CONFUSER", [(GATE_CAM, _vec(0.99, 0.14, 0))])
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(0.90, 0.44, 0))])
        _add(registry, "FILLER-1", [(GATE_CAM, _vec(0.80, 0.60, 0))])

        kept, _ = registry.reid_rank(query, slot_id="B5", k=2)
        assert [c.plate for c in kept] == ["CONFUSER", "TRUE-CAR"]

        registry.set_external_plate_locks(
            {"CONFUSER": {"slot_id": "B26", "camera_id": OTHER_CAM,
                          "locked": True, "locked_at": dt.datetime(2026, 7, 13, 10, 0)}}
        )
        kept, rejected = registry.reid_rank(query, slot_id="B5", k=2)

        assert [c.plate for c in kept] == ["TRUE-CAR", "FILLER-1"], (
            "the rejected candidate must be dropped BEFORE the top-k slice, so the "
            "next car promotes in — the gate is recall-additive, never subtractive"
        )
        assert [(r.plate, r.reason) for r in rejected] == [
            ("CONFUSER", "LOCKED_ELSEWHERE_DB")
        ]
        assert rejected[0].raw_rank == 1
        assert kept[0].rank == 1 and kept[1].rank == 2

    def test_lock_on_this_same_slot_does_not_exclude(self, registry):
        """Re-identifying the car already in THIS slot must not veto itself."""
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(1, 0, 0))])
        registry.set_external_plate_locks(
            {"TRUE-CAR": {"slot_id": "B5", "camera_id": SLOT_CAM,
                          "locked": True, "locked_at": dt.datetime(2026, 7, 13, 10, 0)}}
        )
        kept, rejected = registry.reid_rank(query, slot_id="B5", k=5)
        assert [c.plate for c in kept] == ["TRUE-CAR"]
        assert rejected == []

    def test_reentry_after_the_lock_makes_it_a_ghost(self, registry):
        """The car left and drove back in. The old lock is stale — keep the candidate.

        Deliberately keyed on a re-entry event, NOT a TTL: a car parked overnight holds
        a 12-hour-old lock that is perfectly valid.
        """
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(1, 0, 0))])
        registry.set_external_plate_locks(
            {"TRUE-CAR": {"slot_id": "B26", "camera_id": OTHER_CAM,
                          "locked": True, "locked_at": dt.datetime(2026, 7, 13, 10, 0)}}
        )
        assert registry.reid_rank(query, slot_id="B5", k=5)[0] == []

        registry._last_anpr_entry_at["TRUE-CAR"] = dt.datetime(2026, 7, 13, 11, 30)
        kept, _ = registry.reid_rank(query, slot_id="B5", k=5)
        assert [c.plate for c in kept] == ["TRUE-CAR"]

    def test_an_overnight_lock_is_still_honoured(self, registry):
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(1, 0, 0))])
        registry.set_external_plate_locks(
            {"TRUE-CAR": {"slot_id": "B26", "camera_id": OTHER_CAM,
                          "locked": True, "locked_at": dt.datetime(2026, 7, 12, 20, 0)}}
        )
        # Entered LONG before the lock — it never left. The lock stands.
        registry._last_anpr_entry_at["TRUE-CAR"] = dt.datetime(2026, 7, 12, 19, 0)
        kept, rejected = registry.reid_rank(query, slot_id="B5", k=5)
        assert kept == []
        assert rejected[0].reason == "LOCKED_ELSEWHERE_DB"

    def test_can_be_disabled_by_config(self, registry):
        registry._matching_config.exclude_plates_locked_elsewhere = False
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(1, 0, 0))])
        registry.set_external_plate_locks(
            {"TRUE-CAR": {"slot_id": "B26", "camera_id": OTHER_CAM,
                          "locked": True, "locked_at": dt.datetime(2026, 7, 13, 10, 0)}}
        )
        kept, rejected = registry.reid_rank(query, slot_id="B5", k=5)
        assert [c.plate for c in kept] == ["TRUE-CAR"]
        assert rejected == []


class TestSlotPoseScoring:
    """A reference taught by the SAME camera now looking at the car is not an oblique
    guess — it is the car in the exact pose being scored."""

    def test_same_camera_ref_outranks_a_stranger_gate_photo(self, registry):
        """The whole point: at the old 0.6 weight, a car's own parked pose (0.95 * 0.6
        = 0.57) LOST to a different car's full-weight gate photo (0.70). It could never
        be recognised on a return visit. At 0.8 it wins (0.76 > 0.70)."""
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(0.50, 0.87, 0)),      # weak cross-view
                                    (SLOT_CAM, _vec(0.95, 0.31, 0))])     # strong same-view
        _add(registry, "STRANGER", [(GATE_CAM, _vec(0.70, 0.71, 0))])     # decent gate photo

        without = registry.reid_rank(query, slot_id="B5", k=2)[0]
        assert without[0].plate == "STRANGER", "no slot_camera -> old weighting applies"

        kept, _ = registry.reid_rank(query, slot_id="B5", slot_camera=SLOT_CAM, k=2)
        assert kept[0].plate == "TRUE-CAR"
        assert kept[0].warm is True
        assert kept[0].same_view_score == pytest.approx(0.95, abs=0.02)

    def test_weight_stops_short_of_full_so_regulars_do_not_take_over(self, registry):
        """The uplift is symmetric — it also lifts cars that park here every day. A
        same-view match between DIFFERENT cars (~0.66) can beat a cross-view match on
        the SAME car (~0.55), which is what the discount exists to suppress. At 1.0 the
        regular wins; at 0.8 the true car still does."""
        query = _vec(1, 0, 0)
        # True car has never parked here: only a weak cross-view gate ref.
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(0.72, 0.69, 0))])          # 0.72 * 1.0
        # A regular who parks here daily: same-view ref, but it is a DIFFERENT car.
        _add(registry, "REGULAR", [(GATE_CAM, _vec(0.30, 0.95, 0)),
                                   (SLOT_CAM, _vec(0.85, 0.53, 0))])           # 0.85 * w

        kept, _ = registry.reid_rank(query, slot_id="B5", slot_camera=SLOT_CAM, k=2)
        assert kept[0].plate == "TRUE-CAR", (
            "0.85 * 0.8 = 0.68 < 0.72 — the regular must not steal it"
        )

        registry._matching_config.slot_camera_ref_weight = 1.0
        kept, _ = registry.reid_rank(query, slot_id="B5", slot_camera=SLOT_CAM, k=2)
        assert kept[0].plate == "REGULAR", (
            "0.85 * 1.0 = 0.85 > 0.72 — full weight hands it to the wrong car; this is "
            "why slot_camera_ref_weight is 0.8 and not 1.0"
        )

    def test_cold_car_reports_not_warm(self, registry):
        query = _vec(1, 0, 0)
        _add(registry, "TRUE-CAR", [(GATE_CAM, _vec(0.9, 0.44, 0))])
        kept, _ = registry.reid_rank(query, slot_id="B5", slot_camera=SLOT_CAM, k=2)
        assert kept[0].warm is False
        assert kept[0].same_view_score == 0.0
        assert kept[0].cross_view_score > 0.0


class TestPhantomPlateFilter:
    """ANPR misreads open sessions for cars that do not exist. They carry a plate and no
    photo, so they can never be matched — they can only collide."""

    def _phantom_scenario(self, reg):
        # The real car, seen and photographed at the gate.
        _add(reg, "DJS-7842", [(GATE_CAM, _vec(1, 0, 0))])
        # Two ANPR misreads of that same plate. PMS-AI opened a session for each; VA
        # hydrated them, but there is no car to photograph, so they hold no vectors.
        for junk in ("BJA-7842", "DJA-7842"):
            s = VehicleSession(session_id=str(uuid.uuid4()), plate=junk)
            s.feature_vector = None
            s.reference_feature_vectors = []
            reg._sessions[s.session_id] = s

    def test_phantoms_are_not_candidates(self, registry):
        self._phantom_scenario(registry)
        assert registry.plates_inside() == ["DJS-7842"]
        assert registry.plates_inside(require_appearance=False) == [
            "BJA-7842", "DJA-7842", "DJS-7842",
        ]

    def test_phantoms_would_otherwise_force_confirm_plate_to_abstain(self, registry):
        """The bug this closes: confirm_plate matches on the DIGIT RUN and abstains when
        a read fits more than one candidate. The two phantoms share 7842 with the real
        plate, so a PERFECT read of the real car becomes a three-way tie."""
        from src.matching.plate_ocr_match import confirm_plate

        self._phantom_scenario(registry)
        perfect_read = "7842DJS"  # how PaddleOCR renders DJS-7842 (runs reordered)

        unfiltered = registry.plates_inside(require_appearance=False)
        assert confirm_plate(perfect_read, unfiltered) is None, (
            "three plates share the digits 7842 -> ambiguous -> slot stays NULL"
        )

        filtered = registry.plates_inside()
        assert confirm_plate(perfect_read, filtered) == "DJS-7842"

    def test_reid_path_was_already_immune(self, registry):
        """A phantom has no feature vector, so reid_shortlist never scored it. The filter
        closes the plates_inside() FALLBACK, not the main path."""
        self._phantom_scenario(registry)
        kept, _ = registry.reid_rank(_vec(1, 0, 0), slot_id="B5", k=5)
        assert [c.plate for c in kept] == ["DJS-7842"]

    def test_filter_can_be_disabled(self, registry):
        registry._matching_config.candidates_require_appearance_evidence = False
        self._phantom_scenario(registry)
        assert len(registry.plates_inside()) == 3
