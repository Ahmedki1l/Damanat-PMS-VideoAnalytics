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
    # Never write synthetic fixtures into the real training data. The decision log is a
    # training corpus, and TRUE-CAR / TWIN-A rows would teach the ranker nonsense.
    cfg.decision_log_enabled = False
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


class TestReidSoloFallback:
    """When OCR can never read a slot's plate, appearance decides alone — but only when
    it is genuinely certain. The bar is the MARGIN over the runner-up, not the score:
    measured on 311 real queries a wrong car reaches 0.762 (above any usable score floor)
    but never wins by more than 0.099."""

    def _reg(self, registry):
        registry._matching_config.slot_reid_solo_enabled = True
        registry._matching_config.slot_reid_solo_min_score = 0.70
        registry._matching_config.slot_reid_solo_min_margin = 0.15
        registry._matching_config.slot_reid_solo_min_score_reserved = 0.75
        registry._matching_config.slot_reid_solo_min_margin_reserved = 0.20
        registry._matching_config.decision_log_enabled = False
        return registry

    def test_binds_when_appearance_is_decisive(self, registry, monkeypatch):
        reg = self._reg(registry)
        q = _vec(1, 0, 0)
        _add(reg, "TRUE-CAR", [(GATE_CAM, _vec(0.98, 0.20, 0))])   # ~0.98
        _add(reg, "OTHER", [(GATE_CAM, _vec(0.70, 0.71, 0))])      # ~0.70 -> margin ~0.28
        monkeypatch.setattr(reg.reid_matcher, "extract_feature", lambda _c: q)
        assert reg.try_reid_identify_slot("B1_CRO", np.ones((8, 8, 3), np.uint8),
                                          SLOT_CAM) == "TRUE-CAR"

    def test_abstains_on_a_car_it_has_never_seen(self, registry, monkeypatch):
        """The real B22/B1_CRO case. An unknown vehicle scores ~0.6 against EVERYTHING
        with a flat margin. Binding it would stamp a stranger with a known plate."""
        reg = self._reg(registry)
        q = _vec(1, 0, 0)
        _add(reg, "STRANGER-A", [(GATE_CAM, _vec(0.66, 0.75, 0))])   # 0.66
        _add(reg, "STRANGER-B", [(GATE_CAM, _vec(0.65, 0.76, 0))])   # 0.65 -> margin ~0.01
        monkeypatch.setattr(reg.reid_matcher, "extract_feature", lambda _c: q)
        assert reg.try_reid_identify_slot("B22", np.ones((8, 8, 3), np.uint8),
                                          SLOT_CAM) is None

    def test_high_score_but_flat_margin_is_refused(self, registry, monkeypatch):
        """Score alone is NOT a guard — a wrong car reaches 0.762 in cold. Two cars that
        both look like the query is exactly when we must not guess."""
        reg = self._reg(registry)
        q = _vec(1, 0, 0)
        _add(reg, "TWIN-A", [(GATE_CAM, _vec(0.99, 0.10, 0))])   # ~0.99, well over the floor
        _add(reg, "TWIN-B", [(GATE_CAM, _vec(0.98, 0.17, 0))])   # ~0.98 -> margin ~0.01
        monkeypatch.setattr(reg.reid_matcher, "extract_feature", lambda _c: q)
        assert reg.try_reid_identify_slot("B14", np.ones((8, 8, 3), np.uint8),
                                          SLOT_CAM) is None

    def test_reserved_slots_are_held_to_a_stricter_bar(self, registry, monkeypatch):
        """A wrong plate on a C-level slot raises a false intrusion alert against an
        executive, so reserved slots demand a wider margin than general ones."""
        reg = self._reg(registry)
        q = _vec(1, 0, 0)
        _add(reg, "TRUE-CAR", [(GATE_CAM, _vec(0.90, 0.44, 0))])   # ~0.90
        _add(reg, "OTHER", [(GATE_CAM, _vec(0.79, 0.61, 0))])      # ~0.79 -> margin ~0.11
        monkeypatch.setattr(reg.reid_matcher, "extract_feature", lambda _c: q)
        crop = np.ones((8, 8, 3), np.uint8)
        # margin 0.11 clears neither bar (general needs 0.15) — widen it and retest.
        _add(reg, "PADDING", [(GATE_CAM, _vec(0.10, 0.99, 0))])
        reg._matching_config.slot_reid_solo_min_margin = 0.10       # general: passes
        assert reg.try_reid_identify_slot("B14", crop, SLOT_CAM) == "TRUE-CAR"
        assert reg.try_reid_identify_slot("B13", crop, SLOT_CAM,
                                          is_reserved=True) is None  # reserved: 0.20, fails

    def test_a_solo_bind_never_claims_to_be_ocr_confirmed(self, registry):
        """It is inference, not evidence: it must not set ocr_confirmed, so
        save_parked_reference will not learn from it and the lock gate will not fire.

        It DOES still claim the slot (_locked_slots) — that is what mutual exclusion
        reads, and conflating "claimed" with "frozen against OCR" is what once put
        ERS-7949 in B17 and B19 at the same time. Correctability lives on the state
        machine lock and parking_slots.plate_locked, not here.
        """
        reg = self._reg(registry)
        s = _add(reg, "GUESSED", [(GATE_CAM, _vec(1, 0, 0))])
        assert reg.bind_plate_to_slot("B1_CRO", "GUESSED", SLOT_CAM, source="reid_solo")
        assert s.ocr_confirmed is False, "a guess must never masquerade as a read"
        assert "B1_CRO" in reg._locked_slots, "but it DOES claim the slot"

        s2 = _add(reg, "READ", [(GATE_CAM, _vec(0, 1, 0))])
        assert reg.bind_plate_to_slot("B14", "READ", SLOT_CAM, source="ocr")
        assert s2.ocr_confirmed is True
        assert "B14" in reg._locked_slots

    def test_disabled_by_config(self, registry, monkeypatch):
        reg = self._reg(registry)
        reg._matching_config.slot_reid_solo_enabled = False
        q = _vec(1, 0, 0)
        _add(reg, "TRUE-CAR", [(GATE_CAM, _vec(1, 0, 0))])
        _add(reg, "OTHER", [(GATE_CAM, _vec(0.1, 0.99, 0))])
        monkeypatch.setattr(reg.reid_matcher, "extract_feature", lambda _c: q)
        assert reg.try_reid_identify_slot("B1_CRO", np.ones((8, 8, 3), np.uint8),
                                          SLOT_CAM) is None


class TestOneCarOneSlot:
    """The production contract. It broke once, in exactly the way below: solo binds were
    kept OUT of _locked_slots on the reasoning that "only a READ plate should freeze the
    slot" — but _locked_slots is what mutual exclusion READS, so a solo-bound car became
    invisible to it and ERS-7949 ended up in B17 and B19 at the same time.

    A slot CLAIMS a car whatever named it. "Correctable by OCR" is a different property
    and lives elsewhere (state machine lock / parking_slots.plate_locked)."""

    def test_a_solo_bound_car_cannot_be_claimed_by_a_second_slot(self, registry):
        reg = registry
        reg._matching_config.slot_reid_solo_enabled = True
        q = _vec(1, 0, 0)
        s = _add(reg, "ERS-7949", [(GATE_CAM, _vec(1, 0, 0))])
        _add(reg, "OTHER", [(GATE_CAM, _vec(0.2, 0.98, 0))])

        # B19 takes it by appearance alone — NOT locked, but it IS claimed.
        assert reg.bind_plate_to_slot("B19", "ERS-7949", SLOT_CAM, source="reid_solo")
        assert "B19" in reg._locked_slots, (
            "a solo bind must still CLAIM the slot, or mutual exclusion cannot see it"
        )

        # B17, on the same camera, must no longer be offered that car.
        kept, rejected = reg.reid_rank(q, slot_id="B17", slot_camera=SLOT_CAM, k=5)
        assert "ERS-7949" not in [c.plate for c in kept]
        assert ("ERS-7949", "LOCKED_ELSEWHERE_LOCAL") in [
            (r.plate, r.reason) for r in rejected
        ]

    def test_the_slot_holding_it_can_still_re_confirm_its_own_car(self, registry):
        reg = registry
        _add(reg, "ERS-7949", [(GATE_CAM, _vec(1, 0, 0))])
        reg.bind_plate_to_slot("B19", "ERS-7949", SLOT_CAM, source="reid_solo")
        kept, _ = reg.reid_rank(_vec(1, 0, 0), slot_id="B19", slot_camera=SLOT_CAM, k=5)
        assert [c.plate for c in kept] == ["ERS-7949"]

    def test_relocating_releases_the_old_slot_for_the_db_too(self, registry):
        """Releasing only in memory left parking_slots holding the old row, so the same
        car was reported in two places by /api/slots."""
        reg = registry
        _add(reg, "ERS-7949", [(GATE_CAM, _vec(1, 0, 0))])
        reg.bind_plate_to_slot("B19", "ERS-7949", SLOT_CAM, source="reid_solo")
        reg.bind_plate_to_slot("B17", "ERS-7949", SLOT_CAM, source="ocr")

        assert "B19" not in reg._locked_slots
        assert reg.take_released_slots() == ["B19"]
        assert reg.take_released_slots() == []      # drained once

    def test_a_solo_bind_is_still_correctable_by_ocr(self, registry):
        """Claiming the slot must NOT make it look OCR-confirmed."""
        reg = registry
        s = _add(reg, "GUESSED", [(GATE_CAM, _vec(1, 0, 0))])
        reg.bind_plate_to_slot("B14", "GUESSED", SLOT_CAM, source="reid_solo")
        assert s.ocr_confirmed is False
