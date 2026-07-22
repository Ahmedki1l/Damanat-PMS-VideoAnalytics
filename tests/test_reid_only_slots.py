"""Slots whose plate is never in frame must be identified by appearance ALONE.

Two behaviours, both driven by physical facts about a camera's view:

  * `matching.slot_no_plate_view` — verified 2026-07-15 by pulling live frames: CAM-13
    fills 87% of its frame with B22's flank and OCR reads the burnt-in
    "CAM-13 (B2-PARKING)" caption at 0.805 conf; CAM-21 has logged 455 attempts and
    zero reads on B1_CRO. On those slots OCR is a DEAD path, not a slow one — it costs
    12 x ~200-670ms of the frame loop per park and then defers to appearance anyway.

  * the ReID retry — the 12-attempt budget covers the parking manoeuvre and was then
    spent FOREVER, giving appearance five shots ~4s apart on one pose in one light.
    The gallery is not static, so the same query can clear the margin an hour later.

Neither loosens a gate: `try_reid_identify_slot` is called with the same score/margin
either way. What changes is how often, and how early, it is ASKED.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from src.config import MatchingConfig
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin


class _FakeRegistry:
    def __init__(self, cfg, solo_answer=None):
        self.matching_config = cfg
        self._solo_answer = solo_answer
        self.ocr_calls = 0
        self.reid_calls = 0
        self.bound = []

    def get_slot_plate(self, slot_id):
        return None

    def try_ocr_identify_slot(self, slot_id, crop, cam_id, *, decision_ctx=None):
        self.ocr_calls += 1
        return None

    def plan_slot_ocr(self, slot_id, crop, cam_id, *, decision_ctx=None):
        del slot_id, crop, cam_id, decision_ctx
        return SimpleNamespace(allow_retry=False)

    def read_slot_plate(self, crop, allow_retry, *, slot_id=None):
        del crop, allow_retry, slot_id
        self.ocr_calls += 1
        return "", 0.0

    def confirm_slot_ocr(self, plan, crop, text, confidence):
        del plan, crop, text, confidence
        return None

    def try_reid_identify_slot(self, slot_id, crop, cam_id, *, is_reserved=False,
                               decision_ctx=None):
        self.reid_calls += 1
        return self._solo_answer

    def bind_plate_to_slot(self, slot_id, plate, cam_id, floor=None, source=""):
        self.bound.append((slot_id, plate, source))


class _Harness(ParkingEngineRuntimeMixin):
    """Only the collaborators _try_ocr_identify actually touches."""

    def __init__(self, cfg, solo_answer=None):
        self.vehicle_registry = _FakeRegistry(cfg, solo_answer)
        self.db_manager = None
        self._ocr_id_attempts, self._ocr_id_last_at, self._ocr_armed = {}, {}, {}
        self._reid_retry_last_at = {}
        self._reserved_for_map, self._special_slots = {}, set()
        self.pipelines, self.area_registry = {}, None

    def _bbox_crop(self, frame, detection):
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def _build_slot_snapshot_url(self, slot_id):
        return ""


class _SM:
    def __init__(self):
        self.identity = None

    def bind_identity(self, plate, url, confidence=0.0, lock=False):
        self.identity = (plate, confidence, lock)


def _cfg(no_plate_view=(), interval=60.0, ocr_gap=5.0):
    c = MatchingConfig()
    c.slot_reid_solo_enabled = True
    c.slot_no_plate_view = list(no_plate_view)
    c.slot_reid_retry_interval_s = interval
    c.slot_ocr_min_gap_s = ocr_gap
    return c


FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


class TestNoPlateViewSkipsOCR(unittest.TestCase):
    def test_listed_slot_never_calls_ocr_and_asks_reid_immediately(self):
        h = _Harness(_cfg(no_plate_view=["B22"]))
        h._arm_ocr_for_slot("B22")
        slot, sm = SimpleNamespace(id="B22"), _SM()
        h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        # OCR is the dead path here — it must not be spent at all...
        self.assertEqual(h.vehicle_registry.ocr_calls, 0)
        # ...and appearance is asked on the FIRST attempt, not the ninth.
        self.assertEqual(h.vehicle_registry.reid_calls, 1)

    def test_unlisted_slot_still_uses_ocr_first(self):
        h = _Harness(_cfg(no_plate_view=["B22"]))
        h._arm_ocr_for_slot("B15")
        slot, sm = SimpleNamespace(id="B15"), _SM()
        h._try_ocr_identify("CAM-14", FRAME, slot, sm, object())
        # OCR + ReID stay paired everywhere the camera CAN see a plate.
        self.assertEqual(h.vehicle_registry.ocr_calls, 1)
        self.assertEqual(h.vehicle_registry.reid_calls, 0)  # solo only after attempt 8

    def test_listed_slot_binds_provisionally_when_reid_is_sure(self):
        h = _Harness(_cfg(no_plate_view=["B22"]), solo_answer="ZRS-6511")
        h._arm_ocr_for_slot("B22")
        slot, sm = SimpleNamespace(id="B22"), _SM()
        h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        self.assertEqual(h.vehicle_registry.bound, [("B22", "ZRS-6511", "reid_solo")])
        plate, _conf, lock = sm.identity
        self.assertEqual(plate, "ZRS-6511")
        # NEVER locked: a solo bind is inference, and OCR must be able to overrule it.
        self.assertFalse(lock)


class TestOcrGroupPacing(unittest.TestCase):
    """One PaddleOCR read is 2-8s on the production Xeon and runs INLINE in the
    serial camera loop. The per-slot 0.7s interval cannot pace the loop — with
    ~10 armed slots (every post-restart boot) some slot is always eligible, so
    every frame paid a read: measured 2026-07-16 at ocr=8.8s of a 19s frame on
    b2-a. slot_ocr_min_gap_s is a PROCESS-wide floor between reads."""

    def test_second_slot_in_the_same_window_is_paced(self):
        h = _Harness(_cfg(ocr_gap=5.0))
        h._arm_ocr_for_slot("B15")
        h._arm_ocr_for_slot("B16")
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B15"), _SM(), object())
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B16"), _SM(), object())
        self.assertEqual(h.vehicle_registry.ocr_calls, 1)

    def test_pacing_does_not_consume_the_paced_slots_budget(self):
        # A skip is a deferral, not an attempt: B16 must keep its full budget
        # and an untouched per-slot clock for the next window.
        h = _Harness(_cfg(ocr_gap=5.0))
        h._arm_ocr_for_slot("B15")
        h._arm_ocr_for_slot("B16")
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B15"), _SM(), object())
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B16"), _SM(), object())
        self.assertEqual(h._ocr_id_attempts.get("B16", 0), 0)  # armed init, unspent
        self.assertEqual(h._ocr_id_last_at.get("B16", 0.0), 0.0)

    def test_paced_slot_reads_once_the_window_opens(self):
        h = _Harness(_cfg(ocr_gap=5.0))
        h._arm_ocr_for_slot("B15")
        h._arm_ocr_for_slot("B16")
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B15"), _SM(), object())
        h._ocr_group_last_at -= 6.0  # the gap elapses
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B16"), _SM(), object())
        self.assertEqual(h.vehicle_registry.ocr_calls, 2)

    def test_zero_gap_disables_pacing(self):
        h = _Harness(_cfg(ocr_gap=0.0))
        h._arm_ocr_for_slot("B15")
        h._arm_ocr_for_slot("B16")
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B15"), _SM(), object())
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B16"), _SM(), object())
        self.assertEqual(h.vehicle_registry.ocr_calls, 2)

    def test_no_plate_view_reid_is_not_paced(self):
        # The gate exists to pace OCR, the expensive path. Appearance-only slots
        # must keep answering even when OCR just spent the window.
        h = _Harness(_cfg(no_plate_view=["B22"], ocr_gap=5.0))
        h._arm_ocr_for_slot("B15")
        h._arm_ocr_for_slot("B22")
        h._try_ocr_identify("CAM-14", FRAME, SimpleNamespace(id="B15"), _SM(), object())
        h._try_ocr_identify("CAM-13", FRAME, SimpleNamespace(id="B22"), _SM(), object())
        self.assertEqual(h.vehicle_registry.reid_calls, 1)


class TestReidRetryCadence(unittest.TestCase):
    def test_retry_is_rate_limited(self):
        h = _Harness(_cfg(no_plate_view=["B22"], interval=60.0))
        h._arm_ocr_for_slot("B22")
        slot, sm = SimpleNamespace(id="B22"), _SM()
        for _ in range(5):  # five frames in quick succession
            h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        # Re-asking a noisy scorer until it gets lucky is exactly what the interval
        # exists to prevent — these slots have no OCR witness to catch a wrong answer.
        self.assertEqual(h.vehicle_registry.reid_calls, 1)

    def test_retry_fires_again_after_the_interval(self):
        h = _Harness(_cfg(no_plate_view=["B22"], interval=60.0))
        h._arm_ocr_for_slot("B22")
        slot, sm = SimpleNamespace(id="B22"), _SM()
        h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        h._reid_retry_last_at["B22"] -= 61.0  # pretend a minute passed
        h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        self.assertEqual(h.vehicle_registry.reid_calls, 2)

    def test_zero_interval_disables_the_retry(self):
        h = _Harness(_cfg(no_plate_view=["B22"], interval=0.0))
        h._arm_ocr_for_slot("B22")
        slot, sm = SimpleNamespace(id="B22"), _SM()
        h._try_ocr_identify("CAM-13", FRAME, slot, sm, object())
        self.assertEqual(h.vehicle_registry.reid_calls, 0)

    def test_readable_slot_retries_reid_once_ocr_budget_is_spent(self):
        # The regression that started this: a normal slot whose 12 OCR attempts all
        # failed used to return forever, so ReID was never asked again.
        h = _Harness(_cfg(interval=60.0))
        h._arm_ocr_for_slot("B7_CHRO")
        h._ocr_id_attempts["B7_CHRO"] = h._OCR_ID_MAX_ATTEMPTS  # budget spent
        slot, sm = SimpleNamespace(id="B7_CHRO"), _SM()
        h._try_ocr_identify("CAM-21", FRAME, slot, sm, object())
        self.assertEqual(h.vehicle_registry.ocr_calls, 0)
        self.assertEqual(h.vehicle_registry.reid_calls, 1)


if __name__ == "__main__":
    unittest.main()
