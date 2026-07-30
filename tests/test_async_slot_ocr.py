"""Async slot-identify OCR: the read runs off the camera loop, the bind stays on it.

Two things must hold, and both are safety properties, not performance ones:

  * The worker only ever performs the PaddleOCR READ. Ranking (ReID) and the
    confirm/bind stay on the main thread, because they touch OpenVINO models the
    tracking loop also uses.
  * A read that finishes AFTER the car left — or after the slot was re-armed for a
    different car, or after another path already named it — is DISCARDED. Async may
    bind a frame late; it must never bind a stale plate. `current_plate` stays
    CORRECT-or-NULL, exactly as the inline path guarantees.
"""
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from src.config import MatchingConfig
from src.core.engine.async_slot_ocr import AsyncSlotOcr, SlotOcrJob
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin


# --------------------------------------------------------------------------- #
# The worker in isolation
# --------------------------------------------------------------------------- #
class _Plan:
    allow_retry = True
    decision_ctx = {"is_reserved": False}


def _job(slot="B1", token=1, crop=None):
    return SlotOcrJob(
        slot_id=slot, cam_id="CAM-04",
        crop=crop if crop is not None else np.zeros((8, 8, 3), np.uint8),
        plan=_Plan(), attempts=1, token=token,
    )


class TestAsyncWorker(unittest.TestCase):
    def test_read_runs_and_result_drains(self):
        seen = []
        w = AsyncSlotOcr(read_fn=lambda crop, retry: (seen.append(retry) or ("ABC123", 0.9)))
        w.start()
        try:
            self.assertTrue(w.submit(_job()))
            self._wait(lambda: w.drain_ready())
            results = w.drain()
        finally:
            w.stop()
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0].text, results[0].conf), ("ABC123", 0.9))
        self.assertEqual(seen, [True])  # allow_retry threaded through

    def test_one_read_per_slot_in_flight(self):
        # While a slot's read is IN FLIGHT, a second submit for it is dropped.
        # (A second submit while it is still merely QUEUED coalesces instead —
        # see test_newer_crop_coalesces_before_pickup.)
        started = threading.Event()
        gate = threading.Event()
        def slow(crop, retry):
            started.set()
            gate.wait(2.0)
            return ("X", 1.0)
        w = AsyncSlotOcr(read_fn=slow)
        w.start()
        try:
            self.assertTrue(w.submit(_job()))
            self.assertTrue(started.wait(2.0))  # read has begun -> in flight
            self.assertFalse(w.submit(_job()), "second read for the same slot must drop")
            gate.set()
        finally:
            w.stop()

    def test_newer_crop_coalesces_before_pickup(self):
        gate = threading.Event()
        reads = []
        def blocked(crop, retry):
            gate.wait(2.0)
            reads.append(int(crop[0, 0, 0]))
            return ("X", 1.0)
        w = AsyncSlotOcr(read_fn=blocked, max_pending=8)
        w.start()
        try:
            # Occupy the worker with an unrelated slot so B1's jobs queue up.
            w.submit(_job(slot="OTHER"))
            self._wait(lambda: w.pending_or_inflight("OTHER"))
            w.submit(_job(slot="B1", crop=np.full((8, 8, 3), 1, np.uint8)))
            w.submit(_job(slot="B1", crop=np.full((8, 8, 3), 2, np.uint8)))
            gate.set()
            self._wait(lambda: len(reads) >= 2)
        finally:
            w.stop()
        # B1 read exactly once, on the NEWER crop (value 2).
        self.assertEqual(reads.count(2), 1)
        self.assertNotIn(1, reads)

    def test_forget_drops_a_queued_read(self):
        gate = threading.Event()
        def blocked(crop, retry):
            gate.wait(2.0)
            return ("X", 1.0)
        w = AsyncSlotOcr(read_fn=blocked)
        w.start()
        try:
            w.submit(_job(slot="BUSY"))
            self._wait(lambda: w.pending_or_inflight("BUSY"))
            w.submit(_job(slot="B1"))
            w.forget("B1")
            gate.set()
            time.sleep(0.2)
            self.assertNotIn("B1", [r.job.slot_id for r in w.drain()])
        finally:
            w.stop()

    def test_read_exception_does_not_kill_worker(self):
        calls = []
        def boom(crop, retry):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("paddle went sideways")
            return ("OK", 1.0)
        w = AsyncSlotOcr(read_fn=boom)
        w.start()
        try:
            w.submit(_job(slot="A"))
            self._wait(lambda: w.drain_ready())
            first = w.drain()
            self.assertEqual((first[0].text, first[0].conf), ("", 0.0))
            w.submit(_job(slot="B"))
            self._wait(lambda: w.drain_ready())
            self.assertEqual(w.drain()[0].text, "OK")  # worker survived
        finally:
            w.stop()

    def _wait(self, cond, timeout=2.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.01)
        self.fail("condition not met within timeout")


# --------------------------------------------------------------------------- #
# The drain / re-validation on the engine side
# --------------------------------------------------------------------------- #
class _Reg:
    def __init__(self, plate_for_slot=None, confirm=None):
        self.matching_config = MatchingConfig()
        self._plate = plate_for_slot or {}
        self._confirm = confirm  # (plan, crop, text, conf) -> plate|None
        self.bound = []
        self.saved = []

    def get_slot_plate(self, slot_id):
        return self._plate.get(slot_id)

    def confirm_slot_ocr(self, plan, crop, text, conf):
        return self._confirm(plan, crop, text, conf) if self._confirm else None

    def build_parked_reference_proof(self, plan, plate, text, conf, crop):
        del plan, plate, text, conf, crop
        return SimpleNamespace(authorized=True)

    def bind_plate_to_slot(self, slot_id, plate, cam_id, floor=None, source=""):
        self.bound.append((slot_id, plate, source))

    def save_parked_reference(self, plate, crop, cam_id, proof=None):
        assert proof is not None
        self.saved.append((plate, cam_id))
        return True

    def try_reid_identify_slot(self, *a, **k):
        return None


class _SM:
    def __init__(self, state, plate_number=None):
        self.state = state
        self.plate_number = plate_number
        self.identity = None

    def bind_identity(self, plate, url, confidence=0.0, lock=False):
        self.identity = (plate, confidence, lock)


class _Harness(ParkingEngineRuntimeMixin):
    def __init__(self, reg, sm, occupied_state):
        self.vehicle_registry = reg
        self.db_manager = None
        self._ocr_armed = {"B1": True}
        self._ocr_generation = {"B1": 1}
        self._OCCUPIED = occupied_state
        self.pipelines = {
            "CAM-04": SimpleNamespace(state_machines={"B1": sm}, slots=[])
        }

    def _build_slot_snapshot_url(self, slot_id):
        return ""


def _result(token=1):
    from src.core.engine.async_slot_ocr import SlotOcrResult
    return SlotOcrResult(job=_job(slot="B1", token=token), text="ABC123", conf=0.95)


# Reuse the real SlotState enum so state comparisons match the engine.
from src.models.state_machine import SlotState  # noqa: E402
OCCUPIED = SlotState.OCCUPIED
VACANT = SlotState.VACANT


class TestDrainRevalidation(unittest.TestCase):
    def _apply(self, reg, sm):
        h = _Harness(reg, sm, OCCUPIED)
        h._apply_async_ocr_result(_result())
        return h

    def test_confirmed_read_binds_and_locks_and_teaches(self):
        reg = _Reg(confirm=lambda *_: "ABC123")
        sm = _SM(OCCUPIED)
        self._apply(reg, sm)
        self.assertEqual(reg.bound, [("B1", "ABC123", "")])
        self.assertEqual(reg.saved, [("ABC123", "CAM-04")])
        self.assertEqual(sm.identity, ("ABC123", 1.0, True))  # locked

    def test_result_for_vacated_slot_is_discarded(self):
        reg = _Reg(confirm=lambda *_: "ABC123")
        sm = _SM(OCCUPIED)
        h = _Harness(reg, sm, OCCUPIED)
        h._ocr_armed["B1"] = False  # car left before the read came back
        h._apply_async_ocr_result(_result())
        self.assertEqual(reg.bound, [])
        self.assertIsNone(sm.identity)

    def test_result_for_re_armed_slot_is_discarded(self):
        reg = _Reg(confirm=lambda *_: "ABC123")
        sm = _SM(OCCUPIED)
        h = _Harness(reg, sm, OCCUPIED)
        h._ocr_generation["B1"] = 2  # a new car parked; read belongs to token 1
        h._apply_async_ocr_result(_result(token=1))
        self.assertEqual(reg.bound, [])

    def test_result_when_already_named_is_discarded(self):
        reg = _Reg(plate_for_slot={"B1": "OTHER"}, confirm=lambda *_: "ABC123")
        sm = _SM(OCCUPIED)
        h = self._apply(reg, sm)
        self.assertEqual(reg.bound, [])
        self.assertFalse(h._ocr_armed["B1"])  # disarmed — nothing more to do

    def test_result_when_slot_no_longer_occupied_is_discarded(self):
        reg = _Reg(confirm=lambda *_: "ABC123")
        sm = _SM(VACANT)  # state machine moved on
        self._apply(reg, sm)
        self.assertEqual(reg.bound, [])

    def test_unconfirmed_read_does_not_bind(self):
        reg = _Reg(confirm=lambda *_: None)  # OCR read nothing / confirmed nobody
        sm = _SM(OCCUPIED)
        self._apply(reg, sm)
        self.assertEqual(reg.bound, [])
        self.assertIsNone(sm.identity)


if __name__ == "__main__":
    unittest.main()


class TestUnverifiedRestore(unittest.TestCase):
    """A restored plate is a MEMORY, not an observation.

    Restore assumes "slot still OCCUPIED => same car". That is false whenever a
    different car parked during downtime: the slot never goes VACANT across the
    swap, so the clear-on-vacant path — the only thing that ever retires a
    binding — never runs, and the stale plate has no way out.

    Slot B3 displayed a 0.50/unlocked ERS-7949 the running process had never
    once derived (2026-07-30). The slot is now OCR-armed on restart precisely to
    confirm-or-correct that guess, so the resulting read must NOT be thrown away
    by the "already named" guards, which exist to stop a read racing a *live*
    binding.
    """

    def _harness(self, restored=True):
        reg = _Reg(plate_for_slot={"B1": "STALE-1"}, confirm=lambda *_: "ABC123")
        sm = _SM(OCCUPIED, plate_number="STALE-1")
        h = _Harness(reg, sm, OCCUPIED)
        if restored:
            h._restored_plate_slots = {"B1"}
        return h, reg, sm

    def test_read_overrides_an_unverified_restore(self):
        h, reg, sm = self._harness()
        h._apply_async_ocr_result(_result())
        self.assertEqual(reg.bound, [("B1", "ABC123", "")])
        self.assertEqual(sm.identity, ("ABC123", 1.0, True))  # bound AND locked

    def test_binding_retires_the_restore_marker(self):
        h, reg, sm = self._harness()
        h._apply_async_ocr_result(_result())
        self.assertNotIn("B1", h._restored_plate_slots)

    def test_second_read_cannot_overwrite_the_now_locked_binding(self):
        h, reg, sm = self._harness()
        h._apply_async_ocr_result(_result())
        # A second read was already in flight when the first one bound.
        reg.bound.clear()
        h._ocr_armed["B1"] = True
        sm.plate_number = "ABC123"
        reg._plate["B1"] = "ABC123"
        h._apply_async_ocr_result(_result())
        self.assertEqual(reg.bound, [], "a confirmed binding must not be re-bound")

    def test_a_live_binding_still_blocks_the_read(self):
        # No restore marker: the pre-existing guard must be untouched.
        h, reg, sm = self._harness(restored=False)
        h._apply_async_ocr_result(_result())
        self.assertEqual(reg.bound, [])
        self.assertFalse(h._ocr_armed["B1"])
