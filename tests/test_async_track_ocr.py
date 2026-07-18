"""Async APPROACH OCR: the read of a still-driving car runs off the camera loop.

The sibling of ``test_async_slot_ocr``. The approach read (``_try_ocr_identify_tracks``
-> ``try_ocr_identify_track``) was the last synchronous PaddleOCR left on the consumer
thread: measured in production 2026-07-18 at ocr=517ms as a 50-frame AVERAGE, which was
essentially the entire ``zones`` stage. It had neither of the two fixes the slot path
grew after the 19s-frame incident — no worker, and no process-wide floor between reads.

Three properties, all safety rather than performance:

  * The worker performs ONLY the PaddleOCR read. Narrowing (ReID ``reid_shortlist``),
    the confirm, and the transit hop stay on the main thread, because they touch the
    ReID matcher and mutate registry sessions.
  * The process-wide floor is shared with the slot path, so N unidentified cars in one
    frame can no longer buy N back-to-back reads.
  * A read that finishes after the car was named by another path is DISCARDED.

Plus the refactor guard: the sync path is now composed of the same three phases the
async path uses, so the two cannot drift.
"""
import time
import unittest
from types import SimpleNamespace

import numpy as np

from src.core.engine.async_slot_ocr import SlotOcrJob, SlotOcrResult
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin
from src.core.engine.engine_tracking import ParkingEngineTrackingMixin


# --------------------------------------------------------------------------- #
# Job keying — a track read must never collide with a real slot id
# --------------------------------------------------------------------------- #
class TestTrackKey(unittest.TestCase):
    def test_track_key_is_namespaced(self):
        self.assertEqual(SlotOcrJob.track_key("CAM-21", 37), "track:CAM-21:37")

    def test_default_kind_is_slot(self):
        j = SlotOcrJob(slot_id="B1", cam_id="C", crop=None, plan=None,
                       attempts=1, token=None)
        self.assertEqual(j.kind, "slot")
        self.assertIsNone(j.ctx)


# --------------------------------------------------------------------------- #
# Submit side: pacing, coalescing, and what lands on the worker
# --------------------------------------------------------------------------- #
class _Worker:
    """Stand-in for AsyncSlotOcr that records submissions."""

    def __init__(self):
        self.jobs = []
        self._busy = set()

    def submit(self, job):
        self.jobs.append(job)
        return True

    def pending_or_inflight(self, key):
        return key in self._busy


class _TrackReg:
    def __init__(self, plate_for_track=None):
        self._plate = plate_for_track or {}
        self.sync_calls = []
        self.planned = []

    def get_plate_for_track(self, cam_id, tid):
        return self._plate.get((cam_id, tid))

    def plan_track_ocr(self, cam_id, tid, crop, *, allow_retry=True):
        self.planned.append((cam_id, tid, allow_retry))
        return SimpleNamespace(camera_id=cam_id, track_id=tid, allow_retry=allow_retry)

    def try_ocr_identify_track(self, cam_id, tid, crop):
        self.sync_calls.append((cam_id, tid))
        return None


class _SubmitHarness(ParkingEngineTrackingMixin):
    def __init__(self, reg, worker, gap=5.0):
        self.vehicle_registry = reg
        self._worker = worker
        self._gap = gap
        self.pipelines = {"CAM-21": SimpleNamespace(slots=["B1"], floor="B1")}
        self.adopted = []

    def _ocr_worker_if_async(self):
        return self._worker

    def _ocr_group_gap_s(self):
        return self._gap

    def _crop_detection(self, frame, det):
        return np.zeros((8, 8, 3), np.uint8)

    def _try_adopt_transit_identity(self, *a):
        self.adopted.append(a)


def _dets(*tids):
    return [SimpleNamespace(track_id=t, bbox=(0, 0, 10, 10)) for t in tids]


class TestSubmitSide(unittest.TestCase):
    def _run(self, h, dets, now):
        h._try_ocr_identify_tracks("CAM-21", np.zeros((32, 32, 3), np.uint8), dets, now)

    def test_one_read_per_frame_despite_many_unidentified_cars(self):
        """THE BUG THIS FIXES. Four driving cars used to buy four PaddleOCR reads in a
        single frame; the per-track 1.0s gate paces nothing when some track is always
        eligible."""
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=5.0)
        self._run(h, _dets(1, 2, 3, 4), now=1000.0)
        self.assertEqual(len(w.jobs), 1, "process-wide floor must cap the frame at one read")

    def test_pacing_does_not_cost_a_track_its_attempt_budget(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=5.0)
        self._run(h, _dets(1, 2, 3), now=1000.0)
        # Only the track that actually got a read spent an attempt.
        spent = {k: v for k, v in h._ocr_track_attempts.items() if v}
        self.assertEqual(len(spent), 1)

    def test_every_track_gets_a_first_seen_stamp_even_when_paced_out(self):
        """The transit hop's ordering guard depends on first_seen. The pacing gate
        returns early, so first_seen must be stamped in its own pass beforehand —
        otherwise the guard silently weakens for the tracks that were paced out."""
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=5.0)
        self._run(h, _dets(1, 2, 3, 4), now=1000.0)
        for tid in (1, 2, 3, 4):
            self.assertIn(("CAM-21", tid), h._ocr_track_first_seen)

    def test_second_frame_within_the_gap_reads_nothing(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=5.0)
        self._run(h, _dets(1, 2), now=1000.0)
        self._run(h, _dets(1, 2), now=1001.0)  # 1s later, gap is 5s
        self.assertEqual(len(w.jobs), 1)

    def test_after_the_gap_elapses_a_new_read_is_allowed(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=5.0)
        self._run(h, _dets(1, 2), now=1000.0)
        self._run(h, _dets(1, 2), now=1006.0)
        self.assertEqual(len(w.jobs), 2)

    def test_submitted_job_is_shaped_for_the_track_path(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=0.0)
        self._run(h, _dets(7), now=1000.0)
        job = w.jobs[0]
        self.assertEqual(job.kind, "track")
        self.assertEqual(job.slot_id, "track:CAM-21:7")
        self.assertEqual(job.cam_id, "CAM-21")
        key, tid, det = job.ctx
        self.assertEqual((key, tid), (("CAM-21", 7), 7))
        self.assertIs(det.track_id, 7)

    def test_already_identified_track_is_not_read(self):
        reg = _TrackReg(plate_for_track={("CAM-21", 5): "ABC-123"})
        w = _Worker()
        h = _SubmitHarness(reg, w, gap=0.0)
        self._run(h, _dets(5), now=1000.0)
        self.assertEqual(w.jobs, [])

    def test_in_flight_track_is_not_resubmitted(self):
        reg, w = _TrackReg(), _Worker()
        w._busy.add("track:CAM-21:9")
        h = _SubmitHarness(reg, w, gap=0.0)
        self._run(h, _dets(9), now=1000.0)
        self.assertEqual(w.jobs, [])

    def test_attempt_budget_is_respected(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=0.0)
        h._ocr_track_attempts = {("CAM-21", 3): h._OCR_TRACK_MAX_ATTEMPTS}
        h._ocr_track_last_at, h._ocr_track_first_seen = {}, {}
        self._run(h, _dets(3), now=1000.0)
        self.assertEqual(w.jobs, [])

    def test_sync_path_used_when_no_worker(self):
        reg, w = _TrackReg(), _Worker()
        h = _SubmitHarness(reg, w, gap=0.0)
        h._worker = None  # matching.slot_ocr_async = false
        self._run(h, _dets(2), now=1000.0)
        self.assertEqual(reg.sync_calls, [("CAM-21", 2)])


# --------------------------------------------------------------------------- #
# Fold-back side: dispatch and re-validation
# --------------------------------------------------------------------------- #
class _FoldReg:
    def __init__(self, plate_for_track=None, confirm=None):
        self._plate = plate_for_track or {}
        self._confirm = confirm
        self.confirms = []

    def get_plate_for_track(self, cam_id, tid):
        return self._plate.get((cam_id, tid))

    def confirm_track_ocr(self, plan, crop, text):
        self.confirms.append(text)
        return self._confirm(plan, crop, text) if self._confirm else None


class _FoldHarness(ParkingEngineRuntimeMixin):
    def __init__(self, reg):
        self.vehicle_registry = reg
        self.adopted = []

    def _try_adopt_transit_identity(self, cam_id, tid, key, crop, now_ts, det):
        self.adopted.append((cam_id, tid, key))


def _track_result(text="4920HBR", tid=7, cam="CAM-21"):
    det = SimpleNamespace(track_id=tid)
    job = SlotOcrJob(
        slot_id=SlotOcrJob.track_key(cam, tid), cam_id=cam,
        crop=np.zeros((8, 8, 3), np.uint8),
        plan=SimpleNamespace(camera_id=cam, track_id=tid, allow_retry=True),
        attempts=1, token=None, kind="track", ctx=((cam, tid), tid, det),
    )
    return SlotOcrResult(job=job, text=text, conf=0.9)


class TestFoldBack(unittest.TestCase):
    def test_kind_track_dispatches_to_the_track_handler(self):
        reg = _FoldReg(confirm=lambda *_: "HBR-4920")
        h = _FoldHarness(reg)
        h._apply_async_ocr_result(_track_result())
        self.assertEqual(reg.confirms, ["4920HBR"])
        self.assertEqual(h.adopted, [])  # confirmed, so no transit hop

    def test_unconfirmed_read_falls_through_to_the_transit_hop(self):
        reg = _FoldReg(confirm=lambda *_: None)
        h = _FoldHarness(reg)
        h._apply_async_ocr_result(_track_result())
        self.assertEqual(len(h.adopted), 1)
        self.assertEqual(h.adopted[0][:2], ("CAM-21", 7))

    def test_track_named_while_the_read_was_in_flight_is_discarded(self):
        reg = _FoldReg(plate_for_track={("CAM-21", 7): "OTHER"},
                       confirm=lambda *_: "HBR-4920")
        h = _FoldHarness(reg)
        h._apply_async_ocr_result(_track_result())
        self.assertEqual(reg.confirms, [], "must not re-confirm an already-named track")
        self.assertEqual(h.adopted, [])


# --------------------------------------------------------------------------- #
# Refactor guard: the sync path is the same three phases
# --------------------------------------------------------------------------- #
class TestSyncPathParity(unittest.TestCase):
    def test_sync_path_composes_plan_read_confirm(self):
        from src.vehicle_registry.vehicle_registry_identity import (
            VehicleRegistryIdentityMixin,
        )

        calls = []

        class R(VehicleRegistryIdentityMixin):
            def plan_track_ocr(self, cam, tid, crop, *, allow_retry=True):
                calls.append("plan")
                return SimpleNamespace(camera_id=cam, track_id=tid,
                                       allow_retry=allow_retry)

            def read_slot_plate(self, crop, allow_retry):
                calls.append(("read", allow_retry))
                return ("4920HBR", 0.9)

            def confirm_track_ocr(self, plan, crop, text):
                calls.append(("confirm", text))
                return "HBR-4920"

        out = R().try_ocr_identify_track("CAM-21", 7, np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(out, "HBR-4920")
        self.assertEqual(calls, ["plan", ("read", True), ("confirm", "4920HBR")])

    def test_allow_retry_default_matches_the_historical_inline_call(self):
        """The old inline call passed no allow_retry, and ocr.read defaults it to True.
        The enlarged retry pass is expensive and worth revisiting — but separately, as a
        measured behaviour change, not as a side effect of moving threads."""
        from src.ocr.plate_ocr import PaddlePlateOCR
        import inspect

        sig = inspect.signature(PaddlePlateOCR.read)
        self.assertIs(sig.parameters["allow_retry"].default, True)


if __name__ == "__main__":
    unittest.main()
