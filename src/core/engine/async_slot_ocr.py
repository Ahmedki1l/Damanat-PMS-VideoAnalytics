"""async_slot_ocr.py — run the slot-identify OCR read off the camera loop.

A PaddleOCR read is 2-8s on the production Xeon and, inline, it is the single
largest per-frame cost — it freezes slot flips for EVERY camera in the group
while it runs (measured 2026-07-16: b2-a at 19s/frame, ocr=8.8s of it). This
worker takes only that read off the main thread. The main loop plans the read
(ReID ranking, main-thread) and submits it; the worker performs the PaddleOCR
read; the main loop folds the result back in (confirm + bind, main-thread) a
frame or two later.

WHAT STAYS ON THE MAIN THREAD, and why: the ReID matcher and the colour/type
classifiers share their OpenVINO infer requests with the tracking loop and are
not safe to call from a second thread. So only the PaddleOCR read — which owns
a separate, internally-locked engine — crosses the thread boundary.

BOUNDED AND COALESCED: at most one job per slot is ever in flight; a newer crop
for a slot already queued replaces the old one (a stale crop is useless). The
queue has a hard cap so a wedged read can never grow memory without bound.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlotOcrJob:
    """One identify read in flight. ``plan`` is the registry's SlotOcrPlan (or
    TrackOcrPlan when ``kind == "track"``); ``token`` pins the exact occupancy this
    crop belongs to, so a result that lands after the car left (or after the slot was
    re-armed for a new car) is dropped instead of bound. Everything else is what the
    bind needs.

    ``kind`` discriminates the two producers that share this worker:

      * ``"slot"`` — a car already parked in ``slot_id`` (the original path).
      * ``"track"`` — the APPROACH read: a car still driving up the aisle, keyed by
        track rather than slot. ``slot_id`` then carries the synthetic coalescing key
        ``track:<cam>:<tid>`` so one queue serves both without their keys colliding.

    They share one worker deliberately: PaddleOCR is a single engine with an internal
    lock, so two queues would only let the two paths queue behind each other twice.
    """

    slot_id: str
    cam_id: str
    crop: Any
    plan: Any
    attempts: int
    token: Any
    kind: str = "slot"
    # Engine-side context the fold-back needs but the read does not — currently the
    # approach path's (key, first_seen_ts, detection) for the transit hop. Opaque to
    # this module; the worker only ever carries it.
    ctx: Any = None

    @staticmethod
    def track_key(cam_id: str, track_id: int) -> str:
        """The coalescing key for an approach read. Namespaced so it can never
        collide with a real slot id."""
        return f"track:{cam_id}:{track_id}"


@dataclass
class SlotOcrResult:
    job: SlotOcrJob
    text: str
    conf: float


class AsyncSlotOcr:
    """Single background thread that turns ``SlotOcrJob`` -> ``SlotOcrResult``.

    ``read_fn(crop, allow_retry) -> (text, conf)`` is the only heavy call it
    makes; inject it so the engine can point it at the registry's locked
    ``read_slot_plate`` and tests can pass a stub. Results are collected for the
    main thread to drain once per frame via :meth:`drain`.
    """

    def __init__(
        self,
        read_fn: Callable[[Any, bool], Any],
        *,
        max_pending: int = 64,
    ) -> None:
        self._read_fn = read_fn
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        # slot_id -> pending job (dict, so a newer crop coalesces onto the old).
        self._pending: Dict[str, SlotOcrJob] = {}
        # slots handed to the worker but not yet returned — keeps at most one
        # read per slot in flight without blocking submits for other slots.
        self._inflight: set = set()
        self._results: List[SlotOcrResult] = []
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -------------------------------------------------------- #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="slot-ocr", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    # ---- producer side (main thread) ------------------------------------- #

    def submit(self, job: SlotOcrJob) -> bool:
        """Queue a read. Returns False (dropped) if this slot already has a read
        in flight or queued and this is not a newer crop for it, or if the queue
        is full. Coalesces: a newer crop for a queued slot replaces it."""
        with self._cv:
            if self._stop:
                return False
            # One read per slot in flight; the fresh crop will be submitted again
            # next time the slot is eligible, so dropping here loses nothing.
            if job.slot_id in self._inflight:
                return False
            if job.slot_id in self._pending:
                self._pending[job.slot_id] = job  # coalesce onto the newer crop
                return True
            if len(self._pending) >= self._max_pending:
                return False
            self._pending[job.slot_id] = job
            self._cv.notify()
            return True

    def pending_or_inflight(self, slot_id: str) -> bool:
        with self._lock:
            return slot_id in self._pending or slot_id in self._inflight

    def drain_ready(self) -> bool:
        """True when there is at least one finished result to drain."""
        with self._lock:
            return bool(self._results)

    def drain(self) -> List[SlotOcrResult]:
        """Take everything the worker has finished since the last call."""
        with self._lock:
            if not self._results:
                return []
            out, self._results = self._results, []
            return out

    def forget(self, slot_id: str) -> None:
        """Drop any queued read for a slot whose car has left. A read already in
        flight still returns, but its result is rejected by the token check."""
        with self._lock:
            self._pending.pop(slot_id, None)

    # ---- consumer side (worker thread) ----------------------------------- #

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._pending:
                    self._cv.wait()
                if self._stop:
                    return
                slot_id, job = self._pending.popitem()
                self._inflight.add(slot_id)
            try:
                text, conf = self._read_fn(job.crop, job.plan.allow_retry)
            except Exception as exc:  # never let one bad crop kill the worker
                logger.debug("[slot-ocr] read failed slot=%s: %r", job.slot_id, exc)
                text, conf = "", 0.0
            with self._lock:
                self._inflight.discard(slot_id)
                self._results.append(SlotOcrResult(job=job, text=text, conf=conf))
