"""
engine_single_process.py — single-process async inference loop (Phase 2).

The multi-process supervisor runs one serial ``while`` loop per group, each with
its own OpenVINO pool. Measurement showed the serial loop is the ceiling and the
competing pools inflate the forward pass 24 ms → 123 ms. This loop replaces that
with ONE process that feeds every camera through a single OpenVINO
``AsyncInferQueue`` (THROUGHPUT), so several forward passes are in flight at once
and fill the cores, while all shared state (registry, slots, events, ReID/OCR)
stays on ONE consumer thread — exactly the single-threaded invariant the old main
loop relied on.

Threads:
  * grab threads (per camera) — unchanged (camera_manager).
  * scheduler (this thread) — pure inference feeder: pick a due+fresh+free
    camera, preprocess, ``submit_async``. Touches nothing but the detector; the
    OV ``start_async`` blocks when the pool is full, which paces this thread.
  * OV completion callbacks — run postprocess + this camera's ByteTracker, then
    hand finished detections to the consumer queue. One in-flight frame per
    camera (``_inflight``) keeps each camera's tracker strictly ordered.
  * consumer (one thread) — the old main-loop body: ROI/zones/assign/slot/
    events + OCR drain + periodic janitor/session-sync. Single-threaded, so no
    registry/matcher races. ReID/imwrite/DB stay inline here (offloaded in later
    phases).

Phase 2 success is judged on the [PERF] infer-breakdown ``ov`` dropping toward
~24–40 ms and per-camera fps rising — see the design doc's hard gate.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime

from src import perf_trace


class ParkingEngineSingleProcessMixin:
    def run_single_process(self) -> None:
        """One-process, async-inference processing loop for all cameras."""
        camera_configs = self._bootstrap_camera_runtime()
        if camera_configs is None:
            return

        # Every camera's detector must expose the async core; the shared detector
        # serves all non-override cameras through one queue (the whole point).
        shared = self._shared_detector or self.detector
        if not shared.has_async_core():
            print(
                "[ERROR] single-process mode requires the async OpenVINO core. "
                "Set VA_INFER=async and redeploy."
            )
            self.cam_manager.close_all()
            return

        camera_ids = [c.id for c in camera_configs]
        summary_interval = max(1, len(camera_configs) * 10)
        target_fps = self.config.processing.target_fps_per_camera
        min_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

        nireq = shared._ov_core.nireq
        out_q: "queue.Queue" = queue.Queue(maxsize=max(4, 2 * nireq))
        stop = threading.Event()

        # Feeder pool: preprocessing (frame copy + CLAHE + letterbox + tensor
        # build) is heavy and, done on the single scheduler thread, was the ceiling
        # — the pool sat at ~2/nireq in flight while decode + consumer were idle.
        # OpenCV/numpy release the GIL, so N feeder threads overlap preprocess on
        # different cores and keep the queue full. The picker below stays single-
        # threaded (owns pacing + in-flight bookkeeping) and just hands work off.
        try:
            n_feeders = int(os.environ.get("VA_FEED_THREADS", "4") or "4")
        except ValueError:
            n_feeders = 4
        n_feeders = max(1, n_feeders)
        feed_q: "queue.Queue" = queue.Queue(maxsize=max(2, 2 * n_feeders))

        # Per-camera in-flight guard (one submitted frame per camera at a time →
        # strict per-camera tracker ordering + bounded memory). Cleared in the
        # completion callback so the camera can re-submit as soon as its forward
        # pass + track update finish (~ov ms), independent of the consumer.
        inflight: dict[str, int] = {}
        inflight_lock = threading.Lock()
        last_applied: dict[str, int] = {}
        last_submitted_seq: dict[str, int] = {}
        last_submitted_ts: dict[str, float] = {}
        drops = {"out_full": 0, "stale": 0}

        def on_infer_done(detections, ud) -> None:
            # Runs on an OV worker thread. Keep it minimal: free the camera, then
            # hand off. Never block here (would stall the OV pool).
            cam_id, frame, seq, cap_ts = ud
            with inflight_lock:
                inflight.pop(cam_id, None)
            item = (cam_id, frame, detections, seq, cap_ts)
            try:
                out_q.put_nowait(item)
            except queue.Full:
                # Frames are droppable (newest-wins): evict the oldest and retry.
                try:
                    out_q.get_nowait()
                    drops["out_full"] += 1
                except queue.Empty:
                    pass
                try:
                    out_q.put_nowait(item)
                except queue.Full:
                    pass

        def feeder() -> None:
            # Pull a picked (camera, frame) and do the heavy preprocess + submit.
            # Runs on N threads so preprocessing parallelizes across cores. Only
            # the async-queue enqueue inside submit_async is lock-serialized.
            while not stop.is_set():
                try:
                    cam_id, frame, seq, cap_ts, detector = feed_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    detector.submit_async(
                        frame, cam_id, on_infer_done,
                        userdata=(cam_id, frame, seq, cap_ts),
                    )
                except Exception as exc:
                    print(f"[single] submit error cam={cam_id}: {exc!r}")
                    with inflight_lock:
                        inflight.pop(cam_id, None)

        def _housekeeping() -> None:
            # Registry/DB-touching maintenance — MUST stay on the consumer thread
            # so it never races the pipeline body. All internally time-gated.
            self._drain_slot_ocr_results()
            self._cleanup_stale_data()
            self._exit_janitor_tick()
            self._session_sync_tick()

        def consumer() -> None:
            while not stop.is_set():
                try:
                    cam_id, frame, detections, seq, cap_ts = out_q.get(timeout=0.5)
                except queue.Empty:
                    _housekeeping()  # keep janitor/session-sync alive when quiet
                    continue
                # Drop an out-of-order (older) completion for this camera.
                if seq < last_applied.get(cam_id, -1):
                    drops["stale"] += 1
                    continue
                last_applied[cam_id] = seq
                pipeline = self.pipelines.get(cam_id)
                if pipeline is None:
                    continue
                # Time the busy portion only (excludes the get() wait above) so
                # consumer busy% tells inference-bound from consumer-bound.
                _busy0 = time.perf_counter()
                _housekeeping()
                self.last_processed_at = datetime.now()
                try:
                    self._process_detections_and_events(
                        cam_id, frame, pipeline, detections
                    )
                except Exception as exc:  # one bad frame must not kill the loop
                    print(f"[single] pipeline error cam={cam_id}: {exc!r}")
                self._frame_count += 1
                perf_trace.frame_done()
                perf_trace.record_camera_frame(cam_id)
                perf_trace.record_consumer((time.perf_counter() - _busy0) * 1000.0)
                if self._frame_count % summary_interval == 0:
                    self._emit_full_summary()

        consumer_thread = threading.Thread(
            target=consumer, name="va-postproc", daemon=True
        )
        consumer_thread.start()
        feeder_threads = [
            threading.Thread(target=feeder, name=f"va-feeder-{i}", daemon=True)
            for i in range(n_feeders)
        ]
        for t in feeder_threads:
            t.start()
        print(
            f"[single] scheduler up: {len(camera_ids)} cameras, nireq={nireq}, "
            f"feeders={n_feeders}, target_fps={target_fps}, out_q={out_q.maxsize}"
        )

        last_gauge_report = time.time()
        try:
            while not stop.is_set():
                # Periodic queue-health line. `outstanding` = cameras picked but
                # not yet completed (feed_q + preprocess + OV pool). `feed_q` is the
                # backlog waiting on feeders: feed_q pinned at max = feeders can't
                # keep up (raise VA_FEED_THREADS / give them CPU). The REAL OV pool
                # concurrency is the `~in flight` on the [PERF] async-infer line.
                if perf_trace.enabled() and time.time() - last_gauge_report >= 10.0:
                    with inflight_lock:
                        n_outstanding = len(inflight)
                    print(
                        f"[PERF] sched: outstanding={n_outstanding} "
                        f"feed_q={feed_q.qsize()}/{feed_q.maxsize} "
                        f"out_q={out_q.qsize()}/{out_q.maxsize} "
                        f"drops(out_full={drops['out_full']}, stale={drops['stale']})"
                    )
                    last_gauge_report = time.time()
                submitted_any = False
                for cam_id in camera_ids:
                    now = time.time()
                    if min_interval > 0 and (
                        now - last_submitted_ts.get(cam_id, 0.0) < min_interval
                    ):
                        continue
                    with inflight_lock:
                        if cam_id in inflight:
                            continue
                    ok, frame, cap_ts, seq = self.cam_manager.read_camera_stamped(cam_id)
                    if not ok or frame is None:
                        continue
                    if seq == last_submitted_seq.get(cam_id):
                        continue  # no fresh frame since last submit — skip
                    detector = self._detector_for(cam_id)
                    last_submitted_seq[cam_id] = seq
                    last_submitted_ts[cam_id] = now
                    if detector.has_async_core():
                        with inflight_lock:
                            inflight[cam_id] = seq
                        # Hand the heavy preprocess + submit to the feeder pool.
                        # put() blocks when feeders are saturated → correct
                        # back-pressure without the picker doing the work itself.
                        feed_q.put((cam_id, frame, seq, cap_ts, detector))
                    else:
                        # Rare camera_override on a non-OpenVINO model (e.g. a .pt):
                        # no async core exists, so run one synchronous inference on
                        # the scheduler thread and feed the same consumer. Blocks the
                        # scheduler briefly but keeps the camera live.
                        try:
                            dets = detector.detect_and_track(frame, cam_id)
                        except Exception as exc:
                            print(f"[single] sync detect error cam={cam_id}: {exc!r}")
                            dets = []
                        on_infer_done(dets, (cam_id, frame, seq, cap_ts))
                    submitted_any = True
                    perf_trace.set_gauge("infer_inflight", float(len(inflight)))
                    perf_trace.set_gauge("out_q", float(out_q.qsize()))
                    perf_trace.set_gauge("drop_out_full", float(drops["out_full"]))
                    perf_trace.set_gauge("drop_stale", float(drops["stale"]))
                if not submitted_any:
                    time.sleep(min(min_interval, 0.01) if min_interval > 0 else 0.005)
        except KeyboardInterrupt:
            print("\n[single] interrupted — shutting down.")
        finally:
            stop.set()
            for t in feeder_threads:
                t.join(timeout=1)
            try:
                shared._ov_core.wait_all()
            except Exception:
                pass
            consumer_thread.join(timeout=3)
            self._stop_slot_ocr_worker()
            self.cam_manager.close_all()
            self.event_bus.close()
            print(f"[single] stopped. drops={drops}")
