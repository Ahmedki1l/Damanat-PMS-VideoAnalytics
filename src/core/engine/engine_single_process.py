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
from src.core.fair_latest_output import (
    FairLatestOutputQueue,
    InferenceCompletionGate,
    InferenceOutput,
)
from src.core.motion_scheduler import (
    MotionFrame,
    MotionScheduler,
    motion_iteration_plan,
    motion_bypass_cameras,
    rotated_camera_ids,
)
from src.entry.settings import EntrySettings


class ParkingEngineSingleProcessMixin:
    def run_single_process(self) -> None:
        """One-process, async-inference processing loop for all cameras."""
        camera_configs = self._bootstrap_camera_runtime()
        if camera_configs is None:
            return

        camera_ids = [c.id for c in camera_configs]
        target_fps = self.config.processing.target_fps_per_camera
        entry_settings = EntrySettings.from_env()
        motion_bypass = motion_bypass_cameras(
            self.special_zones,
            entry_settings.primary_cameras | entry_settings.fallback_cameras,
        )
        motion_scheduler = MotionScheduler(
            self.config.motion_scheduler,
            bypass_cameras=motion_bypass,
        )
        try:
            motion_scheduler.validate_target_rate(target_fps, camera_ids)
        except ValueError:
            self._stop_entry_v2_local_bridge()
            self._stop_slot_ocr_worker()
            self.cam_manager.close_all()
            raise

        # Every camera's detector must expose the async core; the shared detector
        # serves all non-override cameras through one queue (the whole point).
        shared = self._shared_detector or self.detector
        if not shared.has_async_core():
            print(
                "[ERROR] single-process mode requires the async OpenVINO core. "
                "Set VA_INFER=async and redeploy."
            )
            self._stop_entry_v2_local_bridge()
            self._stop_slot_ocr_worker()
            self.cam_manager.close_all()
            return
        # Expose read-only snapshots to diagnostics/API code without sharing the
        # mutable per-camera state itself.
        self._motion_scheduler = motion_scheduler
        summary_interval = max(1, len(camera_configs) * 10)
        min_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

        nireq = shared._ov_core.nireq
        outputs = FairLatestOutputQueue()
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
        inflight: dict[str, tuple[int, int]] = {}
        inflight_lock = threading.Lock()
        completion_gate = InferenceCompletionGate()
        last_submitted_ts: dict[str, float] = {}
        drops = {
            "out_replaced": 0,
            "stale": 0,
            "stale_epoch": 0,
            "motion_invalid": 0,
            "infer_error": 0,
        }

        def publish_output(item: InferenceOutput) -> None:
            """Nonblocking, per-camera newest-wins handoff."""
            outputs.put_latest(item)
            drops["out_replaced"] = outputs.replacements

        def on_infer_done(detections, ud) -> None:
            # Runs on an OV worker thread. Keep it minimal: free the camera, then
            # hand off. Never block here (would stall the OV pool).
            # sub_ts/done_ts feed the [SLOTTRACE] end-to-end breakdown so the
            # inference and queue segments are attributable, not lumped together.
            cam_id, frame, seq, cap_ts, stream_epoch, sub_ts = ud
            done_ts = time.time()
            with inflight_lock:
                if inflight.get(cam_id) == (stream_epoch, seq):
                    inflight.pop(cam_id, None)
            publish_output(
                InferenceOutput(
                    camera_id=cam_id,
                    frame=frame,
                    detections=detections,
                    sequence=seq,
                    capture_ts=cap_ts,
                    stream_epoch=stream_epoch,
                    submitted_at=sub_ts,
                    inferred_at=done_ts,
                    unknown_reason="infer_error" if detections is None else "",
                )
            )

        def feeder() -> None:
            # Pull a picked (camera, frame) and do the heavy preprocess + submit.
            # Runs on N threads so preprocessing parallelizes across cores. Only
            # the async-queue enqueue inside submit_async is lock-serialized.
            while not stop.is_set():
                try:
                    (cam_id, frame, seq, cap_ts,
                     stream_epoch, detector, reason) = feed_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    motion_scheduler.mark_submitted(cam_id, reason)
                    detector.submit_async(
                        frame, cam_id, on_infer_done,
                        userdata=(
                            cam_id,
                            frame,
                            seq,
                            cap_ts,
                            stream_epoch,
                            time.time(),
                        ),
                    )
                except Exception as exc:
                    print(f"[single] submit error cam={cam_id}: {exc!r}")
                    on_infer_done(
                        None,
                        (cam_id, frame, seq, cap_ts, stream_epoch, time.time()),
                    )

        def _housekeeping() -> None:
            # Registry/DB-touching maintenance — MUST stay on the consumer thread
            # so it never races the pipeline body. All internally time-gated.
            #
            # GUARDED for the same reason the pipeline call below is: an escape
            # here would kill the consumer daemon thread while the scheduler,
            # grabbers and OV pool all keep running — every slot state machine,
            # event and OCR fold-back stops FOREVER with the process still
            # "healthy". (The serial engine crashes the process instead, which
            # restarts and heals; a silent dead thread never does.) The OCR
            # fold-back path in particular (_apply_async_ocr_result → registry
            # confirm/bind, ReID inference, gallery disk I/O) is not internally
            # try/excepted.
            try:
                self._drain_slot_ocr_results()
                self._cleanup_stale_data()
                self._exit_janitor_tick()
                self._session_sync_tick()
            except Exception as exc:
                print(f"[single] housekeeping error: {exc!r}")

        def consumer() -> None:
            while not stop.is_set():
                try:
                    item = outputs.get(timeout=0.5)
                except queue.Empty:
                    _housekeeping()  # keep janitor/session-sync alive when quiet
                    continue
                cam_id = item.camera_id
                # A completion from before a reconnect can never describe the
                # current stream, regardless of detector confidence.
                current_epoch = self.cam_manager.get_stream_epoch(cam_id)
                rejection = completion_gate.rejection_reason(item, current_epoch)
                if rejection:
                    drops[rejection] += 1
                    continue
                pipeline = self.pipelines.get(cam_id)
                if pipeline is None:
                    continue
                frame_age = max(0.0, time.time() - item.capture_ts)
                if not motion_scheduler.completion_is_valid(
                    cam_id,
                    item.stream_epoch,
                    frame_age,
                ):
                    rejection = completion_gate.rejection_reason(
                        item, self.cam_manager.get_stream_epoch(cam_id)
                    )
                    if rejection:
                        drops[rejection] += 1
                        continue
                    completion_gate.record_acceptance(item)
                    drops["stale"] += 1
                    motion_scheduler.mark_unknown(
                        cam_id,
                        item.stream_epoch,
                        "stale_completion",
                    )
                    self._mark_slot_observations_unknown(pipeline)
                    continue
                if item.detections is None:
                    rejection = completion_gate.rejection_reason(
                        item, self.cam_manager.get_stream_epoch(cam_id)
                    )
                    if rejection:
                        drops[rejection] += 1
                        continue
                    completion_gate.record_acceptance(item)
                    reason = item.unknown_reason or "infer_error"
                    motion_scheduler.mark_unknown(
                        cam_id,
                        item.stream_epoch,
                        reason,
                    )
                    drop_name = (
                        "infer_error" if reason == "infer_error" else "motion_invalid"
                    )
                    drops[drop_name] += 1
                    self._mark_slot_observations_unknown(pipeline)
                    continue
                # Time the busy portion only (excludes the get() wait above) so
                # consumer busy% tells inference-bound from consumer-bound.
                _busy0 = time.perf_counter()
                _housekeeping()
                # Housekeeping may overlap an RTSP reconnect. Revalidate at the
                # last safe point before this frame can mutate occupancy state.
                rejection = completion_gate.rejection_reason(
                    item, self.cam_manager.get_stream_epoch(cam_id)
                )
                if rejection:
                    drops[rejection] += 1
                    continue
                completion_gate.record_acceptance(item)
                self.last_processed_at = datetime.now()
                try:
                    self._process_detections_and_events(
                        cam_id, item.frame, pipeline, item.detections,
                        capture_ts=item.capture_ts,
                        submitted_at=item.submitted_at,
                        inferred_at=item.inferred_at,
                        dequeued_at=time.time(),
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
            f"feeders={n_feeders}, target_fps={target_fps}, "
            f"motion={motion_scheduler.mode}, "
            f"motion_bypass={sorted(motion_bypass)}"
        )

        last_gauge_report = time.time()
        camera_cursor = 0
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
                        f"ready_cameras={outputs.qsize()} "
                        f"drops={drops} motion={motion_scheduler.metrics()}"
                    )
                    last_gauge_report = time.time()
                # Liveness: if the consumer thread has died despite its guards,
                # running headless would mean grabbing/inferring forever while no
                # slot state updates — a silent outage. Exit the loop instead:
                # the finally block tears down and the supervisor/k8s restart
                # heals visibly.
                if not consumer_thread.is_alive():
                    print(
                        "[single] FATAL: consumer thread died — shutting down "
                        "so the supervisor can restart the worker."
                    )
                    break
                submitted_any = False
                for cam_id in rotated_camera_ids(camera_ids, camera_cursor):
                    now = time.time()
                    bypass_motion = motion_scheduler.is_bypassed(cam_id)
                    target_due = (
                        min_interval <= 0
                        or now - last_submitted_ts.get(cam_id, 0.0) >= min_interval
                    )
                    analysis_due = motion_scheduler.analysis_due(cam_id)
                    should_read, should_analyze = motion_iteration_plan(
                        mode=motion_scheduler.mode,
                        bypass=bypass_motion,
                        target_due=target_due,
                        analysis_due=analysis_due,
                    )
                    if not should_read:
                        continue
                    with inflight_lock:
                        if cam_id in inflight:
                            continue
                    ok, frame, cap_ts, seq, stream_epoch = (
                        self.cam_manager.read_camera_stamped_with_epoch(cam_id)
                    )
                    if not ok or frame is None:
                        continue
                    if should_analyze:
                        decision = motion_scheduler.examine(
                            MotionFrame(
                                camera_id=cam_id,
                                image=frame,
                                sequence=seq,
                                stream_epoch=stream_epoch,
                                age_seconds=max(0.0, now - cap_ts),
                            )
                        )
                    else:
                        decision = motion_scheduler.shadow_passthrough(
                            MotionFrame(
                                camera_id=cam_id,
                                image=frame,
                                sequence=seq,
                                stream_epoch=stream_epoch,
                                age_seconds=max(0.0, now - cap_ts),
                            )
                        )
                    if not decision.frame_is_valid:
                        stamp = time.time()
                        publish_output(
                            InferenceOutput(
                                camera_id=cam_id,
                                frame=frame,
                                detections=None,
                                sequence=seq,
                                capture_ts=cap_ts,
                                stream_epoch=stream_epoch,
                                submitted_at=stamp,
                                inferred_at=stamp,
                                unknown_reason=decision.reason,
                            )
                        )
                        continue
                    if not decision.should_infer or not target_due:
                        continue
                    detector = self._detector_for(cam_id)
                    last_submitted_ts[cam_id] = now
                    if detector.has_async_core():
                        with inflight_lock:
                            inflight[cam_id] = (stream_epoch, seq)
                        # Hand the heavy preprocess + submit to the feeder pool.
                        # put() blocks when feeders are saturated → correct
                        # back-pressure without the picker doing the work itself.
                        feed_q.put(
                            (
                                cam_id,
                                frame,
                                seq,
                                cap_ts,
                                stream_epoch,
                                detector,
                                decision.reason,
                            )
                        )
                    else:
                        # Rare camera_override on a non-OpenVINO model (e.g. a .pt):
                        # no async core exists, so run one synchronous inference on
                        # the scheduler thread and feed the same consumer. Blocks the
                        # scheduler briefly but keeps the camera live.
                        motion_scheduler.mark_submitted(cam_id, decision.reason)
                        try:
                            dets = detector.detect_and_track(frame, cam_id)
                        except Exception as exc:
                            print(f"[single] sync detect error cam={cam_id}: {exc!r}")
                            dets = None
                        on_infer_done(
                            dets,
                            (cam_id, frame, seq, cap_ts, stream_epoch, time.time()),
                        )
                    submitted_any = True
                    perf_trace.set_gauge("infer_inflight", float(len(inflight)))
                    perf_trace.set_gauge("out_q", float(outputs.qsize()))
                    perf_trace.set_gauge(
                        "drop_out_replaced",
                        float(drops["out_replaced"]),
                    )
                    perf_trace.set_gauge("drop_stale", float(drops["stale"]))
                if camera_ids:
                    camera_cursor = (camera_cursor + 1) % len(camera_ids)
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
            self._stop_entry_v2_local_bridge()
            self._stop_slot_ocr_worker()
            self.cam_manager.close_all()
            self.event_bus.close()
            print(f"[single] stopped. drops={drops}")
