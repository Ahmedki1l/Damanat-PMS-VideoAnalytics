"""
perf_trace.py — opt-in per-stage performance timing for the processing loop.

Enabled only when the ``PERF_TRACE`` env var is truthy (1/true/yes/on), so it is
a zero-overhead no-op in production — just don't set the var. When on it prints
two kinds of ``[PERF]`` lines:

  * per-frame pipeline stages (main loop, single-threaded): roi / clahe / infer /
    zones / assign / slot, averaged over the last N processed frames.
  * a decode meter (background grabber threads, all cameras): average decode
    time per frame and how many frames/sec are being decoded across all streams —
    the cost the decode-throttle is meant to cut.

Usage:
    PERF_TRACE=1 python main.py --api            # bash
    $env:PERF_TRACE=1; python main.py --api      # PowerShell
    PERF_TRACE_EVERY=100 ...                      # report cadence (default 50 frames)
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

_ENABLED = os.environ.get("PERF_TRACE", "").strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _ENABLED


# --- Per-processed-frame pipeline stages (main loop, single-threaded) -------- #
_sums: dict[str, float] = defaultdict(float)
_frames = 0
_last_report_ts = time.time()
try:
    _report_every = max(1, int(os.environ.get("PERF_TRACE_EVERY", "50") or "50"))
except ValueError:
    _report_every = 50
# Also flush on TIME, not just frame count. A starved group doing one frame
# every 15s never reaches 50 frames inside a log window — exactly the group
# whose stage breakdown we need most (this is how the b1/b2 starvation of
# 2026-07-16 stayed invisible: zero per-frame lines in an 8-minute log).
try:
    _report_max_s = max(1.0, float(os.environ.get("PERF_TRACE_MAX_S", "60") or "60"))
except ValueError:
    _report_max_s = 60.0


@contextmanager
def stage(name: str):
    """Time a named pipeline stage. No-op (and no timing) when disabled.

    Stages may nest (e.g. ``ocr`` runs inside ``slot``); nested stages report
    as their own key but are NOT additive with their parent — read ``ocr`` as
    "of which" under ``slot``.
    """
    if not _ENABLED:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        _sums[name] += (time.perf_counter() - t) * 1000.0


def frame_done() -> None:
    """Call once per *processed* frame. Emits a rolling-average line every
    ``PERF_TRACE_EVERY`` frames, or every ``PERF_TRACE_MAX_S`` seconds if
    frames are coming slower than that (a starving loop is the one that most
    needs a stage breakdown in the logs)."""
    if not _ENABLED:
        return
    global _frames, _last_report_ts
    _frames += 1
    now = time.time()
    if _frames < _report_every and (now - _last_report_ts) < _report_max_s:
        return
    parts = " | ".join(f"{k}={_sums[k] / _frames:.0f}ms" for k in sorted(_sums))
    total = sum(_sums.values()) / _frames
    print(f"[PERF] per-frame: {parts} | total={total:.0f}ms  (avg over {_frames})")
    _sums.clear()

    # Inference breakdown: where the "infer" ms actually go, and how much of the
    # wall-clock is scheduler wait / cgroup throttle vs real OpenVINO compute.
    global _infer_frames
    if _infer_frames > 0:
        n = _infer_frames
        b = _infer_sums
        print(
            "[PERF] infer-breakdown: "
            f"wall={b['wall']/n:.0f}ms | ov={b['ov']/n:.0f}ms | pre={b['pre']/n:.0f}ms | "
            f"post={b['post']/n:.0f}ms | trk={b['trk']/n:.0f}ms | cpu={b['cpu']/n:.0f}ms | "
            f"wait={b['wait']/n:.0f}ms | throttled={b['thr']/n:.0f}ms  (avg over {n})"
        )
        _infer_sums.clear()
        _infer_frames = 0

    # cgroup throttle snapshot (cumulative since boot): the number that reconciles
    # low average CPU with high latency. A high throttled% at <100% core usage IS
    # the CFS-quota-throttling proof.
    thr = read_cgroup_throttle()
    if thr is not None:
        nr_periods, nr_throttled, throttled_usec = thr
        pct = (100.0 * nr_throttled / nr_periods) if nr_periods else 0.0
        print(
            f"[PERF] cgroup-cpu: periods={nr_periods} throttled={nr_throttled} "
            f"({pct:.0f}% of periods) throttled_total={throttled_usec/1e6:.1f}s"
        )

    _frames = 0
    _last_report_ts = now


# --- Inference breakdown (main loop) ---------------------------------------- #
# Splits the single "infer" wall-time into where the milliseconds actually go, so
# we can tell OpenVINO compute apart from scheduler wait / CFS throttling / the
# Ultralytics wrapper. Fed once per processed frame from TrackedDetector.
#
#   wall  = perf_counter around the whole model.track() call
#   cpu   = THIS thread's CPU time over the same call (CLOCK_THREAD_CPUTIME_ID);
#           the main thread blocks while the OV pool computes, so wall - cpu is
#           time the main thread was NOT on a core (pool compute + preempt/throttle)
#   pre/ov/post = Ultralytics' own per-stage timing (results.speed), so `ov` is the
#           wall-time of the pure forward pass, isolated from pre/post
#   trk   = wall - (pre+ov+post): ByteTrack update + Ultralytics wrapper overhead
#   thr   = cgroup CPU throttled-time that elapsed DURING the call (the smoking gun
#           for "low average CPU yet high latency": a frozen cgroup shows here)
_infer_sums: dict[str, float] = defaultdict(float)
_infer_frames = 0


def _thread_cpu_s() -> float:
    """Current thread's consumed CPU seconds. Falls back to process-wide CPU time
    on platforms without per-thread clocks (e.g. Windows)."""
    try:
        return time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)  # POSIX/Linux (prod)
    except (AttributeError, OSError):
        return time.process_time()


def read_cgroup_throttle() -> tuple[int, int, int] | None:
    """(nr_periods, nr_throttled, throttled_usec) from the cgroup, or None.

    Handles cgroup v2 (``cpu.stat`` → ``throttled_usec``) and v1 (``throttled_time``
    in ns). This is THE measurement that reconciles low average CPU with high
    latency: CFS enforces the pod's CPU *limit* per 100ms period, so a bursty
    inference pool can be frozen (throttled) even while the average sits well
    under the core count."""
    # cgroup v2
    try:
        d: dict[str, str] = {}
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                d[k] = v.strip()
        if "throttled_usec" in d:
            return (int(d.get("nr_periods", 0)), int(d.get("nr_throttled", 0)),
                    int(d["throttled_usec"]))
    except OSError:
        pass
    # cgroup v1
    for p in ("/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
        try:
            d = {}
            with open(p) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        d[parts[0]] = parts[1]
            if "throttled_time" in d:  # nanoseconds in v1
                return (int(d.get("nr_periods", 0)), int(d.get("nr_throttled", 0)),
                        int(d["throttled_time"]) // 1000)
        except OSError:
            continue
    return None


def record_infer(wall_ms: float, cpu_ms: float, pre_ms: float, ov_ms: float,
                 post_ms: float, throttled_ms: float) -> None:
    """Record one frame's inference breakdown (no-op when disabled)."""
    if not _ENABLED:
        return
    global _infer_frames
    _infer_sums["wall"] += wall_ms
    _infer_sums["cpu"] += cpu_ms
    _infer_sums["wait"] += max(0.0, wall_ms - cpu_ms)
    _infer_sums["pre"] += pre_ms
    _infer_sums["ov"] += ov_ms
    _infer_sums["post"] += post_ms
    _infer_sums["trk"] += max(0.0, wall_ms - pre_ms - ov_ms - post_ms)
    _infer_sums["thr"] += throttled_ms
    _infer_frames += 1


# --- Per-camera processed-fps + async queue depth (Phase 0 scaffolding) ------ #
# These support the single-process AsyncInferQueue migration: per-camera fps
# tells us whether the scheduler is fair (the original per-camera FPS disparity
# was invisible in the aggregate number), and the queue-depth gauge surfaces
# back-pressure / stale-frame pileup in the async design. Both are no-ops when
# PERF_TRACE is off, and are reported on the same cadence as the decode meter.
_cam_lock = threading.Lock()
_cam_counts: dict[str, int] = defaultdict(int)
_cam_last_report = time.time()

# Named gauges (e.g. queue depths, drop counters) sampled by the async engine.
_gauges: dict[str, float] = {}
_gauge_lock = threading.Lock()


def record_camera_frame(camera_id: str) -> None:
    """Count one processed frame for ``camera_id``; emit a per-camera fps table
    roughly every 10s. No-op when disabled."""
    if not _ENABLED:
        return
    global _cam_last_report
    with _cam_lock:
        _cam_counts[camera_id] += 1
        now = time.time()
        elapsed = now - _cam_last_report
        if elapsed >= 10.0 and _cam_counts:
            rows = sorted(_cam_counts.items())
            body = " ".join(f"{cid}={n / elapsed:.2f}" for cid, n in rows)
            total = sum(_cam_counts.values()) / elapsed
            print(
                f"[PERF] per-camera-fps ({elapsed:.0f}s window, total={total:.1f}/s): "
                f"{body}"
            )
            _cam_counts.clear()
            _cam_last_report = now


_cons_lock = threading.Lock()
_cons_busy_ms = 0.0
_cons_frames = 0
_cons_last_report = time.time()


def record_consumer(busy_ms: float) -> None:
    """Record one consumer-thread frame's processing time (single-process engine).

    ``busy_ms`` is the wall time the ONE consumer thread spent on a frame — the
    post-detection body (ROI/zones/assign/slot/events) plus the inline ReID/OCR/
    snapshot/DB work — excluding the time it sat waiting for the next frame.

    Reports ``busy%`` (share of wall time the consumer was actually working) and
    ms/frame roughly every 10s. This is the metric that tells inference-bound
    from consumer-bound: if ``ov`` drops to ~24-40ms but per-camera fps stays
    capped, a consumer ``busy%`` near 100 says the bottleneck moved onto the
    consumer (inline ReID/DB/snapshots — the Phase 3/4 offload targets); a low
    ``busy%`` says fps is limited upstream (scheduler pace / pool / cameras).
    No-op when disabled."""
    if not _ENABLED:
        return
    global _cons_busy_ms, _cons_frames, _cons_last_report
    with _cons_lock:
        _cons_busy_ms += busy_ms
        _cons_frames += 1
        now = time.time()
        elapsed = now - _cons_last_report
        if elapsed >= 10.0 and _cons_frames:
            busy_pct = 100.0 * (_cons_busy_ms / 1000.0) / elapsed
            avg = _cons_busy_ms / _cons_frames
            print(
                f"[PERF] consumer: busy={busy_pct:.0f}% | {avg:.1f}ms/frame | "
                f"{_cons_frames} frames in {elapsed:.0f}s "
                f"({'consumer-bound' if busy_pct >= 85 else 'not consumer-bound'})"
            )
            _cons_busy_ms = 0.0
            _cons_frames = 0
            _cons_last_report = now


def set_gauge(name: str, value: float) -> None:
    """Record the current value of a named gauge (queue depth, drop count, …).
    Cheap; read back in the async engine's periodic report. No-op when disabled."""
    if not _ENABLED:
        return
    with _gauge_lock:
        _gauges[name] = value


def snapshot_gauges() -> dict[str, float]:
    """Copy of the current gauges (empty when disabled)."""
    if not _ENABLED:
        return {}
    with _gauge_lock:
        return dict(_gauges)


# --- Decode meter (background grabber threads, many) ------------------------- #
_dec_lock = threading.Lock()
_dec_ms = 0.0
_dec_count = 0
_dec_last_report = time.time()


def record_decode(ms: float) -> None:
    """Record one frame decode (cap.read) from a grabber thread. Reports a
    rolling summary roughly every 5s."""
    if not _ENABLED:
        return
    global _dec_ms, _dec_count, _dec_last_report
    with _dec_lock:
        _dec_ms += ms
        _dec_count += 1
        now = time.time()
        elapsed = now - _dec_last_report
        if elapsed >= 5.0 and _dec_count:
            rate = _dec_count / elapsed
            avg = _dec_ms / _dec_count
            # _dec_ms is wall-ms spent decoding across all grabber threads in the
            # window; /elapsed*100 ≈ how many CPU-cores-worth of decode that is.
            cores = _dec_ms / 1000.0 / elapsed
            print(
                f"[PERF] decode: {avg:.1f}ms/frame | {rate:.0f} decodes/s across all cams "
                f"(~{cores:.1f} CPU-core(s) busy decoding)"
            )
            _dec_ms = 0.0
            _dec_count = 0
            _dec_last_report = now
