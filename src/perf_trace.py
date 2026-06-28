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
try:
    _report_every = max(1, int(os.environ.get("PERF_TRACE_EVERY", "50") or "50"))
except ValueError:
    _report_every = 50


@contextmanager
def stage(name: str):
    """Time a named pipeline stage. No-op (and no timing) when disabled."""
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
    ``PERF_TRACE_EVERY`` frames."""
    if not _ENABLED:
        return
    global _frames
    _frames += 1
    if _frames < _report_every:
        return
    parts = " | ".join(f"{k}={_sums[k] / _frames:.0f}ms" for k in sorted(_sums))
    total = sum(_sums.values()) / _frames
    print(f"[PERF] per-frame: {parts} | total={total:.0f}ms  (avg over {_frames})")
    _sums.clear()
    _frames = 0


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
