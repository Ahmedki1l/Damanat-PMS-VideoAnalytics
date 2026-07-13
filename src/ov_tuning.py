"""
ov_tuning.py — force OpenVINO's CPU thread budget for the YOLO detector.

WHY THIS EXISTS
---------------
Ultralytics compiles every OpenVINO model with a hardcoded LATENCY hint and no
thread cap (nn/autobackend.py: ``config={"PERFORMANCE_HINT": inference_mode}``),
and ``YOLO(...)`` gives the caller no way to pass device config through. On CPU
that hint deliberately builds a SMALL pool regardless of how many cores exist.
Measured on a 24-logical-core Ultra 9 285HX:

    PERFORMANCE_HINT=LATENCY    -> INFERENCE_NUM_THREADS=8,  NUM_STREAMS=1
    PERFORMANCE_HINT=THROUGHPUT -> INFERENCE_NUM_THREADS=24, NUM_STREAMS=6

So a 15-core box running the detector shows ~8 busy cores and looks half-idle.
That is the default, not saturation. The only seam is ``Core.compile_model``.

WHAT IT IS WORTH (measured, yolo11m_320_int8, one serial inference)
------------------------------------------------------------------
    threads= 8   13.2 ms   (default)
    threads=15   10.9 ms   1.21x
    threads=16   11.1 ms   1.19x
    threads=24   13.5 ms   0.98x   <-- SLOWER than the 8-thread default

Widening is worth ~1.2x and REGRESSES past ~16 threads: one 320px int8 inference
has too little parallel work to fill a wide pool, so threads synchronise more
than they compute. Do NOT expect 15/8 = 1.9x; it is not there.

The real lever is CONCURRENCY, not thread width. Same box, same model:
serial LATENCY = 109 inf/s, THROUGHPUT with 6 streams = 268 inf/s (2.5x). That
is what the per-group supervisor processes buy.

CONTROL (config.yaml, detector block — YAML-only, not DB-owned)
---------------------------------------------------------------
    detector:
      ov_num_threads: 15         # 0 = leave OpenVINO's default (8)
      ov_performance_hint: ""    # "" = leave LATENCY; or THROUGHPUT

Under the multi-process supervisor, LEAVE ov_num_threads AT 0. Each group is
already pinned to a disjoint core slice (cpu_affinity.py) and OpenVINO sizes its
pool from the affinity mask; a global override would oversubscribe every group
at once — 5 groups x 15 threads on 15 cores is strictly worse than 5 x 3. This
knob is for the SINGLE-process deployment, or a box where the affinity mask is
not honoured (Windows).
"""

from __future__ import annotations

from typing import Dict

# Ultralytics compiles against device "AUTO". The AUTO plugin does not accept
# INFERENCE_NUM_THREADS — it is a CPU-plugin property — so an override left on
# AUTO is silently dropped and you get the 8-thread default anyway. When we are
# overriding the thread count we therefore also pin the compile to CPU, which is
# where this pipeline runs regardless (device: cpu / no CUDA in the image).
_CPU_ONLY_PROPERTIES = ("INFERENCE_NUM_THREADS",)


def apply_openvino_overrides(num_threads: int = 0, performance_hint: str = "") -> None:
    """Patch ``Core.compile_model`` so the detector's pool honours the config.

    Must run BEFORE the model is compiled (i.e. before YOLO() builds its
    AutoBackend): OpenVINO sizes the pool once at compile time and never
    re-reads it. A no-op when both knobs are at their defaults, so the untouched
    config path keeps exactly today's Ultralytics behaviour. Safe to call twice.
    """
    extra: Dict[str, str] = {}

    try:
        n = int(num_threads or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        extra["INFERENCE_NUM_THREADS"] = str(n)

    hint = str(performance_hint or "").strip().upper()
    if hint:
        if hint in ("LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"):
            extra["PERFORMANCE_HINT"] = hint
        else:
            print(f"[OV] ignoring ov_performance_hint={hint!r} (not a valid hint)")

    if not extra:
        return

    try:
        import openvino
    except ImportError:
        return

    if getattr(openvino.Core, "_va_patched", False):
        return

    original = openvino.Core.compile_model
    force_cpu = any(k in extra for k in _CPU_ONLY_PROPERTIES)

    def compile_model(self, model, device_name=None, config=None, **kwargs):
        # Ours wins: ultralytics always passes PERFORMANCE_HINT, so merging the
        # other way round would make ov_performance_hint a no-op.
        merged = dict(config or {})
        merged.update(extra)
        target = device_name
        if force_cpu and (target is None or str(target).upper() == "AUTO"):
            target = "CPU"
        if target is None:
            return original(self, model, config=merged, **kwargs)
        return original(self, model, target, config=merged, **kwargs)

    openvino.Core.compile_model = compile_model
    openvino.Core._va_patched = True
    print(f"[OV] detector pool overridden: {extra}" + (" (device pinned to CPU)" if force_cpu else ""))
