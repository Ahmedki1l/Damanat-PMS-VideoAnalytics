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

WHAT IT IS WORTH — MUCH LESS THAN IT LOOKS
------------------------------------------
Measured on yolo11m_320_int8, 11 interleaved reps on an IDLE 24-core Ultra 9
285HX (median, with the run-to-run spread, because that spread is the point):

    threads= 8   11.27 ms   sigma 0.67   (default)
    threads=14   11.05 ms   sigma 0.54   1.02x  <-- indistinguishable from 8
    threads=16   10.19 ms   sigma 0.28   1.11x

So widening the pool is worth 0-10%, NOT the ~1.5x an early benchmark of mine
suggested — that run was taken while a Docker build was saturating the machine,
which inflated the 8-thread baseline. Beware: a careless measurement here is
worse than none. Run-to-run noise at a FIXED setting spanned 9.9-12.3 ms, which
is larger than the effect being measured. If you tune this, tune it on the
target box (tools/bench_yolo.py), interleaved and repeated, on an idle machine.

One 320px int8 inference simply has too little parallel work to fill a wide
pool. Do NOT expect 15/8 = 1.9x; it is not there.

The real lever is CONCURRENCY, not thread width. Same box, same model, idle:
serial LATENCY = 118 inf/s, THROUGHPUT with an async queue = 250 inf/s (2.1x).
That is what the per-group supervisor processes buy, and it is why this knob is
a footnote rather than the fix.

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
