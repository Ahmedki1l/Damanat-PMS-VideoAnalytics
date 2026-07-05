"""
tools/bench_yolo.py — Measure YOLO OpenVINO inference throughput on THIS box.

Turns the capacity estimate into real numbers: for each combination of model
(FP32 / INT8 IR), input size (imgsz) and CPU-thread budget, it runs the model in
OpenVINO THROUGHPUT mode (AsyncInferQueue, several inferences in flight) and
reports **total inferences/s** and the implied **fps-per-camera** for a given
camera count.

Why this exists
---------------
The production pipeline is a serial single-inference loop, so wall-clock per
frame (~1.5 s) does not reveal the parallel ceiling of the CPU. THROUGHPUT mode
fills the core budget the way a parallel inference pool (or N per-floor
processes) would, so the number here is the realistic upper bound for
"23 cameras at N fps" once inference is parallelised. The one number that
decides 8-core feasibility is INT8 @ imgsz 320 (~40 ms/inf → ~5 fps/cam on 8
cores; ~80 ms → ~2 fps/cam).

Uses the OpenVINO Python API directly (no external benchmark_app needed). To
hard-pin to physical cores on Linux, run the whole script under taskset, e.g.
``taskset -c 0-7 python tools/bench_yolo.py --threads 8 ...``; ``--threads`` sets
OpenVINO's INFERENCE_NUM_THREADS regardless.

Usage
-----

    # FP32 320 vs 640 across 8 and 16 threads, 23 cameras:
    python tools/bench_yolo.py \
        --model models/yolo11m_320_openvino_model --imgsz 320 \
        --model models/yolo11m_openvino_model     --imgsz 640 \
        --threads 8 --threads 16 --cameras 23

    # The number that matters — INT8 @320 on 8 cores (Linux, pinned):
    taskset -c 0-7 python tools/bench_yolo.py \
        --model models/yolo11m_320_int8_openvino_model --imgsz 320 \
        --threads 8 --cameras 23

Notes
-----
* Precision (FP32 vs INT8) is a property of the IR, not a flag — point
  ``--model`` at the FP32 dir or the INT8 dir. The "precision" column is
  inferred from the dir name.
* imgsz reshapes the model input to [1,3,S,S]; keep it matched to the export
  (benchmarking a 320-export at 640 reshapes it and is not a fair number).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import openvino as ov


def find_ir_xml(model_dir: Path) -> Optional[Path]:
    """Return the single OpenVINO ``.xml`` inside an exported model dir."""
    if model_dir.is_file() and model_dir.suffix == ".xml":
        return model_dir
    xmls = sorted(model_dir.glob("*.xml"))
    return xmls[0] if xmls else None


def infer_precision(model_dir: Path) -> str:
    name = model_dir.name.lower()
    if "int8" in name:
        return "int8"
    if "fp16" in name:
        return "fp16"
    return "fp32"


def measure_throughput(
    core: ov.Core,
    ir_xml: Path,
    imgsz: int,
    threads: int,
    duration: float,
    warmup: float = 3.0,
) -> float:
    """Compile in THROUGHPUT mode and return inferences/second."""
    model = core.read_model(str(ir_xml))
    model.reshape([1, 3, imgsz, imgsz])
    config = {"PERFORMANCE_HINT": "THROUGHPUT"}
    if threads and threads > 0:
        config["INFERENCE_NUM_THREADS"] = str(threads)
    compiled = core.compile_model(model, "CPU", config)

    nireq = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    nireq = max(1, int(nireq))
    queue = ov.AsyncInferQueue(compiled, nireq)

    count = {"n": 0}

    def _cb(request, userdata):
        count["n"] += 1

    queue.set_callback(_cb)
    data = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup (not counted)
    t_warm_end = time.perf_counter() + warmup
    while time.perf_counter() < t_warm_end:
        queue.start_async(data)
    queue.wait_all()
    count["n"] = 0

    t0 = time.perf_counter()
    t_end = t0 + duration
    while time.perf_counter() < t_end:
        queue.start_async(data)
    queue.wait_all()
    elapsed = time.perf_counter() - t0
    return count["n"] / elapsed if elapsed > 0 else 0.0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_yolo",
        description="Sweep YOLO OpenVINO throughput across model/imgsz/threads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", action="append", type=Path, required=True,
                   help="Exported OpenVINO model dir (or .xml). Repeatable.")
    p.add_argument("--imgsz", action="append", type=int, default=None,
                   help="Input size(s) to test. Repeatable. Default: 320 640.")
    p.add_argument("--threads", action="append", type=int, default=None,
                   help="INFERENCE_NUM_THREADS value(s). Repeatable. Default: 8 16.")
    p.add_argument("--cameras", type=int, default=23,
                   help="Video-pipeline camera count for the fps/cam column.")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Measurement seconds per config (excludes warmup).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    imgszs = args.imgsz or [320, 640]
    threads_list = args.threads or [8, 16]
    cameras = max(1, args.cameras)
    core = ov.Core()

    rows = []
    for model_dir in args.model:
        ir_xml = find_ir_xml(model_dir)
        if ir_xml is None:
            print(f"[ERROR] No .xml found in {model_dir} — skipping.")
            continue
        precision = infer_precision(model_dir)
        for imgsz in imgszs:
            for threads in threads_list:
                print(f"[RUN] {model_dir.name} | {precision} | imgsz={imgsz} | "
                      f"{threads} threads ...", flush=True)
                try:
                    fps = measure_throughput(core, ir_xml, imgsz, threads, args.duration)
                except Exception as exc:  # noqa: BLE001
                    print(f"[FAIL] {exc!r}")
                    fps = None
                rows.append((model_dir.name, precision, imgsz, threads, fps))

    print(f"\n=== Results (cameras = {cameras}) ===")
    header = f"{'model':32} {'prec':5} {'imgsz':5} {'thr':4} {'inf/s':>9} {'fps/cam':>8}"
    print(header)
    print("-" * len(header))
    for name, prec, imgsz, threads, fps in rows:
        if fps is None:
            print(f"{name:32.32} {prec:5} {imgsz:5} {threads:4} {'ERR':>9} {'-':>8}")
            continue
        per_cam = fps / cameras
        flag = "  <- >=5fps" if per_cam >= 5 else ("  <- >=4fps" if per_cam >= 4 else "")
        print(f"{name:32.32} {prec:5} {imgsz:5} {threads:4} {fps:9.1f} {per_cam:8.2f}{flag}")
    print("\nfps/cam = inf/s ÷ cameras. Target 4–5. Match imgsz to the export.")
    print("On Linux, hard-pin cores by running under `taskset -c 0-(N-1)`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
