"""
tools/bench_speed_forms.py — End-to-end CPU predict speed across model FORMS.

Answers "how fast is the model if I just use the .pt vs export it?" by timing the
SAME Ultralytics predict() call (preprocess + inference + postprocess) on CPU for
each form — raw .pt (PyTorch-CPU), OpenVINO FP32 dir, OpenVINO INT8 dir — on a
real frame. Unlike tools/bench_yolo.py (raw OpenVINO parallel throughput), this is
the realistic per-frame latency the serial camera loop sees, and it works for
.pt too (which bench_yolo cannot load).

Usage
-----
    python tools/bench_speed_forms.py --img data/gold_val/images/<some>.jpg \
        --imgsz 320 --runs 100 \
        --model "26n .pt=runs/.../yolo26n_ft_320/weights/best.pt" \
        --model "26n INT8=runs/.../yolo26n_ft_320/weights/best_int8_openvino_model" \
        --model "11m INT8 (live)=models/yolo11m_320_int8_openvino_model"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional


def bench_one(path: str, im, imgsz: int, runs: int, warmup: int) -> dict:
    from ultralytics import YOLO
    model = YOLO(path, task="detect")
    for _ in range(warmup):
        model.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(im, imgsz=imgsz, device="cpu", verbose=False)
    dt = time.perf_counter() - t0
    ms = 1000.0 * dt / runs
    return {"ms_per_frame": ms, "fps_1thread": 1000.0 / ms if ms > 0 else 0.0}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bench_speed_forms",
                                description="CPU predict speed across model forms.")
    p.add_argument("--model", action="append", default=[], dest="models",
                   help="label=path (.pt or OpenVINO dir). Repeatable.")
    p.add_argument("--img", type=Path, required=True, help="A real frame to time on.")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--cameras", type=int, default=23)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import cv2
    args = parse_args(argv)
    im = cv2.imread(str(args.img))
    if im is None:
        raise SystemExit(f"Cannot read {args.img}")

    rows = []
    for spec in args.models:
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        print(f"[bench] {label} ...", flush=True)
        try:
            r = bench_one(path, im, args.imgsz, args.runs, args.warmup)
            rows.append((label, r["ms_per_frame"], r["fps_1thread"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc!r}")
            rows.append((label, None, None))

    print(f"\n=== CPU end-to-end predict speed (imgsz {args.imgsz}, 1 thread/serial) ===")
    hdr = f"{'form':30} {'ms/frame':>9} {'fps':>7}"
    print(hdr); print("-" * len(hdr))
    base = None
    for label, ms, fps in rows:
        if ms is None:
            print(f"{label:30.30} {'ERR':>9} {'-':>7}"); continue
        if base is None:
            base = ms
        spd = f"{base/ms:.1f}x" if base and ms else ""
        print(f"{label:30.30} {ms:9.1f} {fps:7.1f}  {spd} vs first")
    print("\nms/frame is the serial per-frame latency; lower = faster. fps = 1000/ms "
          "for ONE worker. Your pipeline parallelises across cores/cameras, so total "
          "throughput scales up from here (see tools/bench_yolo.py for the parallel ceiling).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
