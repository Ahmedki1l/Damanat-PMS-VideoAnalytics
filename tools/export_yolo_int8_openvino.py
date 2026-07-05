"""
tools/export_yolo_int8_openvino.py — Quantize the YOLO detector to INT8 (AMX).

Takes the existing FP32 OpenVINO IR (``models/yolo11m_320_openvino_model``) and
produces an INT8 IR via NNCF post-training quantization, calibrated on real
camera frames. On the production Xeon (avx512_vnni + amx_int8) the INT8 IR runs
several × faster than FP32 — the lever that decides whether 8 cores can sustain
4–5 fps across 23 cameras.

Why quantize the IR (not the .pt)
---------------------------------
The FP32 320 export already exists, so there's no need to re-export from the
``.pt``. We quantize the IR directly — same approach as
``tools/export_osnet_openvino.py`` for the ReID backbone (which is how
``models/PS_carMatching`` was already made INT8).

Calibration data — accuracy vs speed
------------------------------------
Quantization speed is independent of calibration quality; **calibration data
only affects accuracy.** So an INT8 IR built from any reasonable full frames is
valid for *measuring latency* on the server. For a production-accuracy IR,
calibrate on full frames from the actual parking cameras (B1/B2/ground), not
crops and not only the ANPR gate view.

YOLO calibration wants **full frames** at the inference resolution, letterboxed
to imgsz the same way Ultralytics does at runtime (resize keeping aspect ratio,
pad to square with 114, RGB, /255, NCHW). ``--min-side`` filters out tiny crops
so pointing ``--calib-dir`` at a mixed folder (e.g. ``vehicle_images/``) still
only feeds genuine frames to the calibrator.

Usage
-----

    # First, speed-oriented build from the full-frame grouped_cars set:
    python tools/export_yolo_int8_openvino.py \
        --calib-dir "C:/Users/moham/Desktop/New folder (2)/grouped_cars" \
        --min-side 640

    # Production-accuracy build once parking-cam frames are gathered:
    python tools/export_yolo_int8_openvino.py \
        --calib-dir data/calib_parking_frames --subset-size 300

Outputs (in ``--output-dir``, default models/yolo11m_320_int8_openvino_model):

    <stem>.xml / <stem>.bin   # INT8 OpenVINO IR
    metadata.yaml             # calibration provenance
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("export_yolo_int8")

DEFAULT_FP32_IR = Path("models/yolo11m_320_openvino_model")
DEFAULT_OUTPUT_DIR = Path("models/yolo11m_320_int8_openvino_model")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def letterbox_chw(bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """YOLO-style letterbox a BGR frame to (imgsz, imgsz): scale keeping aspect
    ratio, pad with 114, BGR→RGB, /255, CHW, add batch → NCHW float32.

    Mirrors Ultralytics' inference preprocessing so the calibration activation
    statistics match what the model sees at runtime."""
    h, w = bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[np.newaxis, ...]


def list_calibration_frames(calib_dir: Path, min_side: int) -> List[Path]:
    """Full-frame images under calib_dir whose shorter side ≥ min_side (so tiny
    crops in a mixed folder are skipped)."""
    paths = [
        Path(p) for p in glob.glob(str(calib_dir / "**" / "*"), recursive=True)
        if Path(p).suffix.lower() in IMAGE_EXTS and Path(p).is_file()
    ]
    kept = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        if min(im.shape[:2]) < min_side:
            continue
        kept.append(p)
    return sorted(kept)


def build_dataset(frames: List[Path], imgsz: int, max_images: int):
    """Wrap calibration frames in an ``nncf.Dataset``."""
    import nncf

    frames = frames[:max_images]
    tensors = []
    for p in frames:
        im = cv2.imread(str(p))
        if im is None:
            continue
        tensors.append(letterbox_chw(im, imgsz))
    if not tensors:
        return None
    return nncf.Dataset(tensors)


def find_ir_xml(ir_dir: Path) -> Path:
    if ir_dir.is_file() and ir_dir.suffix == ".xml":
        return ir_dir
    xmls = sorted(ir_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No .xml in {ir_dir}")
    return xmls[0]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="export_yolo_int8_openvino",
        description="Quantize the FP32 YOLO OpenVINO IR to INT8 via NNCF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fp32-ir", type=Path, default=DEFAULT_FP32_IR,
                   help="FP32 OpenVINO model dir (or .xml) to quantize.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to write the INT8 IR.")
    p.add_argument("--calib-dir", type=Path, required=True,
                   help="Directory of full camera frames for INT8 calibration.")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Inference size (must match the FP32 IR's input).")
    p.add_argument("--min-side", type=int, default=640,
                   help="Skip calibration images whose shorter side is below "
                        "this (filters out crops in a mixed folder).")
    p.add_argument("--subset-size", type=int, default=300,
                   help="Max calibration frames sampled by NNCF.")
    p.add_argument("--min-calibration-images", type=int, default=32,
                   help="Abort if fewer usable frames than this are found.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import nncf
        import openvino as ov
    except ImportError as exc:
        logger.error("Missing dependency: %s. Install with `pip install nncf "
                     "openvino` into the runtime venv.", exc)
        return 2

    fp32_xml = find_ir_xml(args.fp32_ir)
    logger.info("Loading FP32 IR: %s", fp32_xml)
    ov_model = ov.Core().read_model(str(fp32_xml))

    logger.info("Scanning calibration frames in %s (min_side=%d) ...",
                args.calib_dir, args.min_side)
    frames = list_calibration_frames(args.calib_dir, args.min_side)
    logger.info("Found %d usable full-frame image(s).", len(frames))
    if len(frames) < args.min_calibration_images:
        logger.error("Need >= %d calibration frames; got %d. Lower --min-side "
                     "or point --calib-dir at full parking-cam frames.",
                     args.min_calibration_images, len(frames))
        return 3

    dataset = build_dataset(frames, args.imgsz, args.subset_size)
    if dataset is None:
        logger.error("No calibration tensors could be built.")
        return 3

    used = min(len(frames), args.subset_size)
    logger.info("Quantizing to INT8 with NNCF (%d frames)...", used)
    quantized = nncf.quantize(
        ov_model,
        dataset,
        subset_size=used,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        fast_bias_correction=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_xml = args.output_dir / (fp32_xml.stem + "_int8.xml")
    ov.save_model(quantized, str(out_xml), compress_to_fp16=False)
    logger.info("Saved INT8 IR -> %s", out_xml)

    # Carry over the YOLO metadata.yaml (Ultralytics runtime reads names/imgsz
    # from it) and append quantization provenance.
    src_meta = fp32_xml.parent / "metadata.yaml"
    if src_meta.exists():
        shutil.copy2(src_meta, args.output_dir / "metadata.yaml")
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    with (args.output_dir / "metadata.yaml").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n# --- INT8 quantization (appended) ---\n"
            f"quantisation: int8\n"
            f"calibration_dir: {str(args.calib_dir)!r}\n"
            f"calibration_image_count: {used}\n"
            f"calibration_min_side: {args.min_side}\n"
            f"imgsz: {args.imgsz}\n"
            f"quantized_at: {ts}\n"
        )
    logger.info("Done. Point Config.model_path at %s and benchmark with "
                "tools/bench_yolo.py.", args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
