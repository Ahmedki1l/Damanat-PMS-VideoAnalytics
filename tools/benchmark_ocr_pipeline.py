"""
tools/benchmark_ocr_pipeline.py — Full OCR pipeline benchmark across all cameras.

Evaluates the complete OCR pipeline (LPD → preprocessing → PaddleOCR) against
vehicle crop images collected from **every** available parking camera. Produces
per-camera and aggregate statistics:

  * LPD hit rate (plate detected vs. missed)
  * Fallback rate (full-crop OCR used when plate not detected)
  * OCR success rate (non-empty text returned)
  * Character-level accuracy (Levenshtein ratio, when GT is available)
  * Stage-by-stage latency (LPD detect, preprocessing, OCR read)

Usage
-----
::

    # Benchmark all cameras using crops in snapshots/
    python tools/benchmark_ocr_pipeline.py \\
        --input-dir snapshots/ \\
        --config config.yaml

    # Benchmark a specific camera subset
    python tools/benchmark_ocr_pipeline.py \\
        --input-dir snapshots/ \\
        --cameras CAM-24 CAM-25 CAM-26

    # With ground-truth CSV for accuracy metrics
    python tools/benchmark_ocr_pipeline.py \\
        --input-dir tests/data/ocr_eval \\
        --ground-truth-csv tests/data/ocr_eval/plates.csv

Ground-truth CSV schema
-----------------------
::

    crop_path,plate,camera
    CAM-24/slot_1_20260710_120000.jpg,ABC1234,CAM-24
    CAM-25/slot_3_20260710_120500.jpg,XYZ9876,CAM-25

``crop_path`` is resolved relative to ``--input-dir`` if not absolute.
``camera`` column is optional; if absent the parent directory name is used.

Output
------
JSON report on stdout (pipe to ``jq`` for pretty-printing) plus an optional
``--output`` file. Debug images are saved to ``logs/benchmark_debug/`` when
``--debug`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("benchmark_ocr_pipeline")

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Levenshtein utilities
# --------------------------------------------------------------------------- #

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between *a* and *b*."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _char_accuracy(predicted: str, ground_truth: str) -> float:
    """Character-level accuracy as 1 - (edit_dist / max(len(gt), len(pred), 1))."""
    if not ground_truth and not predicted:
        return 1.0
    dist = _levenshtein(predicted.upper().strip(), ground_truth.upper().strip())
    denom = max(len(ground_truth), len(predicted), 1)
    return max(0.0, 1.0 - dist / denom)


# --------------------------------------------------------------------------- #
# Image discovery
# --------------------------------------------------------------------------- #

def _discover_crops(
    input_dir: Path,
    cameras: Optional[List[str]] = None,
    max_per_camera: Optional[int] = None,
) -> Dict[str, List[Path]]:
    """Walk input_dir and group images by camera name (parent directory)."""
    result: Dict[str, List[Path]] = defaultdict(list)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return result

    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        # Camera name = immediate parent directory name
        cam = p.parent.name if p.parent != input_dir else "UNKNOWN"
        if cameras and cam not in cameras:
            continue
        if max_per_camera and len(result[cam]) >= max_per_camera:
            continue
        result[cam].append(p)

    return dict(result)


def _load_ground_truth(
    csv_path: Path, input_dir: Path,
) -> Dict[str, str]:
    """Load ground-truth CSV → {absolute_crop_path: plate_text}."""
    gt: Dict[str, str] = {}
    if not csv_path.exists():
        logger.warning("Ground-truth CSV not found: %s", csv_path)
        return gt
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_rel = row.get("crop_path", "").strip()
            plate = row.get("plate", "").strip()
            if not crop_rel or not plate:
                continue
            crop_abs = Path(crop_rel)
            if not crop_abs.is_absolute():
                crop_abs = input_dir / crop_abs
            gt[str(crop_abs.resolve())] = plate
    logger.info("Loaded %d ground-truth entries", len(gt))
    return gt


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #

def _run_benchmark(
    crops_by_camera: Dict[str, List[Path]],
    *,
    lpd_enabled: bool = True,
    lpd_fallback: bool = True,
    lpd_model_dir: str = "models/yolo11n_lpd_openvino_model",
    lpd_confidence: float = 0.30,
    lpd_iou: float = 0.45,
    lpd_num_threads: int = 2,
    ocr_upscale: float = 1.0,
    ocr_preprocessing: Optional[List[str]] = None,
    ground_truth: Optional[Dict[str, str]] = None,
    debug: bool = False,
    debug_dir: str = "logs/benchmark_debug",
) -> Dict[str, Any]:
    """Run the full OCR pipeline benchmark and return a results dict."""
    import cv2
    import numpy as np

    # --- Build LPD detector ------------------------------------------------ #
    detector = None
    if lpd_enabled:
        try:
            from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector
            detector = OpenVINOPlateRegionDetector(
                model_dir=lpd_model_dir,
                confidence=lpd_confidence,
                iou=lpd_iou,
                num_threads=lpd_num_threads,
            )
            logger.info("Loaded OpenVINO plate detector from %s", lpd_model_dir)
        except Exception as exc:
            logger.warning("Failed to load LPD model: %r — running without LPD", exc)
            lpd_enabled = False

    # --- Build OCR reader -------------------------------------------------- #
    try:
        from src.ocr.plate_ocr import PaddlePlateOCR
        ocr = PaddlePlateOCR(
            upscale_factor=ocr_upscale,
            preprocessing=ocr_preprocessing or [],
        )
        logger.info("PaddlePlateOCR initialised (upscale=%.2f, preprocess=%s)",
                     ocr_upscale, ocr_preprocessing or [])
    except Exception as exc:
        logger.error("Failed to load PaddleOCR: %r", exc)
        return {"error": str(exc)}

    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    gt = ground_truth or {}

    # --- Per-camera loop --------------------------------------------------- #
    camera_results: Dict[str, Dict[str, Any]] = {}
    all_latencies_lpd: List[float] = []
    all_latencies_ocr: List[float] = []

    total_crops = sum(len(v) for v in crops_by_camera.values())
    processed = 0

    for cam_name, crop_paths in sorted(crops_by_camera.items()):
        cam_stats = {
            "total": len(crop_paths),
            "lpd_hits": 0,
            "lpd_misses": 0,
            "fallbacks": 0,
            "ocr_successes": 0,
            "ocr_empty": 0,
            "latencies_lpd_ms": [],
            "latencies_ocr_ms": [],
            "char_accuracies": [],
            "exact_matches": 0,
            "gt_available": 0,
        }

        for crop_path in crop_paths:
            processed += 1
            if processed % 50 == 0:
                logger.info("Progress: %d / %d crops", processed, total_crops)

            img = cv2.imread(str(crop_path))
            if img is None:
                logger.debug("Could not read image: %s", crop_path)
                continue

            ocr_input = img
            lpd_hit = False

            # --- LPD stage ------------------------------------------------ #
            t_lpd_start = time.perf_counter()
            if lpd_enabled and detector is not None:
                try:
                    boxes = detector.detect(img)
                except Exception:
                    boxes = []
                t_lpd = (time.perf_counter() - t_lpd_start) * 1000
                cam_stats["latencies_lpd_ms"].append(t_lpd)
                all_latencies_lpd.append(t_lpd)

                if boxes:
                    lpd_hit = True
                    cam_stats["lpd_hits"] += 1
                    best = max(boxes, key=lambda b: b[4] if len(b) > 4 else 0)
                    x1b, y1b, x2b, y2b = best[0], best[1], best[2], best[3]
                    hi, wi = img.shape[:2]
                    pad_x = (x2b - x1b) * 0.15
                    pad_y = (y2b - y1b) * 0.15
                    xa = max(0, int(x1b - pad_x))
                    ya = max(0, int(y1b - pad_y))
                    xb = min(wi, int(x2b + pad_x))
                    yb = min(hi, int(y2b + pad_y))
                    if xb > xa and yb > ya:
                        plate_crop = img[ya:yb, xa:xb]
                        if plate_crop is not None and plate_crop.size > 0:
                            ocr_input = plate_crop
                        else:
                            lpd_hit = False
                            cam_stats["lpd_misses"] += 1
                    else:
                        lpd_hit = False
                        cam_stats["lpd_misses"] += 1

                if not lpd_hit:
                    cam_stats["lpd_misses"] += 1 if boxes else 1
                    if lpd_fallback:
                        cam_stats["fallbacks"] += 1
                    else:
                        cam_stats["ocr_empty"] += 1
                        continue

                # Debug output
                if debug and lpd_hit:
                    try:
                        vis = img.copy()
                        for box in boxes:
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        fname = f"{cam_name}_{crop_path.stem}_lpd.jpg"
                        cv2.imwrite(os.path.join(debug_dir, fname), vis)
                    except Exception:
                        pass
            else:
                t_lpd = 0.0

            # --- OCR stage ------------------------------------------------ #
            t_ocr_start = time.perf_counter()
            try:
                text, conf = ocr.read(ocr_input, allow_retry=True, apply_plate_roi=False)
            except TypeError:
                text, conf = ocr.read(ocr_input)
            except Exception:
                text, conf = "", 0.0
            t_ocr = (time.perf_counter() - t_ocr_start) * 1000
            cam_stats["latencies_ocr_ms"].append(t_ocr)
            all_latencies_ocr.append(t_ocr)

            if text and text.strip():
                cam_stats["ocr_successes"] += 1
            else:
                cam_stats["ocr_empty"] += 1

            # --- Ground-truth comparison ---------------------------------- #
            gt_plate = gt.get(str(crop_path.resolve()))
            if gt_plate:
                cam_stats["gt_available"] += 1
                acc = _char_accuracy(text, gt_plate)
                cam_stats["char_accuracies"].append(acc)
                if text.upper().strip() == gt_plate.upper().strip():
                    cam_stats["exact_matches"] += 1

        # --- Summarise camera --------------------------------------------- #
        total = cam_stats["total"]
        summary = {
            "total_crops": total,
            "lpd_hit_rate": cam_stats["lpd_hits"] / max(total, 1),
            "lpd_miss_rate": cam_stats["lpd_misses"] / max(total, 1),
            "fallback_rate": cam_stats["fallbacks"] / max(total, 1),
            "ocr_success_rate": cam_stats["ocr_successes"] / max(total, 1),
            "ocr_empty_rate": cam_stats["ocr_empty"] / max(total, 1),
        }
        if cam_stats["latencies_lpd_ms"]:
            lats = cam_stats["latencies_lpd_ms"]
            summary["latency_lpd_ms"] = {
                "mean": round(statistics.mean(lats), 2),
                "median": round(statistics.median(lats), 2),
                "p95": round(sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 2),
            }
        if cam_stats["latencies_ocr_ms"]:
            lats = cam_stats["latencies_ocr_ms"]
            summary["latency_ocr_ms"] = {
                "mean": round(statistics.mean(lats), 2),
                "median": round(statistics.median(lats), 2),
                "p95": round(sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 2),
            }
        if cam_stats["char_accuracies"]:
            accs = cam_stats["char_accuracies"]
            summary["char_accuracy"] = {
                "mean": round(statistics.mean(accs), 4),
                "median": round(statistics.median(accs), 4),
                "min": round(min(accs), 4),
            }
            summary["exact_match_rate"] = cam_stats["exact_matches"] / max(cam_stats["gt_available"], 1)

        camera_results[cam_name] = summary

    # --- Aggregate --------------------------------------------------------- #
    aggregate: Dict[str, Any] = {
        "total_cameras": len(camera_results),
        "total_crops": total_crops,
    }
    if all_latencies_lpd:
        aggregate["latency_lpd_ms"] = {
            "mean": round(statistics.mean(all_latencies_lpd), 2),
            "median": round(statistics.median(all_latencies_lpd), 2),
            "p95": round(sorted(all_latencies_lpd)[int(len(all_latencies_lpd) * 0.95)], 2),
        }
    if all_latencies_ocr:
        aggregate["latency_ocr_ms"] = {
            "mean": round(statistics.mean(all_latencies_ocr), 2),
            "median": round(statistics.median(all_latencies_ocr), 2),
            "p95": round(sorted(all_latencies_ocr)[int(len(all_latencies_ocr) * 0.95)], 2),
        }

    all_char_accs = []
    total_exact = 0
    total_gt = 0
    for cam_name, cam_summary in camera_results.items():
        if "char_accuracy" in cam_summary:
            # weight by crop count
            pass
    # Collect from raw stats
    for cam_name, crop_paths in crops_by_camera.items():
        for crop_path in crop_paths:
            gt_plate = gt.get(str(crop_path.resolve()))
            if gt_plate:
                total_gt += 1

    if total_gt > 0:
        exact_sum = sum(
            camera_results[c].get("exact_match_rate", 0) * camera_results[c].get("total_crops", 0)
            for c in camera_results
            if "exact_match_rate" in camera_results[c]
        )
        acc_values = []
        for c in camera_results:
            if "char_accuracy" in camera_results[c]:
                acc_values.append(camera_results[c]["char_accuracy"]["mean"])
        if acc_values:
            aggregate["char_accuracy_mean"] = round(statistics.mean(acc_values), 4)

    return {
        "config": {
            "lpd_enabled": lpd_enabled,
            "lpd_fallback": lpd_fallback,
            "lpd_model_dir": lpd_model_dir,
            "lpd_confidence": lpd_confidence,
            "lpd_iou": lpd_iou,
            "ocr_upscale": ocr_upscale,
            "ocr_preprocessing": ocr_preprocessing or [],
        },
        "aggregate": aggregate,
        "per_camera": camera_results,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the full OCR pipeline (LPD → preprocess → PaddleOCR) "
                    "across all available cameras.",
    )
    p.add_argument("--input-dir", type=Path, required=True,
                    help="Root directory containing camera subdirectories with crop images.")
    p.add_argument("--cameras", nargs="*", default=None,
                    help="Restrict to these camera names (subdirectory names). "
                         "Default: all cameras found in input-dir.")
    p.add_argument("--max-per-camera", type=int, default=None,
                    help="Max crops to process per camera (for quick smoke tests).")
    p.add_argument("--ground-truth-csv", type=Path, default=None,
                    help="CSV with crop_path,plate columns for accuracy metrics.")
    p.add_argument("--config", type=Path, default=None,
                    help="Path to config.yaml to load matching settings from.")

    # LPD overrides
    p.add_argument("--lpd-enabled", action="store_true", default=True,
                    help="Enable license plate detection (default: true).")
    p.add_argument("--no-lpd", dest="lpd_enabled", action="store_false",
                    help="Disable license plate detection.")
    p.add_argument("--no-fallback", dest="lpd_fallback", action="store_false", default=True,
                    help="Disable fallback to full-crop OCR when LPD misses.")
    p.add_argument("--lpd-model-dir", type=str, default="models/yolo11n_lpd_openvino_model")
    p.add_argument("--lpd-confidence", type=float, default=0.30)
    p.add_argument("--lpd-iou", type=float, default=0.45)
    p.add_argument("--lpd-threads", type=int, default=2)

    # Preprocessing overrides
    p.add_argument("--ocr-upscale", type=float, default=1.0,
                    help="Upscale factor for plate crop before OCR.")
    p.add_argument("--ocr-preprocessing", nargs="*", default=None,
                    help="Preprocessing steps: clahe sharpen denoise threshold")

    # Output
    p.add_argument("--output", type=Path, default=None,
                    help="Write JSON report to this file (in addition to stdout).")
    p.add_argument("--debug", action="store_true",
                    help="Save debug images with LPD bounding boxes.")
    p.add_argument("--debug-dir", type=str, default="logs/benchmark_debug")
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # --- Load config overrides from YAML if provided ----------------------- #
    lpd_enabled = args.lpd_enabled
    lpd_fallback = args.lpd_fallback
    lpd_model_dir = args.lpd_model_dir
    lpd_confidence = args.lpd_confidence
    lpd_iou = args.lpd_iou
    lpd_threads = args.lpd_threads
    ocr_upscale = args.ocr_upscale
    ocr_preprocessing = args.ocr_preprocessing

    if args.config and args.config.exists():
        try:
            from src.config import load_config
            cfg = load_config(str(args.config))
            mc = cfg.matching
            lpd_enabled = mc.slot_lpd_enabled
            lpd_fallback = mc.slot_lpd_fallback_enabled
            lpd_model_dir = mc.slot_lpd_model_dir
            lpd_confidence = mc.slot_lpd_confidence
            lpd_iou = mc.slot_lpd_iou
            lpd_threads = mc.slot_lpd_num_threads
            ocr_upscale = mc.slot_ocr_upscale
            ocr_preprocessing = mc.slot_ocr_preprocessing
            logger.info("Loaded config from %s", args.config)
        except Exception as exc:
            logger.warning("Failed to load config %s: %r — using CLI defaults", args.config, exc)

    # --- Discover images --------------------------------------------------- #
    crops_by_camera = _discover_crops(
        args.input_dir, cameras=args.cameras, max_per_camera=args.max_per_camera,
    )
    if not crops_by_camera:
        logger.error("No images found in %s", args.input_dir)
        sys.exit(1)

    total = sum(len(v) for v in crops_by_camera.values())
    logger.info(
        "Discovered %d crops across %d cameras: %s",
        total, len(crops_by_camera),
        ", ".join(f"{k}({len(v)})" for k, v in sorted(crops_by_camera.items())),
    )

    # --- Ground truth ------------------------------------------------------ #
    gt = {}
    if args.ground_truth_csv:
        gt = _load_ground_truth(args.ground_truth_csv, args.input_dir)

    # --- Run benchmark ----------------------------------------------------- #
    t0 = time.perf_counter()
    report = _run_benchmark(
        crops_by_camera,
        lpd_enabled=lpd_enabled,
        lpd_fallback=lpd_fallback,
        lpd_model_dir=lpd_model_dir,
        lpd_confidence=lpd_confidence,
        lpd_iou=lpd_iou,
        lpd_num_threads=lpd_threads,
        ocr_upscale=ocr_upscale,
        ocr_preprocessing=ocr_preprocessing,
        ground_truth=gt,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    elapsed = time.perf_counter() - t0
    report["wall_time_seconds"] = round(elapsed, 2)

    # --- Output ------------------------------------------------------------ #
    json_str = json.dumps(report, indent=2, ensure_ascii=False)
    print(json_str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_str, encoding="utf-8")
        logger.info("Report written to %s", args.output)

    logger.info("Benchmark completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
