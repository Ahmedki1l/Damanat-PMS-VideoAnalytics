"""
tools/predict_unmatched.py — Identify _Unmatched images via the fine-tuned ReID.

For each sampled image from the source ``_Unmatched/`` bin:

1. Detect the largest vehicle with YOLO11s (same model the engine uses).
2. Crop with mild padding.
3. Extract a 512-D embedding via the production VehicleReIDMatcher pointed
   at the fine-tuned IR (``models/osnet_facility_int8_256x128``).
4. Score against per-plate gallery prototypes built from the eval split
   (mean-normalised embedding per plate).
5. Report top-K candidates with cosine scores.

The script picks a fixed-seed random subset so a re-run produces the same
samples. It also copies the cropped vehicle and the nearest-gallery
exemplar to ``data/unmatched_predictions/<sample-id>/`` so they can be
visually compared.

Usage
-----
    python tools/predict_unmatched.py --num-samples 5
    python tools/predict_unmatched.py --num-samples 5 --top-k 3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Repo-root sys.path so ``import src.*`` works when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("predict_unmatched")


DEFAULT_UNMATCHED_DIR = Path(
    r"D:\Work\Spectech\Projects\Damanat PMS Cameras\PS AI Training Dataset"
    r"\detection_images\detection_images\_Unmatched"
)
DEFAULT_GALLERY_ROOT = Path("data/facility_top20/eval")
DEFAULT_OUTPUT_ROOT = Path("data/unmatched_predictions")
DEFAULT_IR = Path("models/osnet_facility_int8_256x128")
DEFAULT_DETECTOR = "models/yolo11s_openvino_model"
VEHICLE_CLASSES = [2, 5, 7]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="predict_unmatched",
        description="Predict identities for _Unmatched images via fine-tuned ReID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--unmatched-dir", type=Path, default=DEFAULT_UNMATCHED_DIR)
    p.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--ir-dir", type=Path, default=DEFAULT_IR)
    p.add_argument("--detector-model", type=str, default=DEFAULT_DETECTOR)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--detection-conf", type=float, default=0.40)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--padding-ratio", type=float, default=0.10)
    p.add_argument("--min-area-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _expand_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio):
    bw = x2 - x1
    bh = y2 - y1
    px = bw * padding_ratio
    py = bh * padding_ratio
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(img_w, int(round(x2 + px))),
        min(img_h, int(round(y2 + py))),
    )


def _largest_vehicle(model, frame, conf, imgsz):
    results = model(frame, conf=conf, classes=VEHICLE_CLASSES, imgsz=imgsz, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    best = None
    best_area = -1.0
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if a > best_area:
            best_area = a
            cf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            best = (x1, y1, x2, y2, cf, cls)
    return best


def _build_matcher(ir_dir: Path):
    from src.config import MatchingConfig, ReIDPreprocessingConfig
    from src.reid_matcher.reid_matcher import VehicleReIDMatcher

    cfg = MatchingConfig()
    cfg.use_openvino_reid = True
    cfg.reid_openvino_model_dir = str(ir_dir)
    cfg.reid_input_size = (256, 128)
    return VehicleReIDMatcher(
        matching_config=cfg,
        preprocessing_config=ReIDPreprocessingConfig(),
        backend="openvino",
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return v.astype(np.float32) / n
    return v.astype(np.float32)


def _build_gallery_prototypes(
    matcher, gallery_root: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Path]]]:
    """Mean-normalised embedding per plate, plus the raw image list."""
    prototypes: Dict[str, np.ndarray] = {}
    images_by_plate: Dict[str, List[Path]] = {}
    for plate_dir in sorted(p for p in gallery_root.iterdir() if p.is_dir()):
        files = sorted(plate_dir.glob("*.jpg"))
        if not files:
            continue
        embeddings: List[np.ndarray] = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            v = matcher.extract_feature(img)
            if v is None:
                continue
            embeddings.append(_normalize(v))
        if not embeddings:
            continue
        proto = _normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        prototypes[plate_dir.name] = proto
        images_by_plate[plate_dir.name] = files
    return prototypes, images_by_plate


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.unmatched_dir.exists():
        logger.error("Unmatched dir missing: %s", args.unmatched_dir)
        return 1
    if not args.gallery_root.exists():
        logger.error("Gallery root missing: %s (run T-1/T-2 first)", args.gallery_root)
        return 1
    if not (args.ir_dir / "model.xml").exists():
        logger.error("ReID IR missing: %s", args.ir_dir)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)

    # 1. Pick the samples
    all_files = sorted(
        f for f in args.unmatched_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and not f.name.startswith(".")
    )
    if not all_files:
        logger.error("No images in %s", args.unmatched_dir)
        return 1
    rng = random.Random(args.seed)
    samples = rng.sample(all_files, min(args.num_samples, len(all_files)))
    logger.info("Sampled %d images from %d total in _Unmatched/.",
                len(samples), len(all_files))

    # 2. Load detector + matcher + gallery
    from ultralytics import YOLO
    logger.info("Loading detector from %s", args.detector_model)
    detector = YOLO(args.detector_model)
    logger.info("Building matcher with IR %s", args.ir_dir)
    matcher = _build_matcher(args.ir_dir)
    logger.info("Building gallery prototypes from %s", args.gallery_root)
    prototypes, images_by_plate = _build_gallery_prototypes(matcher, args.gallery_root)
    if not prototypes:
        logger.error("No usable gallery prototypes — aborting.")
        return 1
    plate_names = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[p] for p in plate_names], axis=0)

    # 3. Process each sample
    report: List[Dict] = []
    for sample_idx, src in enumerate(samples, start=1):
        logger.info("--- Sample %d: %s", sample_idx, src.name)
        img = cv2.imread(str(src))
        if img is None:
            logger.warning("Unreadable image: %s", src)
            continue
        h, w = img.shape[:2]
        det = _largest_vehicle(detector, img, args.detection_conf, args.imgsz)
        if det is None:
            logger.warning("No vehicle detected.")
            report.append({
                "sample_id": sample_idx, "file": src.name,
                "verdict": "no_detection",
            })
            continue
        x1, y1, x2, y2, det_conf, det_cls = det
        crop_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if crop_area / max(1.0, float(w * h)) < args.min_area_fraction:
            logger.warning("Detection area too small.")
            report.append({
                "sample_id": sample_idx, "file": src.name,
                "verdict": "low_area_fraction",
            })
            continue

        nx1, ny1, nx2, ny2 = _expand_bbox(x1, y1, x2, y2, w, h, args.padding_ratio)
        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            report.append({
                "sample_id": sample_idx, "file": src.name,
                "verdict": "empty_crop",
            })
            continue

        vec = matcher.extract_feature(crop)
        if vec is None:
            report.append({
                "sample_id": sample_idx, "file": src.name,
                "verdict": "embedding_failed",
            })
            continue
        vec_n = _normalize(vec)

        # Score against gallery
        sims = proto_matrix @ vec_n  # (P,)
        order = np.argsort(-sims)
        top_k = []
        for rank_i in order[: args.top_k]:
            plate = plate_names[int(rank_i)]
            top_k.append({"plate": plate, "cosine": float(sims[int(rank_i)])})

        # Stage artifacts for visual comparison
        sample_dir = args.output_root / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Save the original
        shutil.copyfile(str(src), str(sample_dir / f"00_source__{src.name}"))
        # Save the crop
        cv2.imwrite(
            str(sample_dir / "01_crop.jpg"),
            crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        # Save the closest gallery exemplar from each top-k plate
        for rank_i, entry in enumerate(top_k, start=1):
            plate = entry["plate"]
            exemplars = images_by_plate.get(plate, [])
            if exemplars:
                # Pick the first exemplar — deterministic and simple
                ex = exemplars[0]
                dst = sample_dir / f"02_rank{rank_i}_{plate}_{ex.name}"
                shutil.copyfile(str(ex), str(dst))
                entry["gallery_exemplar"] = ex.name

        # Console summary
        print(f"\nSample {sample_idx}: {src.name}")
        print(f"  detection: conf={det_conf:.3f}, area_frac={(crop_area/(w*h)):.3f}")
        print(f"  filename-claimed plate (if any): "
              f"{src.name.split('_')[0] if '_' in src.name else '?'}")
        print(f"  top-{args.top_k} predictions (cosine vs gallery prototype):")
        for r, entry in enumerate(top_k, start=1):
            print(f"    #{r}: {entry['plate']:>10}   cosine={entry['cosine']:+.4f}")
        margin = top_k[0]["cosine"] - top_k[1]["cosine"] if len(top_k) >= 2 else 0.0
        print(f"  top1-vs-top2 margin: {margin:+.4f}")

        report.append({
            "sample_id": sample_idx,
            "file": src.name,
            "filename_prefix": src.name.split("_")[0] if "_" in src.name else "",
            "detection_confidence": det_conf,
            "detection_area_fraction": crop_area / max(1.0, float(w * h)),
            "top_k": top_k,
            "top1_minus_top2": margin,
            "artifact_dir": str(sample_dir),
        })

    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Per-sample artifacts under {args.output_root}/sample_NN/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
