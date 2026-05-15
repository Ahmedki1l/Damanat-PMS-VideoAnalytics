"""
tools/curate_facility_dataset.py — Phase 0.5 / T-1.

Curates the 20 verified plate folders in the external PS AI Training Dataset
into tight, single-vehicle crops suitable for ReID fine-tuning and eval.

Pipeline per image
------------------
1. Read the image.
2. Run YOLO11s OpenVINO detector restricted to vehicle classes [2, 5, 7].
3. Pick the LARGEST detection above the confidence threshold (the dataset
   folders are human-verified; the most prominent vehicle is the labelled
   identity for that folder).
4. Crop the detection bbox with mild padding (default 10 %) so taillights /
   mirrors are not clipped.
5. Save to data/facility_top20/all/<plate>/<source-stem>_<hash>.jpg.

Skip reasons logged to curation_report.json:
  - unreadable image
  - no_detection (no vehicle found at conf >= threshold)
  - crop_too_small (final crop < 32 x 32 px)
  - low_area_fraction (largest crop is < 5 % of image area; usually a far
    background vehicle in a multi-car scene)

Why "largest detection" and not "exactly one vehicle"
-----------------------------------------------------
The strict "1 detection only" rule drops a lot of useful full-frame
captures (part_fielddetection_*) where the target vehicle is the most
prominent of several. The folder name is the authoritative label, so
taking the largest detection trusts the human binning while still filtering
out images where YOLO failed to find any vehicle at all.

Usage
-----
    python tools/curate_facility_dataset.py
    python tools/curate_facility_dataset.py --dry-run --verbose
    python tools/curate_facility_dataset.py \
        --source-root "D:\\path\\to\\detection_images\\detection_images" \
        --output-root data/facility_top20/all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("curate_facility_dataset")


# 20 folders verified label-correct on 2026-05-15 (see plan file).
VERIFIED_TOP20: Tuple[str, ...] = (
    "LRS-9439", "NDD-4141", "HBR-4920", "RDJ-9640", "ZDR-8501",
    "XRD-6663", "ERD-7800", "HUD-9444", "SDD-6707", "HVA-77",
    "NXR-2727", "LNV-94", "AGA-6649", "ZVH-337", "NJS-7894",
    "ZRS-6511", "XBD-5588", "HGD-2926", "EEB-80", "RTB-2016",
)


DEFAULT_SOURCE_ROOT = Path(
    r"D:\Work\Spectech\Projects\Damanat PMS Cameras\PS AI Training Dataset"
    r"\detection_images\detection_images"
)
DEFAULT_OUTPUT_ROOT = Path("data/facility_top20/all")
DEFAULT_DETECTOR_MODEL = "models/yolo11s_openvino_model"
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_PADDING_RATIO = 0.10
DEFAULT_MIN_CROP_SIDE = 32
DEFAULT_MIN_AREA_FRACTION = 0.05  # crop must be at least 5 % of frame area

VEHICLE_CLASSES: List[int] = [2, 5, 7]  # car, bus, truck (COCO)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="curate_facility_dataset",
        description=(
            "Curate the verified top-20 facility plate folders into tight "
            "single-vehicle crops for ReID fine-tuning + eval."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--detector-model", type=str, default=DEFAULT_DETECTOR_MODEL)
    p.add_argument("--confidence", type=float, default=DEFAULT_CONF_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--padding-ratio", type=float, default=DEFAULT_PADDING_RATIO,
                   help="Expand each bbox edge by this fraction before cropping.")
    p.add_argument("--min-crop-side", type=int, default=DEFAULT_MIN_CROP_SIDE)
    p.add_argument("--min-area-fraction", type=float, default=DEFAULT_MIN_AREA_FRACTION,
                   help="Reject crops that are smaller than this fraction of the source image area.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run detections but do not write any crops or the report.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _expand_bbox(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int, padding_ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand a bbox by ``padding_ratio`` on each side, clamped to the frame."""
    bw = x2 - x1
    bh = y2 - y1
    px = bw * padding_ratio
    py = bh * padding_ratio
    nx1 = max(0, int(round(x1 - px)))
    ny1 = max(0, int(round(y1 - py)))
    nx2 = min(img_w, int(round(x2 + px)))
    ny2 = min(img_h, int(round(y2 + py)))
    return nx1, ny1, nx2, ny2


def _hash_filename(name: str) -> str:
    """Short stable hash so the output filename is unique per source file."""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def _load_detector(model_path: str):
    """Lazy-import ultralytics so the tool's --help works without it."""
    from ultralytics import YOLO  # noqa: WPS433 - intentional lazy import
    logger.info("Loading YOLO detector from %s", model_path)
    return YOLO(model_path)


def _largest_vehicle_detection(
    model, frame: np.ndarray, conf: float, imgsz: int,
) -> Optional[Tuple[float, float, float, float, float, int]]:
    """Return (x1, y1, x2, y2, conf, cls) of the largest vehicle, else None."""
    results = model(
        frame,
        conf=conf,
        classes=VEHICLE_CLASSES,
        imgsz=imgsz,
        verbose=False,
    )
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    best = None
    best_area = -1.0
    for i in range(len(r.boxes)):
        xyxy = r.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            cf = float(r.boxes.conf[i].cpu().numpy())
            cls = int(r.boxes.cls[i].cpu().numpy())
            best = (x1, y1, x2, y2, cf, cls)
    return best


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.source_root.exists():
        logger.error("Source root does not exist: %s", args.source_root)
        return 1

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    model = _load_detector(args.detector_model)

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "detector_model": args.detector_model,
        "confidence_threshold": args.confidence,
        "imgsz": args.imgsz,
        "padding_ratio": args.padding_ratio,
        "min_crop_side": args.min_crop_side,
        "min_area_fraction": args.min_area_fraction,
        "dry_run": bool(args.dry_run),
        "per_plate": {},
        "skipped": [],
        "totals": {"kept": 0, "skipped": 0},
    }

    for plate in VERIFIED_TOP20:
        src_dir = args.source_root / plate
        out_dir = args.output_root / plate

        if not src_dir.exists():
            logger.warning("Missing source folder: %s", src_dir)
            report["per_plate"][plate] = {
                "kept": 0, "skipped": 0, "missing": True,
            }
            continue

        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        skipped = 0
        for src in sorted(src_dir.iterdir()):
            if not src.is_file():
                continue
            if src.suffix.lower() not in VALID_EXTS:
                continue
            if src.name.startswith("."):
                # Skip macOS .DS_Store etc.
                continue

            img = cv2.imread(str(src))
            if img is None:
                logger.debug("Unreadable: %s", src.name)
                skipped += 1
                report["skipped"].append({
                    "plate": plate, "file": src.name, "reason": "unreadable",
                })
                continue

            h, w = img.shape[:2]
            det = _largest_vehicle_detection(model, img, args.confidence, args.imgsz)
            if det is None:
                skipped += 1
                report["skipped"].append({
                    "plate": plate, "file": src.name, "reason": "no_detection",
                })
                continue

            x1, y1, x2, y2, det_conf, cls = det
            # Reject crops that take up too little of the source frame —
            # those are usually background vehicles in a multi-car scene,
            # not the labelled identity for this folder.
            crop_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_frac = crop_area / max(1.0, float(w * h))
            if area_frac < args.min_area_fraction:
                skipped += 1
                report["skipped"].append({
                    "plate": plate,
                    "file": src.name,
                    "reason": "low_area_fraction",
                    "area_fraction": round(area_frac, 4),
                })
                continue

            nx1, ny1, nx2, ny2 = _expand_bbox(
                x1, y1, x2, y2, w, h, args.padding_ratio,
            )
            crop = img[ny1:ny2, nx1:nx2]
            if (
                crop.size == 0
                or crop.shape[0] < args.min_crop_side
                or crop.shape[1] < args.min_crop_side
            ):
                skipped += 1
                report["skipped"].append({
                    "plate": plate, "file": src.name, "reason": "crop_too_small",
                })
                continue

            out_name = f"{src.stem}_{_hash_filename(src.name)}.jpg"
            if not args.dry_run:
                out_path = out_dir / out_name
                cv2.imwrite(
                    str(out_path), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
            kept += 1

        report["per_plate"][plate] = {"kept": kept, "skipped": skipped}
        report["totals"]["kept"] += kept
        report["totals"]["skipped"] += skipped
        logger.info("[%s] kept=%d skipped=%d", plate, kept, skipped)

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not args.dry_run:
        report_path = args.output_root.parent / "curation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        logger.info("Report written to %s", report_path)
    else:
        # Print the totals so a dry-run is still informative.
        logger.info(
            "[dry-run] totals: kept=%d skipped=%d",
            report["totals"]["kept"], report["totals"]["skipped"],
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
