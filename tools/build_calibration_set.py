"""
tools/build_calibration_set.py — Build/sample the ReID INT8 calibration set.

Phase 1 / WS-A INT8 quantisation needs roughly 300 representative facility
crops in ``tests/data/calibration_crops/``. This script samples from
``vehicle_images/`` (the on-disk snapshot store the runtime already
populates) and copies a balanced subset into the calibration directory
ready for human review.

The selection policy is intentionally simple:

  1. Bucket images by camera id (parsed from filename ``<plate>_<CAM>_...``).
  2. Round-robin sample across cameras until ``--target`` images are copied
     or all buckets are exhausted.
  3. Skip files that are unreadable, too small, or duplicated.

The script is idempotent — re-running with the same ``--target`` keeps the
existing files and only adds missing ones. Use ``--clean`` to start fresh.

Usage
-----

    python tools/build_calibration_set.py
    python tools/build_calibration_set.py --target 300 --clean
    python tools/build_calibration_set.py \
        --source vehicle_images --output tests/data/calibration_crops
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2

logger = logging.getLogger("build_calibration_set")

DEFAULT_SOURCE = Path("vehicle_images")
DEFAULT_OUTPUT = Path("tests/data/calibration_crops")
DEFAULT_TARGET = 300
MIN_DIMENSION = 32   # discard sub-32 crops as not useful for ReID

CAMERA_RE = re.compile(r"(CAM[-_]\d+)", re.IGNORECASE)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_calibration_set",
        description=(
            "Sample vehicle crops from vehicle_images/ into "
            "tests/data/calibration_crops/ for OpenVINO INT8 quantisation "
            "(data dependency D-1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing images in --output before sampling.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _camera_of(path: Path) -> str:
    m = CAMERA_RE.search(path.name)
    return m.group(1).upper().replace("_", "-") if m else "UNKNOWN"


def _is_usable_image(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
    except Exception:  # pragma: no cover - defensive
        return False
    if img is None:
        return False
    h, w = img.shape[:2]
    return h >= MIN_DIMENSION and w >= MIN_DIMENSION


def _bucket_by_camera(paths: List[Path]) -> Dict[str, List[Path]]:
    buckets: Dict[str, List[Path]] = {}
    for p in paths:
        buckets.setdefault(_camera_of(p), []).append(p)
    for bucket in buckets.values():
        # Stable order so the sample is reproducible.
        bucket.sort()
    return buckets


def _round_robin(buckets: Dict[str, List[Path]], target: int) -> List[Path]:
    keys = sorted(buckets.keys())
    out: List[Path] = []
    cursors = {k: 0 for k in keys}
    while len(out) < target:
        progress = False
        for k in keys:
            if cursors[k] < len(buckets[k]):
                out.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progress = True
                if len(out) >= target:
                    break
        if not progress:
            break
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.source.exists():
        logger.error("Source directory %s does not exist.", args.source)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for p in args.output.glob("*"):
            if p.is_file():
                p.unlink()
        logger.info("Cleaned existing calibration images in %s", args.output)

    existing = {p.name for p in args.output.iterdir() if p.is_file()}
    needed = max(0, args.target - len(existing))
    if needed == 0:
        logger.info(
            "Calibration directory already has %d/%d images. Nothing to do.",
            len(existing), args.target,
        )
        return 0

    image_paths = [
        p for p in args.source.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    if not image_paths:
        logger.error("No image files found in %s.", args.source)
        return 2

    logger.info("Found %d source images in %s", len(image_paths), args.source)

    buckets = _bucket_by_camera(image_paths)
    logger.info(
        "Camera buckets: %s", {k: len(v) for k, v in sorted(buckets.items())}
    )

    candidates = _round_robin(buckets, target=args.target * 2)
    copied = 0
    skipped = 0
    seen: set = set()
    for src in candidates:
        if copied >= needed:
            break
        if src.name in existing or src.name in seen:
            continue
        seen.add(src.name)
        if not _is_usable_image(src):
            skipped += 1
            continue
        dst = args.output / src.name
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            logger.warning("Copy failed: %s -> %s: %s", src, dst, exc)
            continue
        copied += 1

    total = len(existing) + copied
    logger.info(
        "Copied %d new image(s); skipped %d unusable; %d total in %s "
        "(target %d).",
        copied, skipped, total, args.output, args.target,
    )
    if total < args.target:
        logger.warning(
            "Calibration set still short of target (%d/%d). Add more crops "
            "manually before INT8 quantisation.",
            total, args.target,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
