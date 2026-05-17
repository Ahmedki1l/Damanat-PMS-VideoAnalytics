"""
tools/prepare_type_dataset.py — Build a body-type labelling worklist.

This is the dataset-preparation half of WS-C (the trainer is
``tools/train_type_classifier.py``). It walks the snapshot directories
(``vehicle_images/`` and any ``snapshots/`` dirs under tests), pre-fills a
``suggested_type`` column using cheap aspect-ratio / height heuristics, and
emits three CSV worklists split deterministically by file hash (70/15/15
train/val/test).

The heuristic-based ``suggested_type`` is intentionally weak — it exists only
to give the human labeller a sane starting point. The ``confirmed_type``
column is left blank for manual review; only rows with a confirmed label are
used by the trainer.

CSV schema:
    crop_path,suggested_type,confirmed_type,source,split

    crop_path        absolute or repo-relative path to the JPG.
    suggested_type   heuristic guess in {sedan, suv, hatchback, pickup, van,
                     bus, unknown}.
    confirmed_type   human-filled (blank = pending review).
    source           "vehicle_images" / "snapshots" / etc — preserves the
                     directory of origin for audit.
    split            "train" | "val" | "test" — assigned by hashing the path.

Usage:
    python tools/prepare_type_dataset.py --help
    python tools/prepare_type_dataset.py --dry-run
    python tools/prepare_type_dataset.py \\
        --snapshot-dir vehicle_images \\
        --snapshot-dir tests/test_images \\
        --output-dir tests/data/type_classifier

Outputs (one CSV per split):
    tests/data/type_classifier/train.csv
    tests/data/type_classifier/val.csv
    tests/data/type_classifier/test.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LOG = logging.getLogger("prepare_type_dataset")

# Canonical class list — must match the trainer head (and the runtime
# plugin's labels.json). Order matters: it defines the class index used by
# the model. "unknown" is NOT a model class — it is only used as a
# suggestion when the heuristic refuses to guess.
TYPE_CLASSES: Tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "van",
    "bus",
)
ALL_SUGGESTIONS: Tuple[str, ...] = TYPE_CLASSES + ("unknown",)


@dataclass(frozen=True)
class CropInfo:
    """A crop discovered on disk with its image dimensions."""

    path: Path
    source: str  # name of the top-level scan directory
    width: int
    height: int
    file_bytes: int

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0
        return float(self.width) / float(self.height)


def _safe_image_size(path: Path) -> Optional[Tuple[int, int, int]]:
    """Return ``(w, h, file_bytes)`` for an image without requiring cv2.

    Tries Pillow first (lightweight, lazy import). Falls back to a minimal
    JPEG/PNG header parser so the tool still runs in cv2-/PIL-less envs.
    Returns None if the file cannot be read.
    """
    try:
        file_bytes = path.stat().st_size
    except OSError:
        return None

    # Pillow path — preferred, handles every format and is lazy-imported.
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h), int(file_bytes)
    except Exception:
        pass

    # Fallback: parse JPEG SOF / PNG IHDR markers directly.
    try:
        with path.open("rb") as fh:
            head = fh.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                # PNG: IHDR at byte 16, width @ 16, height @ 20 (big-endian).
                w = int.from_bytes(head[16:20], "big")
                h = int.from_bytes(head[20:24], "big")
                return w, h, int(file_bytes)
            if head[:2] == b"\xff\xd8":
                # JPEG: walk markers until SOFn.
                fh.seek(2)
                while True:
                    b = fh.read(1)
                    if not b:
                        return None
                    if b != b"\xff":
                        continue
                    marker = fh.read(1)
                    if not marker:
                        return None
                    code = marker[0]
                    # Skip standalone markers
                    if code in (0xD8, 0xD9):
                        continue
                    # SOFn markers
                    if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                        seg_len_b = fh.read(2)
                        if len(seg_len_b) < 2:
                            return None
                        fh.read(1)  # precision
                        h = int.from_bytes(fh.read(2), "big")
                        w = int.from_bytes(fh.read(2), "big")
                        return w, h, int(file_bytes)
                    # Other marker: skip its segment
                    seg_len_b = fh.read(2)
                    if len(seg_len_b) < 2:
                        return None
                    seg_len = int.from_bytes(seg_len_b, "big")
                    fh.read(max(0, seg_len - 2))
            return None
    except OSError:
        return None


def suggest_type(width: int, height: int, file_bytes: int) -> str:
    """Cheap aspect-ratio + size heuristic.

    Returns one of ``ALL_SUGGESTIONS``. Calibrated against the heuristics
    listed in the plan §WS-C T-C1:

      * wider-than-tall + short      -> pickup / van
      * tall (h > w)                 -> suv (raised stance)
      * roughly square + small       -> hatchback
      * very large + tall            -> bus
      * default                      -> sedan

    "Small" / "large" are derived from pixel count buckets — adequate for
    seeding a worklist, not for production classification.
    """
    if width <= 0 or height <= 0:
        return "unknown"

    pixels = width * height
    aspect = float(width) / float(height)

    # Bus: very large crops AND noticeably tall (buses fill the frame and
    # their height is large in absolute terms).
    if pixels > 400 * 400 and height > 350 and aspect < 1.6:
        return "bus"

    # Van / pickup: wider-than-tall (aspect >= 1.6) and short.
    if aspect >= 1.6:
        # Pickups tend to be even wider thanks to the bed cargo area.
        if aspect >= 2.0:
            return "pickup"
        return "van"

    # SUV: roughly square or taller-than-wide (raised stance).
    if aspect <= 1.05:
        # Small + square + light -> hatchback. Heavier ones -> SUV.
        if pixels < 200 * 200:
            return "hatchback"
        return "suv"

    # Default: sedan covers the typical 1.2..1.5 aspect range.
    return "sedan"


def _path_hash_bucket(path: Path, buckets: int = 100) -> int:
    """Stable hash → integer in ``[0, buckets)``. Uses the filename so the
    split is reproducible across machines (cwd-independent)."""
    key = path.name.lower().encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % buckets


def assign_split(path: Path) -> str:
    """70/15/15 deterministic split by filename hash."""
    bucket = _path_hash_bucket(path, 100)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def discover_crops(directories: Iterable[Path]) -> List[CropInfo]:
    """Walk the given directories, returning every JPG/PNG with known dims."""
    found: List[CropInfo] = []
    for directory in directories:
        if not directory.exists():
            LOG.warning("Skipping missing directory: %s", directory)
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for image in directory.rglob(ext):
                dims = _safe_image_size(image)
                if dims is None:
                    LOG.debug("Skipping unreadable image: %s", image)
                    continue
                w, h, nb = dims
                found.append(
                    CropInfo(
                        path=image,
                        source=directory.name or str(directory),
                        width=w,
                        height=h,
                        file_bytes=nb,
                    )
                )
    return found


def build_rows(crops: Iterable[CropInfo]) -> List[dict]:
    """Build worklist rows with split + suggested type."""
    rows: List[dict] = []
    for crop in crops:
        rows.append(
            {
                "crop_path": str(crop.path).replace("\\", "/"),
                "suggested_type": suggest_type(
                    crop.width, crop.height, crop.file_bytes
                ),
                "confirmed_type": "",
                "source": crop.source,
                "split": assign_split(crop.path),
            }
        )
    return rows


def write_split_csvs(rows: List[dict], output_dir: Path) -> dict:
    """Write three CSVs (train/val/test). Returns ``{split: row_count}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0, "test": 0}
    fieldnames = ["crop_path", "suggested_type", "confirmed_type", "source", "split"]
    handles = {}
    writers = {}
    try:
        for split in counts:
            fh = (output_dir / f"{split}.csv").open(
                "w", encoding="utf-8", newline=""
            )
            handles[split] = fh
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            writers[split] = w
        for row in rows:
            split = row["split"]
            writers[split].writerow(row)
            counts[split] += 1
    finally:
        for fh in handles.values():
            fh.close()
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshot-dir",
        action="append",
        default=[],
        help="Directory to scan for crops (may be repeated). Defaults to "
        "vehicle_images and tests/test_images.",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/data/type_classifier",
        help="Output directory for the train/val/test CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching the filesystem.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="If set, cap the total number of crops written to N (sorted by "
        "filename for determinism). Useful for smoke-testing the pipeline.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    snapshot_dirs = [Path(d) for d in args.snapshot_dir] or [
        Path("vehicle_images"),
        Path("tests/test_images"),
    ]

    crops = discover_crops(snapshot_dirs)
    LOG.info("Discovered %d crops across %s", len(crops), snapshot_dirs)

    # Deterministic order — sorted by filename so re-runs yield identical
    # CSVs and the --max-rows truncation is reproducible.
    crops.sort(key=lambda c: c.path.name.lower())
    if args.max_rows is not None:
        crops = crops[: args.max_rows]
        LOG.info("Truncated to --max-rows=%d", args.max_rows)

    rows = build_rows(crops)

    # Surface a class-suggestion histogram so the human labeller sees the
    # heuristic's bias before they start.
    histogram: dict = {label: 0 for label in ALL_SUGGESTIONS}
    for row in rows:
        histogram[row["suggested_type"]] = (
            histogram.get(row["suggested_type"], 0) + 1
        )
    LOG.info("Suggested-type histogram: %s", histogram)

    if args.dry_run:
        # Print first 25 rows like build_test_set.py does.
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=["crop_path", "suggested_type", "confirmed_type", "source", "split"],
        )
        writer.writeheader()
        for row in rows[:25]:
            writer.writerow(row)
        if len(rows) > 25:
            print(f"... ({len(rows) - 25} more)")
        return 0

    output_dir = Path(args.output_dir)
    counts = write_split_csvs(rows, output_dir)
    LOG.info("Wrote split CSVs to %s: %s", output_dir, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
