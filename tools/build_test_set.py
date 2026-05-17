"""
tools/build_test_set.py — Build candidate pairs for the matching accuracy bench.

Walks the runtime snapshot directories (``vehicle_images/`` by default plus any
``snapshots/`` directories under tests) and emits a CSV worklist that humans
review and label. The output lives at
``tests/data/matching_eval/pairs.csv``.

The script is intentionally simple — it produces a CANDIDATE list, not a
labelled training set. Each row is a guess at an (entry, floor) pair derived
from filename / mtime heuristics; the ``is_match`` column is left blank for
manual review.

Usage:
    python tools/build_test_set.py --dry-run
    python tools/build_test_set.py --snapshot-dir vehicle_images \
        --output tests/data/matching_eval/pairs.csv

CSV schema:
    entry_id,floor_id,plate,is_match
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

LOG = logging.getLogger("build_test_set")

# Conservative regex — accepts the file naming used by VehicleRegistry
# (``b1_bootstrap_<hex>_<YYYYMMDD>_<HHMMSS>.jpg``,
# ``cand_<hex>_<YYYYMMDD>_<HHMMSS>.jpg``, ``session_<hex>_<YYYYMMDD>_<HHMMSS>.jpg``,
# ``extra_<sid>_<YYYYMMDD>_<HHMMSS>.jpg`` etc.). Files that don't match are
# ignored — they are usually orphan thumbnails or out-of-band ANPR captures.
FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-z0-9_]+?)_(?P<token>[a-f0-9]{6,})"
    r"(?:_ref\d+)?_(?P<date>\d{8})_(?P<time>\d{6})\.jpg$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CropRecord:
    """Single crop discovered on disk."""

    path: Path
    prefix: str
    token: str
    timestamp: datetime

    @property
    def kind(self) -> str:
        """Coarse classification: 'entry' (CAM-03 / b1 bootstrap / cand_*),
        'floor' (session_* / extra_*), or 'unknown'."""
        prefix = self.prefix.lower()
        if prefix.startswith("cand_") or prefix.startswith("b1_bootstrap"):
            return "entry"
        if prefix.startswith("cand"):
            return "entry"
        if prefix.startswith("session") or prefix.startswith("extra"):
            return "floor"
        return "unknown"


def _parse_filename(path: Path) -> Optional[CropRecord]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    try:
        ts = datetime.strptime(
            f"{m.group('date')}_{m.group('time')}", "%Y%m%d_%H%M%S"
        )
    except ValueError:
        return None
    return CropRecord(
        path=path,
        prefix=m.group("prefix"),
        token=m.group("token"),
        timestamp=ts,
    )


def discover_crops(directories: List[Path]) -> List[CropRecord]:
    """Walk the given directories and return every crop that parses cleanly."""
    found: List[CropRecord] = []
    for directory in directories:
        if not directory.exists():
            LOG.warning("Skipping missing directory: %s", directory)
            continue
        for jpg in directory.rglob("*.jpg"):
            rec = _parse_filename(jpg)
            if rec is not None:
                found.append(rec)
    return found


def pair_crops(
    crops: List[CropRecord],
    window_seconds: float = 600.0,
) -> List[Tuple[CropRecord, CropRecord]]:
    """Generate (entry, floor) candidate pairs.

    Pairing heuristic: for each entry crop, emit one pair with every floor
    crop seen within ``window_seconds`` afterwards. Humans then label
    ``is_match`` in the produced CSV.
    """
    entries = sorted(
        (c for c in crops if c.kind == "entry"),
        key=lambda c: c.timestamp,
    )
    floors = sorted(
        (c for c in crops if c.kind == "floor"),
        key=lambda c: c.timestamp,
    )
    pairs: List[Tuple[CropRecord, CropRecord]] = []
    for entry in entries:
        for floor in floors:
            dt = (floor.timestamp - entry.timestamp).total_seconds()
            if 0 < dt <= window_seconds:
                pairs.append((entry, floor))
    return pairs


def write_pairs_csv(
    pairs: List[Tuple[CropRecord, CropRecord]],
    output: Path,
    *,
    copy_to: Optional[Path] = None,
) -> int:
    """Write candidate pairs to ``output``. Returns row count.

    If ``copy_to`` is provided, each entry crop is mirrored into
    ``copy_to / 'entry_crops'`` and each floor crop into
    ``copy_to / 'floor_crops'`` so the dataset is self-contained.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if copy_to is not None:
        (copy_to / "entry_crops").mkdir(parents=True, exist_ok=True)
        (copy_to / "floor_crops").mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entry_id", "floor_id", "plate", "is_match"])
        for entry, floor in pairs:
            if copy_to is not None:
                shutil.copy2(entry.path, copy_to / "entry_crops" / entry.path.name)
                shutil.copy2(floor.path, copy_to / "floor_crops" / floor.path.name)
            writer.writerow([entry.path.name, floor.path.name, "", ""])
            rows_written += 1
    return rows_written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        action="append",
        default=[],
        help="Directory to scan for crops (may be repeated). Defaults to "
        "vehicle_images and tests/test_images.",
    )
    parser.add_argument(
        "--output",
        default="tests/data/matching_eval/pairs.csv",
        help="Output CSV path (default: tests/data/matching_eval/pairs.csv).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=600.0,
        help="Pair entries with floor crops seen within this window.",
    )
    parser.add_argument(
        "--copy-into",
        default=None,
        help="If set, copy each crop into <path>/entry_crops or "
        "<path>/floor_crops alongside the CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching the filesystem.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Increase logging verbosity.",
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
    LOG.info("Discovered %d crops from %s", len(crops), snapshot_dirs)
    pairs = pair_crops(crops, window_seconds=args.window_seconds)
    LOG.info("Generated %d candidate pairs", len(pairs))

    if args.dry_run:
        for entry, floor in pairs[:25]:
            print(f"{entry.path.name},{floor.path.name},,")
        if len(pairs) > 25:
            print(f"... ({len(pairs) - 25} more)")
        return 0

    output = Path(args.output)
    copy_to = Path(args.copy_into) if args.copy_into else None
    rows_written = write_pairs_csv(pairs, output, copy_to=copy_to)
    LOG.info("Wrote %d rows to %s", rows_written, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
