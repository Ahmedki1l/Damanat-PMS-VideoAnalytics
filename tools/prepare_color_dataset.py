"""
tools/prepare_color_dataset.py — Build the labelling worklist for WS-B.

Walks one or more source directories (``vehicle_images/``, ``snapshots/``,
custom paths) and emits a CSV with one row per crop that humans need to
label. Each row carries:

    crop_path        Absolute (or repo-relative) path to the image.
    suggested_color  Quick auto-label produced by K-means on dominant HSV.
                     Mapped to the closest of the 11 canonical classes —
                     intentionally noisy; just a starting hint for humans.
    confirmed_color  Empty — humans fill this in. The training pipeline
                     filters down to rows where ``confirmed_color`` is set.
    source           Top-level directory the crop came from (so we can
                     stratify or audit later).

Splits the *labelled subset* deterministically by file hash (sha1 of the
crop_path) into 70 / 15 / 15 (train / val / test). The split is encoded
in a single ``manifest.csv`` that lives at the output root, with columns
``crop_path,split,confirmed_color,source``. Training reads this file.

CLI
~~~

    python tools/prepare_color_dataset.py \\
        --source vehicle_images/ \\
        --source snapshots/ \\
        --out tests/data/color_classifier/ \\
        [--max-per-source 0]   # 0 == unlimited
        [--accept-suggestion]  # auto-set confirmed_color := suggested_color
                               # (fallback path when D-2 has not delivered;
                               # produces a "synthetic" pseudo-labelled
                               # dataset so the training pipeline can still
                               # run end-to-end.)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# Import is heavy but optional — only required for the dominant-color
# auto-labeller. Tests that just exercise CLI parsing do not need cv2.
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Class set — must mirror src/classifiers/color_classifier.COLOR_CLASS_LABELS
# Duplicated here so the prep tool can run before the package is importable
# (no torch/openvino needed in the labelling pipeline).
# --------------------------------------------------------------------------- #

COLOR_CLASS_LABELS: List[str] = [
    "black",
    "white",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "beige",
    "other",
]

# Anchor (H, S, V) representatives for each canonical color. H is in OpenCV's
# 0-180 hue range. Used by the auto-labeller — purely a hint, humans confirm.
COLOR_ANCHORS_HSV = {
    "black":  (0,   0,   20),   # very dark, any hue
    "white":  (0,   10,  235),  # very bright, very low saturation
    "grey":   (0,   10,  128),  # mid-bright, very low saturation
    "silver": (0,   10,  180),  # bright grey
    "red":    (0,   200, 160),  # also wraps at H=179, handled below
    "blue":   (110, 200, 160),
    "green":  (60,  200, 140),
    "yellow": (28,  220, 220),
    "brown":  (10,  180, 80),
    "beige":  (20,  60,  200),
    "other":  (0,   0,   0),    # never selected directly; fallback only
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Crop discovery
# --------------------------------------------------------------------------- #


def iter_image_paths(root: Path) -> Iterable[Path]:
    """Yield image files under ``root`` (recursive) deterministically sorted."""
    if not root.exists():
        return
    yield from sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------------------------------- #
# Auto-labelling — quick K-means dominant HSV
# --------------------------------------------------------------------------- #


def _hue_distance(h1: float, h2: float) -> float:
    """Circular hue distance in OpenCV's [0, 180) range."""
    d = abs(h1 - h2)
    return min(d, 180.0 - d)


def _suggest_color_from_hsv(h: float, s: float, v: float) -> str:
    """Map a dominant (H, S, V) point to the closest of the 11 anchors."""
    # Achromatic short-circuit — saturation is the strongest signal for
    # the grey/silver/white/black axis.
    if s < 30:
        if v < 60:
            return "black"
        if v > 210:
            return "white"
        if v > 160:
            return "silver"
        return "grey"

    # Special-case red: it wraps around hue=0 / hue=179 in OpenCV. Map
    # any high-saturation pixel within 12° of either edge to red.
    if s > 80 and (h <= 10 or h >= 170):
        # Still let beige steal it when V is very high and S is moderate.
        if 60 <= s <= 110 and v > 180:
            return "beige"
        return "red"

    best_label = "other"
    best_dist = float("inf")
    for label, (hh, ss, vv) in COLOR_ANCHORS_HSV.items():
        if label in {"black", "white", "grey", "silver", "other"}:
            continue
        dist = _hue_distance(h, hh) * 2.0 + abs(s - ss) * 0.4 + abs(v - vv) * 0.4
        if dist < best_dist:
            best_dist = dist
            best_label = label
    # Heuristic guard for very dim shots — call them brown rather than red.
    if best_label == "red" and v < 80:
        return "brown"
    return best_label


def suggest_color(crop_bgr: np.ndarray, k: int = 3) -> str:
    """
    Quick auto-label: K-means on small HSV crop, pick the largest cluster,
    map to nearest of the 11 canonical labels.

    Returns ``"other"`` on any error so the pipeline never crashes mid-walk.
    """
    if cv2 is None:  # pragma: no cover - environment-dependent
        return "other"
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3:
        return "other"
    if min(crop_bgr.shape[:2]) < 8:
        return "other"
    try:
        # Resize for speed; sample interior 80% to avoid border padding.
        h, w = crop_bgr.shape[:2]
        m_h = max(1, h // 10)
        m_w = max(1, w // 10)
        inner = crop_bgr[m_h : h - m_h, m_w : w - m_w]
        if inner.size == 0:
            inner = crop_bgr
        small = cv2.resize(inner, (48, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        kk = max(1, min(k, len(pixels)))
        _, labels, centers = cv2.kmeans(
            pixels, kk, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        weights = np.bincount(labels.flatten(), minlength=kk).astype(np.float32)
        weights /= max(1.0, weights.sum())
        # Sort clusters by weight, pick the dominant non-extreme cluster
        # (extreme-dark, extreme-bright clusters with very low weight are
        # often shadows / glare and would mislabel the crop).
        order = np.argsort(-weights)
        for idx in order:
            hh, ss, vv = centers[idx]
            # Skip clusters that look like pure shadow/highlight unless
            # they're dominant — they tend to be roof shadow on a coloured car.
            if weights[idx] < 0.25 and (vv < 30 or vv > 240):
                continue
            return _suggest_color_from_hsv(float(hh), float(ss), float(vv))
        # All clusters were extreme — fall back to the largest.
        hh, ss, vv = centers[int(order[0])]
        return _suggest_color_from_hsv(float(hh), float(ss), float(vv))
    except Exception as exc:
        logger.debug("[prepare_color_dataset] suggest_color failed for crop: %r", exc)
        return "other"


# --------------------------------------------------------------------------- #
# Worklist + manifest writers
# --------------------------------------------------------------------------- #


def split_for_path(path: str, train: float = 0.70, val: float = 0.15) -> str:
    """
    Deterministic 70/15/15 split based on sha1 of the path.

    Returns one of ``"train"``, ``"val"``, ``"test"``.
    """
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    # Take the top 32 bits and divide into 1_000_000 buckets for stability.
    bucket = int(digest[:8], 16) % 1_000_000
    train_cut = int(train * 1_000_000)
    val_cut = int((train + val) * 1_000_000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def write_worklist(
    rows: List[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["crop_path", "suggested_color", "confirmed_color", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(
    rows: List[dict],
    out_root: Path,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    # Only emit manifest rows that have a confirmed label.
    labelled = [r for r in rows if r.get("confirmed_color")]
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["crop_path", "split", "confirmed_color", "source"]
        )
        writer.writeheader()
        for row in labelled:
            split = split_for_path(row["crop_path"])
            writer.writerow(
                {
                    "crop_path": row["crop_path"],
                    "split": split,
                    "confirmed_color": row["confirmed_color"],
                    "source": row.get("source", ""),
                }
            )
    # Also create the empty split subdirs so downstream tools can rely
    # on their existence.
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scan vehicle crops, emit a labelling worklist + train/val/test "
            "manifest for the 11-class color classifier."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "Directory to scan for vehicle crops (recursive). Pass multiple "
            "times to merge sources. Defaults to "
            "['vehicle_images', 'snapshots'] if none given."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/data/color_classifier"),
        help="Output root directory for the worklist + split manifest.",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Cap the number of crops scanned per source (0 == unlimited).",
    )
    p.add_argument(
        "--accept-suggestion",
        action="store_true",
        help=(
            "Auto-confirm the suggested label as the confirmed label. "
            "Use this when the human-labelled data track (D-2) has not "
            "delivered yet — produces a noisy pseudo-labelled set so "
            "training can still run end-to-end. Document the use of this "
            "flag in the training README."
        ),
    )
    p.add_argument(
        "--auto-label-only",
        action="store_true",
        help=(
            "Skip writing the worklist CSV (no human-labelling step). "
            "Combine with --accept-suggestion to go straight to manifest."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_cli().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sources = [Path(s) for s in (args.source or ["vehicle_images", "snapshots"])]
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    if cv2 is None and not args.accept_suggestion:
        logger.warning(
            "OpenCV not available — auto-labelling is disabled; the "
            "suggested_color column will be empty."
        )

    rows: List[dict] = []
    seen_paths = set()
    for source in sources:
        if not source.exists():
            logger.warning("Source '%s' does not exist; skipping.", source)
            continue
        count = 0
        for path in iter_image_paths(source):
            if str(path) in seen_paths:
                continue
            seen_paths.add(str(path))

            # Compute suggested label — cheap but optional.
            suggested = ""
            if cv2 is not None:
                img = cv2.imread(str(path))
                if img is not None:
                    suggested = suggest_color(img)

            confirmed = suggested if args.accept_suggestion else ""
            rows.append(
                {
                    "crop_path": str(path).replace("\\", "/"),
                    "suggested_color": suggested,
                    "confirmed_color": confirmed,
                    "source": source.name,
                }
            )

            count += 1
            if args.max_per_source and count >= args.max_per_source:
                logger.info(
                    "Hit per-source cap (%d) for '%s' — stopping.",
                    args.max_per_source,
                    source,
                )
                break
        logger.info("Scanned %d images under '%s'", count, source)

    if not rows:
        logger.error(
            "No images found in any --source. Pass an explicit --source pointing "
            "to a directory of vehicle crops."
        )
        return 1

    if not args.auto_label_only:
        worklist_path = out_root / "worklist.csv"
        write_worklist(rows, worklist_path)
        logger.info(
            "Wrote labelling worklist with %d rows to %s",
            len(rows),
            worklist_path,
        )

    # Always (re-)write the manifest. It's empty when nothing is confirmed.
    write_manifest(rows, out_root)
    confirmed_count = sum(1 for r in rows if r.get("confirmed_color"))
    logger.info(
        "Wrote manifest.csv with %d labelled rows (of %d total) into %s",
        confirmed_count,
        len(rows),
        out_root,
    )

    if confirmed_count == 0:
        logger.warning(
            "No confirmed labels yet. Edit '%s/worklist.csv', fill in the "
            "`confirmed_color` column, then re-run with --auto-label-only "
            "(or pass --accept-suggestion to use noisy auto-labels).",
            out_root,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
