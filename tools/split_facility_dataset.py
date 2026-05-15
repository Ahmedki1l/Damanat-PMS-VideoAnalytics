"""
tools/split_facility_dataset.py — Phase 0.5 / T-2.

Takes the curated tight-crop output of T-1 (``data/facility_top20/all/``) and
produces a balanced train/eval split with deterministic augmentation:

1. **Cap** mega-folders to ``--cap`` samples per plate (default 30).
2. **Pad** plates below ``--min-train`` (default 10 after the temporal split)
   via simple augmentations (hflip, rotation, brightness) so PK sampling can
   run with K=4 in T-3 fine-tune.
3. **Temporal split** on the date embedded in the source filename:
       date <= 2026-04-30  -> train/
       date >= 2026-05-01  -> eval/
   Plates with no images in either half fall back to a deterministic
   80 / 20 random split and are flagged in ``split_report.json``.
4. **Pairs file**: emit ``pairs_eval.csv`` with within-folder positives and
   stratified cross-folder negatives for T-5's bench.

All randomness uses ``numpy.random.default_rng(seed=42)`` so reruns produce
identical splits.

Usage
-----
    python tools/split_facility_dataset.py
    python tools/split_facility_dataset.py --cap 30 --min-train 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("split_facility_dataset")


DEFAULT_INPUT_ROOT = Path("data/facility_top20/all")
DEFAULT_OUTPUT_ROOT = Path("data/facility_top20")

# Files captured strictly before this date go to train; on/after, to eval.
TEMPORAL_CUTOFF = datetime(2026, 5, 1)

# Filename embeds a date in the form _YYYYMMDD_ (or YYYYMMDD_).
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="split_facility_dataset",
        description=(
            "Balance + temporally split curated facility crops into train/eval "
            "subdirs and emit pairs_eval.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                   help="T-1 output root (per-plate subdirs of tight crops).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Where train/, eval/, pairs_eval.csv and split_report.json are written.")
    p.add_argument("--cap", type=int, default=30,
                   help="Cap per plate before splitting.")
    p.add_argument("--min-train", type=int, default=10,
                   help="Plates with fewer train images get padded via augmentation.")
    p.add_argument("--eval-fraction", type=float, default=0.2,
                   help="Fallback random eval fraction when temporal split degenerates.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-positives-per-plate", type=int, default=10)
    p.add_argument("--num-negatives-per-plate", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Filename date parsing
# --------------------------------------------------------------------------- #


def extract_date(stem: str) -> Optional[datetime]:
    """Find the first YYYYMMDD token in the filename stem."""
    m = _DATE_RE.search(stem)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mo, d)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Augmentation (opencv-only; no extra dependency)
# --------------------------------------------------------------------------- #


def _aug_hflip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def _aug_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _aug_brightness(img: np.ndarray, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


_AUG_RECIPE = (
    ("hflip", _aug_hflip, {}),
    ("rot_p5", _aug_rotate, {"angle_deg": 5.0}),
    ("rot_m5", _aug_rotate, {"angle_deg": -5.0}),
    ("rot_p10", _aug_rotate, {"angle_deg": 10.0}),
    ("rot_m10", _aug_rotate, {"angle_deg": -10.0}),
    ("bright_p15", _aug_brightness, {"beta": 15}),
    ("bright_m15", _aug_brightness, {"beta": -15}),
    ("hflip_rot_p5", lambda im: _aug_rotate(_aug_hflip(im), 5.0), {}),
    ("hflip_rot_m5", lambda im: _aug_rotate(_aug_hflip(im), -5.0), {}),
    ("hflip_bright_p15", lambda im: _aug_brightness(_aug_hflip(im), 15), {}),
)


def augment_image(img: np.ndarray, tag: str) -> np.ndarray:
    for rname, fn, kwargs in _AUG_RECIPE:
        if rname == tag:
            return fn(img, **kwargs) if kwargs else fn(img)
    raise KeyError(f"Unknown aug tag: {tag}")


# --------------------------------------------------------------------------- #
# Split logic
# --------------------------------------------------------------------------- #


def _split_plate(
    images: List[Tuple[Path, Optional[datetime]]],
    rng: np.random.Generator,
    cap: int,
    eval_fraction: float,
) -> Tuple[List[Path], List[Path], str]:
    """Return (train_paths, eval_paths, mode).

    Mode is "temporal" when both halves are non-empty, "random_fallback" when
    one half is empty (e.g. all images dated in March), or "tiny" when there
    are fewer than two images total (everything goes to train).
    """
    # 1. Cap to ``cap`` random samples for balance.
    if len(images) > cap:
        idx = rng.permutation(len(images))[:cap]
        images = [images[i] for i in idx]

    if len(images) < 2:
        return [p for p, _ in images], [], "tiny"

    pre_cutoff = [p for p, d in images if d is not None and d < TEMPORAL_CUTOFF]
    post_cutoff = [p for p, d in images if d is not None and d >= TEMPORAL_CUTOFF]

    if pre_cutoff and post_cutoff:
        return pre_cutoff, post_cutoff, "temporal"

    # Fallback: deterministic random 80/20 split.
    paths = [p for p, _ in images]
    rng.shuffle(paths)
    n_eval = max(1, int(round(len(paths) * eval_fraction)))
    eval_paths = paths[:n_eval]
    train_paths = paths[n_eval:]
    return train_paths, eval_paths, "random_fallback"


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Failed to read {src}")
    cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def _pad_via_augmentation(
    plate_train_dir: Path,
    base_files: List[Path],
    target_count: int,
    rng: np.random.Generator,
) -> int:
    """Add augmented copies under plate_train_dir until count >= target_count.

    Augmented filenames are ``<source-stem>__aug<recipe>.jpg`` so they are
    obviously synthetic and ordered after originals when sorted.
    Returns the number of new files written.
    """
    if len(base_files) == 0:
        return 0
    current = list(plate_train_dir.glob("*.jpg"))
    n_have = len(current)
    n_need = max(0, target_count - n_have)
    if n_need == 0:
        return 0

    written = 0
    aug_idx = 0
    aug_recipes = [r[0] for r in _AUG_RECIPE]
    # Loop until target hit; pick deterministic (file, recipe) pairs.
    while written < n_need:
        # Cycle through original base files so we don't aug the same image 10x.
        src = base_files[written % len(base_files)]
        recipe = aug_recipes[aug_idx % len(aug_recipes)]
        aug_idx += 1

        out_name = f"{src.stem}__aug_{recipe}.jpg"
        out_path = plate_train_dir / out_name
        if out_path.exists():
            continue  # already augmented with this recipe — keep going

        img = cv2.imread(str(src))
        if img is None:
            continue
        try:
            aug = augment_image(img, recipe)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("aug %s on %s failed: %r", recipe, src.name, exc)
            continue
        cv2.imwrite(str(out_path), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written += 1

        # Safety: avoid infinite loops if the recipe list cycles without
        # writing new files.
        if aug_idx > target_count * len(aug_recipes):
            break

    return written


# --------------------------------------------------------------------------- #
# Pairs CSV
# --------------------------------------------------------------------------- #


def _emit_pairs(
    eval_root: Path,
    output_csv: Path,
    rng: np.random.Generator,
    num_pos_per_plate: int,
    num_neg_per_plate: int,
) -> Tuple[int, int]:
    """Emit pairs_eval.csv with within-folder positives + cross-folder negatives.

    Paths are written relative to ``output_csv.parent`` so eval scripts can
    resolve them regardless of absolute working directory.
    """
    plates = sorted(p.name for p in eval_root.iterdir() if p.is_dir())
    plate_files: Dict[str, List[Path]] = {
        plate: sorted((eval_root / plate).glob("*.jpg"))
        for plate in plates
    }
    plate_files = {p: f for p, f in plate_files.items() if len(f) >= 1}

    rel_to = output_csv.parent
    rows: List[Tuple[str, str, int]] = []

    # Positives: within-folder pairs (same plate).
    for plate, files in plate_files.items():
        if len(files) < 2:
            continue
        max_pos = min(num_pos_per_plate, len(files) * (len(files) - 1) // 2)
        seen = set()
        attempts = 0
        while len(seen) < max_pos and attempts < max_pos * 10:
            i, j = rng.integers(0, len(files), size=2)
            if i == j:
                attempts += 1
                continue
            key = tuple(sorted((int(i), int(j))))
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            qp = files[int(i)].relative_to(rel_to).as_posix()
            gp = files[int(j)].relative_to(rel_to).as_posix()
            rows.append((qp, gp, 1))
            attempts += 1

    # Negatives: cross-folder pairs (different plates).
    if len(plate_files) >= 2:
        plate_keys = list(plate_files.keys())
        for plate in plate_keys:
            files_q = plate_files[plate]
            if not files_q:
                continue
            for _ in range(num_neg_per_plate):
                other = plate
                while other == plate:
                    other = plate_keys[int(rng.integers(0, len(plate_keys)))]
                files_g = plate_files[other]
                if not files_g:
                    continue
                qi = int(rng.integers(0, len(files_q)))
                gi = int(rng.integers(0, len(files_g)))
                qp = files_q[qi].relative_to(rel_to).as_posix()
                gp = files_g[gi].relative_to(rel_to).as_posix()
                rows.append((qp, gp, 0))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_path", "gallery_path", "label"])
        for row in rows:
            writer.writerow(row)

    n_pos = sum(1 for r in rows if r[2] == 1)
    n_neg = sum(1 for r in rows if r[2] == 0)
    return n_pos, n_neg


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.input_root.exists():
        logger.error("Input root missing — run T-1 first: %s", args.input_root)
        return 1

    rng = np.random.default_rng(args.seed)
    train_root = args.output_root / "train"
    eval_root = args.output_root / "eval"
    if train_root.exists():
        logger.info("Clearing existing %s", train_root)
        for f in train_root.rglob("*"):
            if f.is_file():
                f.unlink()
    if eval_root.exists():
        logger.info("Clearing existing %s", eval_root)
        for f in eval_root.rglob("*"):
            if f.is_file():
                f.unlink()

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "cap": args.cap,
        "min_train": args.min_train,
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "per_plate": {},
        "totals": {"train": 0, "eval": 0, "augmented": 0},
    }

    plate_dirs = sorted(p for p in args.input_root.iterdir() if p.is_dir())

    for plate_dir in plate_dirs:
        plate = plate_dir.name
        images: List[Tuple[Path, Optional[datetime]]] = []
        for f in sorted(plate_dir.glob("*.jpg")):
            images.append((f, extract_date(f.stem)))
        if not images:
            logger.warning("[%s] no images — skipping", plate)
            report["per_plate"][plate] = {
                "train": 0, "eval": 0, "augmented": 0, "mode": "empty",
            }
            continue

        train_paths, eval_paths, mode = _split_plate(
            images, rng, args.cap, args.eval_fraction,
        )

        plate_train_dir = train_root / plate
        plate_eval_dir = eval_root / plate
        for src in train_paths:
            _copy_or_link(src, plate_train_dir / src.name)
        for src in eval_paths:
            _copy_or_link(src, plate_eval_dir / src.name)

        n_aug = _pad_via_augmentation(
            plate_train_dir,
            base_files=list(train_paths),
            target_count=args.min_train,
            rng=rng,
        )

        n_train_final = len(list(plate_train_dir.glob("*.jpg")))
        n_eval_final = len(list(plate_eval_dir.glob("*.jpg")))
        report["per_plate"][plate] = {
            "train": n_train_final,
            "eval": n_eval_final,
            "augmented": n_aug,
            "mode": mode,
        }
        report["totals"]["train"] += n_train_final
        report["totals"]["eval"] += n_eval_final
        report["totals"]["augmented"] += n_aug
        logger.info(
            "[%s] train=%d eval=%d augmented=%d mode=%s",
            plate, n_train_final, n_eval_final, n_aug, mode,
        )

    # Pairs CSV
    pairs_csv = args.output_root / "pairs_eval.csv"
    n_pos, n_neg = _emit_pairs(
        eval_root, pairs_csv, rng,
        args.num_positives_per_plate, args.num_negatives_per_plate,
    )
    report["pairs"] = {"positives": n_pos, "negatives": n_neg}
    logger.info("pairs_eval.csv: %d positives, %d negatives", n_pos, n_neg)

    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    report_path = args.output_root / "split_report.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    logger.info("Report written to %s", report_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
