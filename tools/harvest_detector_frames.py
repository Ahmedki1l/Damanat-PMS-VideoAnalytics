"""
tools/harvest_detector_frames.py — Pull genuine FULL FRAMES out of the mixed
snapshot dumps so they can be auto-labelled and used to train / evaluate the
vehicle detector.

The problem it solves
---------------------
The snapshot folders (``vehicle_images/``, the downloaded ``images/vi``,
``images/vi2`` …) contain two very different things side by side:

  * genuine full camera frames   — e.g. 2688×1552, the whole scene (USABLE)
  * tight single-vehicle crops    — random small sizes, and the per-plate
    ``gallery/`` folders           (NOT usable for detection training)

Detection needs full frames with boxes; crops are useless (a crop is one
vehicle filling the frame — no localisation signal). This tool keeps only
frames whose shorter side ≥ ``--min-side`` and drops anything under a
``gallery/`` path, then either copies them all to a training pool or draws a
camera-stratified sample for the gold validation set.

Filenames are ``<PLATE>_<CAM>_<YYYYMMDD>_<HHMMSS>.jpg``; camera + date are parsed
for stratification and written to a manifest for provenance.

Usage
-----
    # See what's there without copying anything:
    python tools/harvest_detector_frames.py \
        --src "D:/.../images/vi" --src "D:/.../images/vi2" --dry-run

    # Draw a 250-frame, camera-stratified GOLD-VAL candidate set:
    python tools/harvest_detector_frames.py \
        --src "D:/.../images/vi" --src "D:/.../images/vi2" \
        --sample 250 --out data/gold_val/images --manifest data/gold_val/manifest.json

    # Copy the WHOLE full-frame pool for teacher auto-labelling / training:
    python tools/harvest_detector_frames.py \
        --src "D:/.../images/vi" --src "D:/.../images/vi2" \
        --out data/detector_pool/images --manifest data/detector_pool/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CAM_RE = re.compile(r"CAM[-_]?([A-Z0-9]+)", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_meta(name: str, parent: str = "") -> Dict[str, str]:
    """Robustly pull camera / date / plate from the varied snapshot names:
    ``snapshot_ANPR_CAM-ENTRY_<date>_<time>...``, ``b1_bootstrap_<hash>_<date>_<time>``,
    ``cand_<hash>_<date>_<time>``. Camera is the CAM-token if present (else the
    filename prefix as a coarse source bucket); plate comes from the per-plate
    parent folder when the frames are organised that way (Merged_Vehicles)."""
    cam_m = CAM_RE.search(name)
    if cam_m:
        cam = "CAM-" + cam_m.group(1).upper()
    else:
        cam = name.split("_")[0][:12].lower() or "UNKNOWN"  # e.g. 'cand', 'b1'
    dt = DATE_RE.search(name)
    date, time = (dt.group(1), dt.group(2)) if dt else ("00000000", "000000")
    plate = parent if parent and parent not in {"vi", "images"} else "UNKNOWN"
    return {"plate": plate, "cam": cam, "date": date, "time": time}


def is_full_frame(w: int, h: int, min_side: int, ar_lo: float, ar_hi: float) -> bool:
    """A full camera frame vs. a vehicle crop. Crops have arbitrary aspect ratios
    and are usually small; full frames are ~16:9 and reasonably large. Both the
    2688×1552 gate frames (ar≈1.73) and the 1280×720 parking frames (ar≈1.78)
    pass; square-ish and tiny crops are rejected. Using aspect+min-side (not just
    min-side) is what lets the 1280×720 parking frames — short side 720 — survive
    without letting in the many 700-ish crops."""
    if min(h, w) < min_side:
        return False
    ar = max(w, h) / max(1, min(w, h))
    return ar_lo <= ar <= ar_hi


def scan_full_frames(
    srcs: List[Path], min_side: int, ar_lo: float = 1.5, ar_hi: float = 2.0,
    exclude: Optional[set] = None,
) -> List[Dict]:
    """Return metadata for every full frame (passes is_full_frame, not under a
    gallery/ path) across all source dirs. De-dups by filename so the same
    snapshot present in several source folders is counted once. ``exclude`` is a
    set of filenames to skip (e.g. gold-val frames, to prevent train/val leakage)."""
    seen_names = set(exclude) if exclude else set()
    frames: List[Dict] = []
    for src in srcs:
        if not src.exists():
            print(f"[WARN] source not found: {src}")
            continue
        for p in src.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            if "gallery" in {part.lower() for part in p.parts}:
                continue  # ReID crops
            if p.name in seen_names:
                continue  # same snapshot present in multiple source dirs
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            if not is_full_frame(w, h, min_side, ar_lo, ar_hi):
                continue  # a crop, not a full frame
            seen_names.add(p.name)
            meta = parse_meta(p.name, p.parent.name)
            meta.update({"path": str(p), "w": w, "h": h})
            frames.append(meta)
    return frames


def stratified_sample(frames: List[Dict], n: int, seed: int) -> List[Dict]:
    """Round-robin across cameras so no single gate cam dominates the sample."""
    rng = random.Random(seed)
    by_cam: Dict[str, List[Dict]] = defaultdict(list)
    for f in frames:
        by_cam[f["cam"]].append(f)
    for lst in by_cam.values():
        rng.shuffle(lst)
    order = sorted(by_cam.keys())
    picked: List[Dict] = []
    while len(picked) < n and any(by_cam[c] for c in order):
        for c in order:
            if by_cam[c]:
                picked.append(by_cam[c].pop())
                if len(picked) >= n:
                    break
    return picked


def report(frames: List[Dict]) -> None:
    by_cam: Dict[str, int] = defaultdict(int)
    by_dim: Dict[str, int] = defaultdict(int)
    plates = set()
    for f in frames:
        by_cam[f["cam"]] += 1
        by_dim[f"{f['w']}x{f['h']}"] += 1
        plates.add(f["plate"])
    print(f"\nFull frames: {len(frames)}   cameras: {len(by_cam)}   plates: {len(plates)}")
    print("per-camera:", dict(sorted(by_cam.items())))
    print("resolutions:", dict(sorted(by_dim.items(), key=lambda kv: -kv[1])))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="harvest_detector_frames",
        description="Filter full frames out of mixed snapshot dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", action="append", type=Path, required=True,
                   help="Source snapshot dir(s). Repeatable.")
    p.add_argument("--out", type=Path, default=None,
                   help="Copy selected frames here (images dir). Omit with --dry-run.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Write selection provenance JSON here.")
    p.add_argument("--min-side", type=int, default=1000,
                   help="Keep frames whose shorter side ≥ this (drops crops).")
    p.add_argument("--sample", type=int, default=0,
                   help="Camera-stratified sample size (0 = keep ALL full frames).")
    p.add_argument("--exclude-dir", type=Path, default=None,
                   help="Skip frames whose filename appears in this dir (e.g. the "
                        "gold-val images/ — prevents train/val leakage).")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    p.add_argument("--dry-run", action="store_true", help="Report only; copy nothing.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exclude = None
    if args.exclude_dir and args.exclude_dir.exists():
        exclude = {p.name for p in args.exclude_dir.rglob("*")
                   if p.suffix.lower() in IMAGE_EXTS}
        print(f"Excluding {len(exclude)} frame(s) present in {args.exclude_dir}")
    print(f"Scanning {len(args.src)} source(s), min_side={args.min_side} ...")
    frames = scan_full_frames(args.src, args.min_side, exclude=exclude)
    report(frames)

    selected = stratified_sample(frames, args.sample, args.seed) if args.sample else frames
    if args.sample:
        print(f"\nStratified sample: {len(selected)} frames")

    if args.dry_run or not args.out:
        print("\n[dry-run] nothing copied. Re-run with --out to materialise.")
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(selected, indent=2), encoding="utf-8")
            print(f"Wrote manifest: {args.manifest}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in selected:
        # dedup destination name with camera+date+time (source names are unique already)
        dst = args.out / Path(f["path"]).name
        shutil.copy2(f["path"], dst)
        rec = dict(f); rec["dst"] = str(dst); copied.append(rec)
    print(f"\nCopied {len(copied)} frames -> {args.out}")
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(copied, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
