"""
tools/parity_infer_core.py — Phase 1 parity gate for the async OpenVINO core.

Runs the SAME frames through Ultralytics ``model.track()`` (VA_INFER=ultra) and
through the raw-OpenVINO ``OVInferCore`` (VA_INFER=async), and reports how well
the detections agree. Detection geometry (box/class/conf) must match closely;
track-ids are not compared numerically because tracker state is path-local, but
the per-frame detection COUNT and matched-box IoU tell us the pre/post decode is
faithful.

Usage
-----
    python tools/parity_infer_core.py \
        --model models/yolo26s_stock_320_int8_openvino_model \
        --images tests/test_images --limit 30 --conf 0.25

Reads model + images from disk; needs no DB and no cameras — safe to run on the
dev box. A clean run shows count_match≈100% and mean matched IoU ≳0.97.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.config import DetectorConfig, TrackerConfig, DetectorPreprocessingConfig
from src.detection.tracker import TrackedDetector


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _match(gt: List, cand: List):
    """Greedy IoU match; returns (n_matched, ious, n_gt, n_cand)."""
    used = set()
    ious = []
    for d in gt:
        best, best_j = 0.0, -1
        for j, c in enumerate(cand):
            if j in used:
                continue
            v = _iou(d.bbox, c.bbox)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= 0.5:
            used.add(best_j)
            ious.append(best)
    return len(ious), ious, len(gt), len(cand)


def _build(model_path: str, conf: float, imgsz: int, mode: str) -> TrackedDetector:
    import os
    os.environ["VA_INFER"] = mode
    det_cfg = DetectorConfig(
        model_path=model_path, confidence=conf, classes=[2, 5, 7], imgsz=imgsz
    )
    trk_cfg = TrackerConfig(type="bytetrack")
    pp = DetectorPreprocessingConfig()  # defaults (enabled flag from dataclass)
    return TrackedDetector(det_cfg, trk_cfg, pp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="dir of .jpg frames")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    imgs = sorted(Path(args.images).glob("*.jpg")) or sorted(Path(args.images).rglob("*.jpg"))
    imgs = imgs[: args.limit]
    if not imgs:
        print(f"[ERROR] no .jpg under {args.images}")
        return 1

    print(f"[parity] {len(imgs)} frames | model={Path(args.model).name}")
    ultra = _build(args.model, args.conf, args.imgsz, "ultra")
    asyncd = _build(args.model, args.conf, args.imgsz, "async")
    if asyncd._ov_core is None:
        print("[ERROR] async core failed to build — see warning above.")
        return 2

    tot_gt = tot_cand = tot_matched = 0
    all_ious = []
    count_mismatches = 0
    for i, p in enumerate(imgs):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        cam = f"CAM-{i}"  # distinct per frame → fresh tracker, isolates detection
        u = ultra.detect_and_track(frame, cam)
        a = asyncd.detect_and_track(frame, cam)
        n_match, ious, n_gt, n_cand = _match(u, a)
        tot_gt += n_gt
        tot_cand += n_cand
        tot_matched += n_match
        all_ious += ious
        if n_gt != n_cand:
            count_mismatches += 1
            print(f"  [DIFF] {p.name}: ultra={n_gt} async={n_cand} matched={n_match}")

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print("\n=== Parity summary ===")
    print(f"frames             : {len(imgs)}")
    print(f"ultra detections   : {tot_gt}")
    print(f"async detections   : {tot_cand}")
    print(f"matched (IoU>=0.5) : {tot_matched}")
    print(f"count mismatches   : {count_mismatches} frame(s)")
    print(f"mean matched IoU   : {mean_iou:.4f}")
    ok = (count_mismatches == 0) and (mean_iou >= 0.95 or not all_ious)
    print(f"RESULT             : {'PASS' if ok else 'REVIEW'}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
