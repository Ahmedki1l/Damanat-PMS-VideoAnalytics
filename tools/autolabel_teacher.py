"""
tools/autolabel_teacher.py — Teacher-consensus auto-labelling of vehicle boxes.

Runs a panel of strong teachers over full frames and writes YOLO-format labels
(single class ``vehicle``). A box becomes ground truth only when **at least
``--min-agree`` architecturally-different teachers agree** on it (high IoU); a box
only one teacher saw is *flagged* for human review, not written as GT. This is
what makes the pseudo-labels clean: consensus removes the false positives any
single model hallucinates, while the panel's combined recall (run at high
resolution + TTA) catches the small/distant cars a 640-px pass would miss.

Panel (default): yolo26x + yolo11x + rtdetr-x  — two CNNs and a transformer, so
agreement means genuinely different models saw the same object.

Outputs under ``--out``:
    labels/<stem>.txt     accepted boxes, "0 cx cy w h" normalised (YOLO GT)
    review/<stem>.jpg     frame with accepted (green) + flagged (yellow) boxes
    review_index.json     per-frame counts, worst-first, for the spot-fix pass
    data.yaml             ready for training / eval_detector

Spot-fix workflow (matches "consensus + you spot-fix"):
    Only frames with flagged boxes need a look. Open review_index.json (sorted by
    flags desc), inspect review/<stem>.jpg, and correct labels/<stem>.txt for the
    few that are wrong (add a missed distant car / delete a bad box).

Usage
-----
    python tools/autolabel_teacher.py \
        --images data/gold_val/images --out data/gold_val \
        --teacher models/yolo26x.pt --teacher models/yolo11x.pt \
        --teacher models/rtdetr-x.pt \
        --imgsz 1280 --tta --conf 0.25 --iou-consensus 0.6 --min-agree 2

    python tools/autolabel_teacher.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VEHICLE_COCO = [2, 5, 7]  # car, bus, truck → vehicle


def ensure_cuda_nms() -> None:
    """This box's torch (2.12+cu130) ships a torchvision whose CUDA ``nms`` kernel
    is unregistered, so GPU inference crashes in NMS. Route the (cheap) nms op
    through CPU while conv stays on GPU. No-op if the CUDA kernel works."""
    import torch
    import torchvision
    if not torch.cuda.is_available():
        return
    try:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
        torchvision.ops.nms(b, torch.tensor([0.5], device="cuda"), 0.5)
        return  # CUDA nms works — no patch needed
    except NotImplementedError:
        pass
    _orig = torchvision.ops.nms

    def _nms_cpu(boxes, scores, iou_threshold):
        return _orig(boxes.detach().cpu(), scores.detach().cpu(),
                     iou_threshold).to(boxes.device)

    torchvision.ops.nms = _nms_cpu
    print("[shim] torchvision.ops.nms routed via CPU (CUDA kernel unavailable).")


# --------------------------------------------------------------------------- #
# Consensus clustering                                                        #
# --------------------------------------------------------------------------- #
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cluster_consensus(
    dets: List[Tuple[np.ndarray, float, int]], iou_thr: float, min_agree: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Greedy conf-descending clustering of boxes pooled across teachers.

    dets: list of (box_xyxy, conf, model_id). Returns (accepted, flagged) box
    lists. A cluster is accepted when it contains ≥ min_agree DISTINCT model ids;
    its box is the confidence-weighted mean of members. Clusters seen by fewer
    models are flagged (uncertain — human review)."""
    clusters: List[Dict] = []
    for box, conf, mid in sorted(dets, key=lambda d: -d[1]):
        placed = False
        for c in clusters:
            if iou_xyxy(box, c["rep"]) >= iou_thr:
                c["boxes"].append(box)
                c["confs"].append(conf)
                c["models"].add(mid)
                w = np.asarray(c["confs"])[:, None]
                c["rep"] = (np.asarray(c["boxes"]) * w).sum(0) / w.sum()
                placed = True
                break
        if not placed:
            clusters.append({"boxes": [box], "confs": [conf], "models": {mid},
                             "rep": box.astype(float)})
    accepted = [c["rep"] for c in clusters if len(c["models"]) >= min_agree]
    flagged = [c["rep"] for c in clusters if len(c["models"]) < min_agree]
    return accepted, flagged


# --------------------------------------------------------------------------- #
# Teachers                                                                    #
# --------------------------------------------------------------------------- #
def load_teacher(path: str):
    from ultralytics import RTDETR, YOLO
    name = Path(path).name.lower()
    model = RTDETR(path) if name.startswith("rtdetr") else YOLO(path, task="detect")
    supports_tta = not name.startswith("rtdetr")  # RT-DETR ignores augment
    return model, supports_tta


def teacher_boxes(model, supports_tta, im, imgsz, conf, iou, tta, device, mid):
    """Run one teacher → list of (box_xyxy, conf, mid) for vehicle classes."""
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, device=device,
              classes=VEHICLE_COCO, verbose=False)
    if tta and supports_tta:
        kw["augment"] = True
    res = model.predict(im, **kw)[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, cf):
            out.append((b.astype(float), float(c), mid))
    return out


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def write_yolo_label(path: Path, boxes: List[np.ndarray], w: int, h: int) -> None:
    lines = []
    for b in boxes:
        cx = (b[0] + b[2]) / 2.0 / w
        cy = (b[1] + b[3]) / 2.0 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_review(im, accepted, flagged):
    import cv2
    vis = im.copy()
    for b in accepted:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 200, 0), 3)
    for b in flagged:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 220, 255), 3)
    return vis


# --------------------------------------------------------------------------- #
# Self-test — validates consensus math                                        #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("[selftest] validating consensus clustering ...")
    box = np.array([100, 100, 200, 200], dtype=float)
    jit = box + np.array([3, -2, 1, 4])  # ~same box, slightly moved
    far = np.array([500, 500, 560, 560], dtype=float)

    # two models agree on `box` → accepted; a third model's lone `far` box → flagged
    dets = [(box, 0.9, 0), (jit, 0.8, 1), (far, 0.7, 2)]
    acc, flg = cluster_consensus(dets, iou_thr=0.6, min_agree=2)
    assert len(acc) == 1 and len(flg) == 1, (len(acc), len(flg))
    assert iou_xyxy(acc[0], box) > 0.9, "accepted box should hug the agreed box"
    print(f"  2-agree + 1 lone:   accepted={len(acc)} flagged={len(flg)}  OK")

    # all three agree → 1 accepted, 0 flagged
    dets2 = [(box, 0.9, 0), (jit, 0.8, 1), (box + 1, 0.7, 2)]
    acc2, flg2 = cluster_consensus(dets2, 0.6, 2)
    assert len(acc2) == 1 and len(flg2) == 0, (len(acc2), len(flg2))
    print(f"  3-agree:            accepted={len(acc2)} flagged={len(flg2)}  OK")

    # only one model sees a box → flagged, nothing accepted
    acc3, flg3 = cluster_consensus([(box, 0.9, 0)], 0.6, 2)
    assert len(acc3) == 0 and len(flg3) == 1
    print(f"  lone detection:     accepted={len(acc3)} flagged={len(flg3)}  OK")
    print("[selftest] PASS — consensus math is sound.")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="autolabel_teacher",
        description="Teacher-consensus auto-labelling → YOLO vehicle labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", type=Path, help="Full-frame images dir.")
    p.add_argument("--out", type=Path, help="Output root (labels/ review/ data.yaml).")
    p.add_argument("--teacher", action="append", default=[], dest="teachers",
                   help="Teacher .pt (yolo*/rtdetr*). Repeatable.")
    p.add_argument("--imgsz", type=int, default=1280, help="Teacher inference size.")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (YOLO teachers).")
    p.add_argument("--conf", type=float, default=0.25, help="Per-teacher conf floor.")
    p.add_argument("--iou", type=float, default=0.7, help="Per-teacher NMS IoU.")
    p.add_argument("--iou-consensus", type=float, default=0.6,
                   help="IoU to consider two teachers' boxes the same object.")
    p.add_argument("--min-agree", type=int, default=2,
                   help="Teachers that must agree for a box to be accepted GT.")
    p.add_argument("--min-box-frac", type=float, default=0.04,
                   help="Ignore vehicles whose box height < this fraction of image "
                        "height (drops distant traffic the 320-px detector can't "
                        "see anyway). 0 = keep all. 0.04 ≈ 29px@720, 62px@1552.")
    p.add_argument("--device", type=str, default="0", help="GPU id for teachers.")
    p.add_argument("--no-review-images", action="store_true",
                   help="Skip drawing review JPGs (labels + index only).")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.images and args.out and args.teachers):
        raise SystemExit("Need --images, --out and at least one --teacher (or --selftest).")

    import cv2
    ensure_cuda_nms()
    images = sorted(p for p in args.images.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not images:
        raise SystemExit(f"No images under {args.images}")

    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    review_dir = args.out / "review"
    if not args.no_review_images:
        review_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(args.teachers)} teacher(s): {[Path(t).name for t in args.teachers]}")
    teachers = [(*load_teacher(t), i) for i, t in enumerate(args.teachers)]
    if args.min_agree > len(teachers):
        raise SystemExit(f"--min-agree {args.min_agree} > {len(teachers)} teachers.")

    index = []
    tot_acc = tot_flg = 0
    for k, img_path in enumerate(images, 1):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        dets: List[Tuple[np.ndarray, float, int]] = []
        for model, tta_ok, mid in teachers:
            dets += teacher_boxes(model, tta_ok, im, args.imgsz, args.conf,
                                  args.iou, args.tta, args.device, mid)
        if args.min_box_frac > 0:  # drop operationally-irrelevant tiny/distant cars
            dets = [d for d in dets
                    if (d[0][3] - d[0][1]) / h >= args.min_box_frac]
        accepted, flagged = cluster_consensus(dets, args.iou_consensus, args.min_agree)
        write_yolo_label(args.out / "labels" / (img_path.stem + ".txt"), accepted, w, h)
        # proposals: machine-readable accepted + flagged boxes for the review helper
        prop_dir = args.out / "proposals"
        prop_dir.mkdir(parents=True, exist_ok=True)
        (prop_dir / (img_path.stem + ".json")).write_text(json.dumps({
            "w": w, "h": h,
            "accepted": [[float(x) for x in b] for b in accepted],
            "flagged": [[float(x) for x in b] for b in flagged],
        }), encoding="utf-8")
        if not args.no_review_images:
            cv2.imwrite(str(review_dir / (img_path.stem + ".jpg")),
                        draw_review(im, accepted, flagged))
        index.append({"image": img_path.name, "accepted": len(accepted),
                      "flagged": len(flagged)})
        tot_acc += len(accepted); tot_flg += len(flagged)
        if k % 25 == 0 or k == len(images):
            print(f"  [{k}/{len(images)}] {img_path.name}: "
                  f"{len(accepted)} acc / {len(flagged)} flagged", flush=True)

    index.sort(key=lambda r: -r["flagged"])  # worst-first for the spot-fix pass
    (args.out / "review_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    # data.yaml for training / eval
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        f"train: images\nval: images\nnc: 1\nnames: [vehicle]\n", encoding="utf-8")

    n_need_review = sum(1 for r in index if r["flagged"] > 0)
    print(f"\nDone. {len(index)} frames | {tot_acc} accepted boxes | "
          f"{tot_flg} flagged across {n_need_review} frames.")
    print(f"Labels:  {args.out/'labels'}")
    print(f"Review:  open {args.out/'review_index.json'} (worst-first); "
          f"inspect review/*.jpg for the {n_need_review} flagged frames only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
