"""
tools/eval_detector.py — Measure detection ACCURACY (mAP) of any YOLO model on a
frozen parking gold-validation set. This is the "before/after" yardstick for the
faster-detector project: run it on every stock model now to record the baseline,
then re-run it on each fine-tuned student to prove the win.

What it does
------------
For each (model, imgsz) it runs inference over a gold set of full frames with
hand-verified vehicle boxes and reports, canonical-COCO-style:

    mAP50      — mAP @ IoU 0.50            (the headline "does it find cars" number)
    mAP50-95   — mAP averaged over IoU 0.50:0.95:0.05 (localisation quality)
    P          — precision at best-F1 conf (false-positive pressure)
    R          — recall at best-F1 conf    (miss rate — matters for distant cars)

Apples-to-apples across model families
--------------------------------------
The gold labels use a single class ``vehicle`` (id 0). Stock COCO models predict
80 classes, so their car(2)/bus(5)/truck(7) boxes are kept and **remapped to
vehicle**; single-class fine-tuned / exported models keep class 0 as-is. The
remap is auto-detected from ``model.names`` (1 name → single-class; a COCO-shaped
names dict → COCO). Override with ``--coco`` / ``--single`` per model if needed.

Deployable vs. ceiling
-----------------------
Point ``--model`` at a ``.pt`` to measure the model's FP32 ceiling, or at an
OpenVINO INT8 dir to measure what actually deploys on the CPU box. Running both
for the same architecture quantifies the INT8 accuracy drop (the claim we are
testing for YOLO26: "INT8 retains nearly FP32 mAP").

Gold-set layout (Ultralytics/YOLO convention)
---------------------------------------------
    <root>/images/*.jpg          full frames
    <root>/labels/<same>.txt     one "0 cx cy w h" line per vehicle (normalised)

Pass ``--images`` + ``--labels`` explicitly, or ``--data <root>`` for the layout
above, or ``--data <data.yaml>`` (uses its ``val:`` split).

Usage
-----
    # Baseline sweep — every stock model at its candidate sizes:
    python tools/eval_detector.py --data data/gold_val \
        --model yolo11m=models/yolo11m.pt \
        --model yolo11n=models/yolo11n.pt \
        --model yolo11s=models/yolo11s.pt \
        --model yolo26n=models/yolo26n.pt \
        --model yolo26s=models/yolo26s.pt \
        --model prod_11m_int8=models/yolo11m_320_int8_openvino_model \
        --imgsz 320 --imgsz 416 --conf 0.001 --device 0 \
        --json data/gold_val/baseline_accuracy.json

    # Verify the metric math with no data/models:
    python tools/eval_detector.py --selftest

Notes
-----
* ``--conf`` should be LOW (default 0.001) so the full PR curve is built; the
  reported P/R are taken at the best-F1 operating point, not at ``--conf``.
* OpenVINO exports are static-shaped — evaluate them at the imgsz they were
  exported at (mismatched imgsz reshapes the IR and is not a fair number).
* device: ``0``/``cuda`` for .pt on the GPU; OpenVINO dirs always run on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VEHICLE_COCO = (2, 5, 7)  # car, bus, truck
IOUV = np.linspace(0.5, 0.95, 10)


def ensure_cuda_nms() -> None:
    """torch 2.12+cu130 here ships a torchvision with no CUDA ``nms`` kernel; route
    the cheap nms op via CPU so GPU (.pt) inference works. No-op if it's fine."""
    import torch
    import torchvision
    if not torch.cuda.is_available():
        return
    try:
        b = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
        torchvision.ops.nms(b, torch.tensor([0.5], device="cuda"), 0.5)
        return
    except NotImplementedError:
        pass
    _orig = torchvision.ops.nms
    torchvision.ops.nms = lambda boxes, scores, t: _orig(
        boxes.detach().cpu(), scores.detach().cpu(), t).to(boxes.device)
    print("[shim] torchvision.ops.nms routed via CPU (CUDA kernel unavailable).")


# --------------------------------------------------------------------------- #
# Ground truth                                                                #
# --------------------------------------------------------------------------- #
def _label_path_for(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


def read_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """Read a YOLO label file → xyxy pixel boxes [M,4]. Class is ignored (every
    labelled object is a vehicle) so mixed-id gold files still work."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in parts[:5])
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def resolve_data(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) from --images/--labels, a dataset root, or
    a data.yaml's val split."""
    if args.images and args.labels:
        return Path(args.images), Path(args.labels)
    if not args.data:
        raise SystemExit("Provide --data <root|data.yaml> or --images + --labels.")
    data = Path(args.data)
    if data.is_dir():
        return data / "images", data / "labels"
    # data.yaml
    import yaml
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data.parent))
    val = cfg.get("val") or cfg.get("train")
    val_img = (root / val) if not Path(val).is_absolute() else Path(val)
    # labels dir mirrors images dir with images/->labels/
    val_lbl = Path(str(val_img).replace("images", "labels"))
    return val_img, val_lbl


# --------------------------------------------------------------------------- #
# Matching + mAP (canonical Ultralytics routine)                              #
# --------------------------------------------------------------------------- #
def match_predictions(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iouv: np.ndarray
) -> np.ndarray:
    """Greedy IoU matching → correct[Npred, len(iouv)] bool. Single-class, so any
    pred may match any gt. Mirrors DetectionValidator._process_batch: at each IoU
    threshold, match by descending IoU, one pred per gt and one gt per pred."""
    from ultralytics.utils.metrics import box_iou
    import torch

    npred = pred_boxes.shape[0]
    correct = np.zeros((npred, len(iouv)), dtype=bool)
    if npred == 0 or gt_boxes.shape[0] == 0:
        return correct
    iou = box_iou(
        torch.from_numpy(gt_boxes).float(), torch.from_numpy(pred_boxes).float()
    ).numpy()  # [Ngt, Npred]
    for i, thr in enumerate(iouv):
        cand = np.argwhere(iou >= thr)  # rows of (gt_idx, pred_idx)
        if cand.shape[0] == 0:
            continue
        ious = iou[cand[:, 0], cand[:, 1]]
        cand = cand[ious.argsort()[::-1]]  # strongest overlaps first
        _, keep_pred = np.unique(cand[:, 1], return_index=True)
        cand = cand[keep_pred]
        _, keep_gt = np.unique(cand[:, 0], return_index=True)
        cand = cand[keep_gt]
        correct[cand[:, 1].astype(int), i] = True
    return correct


def compute_map(
    tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray
) -> Dict[str, float]:
    """tp[N,10], conf[N], pred_cls[N], target_cls[Ngt] → summary metrics via the
    canonical ap_per_class. Positional unpack of its 12-tuple is version-guarded."""
    from ultralytics.utils.metrics import ap_per_class

    if tp.shape[0] == 0:
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0,
                "n_pred": 0}
    res = ap_per_class(tp, conf, pred_cls, target_cls, names={0: "vehicle"})
    if len(res) != 12:
        raise RuntimeError(
            f"ap_per_class returned {len(res)} values, expected 12 — ultralytics "
            "API drift; re-check the unpack order for this version."
        )
    p, r, ap = res[2], res[3], res[5]  # per-class precision, recall, AP[nc,10]
    return {
        "map50": float(ap[:, 0].mean()),
        "map5095": float(ap.mean()),
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "n_pred": int(tp.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Per-model evaluation                                                         #
# --------------------------------------------------------------------------- #
def keep_classes_for(model, forced: Optional[str]) -> Tuple[List[int], str]:
    """Decide which raw class ids count as 'vehicle' and how to describe it."""
    names = getattr(model, "names", {}) or {}
    if forced == "coco":
        return list(VEHICLE_COCO), "coco(2,5,7)->vehicle"
    if forced == "single":
        return [0], "single(0)"
    if len(names) == 1:
        return [0], "single(0)"
    # COCO-shaped (or any multi-class): keep car/bus/truck if present
    present = [c for c in VEHICLE_COCO if c in names]
    if present:
        return present, f"coco{tuple(present)}->vehicle"
    return [0], "single(0)"


def _drop_tiny(boxes: np.ndarray, h: int, min_frac: float) -> np.ndarray:
    """Boolean mask keeping boxes whose height ≥ min_frac·image_height."""
    if min_frac <= 0 or boxes.shape[0] == 0:
        return np.ones(boxes.shape[0], dtype=bool)
    return (boxes[:, 3] - boxes[:, 1]) / h >= min_frac


def domain_of(h: int) -> str:
    """Camera domain from frame height: parking cams are 1280×720, gate cams
    2688×1552. Threshold at 1100 cleanly separates the two."""
    return "parking" if h < 1100 else "gate"


def _agg(records) -> Dict[str, float]:
    """Aggregate per-image (correct, conf, gt_cls) records → summary metrics."""
    if not records:
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0,
                "n_pred": 0, "n_gt": 0}
    tp = np.concatenate([r[0] for r in records], 0)
    conf = np.concatenate([r[1] for r in records], 0)
    target = np.concatenate([r[2] for r in records], 0)
    m = compute_map(tp, conf, np.zeros(tp.shape[0]), target)
    m["n_gt"] = int(target.shape[0])
    return m


def evaluate_model(
    label: str, model_path: str, images: List[Path], labels_dir: Path,
    imgsz: int, conf: float, iou: float, device: str, forced: Optional[str],
    min_box_frac: float = 0.0,
) -> Dict:
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    keep, remap_desc = keep_classes_for(model, forced)

    by_domain: Dict[str, list] = {"parking": [], "gate": []}
    all_records = []
    import cv2
    for img_path in images:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = read_gt(_label_path_for(img_path, labels_dir), w, h)
        gt = gt[_drop_tiny(gt, h, min_box_frac)]  # symmetric ignore-tiny

        res = model.predict(
            im, imgsz=imgsz, conf=conf, iou=iou, device=device,
            classes=keep, verbose=False,
        )[0]
        if res.boxes is not None and len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.conf.cpu().numpy()
            keep_mask = _drop_tiny(pb, h, min_box_frac)
            pb, pc = pb[keep_mask], pc[keep_mask]
        else:
            pb = np.zeros((0, 4), dtype=np.float32)
            pc = np.zeros((0,), dtype=np.float32)

        rec = (match_predictions(pb, gt, IOUV), pc,
               np.zeros(gt.shape[0], dtype=np.float32))
        all_records.append(rec)
        by_domain[domain_of(h)].append(rec)

    out = {"label": label, "model": model_path, "imgsz": imgsz,
           "remap": remap_desc, "n_images": len(all_records),
           "overall": _agg(all_records),
           "parking": _agg(by_domain["parking"]),
           "gate": _agg(by_domain["gate"])}
    return out


# --------------------------------------------------------------------------- #
# Self-test — validates the mAP math with no data or models                    #
# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("[selftest] validating match + mAP math ...")
    gt = np.array([[10, 10, 50, 50], [100, 100, 140, 140]], dtype=np.float32)

    # 1) Perfect predictions → mAP≈1.0, P≈1, R≈1
    pb = gt.copy()
    pc = np.array([0.9, 0.8])
    tp = match_predictions(pb, gt, IOUV)
    assert tp.all(), "perfect preds should be TP at every IoU threshold"
    m = compute_map(tp, pc, np.zeros(2), np.zeros(2))
    assert m["map50"] > 0.99 and m["recall"] > 0.99, m
    print(f"  perfect:            map50={m['map50']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f}  OK")

    # 2) One missed gt (only 1 of 2 predicted) → recall ≈ 0.5
    tp2 = match_predictions(gt[:1], gt, IOUV)
    m2 = compute_map(tp2, np.array([0.9]), np.zeros(1), np.zeros(2))
    assert abs(m2["recall"] - 0.5) < 1e-6, m2
    print(f"  half recall:        map50={m2['map50']:.3f} R={m2['recall']:.3f}  OK")

    # 3) A far-off false positive at lower conf → recall stays 1, mAP50 stays high
    pb3 = np.vstack([gt, [[400, 400, 440, 440]]]).astype(np.float32)
    tp3 = match_predictions(pb3, gt, IOUV)
    assert tp3[2].sum() == 0, "the far box must not match any gt"
    m3 = compute_map(tp3, np.array([0.9, 0.8, 0.3]), np.zeros(3), np.zeros(2))
    assert m3["recall"] > 0.99 and m3["map50"] > 0.9, m3
    print(f"  with 1 FP:          map50={m3['map50']:.3f} P={m3['precision']:.3f} "
          f"R={m3['recall']:.3f}  OK")

    # 4) A loosely-localised pred (IoU≈0.34) → counts at mAP50? no; recall low
    shifted = np.array([[30, 30, 70, 70]], dtype=np.float32)  # IoU with gt0 ~0.14
    tp4 = match_predictions(shifted, gt[:1], IOUV)
    assert tp4[0, 0] == 0, "a ~0.14-IoU pred must fail the 0.50 threshold"
    print("  poor-localisation:  correctly rejected at IoU0.50  OK")

    print("[selftest] PASS — metric math is sound.")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_model_specs(specs: List[str]) -> List[Tuple[str, str, Optional[str]]]:
    """Parse ``label=path`` (optionally ``label=path:coco`` / ``:single``)."""
    out = []
    for s in specs:
        forced = None
        if s.endswith(":coco"):
            s, forced = s[:-5], "coco"
        elif s.endswith(":single"):
            s, forced = s[:-7], "single"
        if "=" in s:
            label, path = s.split("=", 1)
        else:
            label, path = Path(s).stem, s
        out.append((label, path, forced))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_detector",
        description="Measure detection mAP of YOLO models on a parking gold set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=str, default=None,
                   help="Gold-set root (images/ + labels/) or a data.yaml.")
    p.add_argument("--images", type=str, default=None, help="Images dir (overrides --data).")
    p.add_argument("--labels", type=str, default=None, help="Labels dir (overrides --data).")
    p.add_argument("--model", action="append", default=[], dest="models",
                   help="label=path[:coco|:single]. Repeatable.")
    p.add_argument("--imgsz", action="append", type=int, default=None,
                   help="Inference size(s). Repeatable. Default: 320.")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Low conf so the full PR curve is built.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored by NMS-free models).")
    p.add_argument("--min-box-frac", type=float, default=0.04,
                   help="Ignore GT and predictions below this box-height fraction "
                        "(match the labeler's floor so tiny distant cars don't "
                        "distort mAP). 0 = evaluate all sizes.")
    p.add_argument("--device", type=str, default="0",
                   help="'0'/'cuda' for .pt on GPU; OpenVINO dirs run on CPU.")
    p.add_argument("--json", type=str, default=None, help="Write results JSON here.")
    p.add_argument("--selftest", action="store_true", help="Validate mAP math and exit.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.models:
        raise SystemExit("No --model given. See --help (or run --selftest).")

    ensure_cuda_nms()
    images_dir, labels_dir = resolve_data(args)
    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No images under {images_dir}")
    imgszs = args.imgsz or [320]
    specs = parse_model_specs(args.models)

    print(f"Gold set: {len(images)} images from {images_dir}")
    print(f"Labels:   {labels_dir}\n")

    results = []
    for label, path, forced in specs:
        for imgsz in imgszs:
            print(f"[RUN] {label:16} imgsz={imgsz} ...", flush=True)
            try:
                r = evaluate_model(label, path, images, labels_dir, imgsz,
                                   args.conf, args.iou, args.device, forced,
                                   args.min_box_frac)
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {exc!r}")
                r = {"label": label, "model": path, "imgsz": imgsz, "error": repr(exc)}
            results.append(r)

    # Table — one row per (model, imgsz, domain)
    hdr = (f"{'model':16} {'imgsz':5} {'domain':8} {'mAP50':>7} {'mAP50-95':>9} "
           f"{'P':>6} {'R':>6} {'n_gt':>6}")
    print("\n=== Detection accuracy on gold set (by camera domain) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['label']:16.16} {r['imgsz']:5} {'ERR':>8}  {r['error'][:40]}")
            continue
        for dom in ("parking", "gate", "overall"):
            m = r[dom]
            if m["n_gt"] == 0:
                continue
            print(f"{r['label']:16.16} {r['imgsz']:5} {dom:8} {m['map50']:7.3f} "
                  f"{m['map5095']:9.3f} {m['precision']:6.3f} {m['recall']:6.3f} "
                  f"{m['n_gt']:6}")
        print()
    print("PARKING (1280×720) is the deployment domain — that mAP50/recall is the "
          "number that matters. GATE (2688) is easier (one big frontal car).")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
