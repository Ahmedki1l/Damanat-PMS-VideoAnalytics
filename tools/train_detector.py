"""
tools/train_detector.py — Fine-tune vehicle detectors on the teacher-labelled
parking pool (single class ``vehicle``), starting from COCO-pretrained weights.

Trains one or more candidates (yolo26n/s, yolo11n/s …) so their accuracy can be
compared back to the frozen gold-val baseline with tools/eval_detector.py. Each
run starts from the stock .pt (transfer learning), collapses detection to a
single ``vehicle`` class, and trains at the requested imgsz.

Data
----
Point ``--pool`` at the labelled pool (``images/`` + ``labels/`` from
autolabel_teacher.py). A random train/val split (default 95/5) is written as
filelists so Ultralytics validates on held-out pool frames during training. The
REAL yardstick is still the gold-val set via eval_detector.py after training —
this internal val only drives early-stopping / monitoring.

The gold-val frames were already excluded from the pool at harvest time
(harvest_detector_frames.py --exclude-dir), so this split cannot leak them.

Usage
-----
    # Fine-tune the two front-runners at deployment size:
    python tools/train_detector.py --pool data/detector_pool \
        --model models/yolo26n.pt --model models/yolo26s.pt \
        --imgsz 320 --epochs 100 --batch 32 --device 0

    # Imgsz sweep for one model:
    python tools/train_detector.py --pool data/detector_pool \
        --model models/yolo26n.pt --imgsz 256 --imgsz 320 --imgsz 416 --epochs 80

Outputs: runs/detect/<model>_<imgsz>/weights/best.pt  (then export INT8 with
tools/export_yolo_int8_openvino.py and re-evaluate).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_cuda_nms() -> None:
    """torch 2.12+cu130 here has no CUDA torchvision ``nms`` kernel; route it via
    CPU so per-epoch validation doesn't crash. No-op if the kernel works."""
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


def make_split(pool: Path, val_frac: float, seed: int) -> Path:
    """Write train/val filelists + a data.yaml; return the data.yaml path.

    Only images that have a NON-EMPTY label file are used: a full frame the
    teachers found no vehicle in is ambiguous (missed vs. genuinely empty) and is
    skipped to avoid teaching the student false negatives."""
    images_dir, labels_dir = pool / "images", pool / "labels"
    imgs = []
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        lbl = labels_dir / (p.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            imgs.append(p.resolve())
    if not imgs:
        raise SystemExit(f"No labelled images under {images_dir} (run autolabel first).")

    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac))
    val, train = imgs[:n_val], imgs[n_val:]

    (pool / "train.txt").write_text("\n".join(p.as_posix() for p in train), encoding="utf-8")
    (pool / "val.txt").write_text("\n".join(p.as_posix() for p in val), encoding="utf-8")
    data_yaml = pool / "data_split.yaml"
    data_yaml.write_text(
        f"path: {pool.resolve().as_posix()}\n"
        f"train: train.txt\nval: val.txt\nnc: 1\nnames: [vehicle]\n",
        encoding="utf-8")
    print(f"Split: {len(train)} train / {len(val)} val (of {len(imgs)} labelled). "
          f"data.yaml -> {data_yaml}")
    return data_yaml


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_detector",
        description="Fine-tune vehicle detectors on the teacher-labelled pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pool", type=Path, required=True,
                   help="Labelled pool dir (images/ + labels/).")
    p.add_argument("--model", action="append", default=[], dest="models",
                   help="Pretrained .pt to fine-tune (yolo26n/s, yolo11n/s). Repeatable.")
    p.add_argument("--imgsz", action="append", type=int, default=None,
                   help="Train size(s). Repeatable. Default: 320.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32, help="-1 for Ultralytics auto-batch.")
    p.add_argument("--patience", type=int, default=25, help="Early-stopping patience.")
    p.add_argument("--workers", type=int, default=8,
                   help="Dataloader workers. LOWER THIS ON WINDOWS if training dies "
                        "with 'bad allocation' or 'The paging file is too small': "
                        "workers spawn (not fork) on Windows, so each re-imports "
                        "torch and costs GB of COMMIT charge — the limit that runs "
                        "out first here, well before VRAM or physical RAM.")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--dry-run", action="store_true", help="Only build the split, don't train.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.models and not args.dry_run:
        raise SystemExit("No --model given.")
    ensure_cuda_nms()
    data_yaml = make_split(args.pool, args.val_frac, args.seed)
    if args.dry_run:
        return 0

    from ultralytics import YOLO
    imgszs = args.imgsz or [320]
    for model_path in args.models:
        stem = Path(model_path).stem
        for imgsz in imgszs:
            name = f"{stem}_ft_{imgsz}"
            print(f"\n=== Training {name} (from {model_path}, imgsz={imgsz}) ===")
            model = YOLO(model_path, task="detect")
            model.train(
                data=str(data_yaml), imgsz=imgsz, epochs=args.epochs,
                batch=args.batch, patience=args.patience, device=args.device,
                workers=args.workers,
                project=args.project, name=name, exist_ok=True,
                # fine-tune-friendly: keep strong aug for small dataset, single class
                pretrained=True, optimizer="auto", cos_lr=True, plots=True,
            )
            print(f"  -> {args.project}/{name}/weights/best.pt")
    print("\nDone. Export best.pt to INT8 OpenVINO, then re-run eval_detector.py "
          "against data/gold_val to compare with the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
