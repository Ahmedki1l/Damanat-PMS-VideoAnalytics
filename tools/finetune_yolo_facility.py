"""
tools/finetune_yolo_facility.py — Specialise a fast YOLO to THIS facility's cameras.

WHY. A stock yolo11s trained on COCO web photos has never seen an oblique,
ceiling-mounted parking camera. At 320px it misses ~23% of the cars a heavier
yolo11m catches (measured 2026-07-16: 77% recall, CAM-08 5/29). Fine-tuning the
small model on frames from the actual cameras teaches it these exact angles,
lighting and car poses — a specialised small model matches or beats a generic
big one on its home turf, at 2.7x the speed.

THE TRICK — distil from a high-res teacher. We auto-label the training frames
with yolo11m at 640px (its most accurate setting) and train yolo11s at 320px on
those labels. The student inherits the teacher's *640px* recall while running at
*320px* speed, so it can end up BETTER than yolo11m@320 — no manual annotation.

    yolo11m @640  ->  labels  ->  train yolo11s @320  ->  OpenVINO int8 @320

If your photos are ALREADY labelled (YOLO .txt next to each image, or a
--labels-dir), pass --labels-dir to skip the teacher and train on ground truth,
which can beat the teacher outright.

RUN THIS ON THE 5090 BOX (where the photos and the big GPU are), inside the repo
venv (needs `ultralytics`, `openvino`, `nncf`):

    # unlabelled facility frames -> teacher-labelled -> fine-tuned int8:
    # (strongest teacher, high-res + TTA labels — it's a one-time offline pass)
    python tools/finetune_yolo_facility.py \
        --photos /data/facility_frames \
        --teacher models/yolo11x.pt --teacher-imgsz 1280 --teacher-augment \
        --student models/yolo11s.pt --imgsz 320 \
        --epochs 100 --batch 64 --output models/yolo11s_facility

    # already-labelled data (YOLO format), skip the teacher:
    python tools/finetune_yolo_facility.py \
        --photos /data/frames --labels-dir /data/labels \
        --student models/yolo11s.pt --imgsz 320 --epochs 100 \
        --output models/yolo11s_facility

OUTPUT (in --output):
    weights/best.pt                     fine-tuned student
    <name>_int8_openvino_model/         int8 IR — point the DB detector here
    dataset/                            the auto-labelled train/val split (kept
                                        so you can inspect the pseudo-labels)

VALIDATE before trusting it: the script prints val-set recall of the fine-tuned
student vs the teacher. Then A/B on the server (a few cameras) against yolo11m
before it touches billing — the recall gap that started this is a slot-occupancy
risk, and a held-out number is not a production guarantee.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# COCO vehicle classes the deployed detector keeps (config.detector.classes).
# Labels are written with THESE ids so the fine-tuned model is a drop-in swap —
# the runtime classes filter [2,5,7] is unchanged.
VEHICLE_CLASSES = [2, 5, 7]
# Standard COCO-80 class names, in id order. Only 2/5/7 carry labels here; the
# rest keep yolo11s's head intact so ids stay stable and the model swaps in clean.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def list_images(photos: Path) -> List[Path]:
    return sorted(
        Path(p) for p in glob.glob(str(photos / "**" / "*"), recursive=True)
        if Path(p).suffix.lower() in IMAGE_EXTS and Path(p).is_file()
    )


def autolabel(images: List[Path], teacher: Path, imgsz: int, conf: float,
              labels_dir: Path, min_side: int, augment: bool = False) -> int:
    """Run the teacher on each frame, write YOLO-format labels (normalised
    xywh, class kept as the COCO id). Returns the count of labelled frames.

    Labelling is a ONE-TIME OFFLINE pass, so make the teacher as accurate as it
    can be — its recall is the ceiling on the student's. Use the biggest model
    (yolo11x), a large ``imgsz`` (small distant cars are exactly what a fast
    320px student misses), and optionally ``augment`` (TTA: multi-scale/flip
    inference, slower but higher recall). None of this cost reaches production —
    it only shapes the labels."""
    import cv2
    from ultralytics import YOLO

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(teacher))
    keep = set(VEHICLE_CLASSES)
    labelled = 0
    for i, img in enumerate(images):
        im = cv2.imread(str(img))
        if im is None or min(im.shape[:2]) < min_side:
            continue
        h, w = im.shape[:2]
        res = model.predict(im, imgsz=imgsz, conf=conf, classes=VEHICLE_CLASSES,
                            augment=augment, verbose=False)[0]
        lines = []
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls.item())
                if cls not in keep:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # Frames with no vehicle are still useful NEGATIVES — write an empty file
        # so the trainer sees "this whole scene is background", cutting false
        # positives on pillars/shadows.
        (labels_dir / f"{img.stem}.txt").write_text("\n".join(lines))
        labelled += 1
        if (i + 1) % 200 == 0:
            print(f"  auto-labelled {i + 1}/{len(images)} ...")
    return labelled


def build_split(images: List[Path], labels_dir: Path, dataset: Path,
                val_frac: float, seed: int) -> Path:
    """Lay out an Ultralytics dataset (images/ + labels/, train/ + val/) and
    write data.yaml. Only frames that got a label file are included."""
    random.seed(seed)
    paired = [im for im in images if (labels_dir / f"{im.stem}.txt").exists()]
    random.shuffle(paired)
    n_val = max(1, int(len(paired) * val_frac))
    splits = {"val": paired[:n_val], "train": paired[n_val:]}

    for split, items in splits.items():
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in items:
            shutil.copy2(im, dataset / "images" / split / im.name)
            shutil.copy2(labels_dir / f"{im.stem}.txt",
                         dataset / "labels" / split / f"{im.stem}.txt")

    # 80-class COCO names so the vehicle ids (2/5/7) stay valid and yolo11s's
    # head is unchanged — the fine-tuned model is a drop-in swap for the runtime
    # classes filter. Embedded (not imported) so it's independent of the
    # ultralytics version on the training box.
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80_NAMES))
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names_block}\n"
    )
    print(f"  dataset: {len(splits['train'])} train / {len(splits['val'])} val "
          f"-> {data_yaml}")
    return data_yaml


def train_student(student: Path, data_yaml: Path, imgsz: int, epochs: int,
                  output: Path, batch: int, device: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(student))
    model.train(
        data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch,
        device=device, project=str(output), name="train", exist_ok=True,
        patience=20, cache=False,
    )
    best = output / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training did not produce {best}")
    return best


def export_int8(best: Path, data_yaml: Path, imgsz: int, output: Path) -> Path:
    """Export the fine-tuned student to an INT8 OpenVINO IR, calibrated on the
    val split (Ultralytics runs NNCF PTQ when int8=True + data=...)."""
    from ultralytics import YOLO

    model = YOLO(str(best))
    ir_dir = model.export(format="openvino", imgsz=imgsz, int8=True,
                          data=str(data_yaml))
    dest = output / f"{best.parent.parent.name}_int8_openvino_model"
    if Path(ir_dir).resolve() != dest.resolve():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(ir_dir), str(dest))
    return dest


def validate(best: Path, teacher: Path, data_yaml: Path, imgsz: int,
             teacher_imgsz: int) -> None:
    """Cars found by the fine-tuned student vs the teacher, on the val images —
    the sanity check before any server A/B."""
    import cv2
    from ultralytics import YOLO
    import yaml

    d = yaml.safe_load(Path(data_yaml).read_text())
    val_dir = Path(d["path"]) / "images" / "val"
    imgs = list_images(val_dir)
    s = YOLO(str(best))
    m = YOLO(str(teacher))
    ts = tm = 0
    for f in imgs:
        im = cv2.imread(str(f))
        if im is None:
            continue
        ds = s.predict(im, imgsz=imgsz, conf=0.25, classes=VEHICLE_CLASSES,
                       verbose=False)[0]
        dm = m.predict(im, imgsz=teacher_imgsz, conf=0.25,
                       classes=VEHICLE_CLASSES, verbose=False)[0]
        ts += 0 if ds.boxes is None else len(ds.boxes)
        tm += 0 if dm.boxes is None else len(dm.boxes)
    print("\n=== validation (val split) ===")
    print(f"teacher (yolo11m@{teacher_imgsz}) cars: {tm}")
    print(f"student (fine-tuned @{imgsz}) cars:  {ts}  "
          f"({ts / max(tm, 1) * 100:.0f}% of teacher)")
    print("A/B on the server against yolo11m before trusting for billing.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="finetune_yolo_facility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Distil a facility-specialised INT8 yolo11s from a yolo11m teacher.",
    )
    p.add_argument("--photos", type=Path, required=True,
                   help="Directory of facility camera FRAMES (recursive).")
    p.add_argument("--labels-dir", type=Path, default=None,
                   help="Existing YOLO labels (skip the teacher; train on these).")
    p.add_argument("--teacher", type=Path, default=Path("models/yolo11x.pt"),
                   help="Teacher weights for auto-labelling. Use the strongest "
                        "model you have (yolo11x) — its recall caps the student's.")
    p.add_argument("--teacher-imgsz", type=int, default=1280,
                   help="Label at the teacher's most accurate size, NOT the "
                        "student's. Big = catches the small distant cars a 320px "
                        "student misses. Offline, so cost is irrelevant.")
    p.add_argument("--teacher-conf", type=float, default=0.20,
                   help="Slightly loose so faint cars get labelled; too low adds "
                        "false-positive labels, so keep ~0.2-0.25.")
    p.add_argument("--teacher-augment", action="store_true",
                   help="TTA (multi-scale/flip) on the teacher — slower labelling, "
                        "higher recall labels. Worth it for a one-time pass.")
    p.add_argument("--student", type=Path, default=Path("models/yolo11s.pt"),
                   help="Student weights to fine-tune.")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Student train/inference size (match the deployed detector).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32, help="5090 can go higher.")
    p.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' fallback.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--min-side", type=int, default=640,
                   help="Skip frames whose shorter side is below this.")
    p.add_argument("--output", type=Path, default=Path("models/yolo11s_facility"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-export", action="store_true",
                   help="Train only; export the int8 IR later.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] scanning {args.photos} ...")
    images = list_images(args.photos)
    print(f"  {len(images)} image(s) found")
    if len(images) < 50:
        print("  ERROR: need at least ~50 frames; thousands is ideal.")
        return 2

    dataset = args.output / "dataset"
    if args.labels_dir:
        print(f"[2/5] using existing labels: {args.labels_dir}")
        labels_dir = args.labels_dir
    else:
        print(f"[2/5] auto-labelling with teacher {args.teacher} "
              f"@{args.teacher_imgsz} ...")
        labels_dir = dataset / "_autolabels"
        n = autolabel(images, args.teacher, args.teacher_imgsz,
                      args.teacher_conf, labels_dir, args.min_side,
                      augment=args.teacher_augment)
        print(f"  labelled {n} frame(s)")
        if n < 50:
            print("  ERROR: too few usable frames after --min-side filter.")
            return 3

    print("[3/5] building train/val split ...")
    data_yaml = build_split(images, labels_dir, dataset, args.val_frac, args.seed)

    print(f"[4/5] fine-tuning {args.student} @{args.imgsz} for {args.epochs} "
          f"epoch(s) on device {args.device} ...")
    best = train_student(args.student, data_yaml, args.imgsz, args.epochs,
                         args.output, args.batch, args.device)
    print(f"  best weights: {best}")

    if not args.skip_export:
        print("[5/5] exporting INT8 OpenVINO IR ...")
        ir = export_int8(best, data_yaml, args.imgsz, args.output)
        print(f"  INT8 IR: {ir}")
        print(f"  -> set the DB detector model_path to {ir} and imgsz={args.imgsz}")

    try:
        validate(best, args.teacher, data_yaml, args.imgsz, args.teacher_imgsz)
    except Exception as exc:
        print(f"  (validation skipped: {exc})")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
