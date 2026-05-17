"""
tools/train_type_classifier.py - Train the 6-class body-type classifier.

Pipeline:

  1. Read confirmed labels from
     ``tests/data/type_classifier/{train,val,test}.csv``
     (produced by ``tools/prepare_type_dataset.py``).
  2. If real labelled data is too sparse, fall back to a tiny synthetic
     dataset of geometric primitives. The fallback exists so Phase 2 has a
     working IR even when the data-work track (D-3) has not finished
     labelling. The fallback model is intentionally degraded - its
     accuracy bar is >=0.65 on the synthetic test split, not >=0.85.
  3. Fine-tune torchvision's MobileNetV3-Small (ImageNet-pretrained
     backbone) with AdamW + cosine LR + light augmentation. Same training
     stack as WS-B's colour classifier.
  4. Save the best torch checkpoint, export ONNX, convert to OpenVINO IR
     at ``models/type_classifier_openvino/model.{xml,bin}``.
  5. Write ``labels.json`` and a README explaining the data dependency.

Usage:
    python tools/train_type_classifier.py --help
    python tools/train_type_classifier.py             # auto: real or fallback
    python tools/train_type_classifier.py --epochs 25 --batch-size 64
    python tools/train_type_classifier.py --synthetic # force synthetic
    python tools/train_type_classifier.py --no-export # skip OpenVINO conversion

Heavy deps (torch / torchvision / openvino) are imported lazily - running
``--help`` works in CI sanity environments without them.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("train_type_classifier")

# Canonical class list — MUST match src/classifiers/type_classifier.py.
CLASSES: Tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "van",
    "bus",
)
CLASS_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

INPUT_SIZE: Tuple[int, int] = (224, 224)  # (H, W) — MobileNetV3 standard
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_DATA_DIR = Path("tests/data/type_classifier")
DEFAULT_OUTPUT_DIR = Path("models/type_classifier_openvino")

# Minimum confirmed labels per class before we treat the dataset as
# "real". Below this we fall back to the synthetic generator.
MIN_LABELS_PER_CLASS = 5


@dataclass
class LabeledCrop:
    """A single confirmed-labelled crop usable for training."""

    path: Path
    label_index: int
    split: str  # "train" | "val" | "test"


# --------------------------------------------------------------------------- #
# Data loading from CSV
# --------------------------------------------------------------------------- #


def load_labeled_crops(data_dir: Path) -> List[LabeledCrop]:
    """Read the three CSVs and return only rows with a confirmed_type."""
    crops: List[LabeledCrop] = []
    for split in ("train", "val", "test"):
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            LOG.debug("CSV not found: %s", csv_path)
            continue
        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = (row.get("confirmed_type") or "").strip().lower()
                if not label or label == "unknown":
                    continue
                if label not in CLASS_INDEX:
                    LOG.warning(
                        "Skipping row with unknown label %r in %s",
                        label,
                        csv_path,
                    )
                    continue
                path = Path(row["crop_path"])
                if not path.exists():
                    LOG.debug("Skipping missing crop: %s", path)
                    continue
                crops.append(
                    LabeledCrop(
                        path=path,
                        label_index=CLASS_INDEX[label],
                        split=split,
                    )
                )
    return crops


def labels_meet_threshold(crops: List[LabeledCrop]) -> bool:
    """True iff every class has ≥ MIN_LABELS_PER_CLASS examples."""
    counts: Dict[int, int] = {i: 0 for i in range(len(CLASSES))}
    for c in crops:
        counts[c.label_index] = counts.get(c.label_index, 0) + 1
    for cls_idx, n in counts.items():
        if n < MIN_LABELS_PER_CLASS:
            LOG.warning(
                "Class %s has only %d labels (need ≥ %d)",
                CLASSES[cls_idx],
                n,
                MIN_LABELS_PER_CLASS,
            )
            return False
    return True


# --------------------------------------------------------------------------- #
# Synthetic fallback dataset
# --------------------------------------------------------------------------- #


def _make_synthetic_sample(label_index: int, rng: random.Random):
    """Render a deterministic geometric primitive per class.

    Each class has a unique colour + shape so a small CNN can still hit
    ~0.65+ accuracy. Imported lazily because numpy is the only dep.
    """
    import numpy as np

    h, w = INPUT_SIZE
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Per-class palette + shape — chosen to be geometrically distinct so
    # the synthetic accuracy floor (0.65) is reachable on CPU in a few
    # epochs.
    palette = [
        (200, 60, 60),    # sedan     - red rect, low aspect
        (60, 60, 200),    # suv       - blue, square
        (60, 200, 60),    # hatchback - green, small square
        (200, 160, 40),   # pickup    - orange, very wide
        (160, 60, 200),   # van       - purple, wide tall
        (40, 200, 200),   # bus       - cyan, very large
    ]
    color = palette[label_index]

    # Background colour: random low-saturation so the model can't just learn
    # "any pixel of color X = class X".
    bg = (
        rng.randint(0, 60),
        rng.randint(0, 60),
        rng.randint(0, 60),
    )
    img[:, :, 0] = bg[2]  # OpenCV/BGR style
    img[:, :, 1] = bg[1]
    img[:, :, 2] = bg[0]

    # Pick shape dims per class
    shape_specs = [
        (0.45, 0.65),  # sedan      width%, height% of image
        (0.55, 0.55),  # suv
        (0.30, 0.30),  # hatchback
        (0.85, 0.40),  # pickup
        (0.70, 0.60),  # van
        (0.95, 0.85),  # bus
    ]
    pw, ph = shape_specs[label_index]
    rect_w = int(w * pw * rng.uniform(0.85, 1.0))
    rect_h = int(h * ph * rng.uniform(0.85, 1.0))
    x0 = (w - rect_w) // 2 + rng.randint(-10, 10)
    y0 = (h - rect_h) // 2 + rng.randint(-10, 10)
    x0 = max(0, min(w - rect_w, x0))
    y0 = max(0, min(h - rect_h, y0))
    img[y0 : y0 + rect_h, x0 : x0 + rect_w, 0] = color[2]
    img[y0 : y0 + rect_h, x0 : x0 + rect_w, 1] = color[1]
    img[y0 : y0 + rect_h, x0 : x0 + rect_w, 2] = color[0]

    # Add a little class-correlated noise so it isn't trivially separable.
    noise = (
        np.random.default_rng(rng.randint(0, 2**32 - 1))
        .integers(-20, 20, size=img.shape, dtype=np.int16)
    )
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def synthetic_dataset(
    samples_per_class: int = 80,
    seed: int = 1337,
) -> Dict[str, List]:
    """Return ``{"train": [(image, label), ...], "val": [...], "test": [...]}``.

    Splits are 70/15/15 within each class so every class is represented
    in every split.
    """
    import numpy as np  # noqa: F401  - asserts numpy is available

    rng = random.Random(seed)
    bucket: Dict[str, List] = {"train": [], "val": [], "test": []}
    for label_index in range(len(CLASSES)):
        for i in range(samples_per_class):
            img = _make_synthetic_sample(label_index, rng)
            r = rng.random()
            if r < 0.70:
                split = "train"
            elif r < 0.85:
                split = "val"
            else:
                split = "test"
            bucket[split].append((img, label_index))
    # Shuffle for good measure
    for k in bucket:
        rng.shuffle(bucket[k])
    return bucket


# --------------------------------------------------------------------------- #
# Torch training
# --------------------------------------------------------------------------- #


def _build_model(num_classes: int):
    """Instantiate MobileNetV3-Small with a fresh classifier head."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import (  # type: ignore
        MobileNet_V3_Small_Weights,
        mobilenet_v3_small,
    )

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    try:
        model = mobilenet_v3_small(weights=weights)
    except Exception:
        # Offline / no-network env — fall back to random init.
        LOG.warning(
            "Pretrained weights unavailable; falling back to random init."
        )
        model = mobilenet_v3_small(weights=None)

    # Replace the head: MobileNetV3-Small has classifier[3] = Linear(1024, 1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _preprocess_for_train(img_bgr, augment: bool):
    """Resize + augment + ImageNet-normalise. Returns a torch tensor.

    ``img_bgr`` is an HxWx3 uint8 numpy array (OpenCV style). Augmentations:
    random horizontal flip, brightness jitter, gaussian noise — light
    enough to converge fast on small datasets.
    """
    import numpy as np
    import torch  # type: ignore

    h, w = INPUT_SIZE
    # Resize via numpy if cv2 is missing — happens in CI smoke
    try:
        import cv2  # type: ignore

        img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        sh, sw = img_bgr.shape[:2]
        yi = np.linspace(0, sh - 1, h).astype(np.int64)
        xi = np.linspace(0, sw - 1, w).astype(np.int64)
        img = img_bgr[np.ix_(yi, xi)][..., ::-1]

    if augment:
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
        # Brightness
        if random.random() < 0.5:
            scale = 0.7 + random.random() * 0.6
            img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(
                np.uint8
            )
        # Gaussian noise
        if random.random() < 0.3:
            n = np.random.normal(0, 6, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)

    f = img.astype(np.float32) / 255.0
    f = (f - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    f = np.transpose(f, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(f))


def _load_bgr(path: Path):
    """Load a JPG/PNG into an HxWx3 BGR uint8 numpy array. Prefers cv2."""
    import numpy as np

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    # Fall back to Pillow + RGB→BGR conversion.
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        return rgb[..., ::-1].copy()  # RGB -> BGR
    except Exception as exc:
        LOG.warning("Failed to read %s: %r", path, exc)
        return None


def _iter_batches(samples, batch_size: int, augment: bool):
    """Yield ``(images_tensor, labels_tensor)`` batches.

    ``samples`` is a list of ``(image_bgr, label_index)`` tuples OR a list
    of ``LabeledCrop`` objects — the latter are loaded lazily so we don't
    need to keep the full dataset in memory.
    """
    import torch  # type: ignore

    rng_seed = 0
    random.shuffle(samples)
    batch_imgs = []
    batch_labels = []
    for s in samples:
        if isinstance(s, tuple):
            img, label = s
        else:  # LabeledCrop
            img = _load_bgr(s.path)
            if img is None:
                continue
            label = s.label_index
        try:
            tensor = _preprocess_for_train(img, augment)
        except Exception as exc:
            LOG.debug("preprocess skip: %r", exc)
            continue
        batch_imgs.append(tensor)
        batch_labels.append(label)
        if len(batch_imgs) >= batch_size:
            yield (
                torch.stack(batch_imgs, dim=0),
                torch.tensor(batch_labels, dtype=torch.long),
            )
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        yield (
            torch.stack(batch_imgs, dim=0),
            torch.tensor(batch_labels, dtype=torch.long),
        )


def train_model(
    train_samples,
    val_samples,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_path: Path,
) -> Tuple[float, float]:
    """Run AdamW + cosine LR fine-tune. Returns ``(best_val_acc, train_secs)``."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Training on %s", device)

    model = _build_model(len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total = correct = 0
        train_loss = 0.0
        n_batches = 0
        for imgs, labels in _iter_batches(
            list(train_samples), batch_size, augment=True
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
        scheduler.step()

        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_samples, batch_size=batch_size, device=device)
        LOG.info(
            "Epoch %d/%d: loss=%.4f train_acc=%.3f val_acc=%.3f",
            epoch + 1,
            epochs,
            train_loss / max(1, n_batches),
            train_acc,
            val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            LOG.info("  ↳ new best, saved %s", output_path)

    return best_val_acc, time.time() - start


def evaluate(model, samples, *, batch_size: int, device) -> float:
    """Top-1 accuracy on the provided samples."""
    import torch  # type: ignore

    model.eval()
    total = correct = 0
    with torch.no_grad():
        for imgs, labels in _iter_batches(
            list(samples), batch_size, augment=False
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    if total == 0:
        return 0.0
    return correct / total


# --------------------------------------------------------------------------- #
# Export ONNX -> OpenVINO IR
# --------------------------------------------------------------------------- #


def export_to_openvino(
    checkpoint: Path,
    output_dir: Path,
    *,
    no_export: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Export the trained checkpoint to ONNX and convert to OpenVINO IR.

    Returns ``(onnx_path, ir_xml_path)`` — either may be None if the
    corresponding step is skipped or fails. The caller should treat None
    as "not produced this run".
    """
    if no_export:
        LOG.info("Skipping export (--no-export).")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    xml_path = output_dir / "model.xml"

    try:
        import torch  # type: ignore
    except Exception as exc:
        LOG.warning("Cannot export ONNX without torch: %r", exc)
        return None, None

    model = _build_model(len(CLASSES))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=13,
        )
        LOG.info("Exported ONNX -> %s", onnx_path)
    except Exception as exc:
        LOG.warning("ONNX export failed: %r", exc)
        return None, None

    # ONNX -> OpenVINO IR
    try:
        import openvino as ov  # type: ignore

        # OpenVINO 2024+ supports ov.convert_model directly. Older versions
        # require the ``mo`` CLI — we prefer the API.
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(xml_path))
        LOG.info("Converted ONNX -> OpenVINO IR at %s", xml_path)
        return onnx_path, xml_path
    except Exception as exc:
        LOG.warning(
            "OpenVINO conversion failed: %r — leaving ONNX in place.", exc
        )
        return onnx_path, None


# --------------------------------------------------------------------------- #
# README + labels.json artifacts
# --------------------------------------------------------------------------- #


def write_artifacts(
    output_dir: Path,
    classes: Tuple[str, ...],
    metrics: Dict[str, float],
    used_fallback: bool,
) -> None:
    """Write ``labels.json`` and ``README.md`` next to the IR."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels.json"
    payload = {
        "classes": list(classes),
        "input_size": list(INPUT_SIZE),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "metrics": metrics,
        "fallback": used_fallback,
    }
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    LOG.info("Wrote %s", labels_path)

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _readme_body(classes, metrics, used_fallback),
        encoding="utf-8",
    )
    LOG.info("Wrote %s", readme_path)


def _readme_body(
    classes: Tuple[str, ...],
    metrics: Dict[str, float],
    used_fallback: bool,
) -> str:
    """Generate the README text. Plain Markdown, no emojis."""
    fallback_note = (
        "**This artifact was trained on the synthetic fallback dataset.** "
        "Accuracy is intentionally degraded; replace it once the human-owned "
        "data track D-3 delivers ≥500 labelled crops per class."
        if used_fallback
        else "This artifact was trained on real labelled crops produced by "
        "`tools/prepare_type_dataset.py` (data track D-3)."
    )
    metrics_lines = "\n".join(
        f"- {k}: {v:.4f}" for k, v in sorted(metrics.items())
    )
    classes_listing = "\n".join(
        f"  {i}. {name}" for i, name in enumerate(classes)
    )
    return (
        "# Type Classifier (OpenVINO IR)\n\n"
        "MobileNetV3-Small body-type classifier — 6 classes.\n\n"
        "## Classes (in IR output order)\n\n"
        f"{classes_listing}\n\n"
        "## Data dependency\n\n"
        f"{fallback_note}\n\n"
        "## Training metrics\n\n"
        f"{metrics_lines}\n\n"
        "## Files\n\n"
        "- `model.xml` / `model.bin` — OpenVINO IR (FP32).\n"
        "- `model.onnx` — intermediate ONNX export.\n"
        "- `labels.json` — class index → label map plus preprocessing constants.\n"
        "- `README.md` — this file.\n\n"
        "## Loading\n\n"
        "```python\n"
        "from src.classifiers.type_classifier import OpenVINOTypeClassifier\n"
        "clf = OpenVINOTypeClassifier('models/type_classifier_openvino/model.xml')\n"
        "label, conf = clf.predict(crop_bgr)\n"
        "```\n\n"
        "Retrain via `python tools/train_type_classifier.py --help`.\n"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing train.csv / val.csv / test.csv produced "
        "by tools/prepare_type_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory the OpenVINO IR + labels.json + README.md are "
        "written to.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force the synthetic-data fallback even if real labels exist. "
        "Useful for smoke-testing the pipeline.",
    )
    parser.add_argument(
        "--synthetic-samples-per-class",
        type=int,
        default=80,
        help="How many synthetic crops to render per class when the "
        "fallback path is used.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip ONNX -> OpenVINO conversion (only saves the torch checkpoint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
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

    random.seed(args.seed)
    try:
        import numpy as np  # noqa: F401

        np.random.seed(args.seed)
    except Exception:
        pass

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 1. Decide real vs synthetic
    real_crops = load_labeled_crops(data_dir) if data_dir.exists() else []
    use_fallback = args.synthetic or not labels_meet_threshold(real_crops)
    if use_fallback:
        LOG.info(
            "Using synthetic fallback dataset (%d real labels found)",
            len(real_crops),
        )
        synth = synthetic_dataset(
            samples_per_class=args.synthetic_samples_per_class,
            seed=args.seed,
        )
        train_samples = synth["train"]
        val_samples = synth["val"]
        test_samples = synth["test"]
    else:
        LOG.info("Using %d real labelled crops", len(real_crops))
        train_samples = [c for c in real_crops if c.split == "train"]
        val_samples = [c for c in real_crops if c.split == "val"]
        test_samples = [c for c in real_crops if c.split == "test"]

    # 2. Heavy deps available?
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:
        LOG.error(
            "Cannot train without torch — install requirements.txt. (%r)",
            exc,
        )
        return 2

    # 3. Train
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best.pt"
    best_val_acc, secs = train_model(
        train_samples,
        val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_path=ckpt_path,
    )
    LOG.info(
        "Training finished in %.1fs. best_val_acc=%.4f (synthetic=%s)",
        secs,
        best_val_acc,
        use_fallback,
    )

    # 4. Final test accuracy
    import torch  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(len(CLASSES)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = evaluate(
        model, test_samples, batch_size=args.batch_size, device=device
    )
    LOG.info("test_acc=%.4f", test_acc)

    # 5. Export
    onnx_path, xml_path = export_to_openvino(
        ckpt_path,
        output_dir,
        no_export=args.no_export,
    )

    # 6. labels.json + README.md
    write_artifacts(
        output_dir,
        CLASSES,
        metrics={
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "train_seconds": float(secs),
        },
        used_fallback=use_fallback,
    )

    LOG.info("Done. IR -> %s", xml_path or "<not produced>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
