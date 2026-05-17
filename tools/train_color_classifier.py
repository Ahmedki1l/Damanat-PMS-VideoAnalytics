"""
tools/train_color_classifier.py — Train the 11-class vehicle color head.

Pipeline:

    manifest.csv (from tools/prepare_color_dataset.py)
        ↓
    MobileNetV3-Small + 11-class head (torchvision)
        ↓
    best.pt  (PyTorch checkpoint, val-acc selected)
        ↓
    model.onnx  (torch.onnx.export, opset 13)
        ↓
    model.xml / model.bin   (OpenVINO IR via openvino.convert_model)
        ↓
    labels.json  +  README.md

CLI
~~~

    python tools/train_color_classifier.py \\
        --manifest tests/data/color_classifier/manifest.csv \\
        --out models/color_classifier_openvino/ \\
        --epochs 30 \\
        --batch-size 32 \\
        --img-size 224 \\
        --lr 1e-3 \\
        --patience 6

Fallback (no labelled data yet)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``manifest.csv`` does not exist or is empty, the trainer falls back to a
**synthetic per-class swatch generator** that produces ~50 procedurally-coloured
images per class. The resulting model is intentionally low-accuracy but
end-to-end: it lets Phase 2 wire the plugin in without blocking on D-2. The
generated README clearly documents the fallback so the model is not mistaken
for the production head.

Use ``--synthetic`` to force the fallback even when a manifest exists (handy
for smoke-testing the export pipeline).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# torch / torchvision are required for training. We import lazily and
# gracefully surface a friendly error so `--help` always works even when the
# heavy ML stack is missing.
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import DataLoader, Dataset  # type: ignore
    from torchvision import models, transforms  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _TORCH_AVAILABLE = False

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Class set — duplicated from src/classifiers/color_classifier.py so the
# trainer can run before that package is importable.
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Synthetic data — used when no labelled manifest is available so the
# training + export pipeline still runs end-to-end.
# --------------------------------------------------------------------------- #


SYNTHETIC_COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "black":  (25, 25, 25),
    "white":  (235, 235, 235),
    "grey":   (128, 128, 128),
    "silver": (190, 190, 195),
    "red":    (190, 30, 30),
    "blue":   (30, 60, 190),
    "green":  (40, 150, 60),
    "yellow": (230, 210, 40),
    "brown":  (110, 70, 40),
    "beige":  (215, 195, 165),
    # The "other" class is intentionally a noisy multi-colour swatch — the
    # plugin treats it as "do not refute" anyway.
    "other":  (140, 100, 160),
}


def _make_synthetic_swatch(
    label: str,
    size: Tuple[int, int],
    rng: random.Random,
) -> np.ndarray:
    """Build a 3-channel BGR swatch resembling a vehicle of the given color.

    The swatch is the base colour with: a horizontal lighter-gradient
    (windshield band), random Gaussian noise, and a small darker patch
    representing tyres / shadow. Plenty of intra-class variance so the
    network learns a non-trivial decision boundary.
    """
    h, w = size
    base_rgb = SYNTHETIC_COLOR_RGB[label]
    base_bgr = np.array(base_rgb[::-1], dtype=np.float32)
    img = np.tile(base_bgr.reshape(1, 1, 3), (h, w, 1))

    # Per-channel global brightness shift
    shift = rng.uniform(-20, 20)
    img += shift

    # Horizontal "windshield" band (lighter middle row)
    band_top = int(h * (0.25 + rng.uniform(-0.05, 0.05)))
    band_bot = int(h * (0.55 + rng.uniform(-0.05, 0.05)))
    img[band_top:band_bot, :, :] += rng.uniform(15, 40)

    # Wheels (darker corners)
    wheel_w = int(w * 0.18)
    wheel_h = int(h * 0.18)
    img[h - wheel_h :, : wheel_w, :] -= rng.uniform(30, 60)
    img[h - wheel_h :, w - wheel_w :, :] -= rng.uniform(30, 60)

    # "other" class — overlay a contrasting blob so it visibly differs from
    # the named classes.
    if label == "other":
        blob_x = int(w * rng.uniform(0.3, 0.7))
        blob_y = int(h * rng.uniform(0.3, 0.7))
        blob_r = int(min(h, w) * rng.uniform(0.1, 0.2))
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - blob_x) ** 2 + (yy - blob_y) ** 2 < blob_r ** 2
        contrast = np.array(
            [
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
            ],
            dtype=np.float32,
        )
        img[mask] = contrast

    # Gaussian noise — seed off the python rng so different samples get
    # different noise patterns (the python rng is itself seeded for
    # reproducibility from the outer caller).
    seed = rng.randint(0, 2**31 - 1)
    img += np.random.RandomState(seed).normal(0, 6, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def build_synthetic_manifest(
    out_root: Path,
    per_class: int = 60,
    seed: int = 1337,
) -> Path:
    """
    Generate a small per-class swatch dataset under
    ``out_root / synthetic / <label>/`` and write a paired manifest.csv.

    Returns the manifest path. Splits 70/15/15 deterministically by sample
    index within each class so each split contains every class.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV is required to write synthetic swatch images. "
            "Install `opencv-python-headless`."
        )

    rng = random.Random(seed)
    synth_root = out_root / "synthetic"
    if synth_root.exists():
        shutil.rmtree(synth_root, ignore_errors=True)
    synth_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_idx, label in enumerate(COLOR_CLASS_LABELS):
        cls_dir = synth_root / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # Vary the size slightly for realism.
            h = rng.randint(80, 200)
            w = int(h * rng.uniform(1.2, 2.0))
            img = _make_synthetic_swatch(label, (h, w), rng)
            fname = f"{label}_{i:04d}.png"
            fpath = cls_dir / fname
            cv2.imwrite(str(fpath), img)
            # 70/15/15 split by within-class index
            if i % 20 < 14:
                split = "train"
            elif i % 20 < 17:
                split = "val"
            else:
                split = "test"
            rows.append(
                {
                    "crop_path": str(fpath).replace("\\", "/"),
                    "split": split,
                    "confirmed_color": label,
                    "source": "synthetic",
                }
            )

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["crop_path", "split", "confirmed_color", "source"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(
        "Synthetic manifest written: %d rows across %d classes -> %s",
        len(rows),
        len(COLOR_CLASS_LABELS),
        manifest_path,
    )
    return manifest_path


# --------------------------------------------------------------------------- #
# Dataset / DataLoader
# --------------------------------------------------------------------------- #


def _load_manifest(path: Path) -> Dict[str, List[dict]]:
    """Return {split_name -> list of rows}. Missing splits map to empty list."""
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    if not path.exists():
        return splits
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split = (row.get("split") or "train").lower()
            if split not in splits:
                continue
            label = row.get("confirmed_color") or ""
            if not label:
                continue
            if label not in COLOR_CLASS_LABELS:
                logger.warning(
                    "Skipping row with unknown color label %r (path=%s)",
                    label,
                    row.get("crop_path"),
                )
                continue
            splits[split].append(row)
    return splits


if _TORCH_AVAILABLE:
    class _ColorDataset(Dataset):
        """torch Dataset reading rows from the manifest."""

        def __init__(
            self,
            rows: List[dict],
            img_size: int,
            train: bool,
            *,
            heavy_aug: bool = True,
        ):
            self._rows = rows
            self._img_size = img_size
            self._train = train

            base_tx = [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
            ]
            if train:
                aug = [transforms.RandomHorizontalFlip(p=0.5)]
                if heavy_aug:
                    # Hue jitter MUST stay small — we are predicting color.
                    aug.append(
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.15,
                            hue=0.02,
                        )
                    )
                    aug.append(
                        transforms.RandomResizedCrop(
                            img_size, scale=(0.80, 1.0), ratio=(0.95, 1.05)
                        )
                    )
                else:
                    # Light augmentation only — keeps colors pristine.
                    aug.append(
                        transforms.ColorJitter(
                            brightness=0.10, contrast=0.10
                        )
                    )
                base_tx.extend(aug)
            base_tx.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            self._transform = transforms.Compose(base_tx)

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, idx: int):
            row = self._rows[idx]
            path = row["crop_path"]
            label = row["confirmed_color"]
            class_idx = COLOR_CLASS_LABELS.index(label)

            img = cv2.imread(path)
            if img is None:
                # Fall back to a neutral grey patch — the loss can still flow.
                img = np.full((self._img_size, self._img_size, 3), 114, dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            tensor = self._transform(img)
            return tensor, torch.tensor(class_idx, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def _build_model(num_classes: int = len(COLOR_CLASS_LABELS), pretrained: bool = True):
    """MobileNetV3-Small with a fresh ``num_classes`` head.

    Args:
        num_classes: output dimension of the new head.
        pretrained: when True, attempts to load torchvision's ImageNet weights
            so BatchNorm running stats start sensibly (matters when training
            on small datasets — without it the model often collapses on the
            held-out split). Falls back to random init if download fails
            (e.g. air-gapped CI box).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch / torchvision are required for training.")
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    try:
        backbone = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        logger.warning(
            "Pretrained MobileNetV3 weights unavailable (%r) — falling back "
            "to random init.",
            exc,
        )
        backbone = models.mobilenet_v3_small(weights=None)
    # MobileNetV3 final classifier is Sequential(Linear, Hardswish, Dropout, Linear)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(in_features, num_classes)
    return backbone


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


@dataclass
class TrainingReport:
    best_val_acc: float = 0.0
    train_history: List[Dict[str, float]] = field(default_factory=list)
    test_acc: float = 0.0
    used_synthetic: bool = False
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    epochs_run: int = 0


def _accuracy(logits, labels) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_loop(
    splits: Dict[str, List[dict]],
    out_root: Path,
    *,
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    num_workers: int = 0,
    device: str = "cpu",
    used_synthetic: bool = False,
    pretrained: bool = True,
) -> TrainingReport:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch / torchvision not installed — see requirements.txt."
        )

    out_root.mkdir(parents=True, exist_ok=True)
    report = TrainingReport(used_synthetic=used_synthetic)

    train_rows = splits.get("train", [])
    val_rows = splits.get("val", [])
    test_rows = splits.get("test", [])
    report.num_train = len(train_rows)
    report.num_val = len(val_rows)
    report.num_test = len(test_rows)
    if not train_rows:
        raise RuntimeError(
            "No training rows in manifest. Either fill in confirmed_color in "
            "the labelling worklist, or pass --synthetic for a smoke-test run."
        )

    # When training on synthetic data, fall back to lighter augmentation —
    # heavy ColorJitter destroys the only signal the synthetic swatches carry.
    train_ds = _ColorDataset(
        train_rows, img_size, train=True, heavy_aug=not used_synthetic
    )
    val_ds = _ColorDataset(val_rows, img_size, train=False) if val_rows else None
    test_ds = _ColorDataset(test_rows, img_size, train=False) if test_rows else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        if test_ds is not None
        else None
    )

    model = _build_model(pretrained=pretrained).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss_sum += float(loss.item())
            train_acc_sum += _accuracy(logits, labels)
            n_batches += 1
        scheduler.step()

        train_loss = train_loss_sum / max(1, n_batches)
        train_acc = train_acc_sum / max(1, n_batches)

        val_acc = train_acc  # Default in absence of val set
        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            v_loss_sum = 0.0
            v_acc_sum = 0.0
            v_batches = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    v_loss_sum += float(criterion(logits, labels).item())
                    v_acc_sum += _accuracy(logits, labels)
                    v_batches += 1
            val_loss = v_loss_sum / max(1, v_batches)
            val_acc = v_acc_sum / max(1, v_batches)

        dt = time.time() - t0
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "seconds": round(dt, 2),
        }
        report.train_history.append(record)
        report.epochs_run = epoch
        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f "
            "val_loss=%.4f val_acc=%.4f (%.2fs)",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            dt,
        )

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping: no val improvement for %d epochs.", patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out_root / "best.pt")
        logger.info("Saved best checkpoint to %s", out_root / "best.pt")
    report.best_val_acc = best_acc

    # Held-out test evaluation
    if test_loader is not None:
        model.eval()
        t_acc_sum = 0.0
        t_batches = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                t_acc_sum += _accuracy(logits, labels)
                t_batches += 1
        report.test_acc = t_acc_sum / max(1, t_batches)
        logger.info("Test accuracy: %.4f", report.test_acc)

    return report


# --------------------------------------------------------------------------- #
# Export ONNX → OpenVINO IR
# --------------------------------------------------------------------------- #


def export_openvino(
    model,
    out_root: Path,
    *,
    img_size: int = 224,
    labels: List[str] = COLOR_CLASS_LABELS,
) -> Tuple[Path, Path]:
    """Export the trained model to ONNX then convert to OpenVINO IR.

    Returns the (xml, bin) paths.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to export ONNX.")
    out_root.mkdir(parents=True, exist_ok=True)
    onnx_path = out_root / "model.onnx"
    xml_path = out_root / "model.xml"
    bin_path = out_root / "model.bin"
    labels_path = out_root / "labels.json"

    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    # Use the legacy TorchScript exporter explicitly. The new dynamo-based
    # exporter is the default in torch 2.9+ but it pulls in onnxscript and
    # may emit emoji-laced status lines that crash on Windows cp1252 stdout.
    # The legacy path is sufficient for a vanilla classification head.
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch (<2.9) does not accept ``dynamo=`` kwarg — drop it.
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    logger.info("Wrote ONNX -> %s", onnx_path)

    # OpenVINO conversion. Newer API exposes ``openvino.convert_model`` /
    # ``openvino.save_model``; on older releases (<= 2024.0) we fall back
    # to ``openvino.tools.mo.convert_model``.
    saved = False
    try:
        import openvino as ov  # type: ignore

        if hasattr(ov, "convert_model"):
            ov_model = ov.convert_model(str(onnx_path))
            ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
            saved = True
    except Exception as exc:
        logger.warning(
            "openvino.convert_model path failed (%r) — trying tools.mo fallback.",
            exc,
        )

    if not saved:
        try:
            from openvino.tools import mo  # type: ignore
            ov_model = mo.convert_model(str(onnx_path))
            from openvino import serialize  # type: ignore
            serialize(ov_model, str(xml_path))
            saved = True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert ONNX to OpenVINO IR: {exc!r}"
            ) from exc

    logger.info("Wrote OpenVINO IR -> %s + %s", xml_path, bin_path)

    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "labels": list(labels),
                "input_size": [img_size, img_size],
                "preprocessing": {
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                    "color_order": "RGB",
                    "letterbox": True,
                },
            },
            fh,
            indent=2,
        )
    logger.info("Wrote labels.json -> %s", labels_path)
    return xml_path, bin_path


def write_artifact_readme(
    out_root: Path,
    report: TrainingReport,
    args,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    readme = out_root / "README.md"
    fallback_note = ""
    if report.used_synthetic:
        fallback_note = (
            "\n\n## Fallback synthetic data\n\n"
            "This artifact was trained on the **synthetic swatch fallback** "
            "(see `tools/train_color_classifier.py::build_synthetic_manifest`)\n"
            "because no human-labelled manifest was available at training time.\n"
            "Accuracy is intentionally low and the model is suitable only for "
            "pipeline smoke-testing, not production matching. Re-train on the "
            "real D-2 deliverable before relying on this output.\n"
        )
    content = (
        "# Color classifier (WS-B) — OpenVINO artifacts\n\n"
        f"* Backbone: MobileNetV3-Small\n"
        f"* Classes ({len(COLOR_CLASS_LABELS)}): "
        f"{', '.join(COLOR_CLASS_LABELS)}\n"
        f"* Input: {args.img_size}x{args.img_size} RGB, ImageNet-normalised, "
        "letterbox-resized (BGR -> RGB inside the plugin)\n"
        f"* Trained for {report.epochs_run} epochs (early-stopped on val acc)\n"
        f"* Best val accuracy: {report.best_val_acc:.4f}\n"
        f"* Held-out test accuracy: {report.test_acc:.4f}\n"
        f"* Dataset rows — train: {report.num_train}, val: {report.num_val}, "
        f"test: {report.num_test}\n\n"
        "## Files\n"
        "* `model.xml` / `model.bin` — OpenVINO IR (consumed by "
        "`src.classifiers.color_classifier.OpenVINOColorClassifier`).\n"
        "* `model.onnx` — intermediate ONNX export (kept for reproducibility).\n"
        "* `best.pt` — torch state dict at the best-val-acc epoch.\n"
        "* `labels.json` — `{labels, input_size, preprocessing}` consumed by "
        "the plugin at load time.\n"
        + fallback_note
    )
    readme.write_text(content, encoding="utf-8")
    logger.info("Wrote artifact README -> %s", readme)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train the 11-class vehicle-color MobileNetV3-Small head and "
            "export it to OpenVINO IR."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/data/color_classifier/manifest.csv"),
        help="Path to the train/val/test manifest CSV.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models/color_classifier_openvino"),
        help="Output directory for the trained artifacts.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cpu', or 'cuda'.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Skip the real manifest and train on a procedurally-generated "
            "swatch dataset. Use this to smoke-test the export pipeline."
        ),
    )
    p.add_argument(
        "--synthetic-per-class",
        type=int,
        default=60,
        help="When --synthetic is used: number of generated images per class.",
    )
    p.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Skip training; load best.pt from --out and re-export ONNX + IR. "
            "Useful when iterating on the export step only."
        ),
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Train the backbone from random init instead of ImageNet weights. "
            "Pretrained start is strongly preferred — it stabilises BatchNorm "
            "running stats with small datasets."
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

    if not _TORCH_AVAILABLE:
        logger.error(
            "PyTorch / torchvision are not installed. Install them via "
            "`pip install -r requirements.txt` then re-run."
        )
        return 2
    if not _CV2_AVAILABLE:
        logger.error("OpenCV is required (`pip install opencv-python-headless`).")
        return 2

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    used_synthetic = False
    manifest_path = args.manifest

    # Resolve dataset source: real manifest, else synthetic fallback.
    if args.synthetic or not manifest_path.exists():
        logger.warning(
            "Using synthetic swatch dataset (manifest='%s', --synthetic=%s). "
            "Resulting model is for pipeline smoke-test only.",
            manifest_path,
            args.synthetic,
        )
        synth_out_root = out_root / "synthetic_dataset"
        synth_out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = build_synthetic_manifest(
            synth_out_root, per_class=args.synthetic_per_class
        )
        used_synthetic = True

    splits = _load_manifest(manifest_path)
    if not any(splits[k] for k in ("train", "val", "test")):
        logger.error(
            "Manifest at %s has no labelled rows. Fill in confirmed_color in "
            "the worklist or re-run with --synthetic.",
            manifest_path,
        )
        return 1

    pretrained = not args.no_pretrained
    if args.export_only:
        logger.info("Export-only mode — loading best.pt and re-exporting.")
        model = _build_model(pretrained=False)
        state_path = out_root / "best.pt"
        if not state_path.exists():
            logger.error("No checkpoint at %s — cannot export.", state_path)
            return 1
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        report = TrainingReport(used_synthetic=used_synthetic)
    else:
        report = train_loop(
            splits,
            out_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
            used_synthetic=used_synthetic,
            pretrained=pretrained,
        )
        model = _build_model(pretrained=False).to(device)
        state_path = out_root / "best.pt"
        if state_path.exists():
            model.load_state_dict(torch.load(state_path, map_location="cpu"))
            model.to("cpu")

    export_openvino(model.cpu(), out_root, img_size=args.img_size)
    write_artifact_readme(out_root, report, args)

    metrics_path = out_root / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_val_acc": report.best_val_acc,
                "test_acc": report.test_acc,
                "used_synthetic": report.used_synthetic,
                "num_train": report.num_train,
                "num_val": report.num_val,
                "num_test": report.num_test,
                "epochs_run": report.epochs_run,
                "history": report.train_history,
            },
            fh,
            indent=2,
        )
    logger.info("Wrote training metrics -> %s", metrics_path)
    logger.info(
        "Done. best_val_acc=%.4f test_acc=%.4f synthetic=%s",
        report.best_val_acc,
        report.test_acc,
        report.used_synthetic,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
