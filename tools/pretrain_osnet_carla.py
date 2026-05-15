"""
tools/pretrain_osnet_carla.py — Pretrain OSNet on the VeRi-CARLA synthetic dataset.

Stage-1 of the two-stage facility training:
   ImageNet (torchreid default) -> CARLA (this script) -> Facility top-20 (finetune_osnet_top20.py --init-from ...)

CARLA gives the backbone hundreds of distinct vehicle identities seen across
many camera angles, training the embedding to be discriminative on
hard-negative pairs (same body type, same colour, different vehicle) before
it ever sees facility data.

Dataset source
--------------
sekilab/VehicleReIdentificationDataset (Apache-2.0). ~55 k images, 4 classes
(car/truck/motorcycle/bicycle), 85 camera angles per vehicle, generated
in the CARLA driving simulator.

Layout autodetection
--------------------
The script accepts either of two on-disk layouts:

  1. **folder-per-identity**:  <root>/<vehicle_id>/<image>.jpg
  2. **filename-encoded id**:  <root>/<...>/<vehicle_id>_<camera>_<seq>.jpg

When a `--id-regex` is supplied the regex's first capture group is the
identity label; otherwise the parent directory name is used.

Usage
-----
    python tools/pretrain_osnet_carla.py --data-root data/external/veri_carla_unpacked
    python tools/pretrain_osnet_carla.py --data-root data/external/veri_carla_unpacked \
        --id-regex '^(\d+)_'    # for VeRi-style filename labels
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Repo root on sys.path so ``src.*`` imports work when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse all the training machinery from the facility finetune.
from tools.finetune_osnet_top20 import (  # noqa: E402
    NORM_MEAN, NORM_STD,
    _CropDataset, _PKSampler, _build_backbone, _Finetuner,
    _batch_hard_triplet_loss, _label_smoothed_ce,
    _make_transforms,
)

logger = logging.getLogger("pretrain_osnet_carla")


DEFAULT_DATA_ROOT = Path("data/external/veri_carla_unpacked")
DEFAULT_OUTPUT_DIR = Path("models")
DEFAULT_INPUT_HW = (256, 128)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pretrain_osnet_carla",
        description="Pretrain OSNet on the VeRi-CARLA synthetic dataset (Apache-2.0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--id-regex", type=str, default=None,
                   help="Optional regex with one capture group extracting the vehicle "
                        "ID from each filename stem. If unset, the immediate parent "
                        "directory name is used.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--arch", type=str, default="osnet_ain_x1_0")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--P", type=int, default=16,
                   help="Identities per batch (PK sampler).")
    p.add_argument("--K", type=int, default=4,
                   help="Instances per identity per batch (PK sampler).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--input-h", type=int, default=DEFAULT_INPUT_HW[0])
    p.add_argument("--input-w", type=int, default=DEFAULT_INPUT_HW[1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 avoids Windows multiprocessing issues.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-iters-per-epoch", type=int, default=600,
                   help="Cap so a single CPU epoch fits in ~10 min.")
    p.add_argument("--limit-ids", type=int, default=0,
                   help="If >0, subsample this many random identities for a fast run.")
    p.add_argument("--limit-images-per-id", type=int, default=0,
                   help="If >0, cap each identity to N random images.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Dataset scanning
# --------------------------------------------------------------------------- #


def _scan_carla(
    root: Path,
    id_regex: Optional[str],
    rng: np.random.Generator,
    limit_ids: int,
    limit_images_per_id: int,
) -> Tuple[List[Path], List[int], List[str]]:
    """Walk ``root`` and return (paths, contiguous_labels, id_names)."""
    if not root.exists():
        raise FileNotFoundError(root)

    by_id: Dict[str, List[Path]] = {}
    regex = re.compile(id_regex) if id_regex else None

    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        # Identity: regex on stem, else parent folder name.
        if regex is not None:
            m = regex.search(f.stem)
            if not m:
                continue
            vid = m.group(1)
        else:
            vid = f.parent.name
        by_id.setdefault(vid, []).append(f)

    if not by_id:
        raise RuntimeError(
            f"No images found under {root} (id_regex={id_regex!r}). "
            "Inspect the layout and pass a regex with one capture group."
        )

    # Drop identities with < 2 images — useless for triplet training.
    by_id = {k: v for k, v in by_id.items() if len(v) >= 2}
    if not by_id:
        raise RuntimeError("Every identity has <2 images — cannot train.")

    id_names = sorted(by_id.keys())
    if limit_ids > 0 and len(id_names) > limit_ids:
        chosen = rng.choice(id_names, size=limit_ids, replace=False)
        id_names = sorted(chosen.tolist())
    id_to_idx = {n: i for i, n in enumerate(id_names)}

    images: List[Path] = []
    labels: List[int] = []
    for vid in id_names:
        files = sorted(by_id[vid])
        if limit_images_per_id > 0 and len(files) > limit_images_per_id:
            idx = rng.choice(len(files), size=limit_images_per_id, replace=False)
            files = [files[int(i)] for i in idx]
        for f in files:
            images.append(f)
            labels.append(id_to_idx[vid])

    return images, labels, id_names


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch
    from torch.utils.data import DataLoader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    logger.info("Scanning %s ...", args.data_root)
    images, labels, id_names = _scan_carla(
        args.data_root,
        id_regex=args.id_regex,
        rng=rng,
        limit_ids=args.limit_ids,
        limit_images_per_id=args.limit_images_per_id,
    )
    n_ids = len(id_names)
    logger.info(
        "Loaded %d images across %d identities (avg %.1f per id).",
        len(images), n_ids, len(images) / max(1, n_ids),
    )

    train_tf = _make_transforms(args.input_h, args.input_w, train=True)
    train_ds = _CropDataset(images, labels, train_tf)

    batch_size = args.P * args.K
    iters = max(1, min(args.max_iters_per_epoch, len(images) // batch_size))
    sampler = _PKSampler(
        labels, P=args.P, K=args.K, num_iter=iters, rng=rng,
    )
    if not sampler.supports_pk:
        logger.warning(
            "PK sampler degraded: some identities have <%d instances; "
            "drawing with replacement for those.",
            args.K,
        )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone = _build_backbone(args.arch)
    model = _Finetuner(backbone, num_classes=n_ids)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable params: %.2fM",
                sum(p.numel() for p in trainable) / 1e6)

    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )

    today = _dt.date.today().strftime("%Y%m%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"osnet_carla_pretrain_{today}.pt"
    log_path = args.output_dir / f"osnet_carla_pretrain_{today}.log"

    log_fh = log_path.open("w", encoding="utf-8")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = _dt.datetime.now()
            losses = {"triplet": 0.0, "ce": 0.0, "total": 0.0}
            n_batches = 0
            for batch in train_loader:
                tensors, targets = batch
                tensors = tensors.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits, embeddings = model.forward(tensors)
                loss_triplet = _batch_hard_triplet_loss(
                    embeddings, targets, margin=args.triplet_margin,
                )
                loss_ce = _label_smoothed_ce(
                    logits, targets, smoothing=args.label_smoothing,
                )
                total = loss_triplet + loss_ce
                total.backward()
                optimizer.step()

                losses["triplet"] += float(loss_triplet.detach())
                losses["ce"] += float(loss_ce.detach())
                losses["total"] += float(total.detach())
                n_batches += 1

            scheduler.step()
            for k in losses:
                losses[k] /= max(1, n_batches)
            elapsed = (_dt.datetime.now() - t0).total_seconds()

            log_entry = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_triplet": round(losses["triplet"], 4),
                "loss_ce": round(losses["ce"], 4),
                "loss_total": round(losses["total"], 4),
                "iters": n_batches,
                "elapsed_s": round(elapsed, 1),
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            log_fh.flush()
            logger.info(
                "epoch %d/%d: loss=%.4f (trip=%.4f, ce=%.4f), iters=%d, t=%.1fs",
                epoch, args.epochs, losses["total"],
                losses["triplet"], losses["ce"], n_batches, elapsed,
            )

            # Save after every epoch so a kill doesn't lose all progress.
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "arch": args.arch,
                    "num_classes": n_ids,
                    "id_names": id_names,
                    "input_hw": [args.input_h, args.input_w],
                    "source": "VeRi-CARLA (sekilab/VehicleReIdentificationDataset)",
                    "license": "Apache-2.0",
                },
                ckpt_path,
            )

    finally:
        log_fh.close()

    logger.info("Pretrain done. Best ckpt at %s", ckpt_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
