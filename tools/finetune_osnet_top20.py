"""
tools/finetune_osnet_top20.py — Phase 0.5 / T-3.

Fine-tunes torchreid's ``osnet_ain_x1_0`` backbone on the curated top-20
facility dataset produced by T-1 (``curate_facility_dataset.py``) and T-2
(``split_facility_dataset.py``).

Distinct from ``tools/finetune_osnet_facility.py`` which mines pairs from
production logs + DB. This one trains on a hand-organised folder dataset
where each subfolder name is the identity label.

Loss
----
* Triplet (margin 0.3) on L2-normalised 512-D embeddings — matches the
  production cosine-similarity decision boundary.
* Label-smoothed cross-entropy (smoothing 0.1) on a fresh N-class head
  attached to the backbone embedding.

Sampler
-------
PK sampling — at each step, pick ``P`` identities, then ``K`` instances per
identity. ``P × K`` becomes the per-batch size. Falls back to random
sampling with replacement when any identity has fewer than ``K`` training
images, so thin folders don't crash the run.

Outputs
-------
* ``models/osnet_facility_finetune_<YYYYMMDD>.pt`` — best checkpoint by
  eval rank-1 accuracy. Contains ``state_dict``, ``epoch``, ``rank1``,
  ``margin``, ``arch``, ``plate_names``, ``input_hw``.
* ``models/osnet_facility_finetune_<YYYYMMDD>.log`` — per-epoch training
  log (JSON lines).

Usage
-----
    python tools/finetune_osnet_top20.py
    python tools/finetune_osnet_top20.py --epochs 30 --P 8 --K 4
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("finetune_osnet_top20")


DEFAULT_TRAIN_DIR = Path("data/facility_top20/train")
DEFAULT_EVAL_DIR = Path("data/facility_top20/eval")
DEFAULT_OUTPUT_DIR = Path("models")
DEFAULT_INPUT_HW = (256, 128)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="finetune_osnet_top20",
        description="Fine-tune OSNet-AIN on the curated top-20 facility set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--arch", type=str, default="osnet_ain_x1_0")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--P", type=int, default=8,
                   help="Identities per batch (PK sampler).")
    p.add_argument("--K", type=int, default=4,
                   help="Instances per identity per batch (PK sampler).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--triplet-margin", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--triplet-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--input-h", type=int, default=DEFAULT_INPUT_HW[0])
    p.add_argument("--input-w", type=int, default=DEFAULT_INPUT_HW[1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 avoids Windows multiprocessing issues.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--freeze-stem", action="store_true", default=True)
    p.add_argument("--no-freeze-stem", dest="freeze_stem", action="store_false")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Optional checkpoint to load as the backbone init instead of "
                        "torchreid's ImageNet pretrain. Use the .pt produced by "
                        "tools/pretrain_osnet_carla.py for CARLA-pretrained init.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _scan_dir(root: Path) -> Tuple[List[Path], List[int], List[str]]:
    """Return (image_paths, labels, plate_names). Labels are contiguous ints."""
    if not root.exists():
        raise FileNotFoundError(root)
    plate_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not plate_dirs:
        raise RuntimeError(f"No plate subdirs in {root}")
    plate_names = [d.name for d in plate_dirs]
    plate_to_idx = {name: i for i, name in enumerate(plate_names)}

    images: List[Path] = []
    labels: List[int] = []
    for d in plate_dirs:
        for f in sorted(d.glob("*.jpg")):
            images.append(f)
            labels.append(plate_to_idx[d.name])
    return images, labels, plate_names


def _make_transforms(input_h: int, input_w: int, train: bool):
    """Build a torchvision transform pipeline (lazy torch import)."""
    import torchvision.transforms as T

    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((input_h + 16, input_w + 8)),
            T.RandomCrop((input_h, input_w)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
            T.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((input_h, input_w)),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class _CropDataset:
    """Minimal dataset returning (transformed_tensor, label)."""

    def __init__(self, images: List[Path], labels: List[int], transform):
        import cv2
        self._images = images
        self._labels = labels
        self._transform = transform
        self._cv2 = cv2

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        path = self._images[idx]
        img = self._cv2.imread(str(path))
        if img is None:
            img = np.zeros((64, 32, 3), dtype=np.uint8)
        img = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        tensor = self._transform(img)
        return tensor, self._labels[idx]


# --------------------------------------------------------------------------- #
# PK sampler
# --------------------------------------------------------------------------- #


class _PKSampler:
    """PK sampler producing P*K-sized batches."""

    def __init__(
        self,
        labels: List[int],
        P: int,
        K: int,
        num_iter: int,
        rng: np.random.Generator,
    ):
        self._labels = np.asarray(labels)
        self._P = int(P)
        self._K = int(K)
        self._num_iter = int(num_iter)
        self._rng = rng

        self._idx_by_label: Dict[int, np.ndarray] = {}
        for lab in np.unique(self._labels):
            self._idx_by_label[int(lab)] = np.where(self._labels == lab)[0]

        self._labels_below_K = [
            lab for lab, idxs in self._idx_by_label.items() if len(idxs) < K
        ]

    @property
    def supports_pk(self) -> bool:
        return not self._labels_below_K

    def __iter__(self):
        all_labels = list(self._idx_by_label.keys())
        for _ in range(self._num_iter):
            picked = self._rng.choice(
                all_labels, size=self._P,
                replace=(self._P > len(all_labels)),
            )
            batch_indices: List[int] = []
            for lab in picked:
                pool = self._idx_by_label[int(lab)]
                replace = len(pool) < self._K
                draw = self._rng.choice(pool, size=self._K, replace=replace)
                batch_indices.extend(int(i) for i in draw)
            yield batch_indices

    def __len__(self):
        return self._num_iter


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #


def _build_backbone(arch: str):
    import torch
    import torchreid

    logger.info("Building torchreid backbone %s (pretrained=True)...", arch)
    model = torchreid.models.build_model(
        name=arch, num_classes=1000, pretrained=True,
    )
    for m in model.modules():
        if isinstance(m, torch.nn.InstanceNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return model


class _Finetuner:
    """Wraps the torchreid OSNet backbone with a fresh N-class classifier."""

    def __init__(self, backbone, num_classes: int, feature_dim: int = 512):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)
        self._has_osnet_api = (
            hasattr(backbone, "featuremaps")
            and hasattr(backbone, "global_avgpool")
        )

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.classifier.parameters())

    def to(self, device):
        self.backbone.to(device)
        self.classifier.to(device)
        return self

    def train(self):
        self.backbone.train()
        self.classifier.train()
        return self

    def eval(self):
        self.backbone.eval()
        self.classifier.eval()
        return self

    def freeze_stem(self):
        if hasattr(self.backbone, "conv1"):
            for p in self.backbone.conv1.parameters():
                p.requires_grad = False
            logger.info("Froze backbone.conv1 (stem).")

    def _backbone_embedding(self, x):
        if self._has_osnet_api:
            f = self.backbone.featuremaps(x)
            v = self.backbone.global_avgpool(f)
            v = v.view(v.size(0), -1)
            if hasattr(self.backbone, "fc") and self.backbone.fc is not None:
                v = self.backbone.fc(v)
            return v
        # Fallback for non-OSNet backbones — eval-mode forward returns features
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            v = self.backbone(x)
        finally:
            if was_training:
                self.backbone.train()
        return v

    def forward(self, x):
        v = self._backbone_embedding(x)
        logits = self.classifier(v)
        v_norm = self._torch.nn.functional.normalize(v, p=2, dim=1)
        return logits, v_norm

    def state_dict(self):
        return {
            "backbone": self.backbone.state_dict(),
            "classifier": self.classifier.state_dict(),
        }


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def _batch_hard_triplet_loss(embeddings, labels, margin: float):
    """Cosine-distance batch-hard triplet on L2-normalised embeddings."""
    import torch

    sim = embeddings @ embeddings.t()
    dist = 1.0 - sim
    labels = labels.view(-1, 1)
    same = labels.eq(labels.t())
    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid = has_pos & has_neg
    if not valid.any():
        return torch.tensor(0.0, device=dist.device, requires_grad=True)

    big = torch.full_like(dist, fill_value=-1.0)
    dist_pos = torch.where(pos_mask, dist, big).max(dim=1).values
    small = torch.full_like(dist, fill_value=2.0)
    dist_neg = torch.where(neg_mask, dist, small).min(dim=1).values

    triplets = torch.relu(dist_pos[valid] - dist_neg[valid] + margin)
    return triplets.mean()


def _label_smoothed_ce(logits, targets, smoothing: float):
    """Cross-entropy with label smoothing."""
    import torch
    import torch.nn.functional as F

    n_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true = torch.zeros_like(log_probs).fill_(smoothing / max(1, n_classes - 1))
        true.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true * log_probs, dim=1))


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #


def _compute_eval_metrics(
    model: _Finetuner,
    eval_images: List[Path],
    eval_labels: List[int],
    transform,
    device,
    batch_size: int = 64,
) -> Tuple[float, float, float]:
    """Return (rank1, mean_same_cos, mean_diff_cos)."""
    import torch

    if len(eval_images) < 2:
        return 0.0, 0.0, 0.0

    ds = _CropDataset(eval_images, eval_labels, transform)
    feats: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = [ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]
            tensor = torch.stack(batch).to(device)
            _, emb = model.forward(tensor)
            feats.append(emb.cpu().numpy())
    feats_np = np.concatenate(feats, axis=0)
    labels_np = np.asarray(eval_labels)

    sim = feats_np @ feats_np.T
    n = len(labels_np)
    sim_for_rank = sim.copy()
    np.fill_diagonal(sim_for_rank, -np.inf)
    nearest = np.argmax(sim_for_rank, axis=1)
    rank1 = float(np.mean(labels_np[nearest] == labels_np))

    mask_self = ~np.eye(n, dtype=bool)
    label_eq = labels_np[:, None] == labels_np[None, :]
    same_mask = label_eq & mask_self
    diff_mask = (~label_eq) & mask_self
    mean_same = float(sim[same_mask].mean()) if same_mask.any() else 0.0
    mean_diff = float(sim[diff_mask].mean()) if diff_mask.any() else 0.0
    return rank1, mean_same, mean_diff


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

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    train_images, train_labels, train_plates = _scan_dir(args.train_dir)
    eval_images, eval_labels, eval_plates = _scan_dir(args.eval_dir)
    logger.info(
        "Train: %d images / %d ids. Eval: %d images / %d ids.",
        len(train_images), len(train_plates),
        len(eval_images), len(eval_plates),
    )

    # Remap eval labels to train label space so rank-1 is on the right axes.
    train_plate_to_idx = {n: i for i, n in enumerate(train_plates)}
    keep_idx: List[int] = []
    remapped: List[int] = []
    for i, img in enumerate(eval_images):
        plate = img.parent.name
        if plate in train_plate_to_idx:
            keep_idx.append(i)
            remapped.append(train_plate_to_idx[plate])
    eval_images = [eval_images[i] for i in keep_idx]
    eval_labels = remapped

    train_tf = _make_transforms(args.input_h, args.input_w, train=True)
    eval_tf = _make_transforms(args.input_h, args.input_w, train=False)

    train_ds = _CropDataset(train_images, train_labels, train_tf)

    batch_size = args.P * args.K
    iters_per_epoch = max(1, len(train_images) // batch_size)
    rng = np.random.default_rng(args.seed)

    sampler = _PKSampler(
        train_labels, P=args.P, K=args.K,
        num_iter=iters_per_epoch, rng=rng,
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
    if args.init_from is not None:
        if not args.init_from.exists():
            raise FileNotFoundError(args.init_from)
        logger.info("Loading backbone init from %s", args.init_from)
        ckpt = torch.load(str(args.init_from), map_location="cpu", weights_only=False)
        backbone_sd = None
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                inner = ckpt["state_dict"]
                if "backbone" in inner and isinstance(inner["backbone"], dict):
                    backbone_sd = inner["backbone"]
                else:
                    backbone_sd = inner
            elif "backbone" in ckpt and isinstance(ckpt["backbone"], dict):
                backbone_sd = ckpt["backbone"]
            else:
                backbone_sd = ckpt
        if backbone_sd is None:
            raise RuntimeError(f"Could not extract backbone state_dict from {args.init_from}")
        missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
        logger.info(
            "Loaded init weights: %d missing, %d unexpected keys.",
            len(missing), len(unexpected),
        )
        # Re-zero InstanceNorm buffers in case the init reintroduced them.
        for m in backbone.modules():
            if isinstance(m, torch.nn.InstanceNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                m.num_batches_tracked = None
    model = _Finetuner(backbone, num_classes=len(train_plates))
    if args.freeze_stem:
        model.freeze_stem()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    logger.info("Trainable params: %.2fM", n_train_params / 1e6)

    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )

    today = _dt.date.today().strftime("%Y%m%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"osnet_facility_finetune_{today}.pt"
    log_path = args.output_dir / f"osnet_facility_finetune_{today}.log"

    best_rank1 = -1.0
    best_margin = 0.0
    patience_counter = 0

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
                total = (
                    args.triplet_weight * loss_triplet
                    + args.ce_weight * loss_ce
                )
                total.backward()
                optimizer.step()

                losses["triplet"] += float(loss_triplet.detach())
                losses["ce"] += float(loss_ce.detach())
                losses["total"] += float(total.detach())
                n_batches += 1

            scheduler.step()
            for k in losses:
                losses[k] /= max(1, n_batches)

            rank1, mean_same, mean_diff = _compute_eval_metrics(
                model, eval_images, eval_labels, eval_tf, device,
            )
            margin = mean_same - mean_diff
            elapsed = (_dt.datetime.now() - t0).total_seconds()

            log_entry = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_triplet": round(losses["triplet"], 4),
                "loss_ce": round(losses["ce"], 4),
                "loss_total": round(losses["total"], 4),
                "rank1": round(rank1, 4),
                "mean_same": round(mean_same, 4),
                "mean_diff": round(mean_diff, 4),
                "margin": round(margin, 4),
                "elapsed_s": round(elapsed, 1),
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            log_fh.flush()
            logger.info(
                "epoch %d/%d: loss=%.4f (trip=%.4f, ce=%.4f), "
                "rank1=%.4f, margin=%.4f, t=%.1fs",
                epoch, args.epochs, losses["total"],
                losses["triplet"], losses["ce"], rank1, margin, elapsed,
            )

            if rank1 > best_rank1:
                best_rank1 = rank1
                best_margin = margin
                patience_counter = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "rank1": best_rank1,
                        "margin": best_margin,
                        "arch": args.arch,
                        "num_classes": len(train_plates),
                        "plate_names": train_plates,
                        "input_hw": [args.input_h, args.input_w],
                    },
                    ckpt_path,
                )
                logger.info(
                    "  -> new best rank1=%.4f saved to %s",
                    best_rank1, ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(
                        "Early stop at epoch %d (no improvement for %d epochs).",
                        epoch, patience_counter,
                    )
                    break

    finally:
        log_fh.close()

    logger.info(
        "Done. best_rank1=%.4f best_margin=%.4f ckpt=%s",
        best_rank1, best_margin, ckpt_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
