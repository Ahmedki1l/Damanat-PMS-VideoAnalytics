"""
tools/eval_identity_disjoint.py — the HONEST ReID eval (defect report D15).

The existing eval tools (``tools/verify_on_eval.py``,
``tests/test_facility_match_accuracy.py``) grade the fine-tuned model on the
``facility_top20`` identities — the SAME plates the model was fine-tuned on
(``tools/finetune_osnet_top20.py``). They are image-disjoint (a query image is
never its own gallery image) but NOT identity-disjoint, so the reported rank-1
is inflated by identity memorisation. This tool fixes the one thing that is
wrong there: the DATA SPLIT.

It evaluates ONLY on identities the model has never seen — every identity in
``--source-root`` that is NOT in the training set — and it **hard-fails** if any
training identity leaks into the eval, so a contaminated run can never quietly
report an inflated number again. The metric machinery (rank-k, mAP) is imported
from ``tests.test_facility_match_accuracy`` so there is a single vetted
implementation.

Data layout expected at ``--source-root``::

    <source-root>/
        AGA-6649/  img0.jpg img1.jpg ...     # one folder per identity (plate)
        NDD-4141/  ...
        ...

Training identities are read from a split report (``--train-report``, defaults
to ``data/facility_top20/split_report.json`` whose ``per_plate`` keys are the
fine-tune identities) or given explicitly with ``--train-id`` (repeatable).

Usage
-----
    # Once the held-out crops are on disk:
    python tools/eval_identity_disjoint.py \
        --source-root data/facility_all_identities \
        --ir-dir models/osnet_facility_int8_256x128

    # Compare two IRs on the SAME held-out identities:
    python tools/eval_identity_disjoint.py --source-root <root> \
        --ir-dir models/osnet_openvino_int8_256x128 \
        --ir-dir models/osnet_facility_int8_256x128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("eval_identity_disjoint")

DEFAULT_TRAIN_REPORT = Path("data/facility_top20/split_report.json")
DEFAULT_IR = Path("models/osnet_facility_int8_256x128")
DEFAULT_REPORT_PATH = Path("data/identity_disjoint_eval_report.json")
# An identity needs at least this many crops to be usable: one to act as the
# query and at least one OTHER same-identity crop to be retrievable.
MIN_IMAGES_PER_IDENTITY = 2


class ContaminationError(RuntimeError):
    """Raised when training identities leak into the held-out eval set — the
    exact failure this tool exists to make impossible to ignore."""


# --------------------------------------------------------------------------- #
# Pure split logic (unit-tested without a model)
# --------------------------------------------------------------------------- #


def normalize_identity(name: str) -> str:
    """Canonicalise an identity/plate for comparison. Folder names and manifests
    disagree on the plate separator (``EEB_80`` vs ``EEB-80``) and case; without
    this, a training car spelled with the other separator would slip past the
    disjointness check and silently contaminate the eval."""
    return name.strip().upper().replace("_", "-")


def load_train_identities(train_report: Path) -> Set[str]:
    """Training identities = the ``per_plate`` keys of a split report
    (``facility_top20/split_report.json`` shape)."""
    data = json.loads(train_report.read_text(encoding="utf-8"))
    per_plate = data.get("per_plate")
    if not isinstance(per_plate, dict) or not per_plate:
        raise ValueError(
            f"{train_report} has no non-empty 'per_plate' map — cannot "
            "determine the training identities."
        )
    return set(per_plate.keys())


def load_train_identities_from_splits(splits_json: Path) -> Set[str]:
    """Training identities from a ``reid_data/splits.json`` manifest: the
    identity is the parent folder of each ``train`` entry's crop path. This is
    the ACTUAL train set the deployed OSNet facility checkpoint saw (its
    ``query``/``gallery`` identities are the held-out test set and must NOT be
    treated as training)."""
    data = json.loads(splits_json.read_text(encoding="utf-8"))
    train = data.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError(
            f"{splits_json} has no non-empty 'train' list — cannot determine "
            "the training identities."
        )
    ids: Set[str] = set()
    for rec in train:
        # rec = [crop_path, pid, camid]; identity = parent folder of the path.
        path = rec[0] if isinstance(rec, (list, tuple)) else rec
        ids.add(Path(str(path).replace("\\", "/")).parent.name)
    return ids


def scan_identities(source_root: Path) -> Dict[str, List[Path]]:
    """Map each identity subfolder to its list of .jpg crops (sorted)."""
    if not source_root.exists():
        raise FileNotFoundError(f"source-root missing: {source_root}")
    out: Dict[str, List[Path]] = {}
    for d in sorted(p for p in source_root.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.jpg"))
        if files:
            out[d.name] = files
    return out


@dataclass
class Partition:
    held_out: Dict[str, List[Path]] = field(default_factory=dict)
    # identity -> reason it was excluded ("in_training" | "too_few_images")
    dropped: Dict[str, str] = field(default_factory=dict)


def partition_identities(
    source: Dict[str, List[Path]],
    train_ids: Set[str],
    min_images: int = MIN_IMAGES_PER_IDENTITY,
) -> Partition:
    """Split scanned identities into the held-out eval set and dropped ones.

    An identity is held out iff it is NOT in ``train_ids`` AND has at least
    ``min_images`` crops. Training identities are dropped as ``in_training``;
    thin held-out identities as ``too_few_images``.
    """
    train_norm = {normalize_identity(t) for t in train_ids}
    part = Partition()
    for ident, files in source.items():
        if normalize_identity(ident) in train_norm:
            part.dropped[ident] = "in_training"
        elif len(files) < min_images:
            part.dropped[ident] = "too_few_images"
        else:
            part.held_out[ident] = files
    return part


def assert_identity_disjoint(held_out: Set[str], train_ids: Set[str]) -> None:
    """Fail loudly if any training identity is present in the eval set, or if
    the eval set is empty (nothing to measure)."""
    held_norm = {normalize_identity(x) for x in held_out}
    train_norm = {normalize_identity(x) for x in train_ids}
    overlap = held_norm & train_norm
    if overlap:
        raise ContaminationError(
            "Eval set is contaminated: these training identities leaked into "
            f"the held-out set (normalised): {sorted(overlap)}"
        )
    if not held_out:
        raise ContaminationError(
            "No held-out identities remain — every identity in the source is "
            "in the training set (or too thin). An identity-disjoint eval "
            "requires identities the model has never seen."
        )


def build_index(
    held_out: Dict[str, List[Path]],
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Flatten held-out identities into (image_paths, int_labels, id_names)."""
    id_names = sorted(held_out.keys())
    id_to_label = {name: i for i, name in enumerate(id_names)}
    images: List[Path] = []
    labels: List[int] = []
    for name in id_names:
        for f in held_out[name]:
            images.append(f)
            labels.append(id_to_label[name])
    return images, np.asarray(labels, dtype=np.int64), id_names


# --------------------------------------------------------------------------- #
# Eval driver (uses the real matcher; reuses vetted metric code)
# --------------------------------------------------------------------------- #


def evaluate_ir(
    ir_dir: Path,
    images: List[Path],
    labels: np.ndarray,
) -> Optional[Dict]:
    """Embed every held-out crop through the IR and compute rank-1/5 + mAP on a
    self-as-query gallery (each crop queried against every OTHER crop). Returns
    None when the IR is missing so a sweep skips it gracefully."""
    if not (ir_dir / "model.xml").exists():
        logger.warning("IR not found at %s — skipping.", ir_dir)
        return None
    # Reuse the ONE vetted implementation of matcher-build and the metrics.
    from tests.test_facility_match_accuracy import (
        _build_matcher,
        _extract_features,
        _mean_average_precision,
        _rank_k,
    )

    matcher = _build_matcher(ir_dir)
    feats = _extract_features(matcher, images)
    sim = feats @ feats.T
    return {
        "ir_dir": str(ir_dir),
        "n_images": int(feats.shape[0]),
        "rank1": round(_rank_k(sim, labels, 1), 4),
        "rank5": round(_rank_k(sim, labels, 5), 4),
        "mAP": round(_mean_average_precision(sim, labels), 4),
    }


def run(
    source_root: Path,
    train_report: Optional[Path],
    train_splits: Optional[Path],
    explicit_train_ids: Optional[Set[str]],
    ir_dirs: List[Path],
    report_path: Path,
    min_images: int = MIN_IMAGES_PER_IDENTITY,
) -> Dict:
    train_ids = set(explicit_train_ids or set())
    if train_report is not None:
        train_ids |= load_train_identities(train_report)
    if train_splits is not None:
        train_ids |= load_train_identities_from_splits(train_splits)
    if not train_ids:
        raise ValueError(
            "No training identities supplied — pass --train-report or "
            "--train-id, else the disjointness check is meaningless."
        )

    source = scan_identities(source_root)
    part = partition_identities(source, train_ids, min_images)
    # The guard: refuse to proceed on a contaminated or empty eval set.
    assert_identity_disjoint(set(part.held_out), train_ids)

    images, labels, id_names = build_index(part.held_out)
    logger.info(
        "Identity-disjoint eval: %d held-out identities, %d images "
        "(dropped %d identities).",
        len(id_names), len(images), len(part.dropped),
    )

    results = [r for r in (evaluate_ir(ir, images, labels) for ir in ir_dirs) if r]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "train_identity_count": len(train_ids),
        "source_identity_count": len(source),
        "held_out_identity_count": len(id_names),
        "held_out_identities": id_names,
        "dropped_identities": part.dropped,
        "min_images_per_identity": min_images,
        "n_eval_images": len(images),
        "disjointness_verified": True,  # assert_identity_disjoint passed above
        "per_ir": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)
    return report


def _print_summary(report: Dict) -> None:
    print()
    print("=" * 70)
    print("Identity-disjoint ReID eval (D15 — honest, no training identities)")
    print("=" * 70)
    print(
        f"held-out identities: {report['held_out_identity_count']}  "
        f"| eval images: {report['n_eval_images']}  "
        f"| training identities excluded: {report['train_identity_count']}"
    )
    print(f"disjointness verified: {report['disjointness_verified']}")
    print("-" * 70)
    if not report["per_ir"]:
        print("No IR evaluated (missing model.xml). Split is ready; add an IR.")
    else:
        print(f"{'IR':40} {'rank1':>7} {'rank5':>7} {'mAP':>7}")
        for r in report["per_ir"]:
            print(
                f"{Path(r['ir_dir']).name:40.40} "
                f"{r['rank1']:>7.4f} {r['rank5']:>7.4f} {r['mAP']:>7.4f}"
            )
    print("=" * 70)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval_identity_disjoint",
        description="Honest ReID eval on identities the model never trained on.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-root", type=Path, required=True,
                   help="Dir of per-identity crop subfolders.")
    p.add_argument("--train-report", type=Path, default=DEFAULT_TRAIN_REPORT,
                   help="Split report whose per_plate keys are training IDs "
                        "(facility_top20/split_report.json shape).")
    p.add_argument("--train-splits", type=Path, default=None,
                   help="reid_data/splits.json manifest; its 'train' entries' "
                        "folders are the deployed model's training IDs.")
    p.add_argument("--train-id", action="append", default=None,
                   help="Extra training identity to exclude (repeatable).")
    p.add_argument("--ir-dir", action="append", type=Path, default=None,
                   help="OpenVINO IR dir to evaluate (repeatable).")
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--min-images", type=int, default=MIN_IMAGES_PER_IDENTITY)
    p.add_argument("--no-train-report", action="store_true",
                   help="Ignore --train-report; use only --train-id values.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # When a splits.json manifest is given, it is the authoritative training set
    # — don't also pull in the unrelated facility_top20 report by default.
    if args.train_splits is not None or args.no_train_report:
        train_report = None
    else:
        train_report = args.train_report
    if train_report is not None and not train_report.exists():
        logger.error("train-report missing: %s (use --train-splits or "
                     "--no-train-report + --train-id instead).", train_report)
        return 1
    if args.train_splits is not None and not args.train_splits.exists():
        logger.error("train-splits missing: %s", args.train_splits)
        return 1
    try:
        run(
            source_root=args.source_root,
            train_report=train_report,
            train_splits=args.train_splits,
            explicit_train_ids=set(args.train_id or []),
            ir_dirs=args.ir_dir or [DEFAULT_IR],
            report_path=args.report_path,
            min_images=args.min_images,
        )
    except ContaminationError as exc:
        logger.error("REFUSING TO REPORT A CONTAMINATED EVAL: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
