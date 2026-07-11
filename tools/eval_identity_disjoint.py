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
# View (camera-proxy) labels + the stricter cross-view protocols
# --------------------------------------------------------------------------- #
#
# The held-out crops carry NO camera id in their filenames (all parse to the
# training project's camid 0), so a literal Market-1501 cross-camera protocol is
# impossible. The filename PREFIX, however, is a genuine viewpoint signal:
# ``floor_*`` (a car on the parking floor) vs ``gate_*`` (a car at the entry
# gate) are different views of the same car. We use that as a camera PROXY: the
# cross-view protocol credits a match only when query and gallery come from
# different views, which removes the same-camera near-duplicate frames that
# saturate the self-gallery rank-1.

VIEW_CODES = {"floor": 0, "gate": 1, "other": 2}


def parse_view(filename: str) -> int:
    """Map a crop filename to a coarse viewpoint code from its prefix token.
    ``floor``/``gate`` are the two real facility viewpoints; everything else
    (``crop_*`` etc.) is ``other`` and never yields a cross-view match."""
    prefix = filename.split("_", 1)[0].lower()
    return VIEW_CODES.get(prefix, VIEW_CODES["other"])


def _evaluate_cross_view(
    sim: np.ndarray, pids: np.ndarray, views: np.ndarray,
) -> Optional[Dict]:
    """Market-1501 evaluation with VIEW as the camera. Each image is a query in
    turn; the gallery is every other image, but same-identity + same-view images
    are dropped as junk so only CROSS-VIEW matches count. Queries with no
    cross-view gallery mate of their identity are skipped (standard). Returns
    rank-1/5 + mAP averaged over valid queries."""
    n = sim.shape[0]
    r1 = r5 = 0
    aps: List[float] = []
    valid = 0
    for i in range(n):
        order = np.argsort(-sim[i])
        order = order[order != i]  # drop the query itself
        # Junk = same identity AND same view (near-duplicate same-camera frames).
        keep = ~((pids[order] == pids[i]) & (views[order] == views[i]))
        order = order[keep]
        good = (pids[order] == pids[i]).astype(np.float64)  # same id, other view
        if good.sum() == 0:
            continue  # no cross-view mate — not a valid cross-view query
        valid += 1
        first = int(np.flatnonzero(good)[0])
        if first < 1:
            r1 += 1
        if first < 5:
            r5 += 1
        cum = np.cumsum(good)
        ranks = np.arange(1, len(good) + 1)
        aps.append(float((cum / ranks * good).sum() / good.sum()))
    if valid == 0:
        return None
    return {
        "rank1": round(r1 / valid, 4),
        "rank5": round(r5 / valid, 4),
        "mAP": round(float(np.mean(aps)), 4),
        "valid_queries": valid,
    }


def _evaluate_single_shot_cross_view(
    feats: np.ndarray, pids: np.ndarray, views: np.ndarray,
    n_trials: int, base_seed: int,
) -> Optional[Dict]:
    """Strictest protocol: the gallery is ONE random crop per (identity, view);
    every other crop is a query, scored cross-view (same id+view junked). This
    removes the many-frames-per-car crutch entirely — a query must match a single
    reference shot of its car taken from a different view, ranked against one
    reference shot of every other car. Averaged over ``n_trials`` random gallery
    draws for stability."""
    from collections import defaultdict

    n = len(pids)
    pair_to_idxs: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx in range(n):
        pair_to_idxs[(int(pids[idx]), int(views[idx]))].append(idx)

    r1s: List[float] = []
    maps: List[float] = []
    valids: List[int] = []
    for t in range(n_trials):
        rng = np.random.RandomState(base_seed + t)
        gallery = np.array(
            sorted(int(rng.choice(idxs)) for idxs in pair_to_idxs.values())
        )
        gset = set(gallery.tolist())
        queries = np.array([i for i in range(n) if i not in gset])
        if queries.size == 0:
            continue
        g_pid, g_view = pids[gallery], views[gallery]
        sims = feats[queries] @ feats[gallery].T
        r1 = valid = 0
        aps: List[float] = []
        for qi, i in enumerate(queries):
            order = np.argsort(-sims[qi])
            keep = ~((g_pid[order] == pids[i]) & (g_view[order] == views[i]))
            order = order[keep]
            good = (g_pid[order] == pids[i]).astype(np.float64)
            if good.sum() == 0:
                continue
            valid += 1
            if int(np.flatnonzero(good)[0]) < 1:
                r1 += 1
            cum = np.cumsum(good)
            ranks = np.arange(1, len(good) + 1)
            aps.append(float((cum / ranks * good).sum() / good.sum()))
        if valid:
            r1s.append(r1 / valid)
            maps.append(float(np.mean(aps)))
            valids.append(valid)
    if not r1s:
        return None
    return {
        "rank1": round(float(np.mean(r1s)), 4),
        "mAP": round(float(np.mean(maps)), 4),
        "rank1_std": round(float(np.std(r1s)), 4),
        "avg_valid_queries": round(float(np.mean(valids)), 1),
        "trials": len(r1s),
    }


# --------------------------------------------------------------------------- #
# Eval driver (uses the real matcher; reuses vetted metric code)
# --------------------------------------------------------------------------- #


def evaluate_ir(
    ir_dir: Path,
    images: List[Path],
    labels: np.ndarray,
    views: Optional[np.ndarray] = None,
    single_shot_trials: int = 10,
    single_shot_seed: int = 42,
) -> Optional[Dict]:
    """Embed every held-out crop and score three protocols:

    * ``self_gallery`` — each crop queried against every other (the lenient
      number; rank-1 saturates because same-view near-duplicate frames count).
    * ``cross_view`` — Market-1501 with VIEW as the camera: only cross-view
      matches count (same id+view junked). The honest stricter number.
    * ``cross_view_single_shot`` — gallery is one crop per (id, view); strictest.

    Returns None when the IR is missing so a sweep skips it gracefully."""
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
    out: Dict = {
        "ir_dir": str(ir_dir),
        "n_images": int(feats.shape[0]),
        "self_gallery": {
            "rank1": round(_rank_k(sim, labels, 1), 4),
            "rank5": round(_rank_k(sim, labels, 5), 4),
            "mAP": round(_mean_average_precision(sim, labels), 4),
        },
    }
    if views is not None:
        out["cross_view"] = _evaluate_cross_view(sim, labels, views)
        out["cross_view_single_shot"] = _evaluate_single_shot_cross_view(
            feats, labels, views, single_shot_trials, single_shot_seed
        )
    return out


def run(
    source_root: Path,
    train_report: Optional[Path],
    train_splits: Optional[Path],
    explicit_train_ids: Optional[Set[str]],
    ir_dirs: List[Path],
    report_path: Path,
    min_images: int = MIN_IMAGES_PER_IDENTITY,
    single_shot_trials: int = 10,
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
    views = np.array([parse_view(p.name) for p in images], dtype=np.int64)
    view_hist = {
        name: int((views == code).sum()) for name, code in VIEW_CODES.items()
    }
    logger.info(
        "Identity-disjoint eval: %d held-out identities, %d images "
        "(dropped %d identities). views=%s",
        len(id_names), len(images), len(part.dropped), view_hist,
    )

    results = [
        r
        for r in (
            evaluate_ir(ir, images, labels, views, single_shot_trials)
            for ir in ir_dirs
        )
        if r
    ]

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
        "view_histogram": view_hist,
        "single_shot_trials": single_shot_trials,
        "disjointness_verified": True,  # assert_identity_disjoint passed above
        "protocols": {
            "self_gallery": "each crop vs every other (lenient; rank-1 saturates)",
            "cross_view": "Market-1501, VIEW as camera; only cross-view matches "
                          "count (same id+view junked)",
            "cross_view_single_shot": "gallery = 1 crop per (id, view); strictest",
        },
        "view_note": "views are a floor/gate filename-prefix PROXY for camera — "
                     "the crops carry no camera id.",
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
    print(f"disjointness verified: {report['disjointness_verified']}  "
          f"| views (camera proxy): {report.get('view_histogram')}")
    print("-" * 70)
    if not report["per_ir"]:
        print("No IR evaluated (missing model.xml). Split is ready; add an IR.")
        print("=" * 70)
        return

    def _row(label: str, block: Optional[Dict]) -> None:
        if not block:
            print(f"  {label:26} (no valid queries)")
            return
        r1 = block.get("rank1")
        r5 = block.get("rank5")
        mp = block.get("mAP")
        extra = ""
        if "valid_queries" in block:
            extra = f"  [{block['valid_queries']} valid q]"
        elif "avg_valid_queries" in block:
            extra = (f"  [~{block['avg_valid_queries']:.0f} q x "
                     f"{block['trials']} trials, r1σ={block.get('rank1_std')}]")
        r5s = f"{r5:>7.4f}" if isinstance(r5, float) else f"{'—':>7}"
        print(f"  {label:26} rank1={r1:>7.4f} rank5={r5s} mAP={mp:>7.4f}{extra}")

    for r in report["per_ir"]:
        print(f"{Path(r['ir_dir']).name}  (n={r['n_images']})")
        _row("self-gallery (lenient)", r.get("self_gallery"))
        _row("cross-view (Market)", r.get("cross_view"))
        _row("cross-view single-shot", r.get("cross_view_single_shot"))
        print()
    print("mAP is the metric to trust; cross-view removes same-camera "
          "near-duplicates. Views are a floor/gate proxy (no camera id in crops).")
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
    p.add_argument("--single-shot-trials", type=int, default=10,
                   help="Random gallery draws for the cross-view single-shot "
                        "protocol (averaged for stability).")
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
            single_shot_trials=args.single_shot_trials,
        )
    except ContaminationError as exc:
        logger.error("REFUSING TO REPORT A CONTAMINATED EVAL: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
