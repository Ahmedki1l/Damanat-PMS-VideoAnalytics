"""
tests/test_facility_match_accuracy.py — Phase 0.5 / T-5.

Eval benchmark on the curated facility eval set produced by T-2. Runs ReID
embedding extraction through ``VehicleReIDMatcher`` for two OpenVINO IRs
side-by-side:

* ``baseline``     — the current production IR (``models/osnet_openvino_int8_256x128``).
* ``fine_tuned``   — the IR re-exported from the T-3 fine-tune
                     (``models/osnet_facility_int8_256x128``). Skipped
                     gracefully when the directory is missing.

Metrics
-------
1. **rank-1 / rank-5** — for each eval image, treat it as a query against
   every other eval image. Rank-k accuracy = fraction of queries whose
   top-k nearest neighbours contain at least one same-identity gallery.
2. **mAP** — mean Average Precision over all queries.
3. **ROC AUC** — on the (query, gallery, label) triples from
   ``pairs_eval.csv`` (1=same plate, 0=different).
4. **Precision / recall** at the current production ``b1_zone`` threshold
   (0.53). Reported on the pairs split.
5. **Per-modality contribution** — average colour / type cascade agreement
   rates on positive vs negative pairs (sanity check on the wider cascade,
   not just ReID).

Outputs ``data/facility_top20/eval_report.json`` so the result is durable
and CI-comparable across runs.

CLI usage
---------

    python -m tests.test_facility_match_accuracy
    python -m tests.test_facility_match_accuracy --skip-cascade
    pytest tests/test_facility_match_accuracy.py -k facility_eval

The pytest entry-point ``test_facility_eval_smoke`` only asserts the
benchmark *runs* end-to-end; the actual numbers are written to disk for
the operator to inspect, not gated as a hard pass/fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the repo root importable when this script is run directly
# (``python tests/test_facility_match_accuracy.py``). pytest already inserts
# the rootdir automatically; this guard is for the CLI entry-point.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("test_facility_match_accuracy")


DEFAULT_EVAL_DIR = Path("data/facility_top20/eval")
DEFAULT_PAIRS_CSV = Path("data/facility_top20/pairs_eval.csv")
DEFAULT_REPORT_PATH = Path("data/facility_top20/eval_report.json")
DEFAULT_BASELINE_IR = Path("models/osnet_openvino_int8_256x128")
DEFAULT_FINETUNED_IR = Path("models/osnet_facility_int8_256x128")
DEFAULT_THRESHOLD = 0.53  # b1_zone


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_facility_match_accuracy",
        description="Bench ReID accuracy on the top-20 facility eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--baseline-ir", type=Path, default=DEFAULT_BASELINE_IR)
    p.add_argument("--finetuned-ir", type=Path, default=DEFAULT_FINETUNED_IR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--skip-cascade", action="store_true",
                   help="Skip color/type per-modality contribution measurement.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class EvalIndex:
    images: List[Path]
    labels: List[int]
    plate_names: List[str]
    plate_to_idx: Dict[str, int] = field(default_factory=dict)


def _scan_eval(root: Path) -> EvalIndex:
    if not root.exists():
        raise FileNotFoundError(f"Eval dir missing: {root}")
    plate_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    plate_names = [d.name for d in plate_dirs]
    plate_to_idx = {n: i for i, n in enumerate(plate_names)}
    images: List[Path] = []
    labels: List[int] = []
    for d in plate_dirs:
        for f in sorted(d.glob("*.jpg")):
            images.append(f)
            labels.append(plate_to_idx[d.name])
    return EvalIndex(images, labels, plate_names, plate_to_idx)


def _load_pairs(csv_path: Path) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((row["query_path"], row["gallery_path"], int(row["label"])))
    return rows


# --------------------------------------------------------------------------- #
# ReID matcher swap (one matcher per IR)
# --------------------------------------------------------------------------- #


def _build_matcher(ir_dir: Path):
    """Build a ``VehicleReIDMatcher`` pointed at the given OpenVINO IR dir.

    Imports are lazy so this module's --help works without torch.
    """
    from src.config import MatchingConfig, ReIDPreprocessingConfig
    from src.reid_matcher.reid_matcher import VehicleReIDMatcher

    cfg = MatchingConfig()
    cfg.use_openvino_reid = True
    cfg.reid_openvino_model_dir = str(ir_dir)
    cfg.reid_input_size = (256, 128)
    matcher = VehicleReIDMatcher(
        matching_config=cfg,
        preprocessing_config=ReIDPreprocessingConfig(),
        backend="openvino",
    )
    return matcher


def _extract_features(
    matcher,
    image_paths: List[Path],
) -> np.ndarray:
    """Return an (N, 512) float32 array, L2-normalised. Skipped paths use zeros."""
    import cv2

    feats: List[np.ndarray] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            feats.append(np.zeros(512, dtype=np.float32))
            continue
        vec = matcher.extract_feature(img)
        if vec is None:
            vec = np.zeros(512, dtype=np.float32)
        else:
            n = float(np.linalg.norm(vec))
            if n > 1e-9:
                vec = vec.astype(np.float32) / n
            else:
                vec = vec.astype(np.float32)
        feats.append(vec)
    return np.vstack(feats) if feats else np.zeros((0, 512), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


def _rank_k(sim: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Rank-k accuracy over self-as-query gallery (excluding the query itself)."""
    n = sim.shape[0]
    if n < 2:
        return 0.0
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    # argsort descending; take top-k indices
    top_k_idx = np.argsort(-s, axis=1)[:, :k]
    top_k_labels = labels[top_k_idx]  # (n, k)
    correct = (top_k_labels == labels[:, None]).any(axis=1)
    return float(np.mean(correct))


def _mean_average_precision(sim: np.ndarray, labels: np.ndarray) -> float:
    """Standard ReID mAP: AP per query, then mean."""
    n = sim.shape[0]
    if n < 2:
        return 0.0
    aps: List[float] = []
    for q in range(n):
        gallery_idx = np.array([i for i in range(n) if i != q])
        if gallery_idx.size == 0:
            continue
        gallery_sim = sim[q, gallery_idx]
        gallery_lab = labels[gallery_idx]
        order = np.argsort(-gallery_sim)
        sorted_lab = gallery_lab[order]
        same = (sorted_lab == labels[q]).astype(np.float64)
        if same.sum() == 0:
            continue
        cum = np.cumsum(same)
        rank = np.arange(1, len(same) + 1)
        precision_at_k = cum / rank
        ap = (precision_at_k * same).sum() / same.sum()
        aps.append(float(ap))
    if not aps:
        return 0.0
    return float(np.mean(aps))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC manually (avoid sklearn dependency).

    Uses the rank-based formula: AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _precision_recall_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float,
) -> Tuple[float, float, int, int, int, int]:
    """Return (precision, recall, TP, FP, FN, TN)."""
    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, tp, fp, fn, tn


def _pair_cosine_stats(
    feats: np.ndarray, labels: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (mean_same, mean_diff, margin) — same as fine-tune metric."""
    n = feats.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0
    sim = feats @ feats.T
    mask = ~np.eye(n, dtype=bool)
    label_eq = labels[:, None] == labels[None, :]
    same = sim[label_eq & mask]
    diff = sim[(~label_eq) & mask]
    if same.size == 0 or diff.size == 0:
        return 0.0, 0.0, 0.0
    mean_same = float(same.mean())
    mean_diff = float(diff.mean())
    return mean_same, mean_diff, mean_same - mean_diff


# --------------------------------------------------------------------------- #
# Cascade contribution (optional)
# --------------------------------------------------------------------------- #


def _measure_cascade_contribution(
    pairs: List[Tuple[str, str, int]],
    pairs_root: Path,
) -> Optional[Dict[str, float]]:
    """Run color and type classifiers on each pair; report agreement rates.

    Returns ``None`` when the classifiers are unavailable.
    """
    try:
        from src.config import MatchingConfig
        from src.classifiers.color_classifier import OpenVINOColorClassifier
        from src.classifiers.type_classifier import OpenVINOTypeClassifier
    except Exception as exc:
        logger.warning("Cascade contribution skipped: %r", exc)
        return None

    import os
    import cv2

    cfg = MatchingConfig()
    color = None
    type_clf = None
    if cfg.color_classifier_model and os.path.exists(cfg.color_classifier_model):
        try:
            color = OpenVINOColorClassifier(cfg.color_classifier_model, config=cfg)
        except Exception as exc:
            logger.warning("Color classifier load failed: %r", exc)
    if cfg.type_classifier_model and os.path.exists(cfg.type_classifier_model):
        try:
            type_clf = OpenVINOTypeClassifier(model_path=cfg.type_classifier_model)
        except Exception as exc:
            logger.warning("Type classifier load failed: %r", exc)
    if color is None and type_clf is None:
        return None

    pos_color = pos_color_total = 0
    neg_color = neg_color_total = 0
    pos_type = pos_type_total = 0
    neg_type = neg_type_total = 0

    cache: Dict[str, Tuple[Optional[Tuple[str, float]], Optional[Tuple[str, float]]]] = {}

    def _classify(path: str):
        if path in cache:
            return cache[path]
        img = cv2.imread(str(pairs_root / path))
        if img is None:
            cache[path] = (None, None)
            return cache[path]
        c = None
        t = None
        if color is not None:
            try:
                c = color.predict(img)
            except Exception:
                c = None
        if type_clf is not None:
            try:
                t = type_clf.predict(img)
            except Exception:
                t = None
        cache[path] = (c, t)
        return cache[path]

    for qp, gp, label in pairs:
        cq, tq = _classify(qp)
        cg, tg = _classify(gp)
        if cq is not None and cg is not None:
            agree = cq[0] == cg[0]
            if label == 1:
                pos_color_total += 1
                pos_color += int(agree)
            else:
                neg_color_total += 1
                neg_color += int(agree)
        if tq is not None and tg is not None:
            agree_t = tq[0] == tg[0]
            if label == 1:
                pos_type_total += 1
                pos_type += int(agree_t)
            else:
                neg_type_total += 1
                neg_type += int(agree_t)

    result: Dict[str, float] = {}
    if pos_color_total > 0:
        result["color_agreement_positives"] = pos_color / pos_color_total
    if neg_color_total > 0:
        result["color_agreement_negatives"] = neg_color / neg_color_total
    if pos_type_total > 0:
        result["type_agreement_positives"] = pos_type / pos_type_total
    if neg_type_total > 0:
        result["type_agreement_negatives"] = neg_type / neg_type_total
    return result if result else None


# --------------------------------------------------------------------------- #
# Bench driver
# --------------------------------------------------------------------------- #


def _bench_ir(
    name: str,
    ir_dir: Path,
    eval_idx: EvalIndex,
    pairs: List[Tuple[str, str, int]],
    pairs_root: Path,
    threshold: float,
) -> Optional[Dict]:
    """Run the full bench against a single IR. Returns None if IR missing."""
    if not (ir_dir / "model.xml").exists():
        logger.warning("[%s] IR not found at %s — skipping.", name, ir_dir)
        return None
    logger.info("[%s] Building matcher with IR %s", name, ir_dir)
    matcher = _build_matcher(ir_dir)

    t0 = time.perf_counter()
    feats = _extract_features(matcher, eval_idx.images)
    embed_s = time.perf_counter() - t0
    logger.info(
        "[%s] embedded %d images in %.2fs (%.1f ms/img)",
        name, feats.shape[0], embed_s,
        1000.0 * embed_s / max(1, feats.shape[0]),
    )

    labels = np.asarray(eval_idx.labels)
    sim = feats @ feats.T

    rank1 = _rank_k(sim, labels, 1)
    rank5 = _rank_k(sim, labels, 5)
    mAP = _mean_average_precision(sim, labels)
    mean_same, mean_diff, margin = _pair_cosine_stats(feats, labels)

    # Pairs CSV — compute per-pair cosine using the same features.
    path_to_row: Dict[str, int] = {
        p.relative_to(pairs_root).as_posix(): i for i, p in enumerate(eval_idx.images)
    }
    pair_scores: List[float] = []
    pair_labels: List[int] = []
    skipped = 0
    for qp, gp, label in pairs:
        qi = path_to_row.get(qp)
        gi = path_to_row.get(gp)
        if qi is None or gi is None:
            skipped += 1
            continue
        pair_scores.append(float(feats[qi] @ feats[gi]))
        pair_labels.append(label)
    pair_scores_np = np.asarray(pair_scores, dtype=np.float32)
    pair_labels_np = np.asarray(pair_labels, dtype=np.int32)

    auc = _roc_auc(pair_scores_np, pair_labels_np)
    precision, recall, tp, fp, fn, tn = _precision_recall_at_threshold(
        pair_scores_np, pair_labels_np, threshold,
    )

    return {
        "ir_dir": str(ir_dir),
        "n_eval": int(feats.shape[0]),
        "embed_time_s": round(embed_s, 3),
        "embed_ms_per_image": round(1000.0 * embed_s / max(1, feats.shape[0]), 2),
        "rank1": round(rank1, 4),
        "rank5": round(rank5, 4),
        "mAP": round(mAP, 4),
        "mean_same_cosine": round(mean_same, 4),
        "mean_diff_cosine": round(mean_diff, 4),
        "cosine_margin": round(margin, 4),
        "pairs": {
            "total": len(pair_labels),
            "positives": int((pair_labels_np == 1).sum()),
            "negatives": int((pair_labels_np == 0).sum()),
            "skipped_due_to_missing_paths": skipped,
            "roc_auc": round(auc, 4),
            "threshold_used": threshold,
            "precision_at_threshold": round(precision, 4),
            "recall_at_threshold": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
    }


def run_bench(
    eval_dir: Path,
    pairs_csv: Path,
    report_path: Path,
    baseline_ir: Path,
    finetuned_ir: Path,
    threshold: float,
    measure_cascade: bool = True,
) -> Dict:
    eval_idx = _scan_eval(eval_dir)
    pairs = _load_pairs(pairs_csv)
    pairs_root = pairs_csv.parent

    baseline = _bench_ir("baseline", baseline_ir, eval_idx, pairs, pairs_root, threshold)
    fine_tuned = _bench_ir("fine_tuned", finetuned_ir, eval_idx, pairs, pairs_root, threshold)

    cascade = None
    if measure_cascade:
        cascade = _measure_cascade_contribution(pairs, pairs_root)

    # Side-by-side comparison summary at the top of the report.
    summary = {
        "baseline_vs_fine_tuned": {
            metric: {
                "baseline": baseline[metric] if baseline else None,
                "fine_tuned": fine_tuned[metric] if fine_tuned else None,
                "delta": (
                    round(fine_tuned[metric] - baseline[metric], 4)
                    if (baseline is not None and fine_tuned is not None
                        and isinstance(baseline[metric], (int, float)))
                    else None
                ),
            }
            for metric in (
                "rank1", "rank5", "mAP",
                "cosine_margin",
                "embed_ms_per_image",
            )
            if baseline is not None and metric in baseline
        },
        "pair_roc_auc": {
            "baseline": baseline["pairs"]["roc_auc"] if baseline else None,
            "fine_tuned": fine_tuned["pairs"]["roc_auc"] if fine_tuned else None,
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "pairs_csv": str(pairs_csv),
        "n_eval_images": len(eval_idx.images),
        "n_plates": len(eval_idx.plate_names),
        "plate_names": eval_idx.plate_names,
        "threshold": threshold,
        "summary": summary,
        "baseline": baseline,
        "fine_tuned": fine_tuned,
        "cascade_contribution": cascade,
        "acceptance": _acceptance_verdict(baseline, fine_tuned),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Print the summary table to stdout so a quick run sees it.
    _print_summary(report)
    return report


def _acceptance_verdict(baseline, fine_tuned) -> Dict[str, object]:
    """T-5 acceptance: fine-tuned must beat baseline on at least 3 of 4 metrics."""
    if baseline is None or fine_tuned is None:
        return {
            "verdict": "incomplete",
            "reason": "missing baseline or fine-tuned IR",
            "metrics_beaten": 0,
            "metrics_total": 4,
        }
    keys = ("rank1", "rank5", "mAP", "cosine_margin")
    beats = 0
    deltas: Dict[str, float] = {}
    for k in keys:
        d = fine_tuned[k] - baseline[k]
        deltas[k] = round(d, 4)
        if d > 0:
            beats += 1
    return {
        "verdict": "pass" if beats >= 3 else "fail",
        "metrics_beaten": beats,
        "metrics_total": len(keys),
        "deltas": deltas,
    }


def _print_summary(report: Dict) -> None:
    s = report["summary"]["baseline_vs_fine_tuned"]
    print()
    print("=" * 78)
    print(f"Eval bench at {report['generated_at']}")
    print(
        f"Eval set: {report['n_eval_images']} images / {report['n_plates']} plates"
    )
    print("=" * 78)
    header = f"{'Metric':>22} | {'Baseline':>10} | {'Fine-Tuned':>10} | {'Delta':>10}"
    print(header)
    print("-" * len(header))
    for metric, vals in s.items():
        b = vals["baseline"]
        f_ = vals["fine_tuned"]
        d = vals["delta"]
        b_str = "n/a" if b is None else f"{b:.4f}" if isinstance(b, float) else str(b)
        f_str = "n/a" if f_ is None else f"{f_:.4f}" if isinstance(f_, float) else str(f_)
        d_str = "n/a" if d is None else f"{d:+.4f}"
        print(f"{metric:>22} | {b_str:>10} | {f_str:>10} | {d_str:>10}")
    print("=" * 78)
    print(f"Acceptance: {report['acceptance']['verdict']} "
          f"({report['acceptance']['metrics_beaten']}/"
          f"{report['acceptance']['metrics_total']} metrics improved)")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_bench(
        eval_dir=args.eval_dir,
        pairs_csv=args.pairs_csv,
        report_path=args.report_path,
        baseline_ir=args.baseline_ir,
        finetuned_ir=args.finetuned_ir,
        threshold=args.threshold,
        measure_cascade=not args.skip_cascade,
    )
    return 0


# --------------------------------------------------------------------------- #
# pytest entry point
# --------------------------------------------------------------------------- #


def test_facility_eval_smoke():
    """Smoke test: run the bench. Skips when the eval set hasn't been built."""
    import pytest

    if not DEFAULT_EVAL_DIR.exists():
        pytest.skip("eval set not built — run T-1/T-2 first")
    if not DEFAULT_PAIRS_CSV.exists():
        pytest.skip("pairs CSV not built — run T-2 first")
    if not (DEFAULT_BASELINE_IR / "model.xml").exists():
        pytest.skip("baseline IR missing")

    report = run_bench(
        eval_dir=DEFAULT_EVAL_DIR,
        pairs_csv=DEFAULT_PAIRS_CSV,
        report_path=DEFAULT_REPORT_PATH,
        baseline_ir=DEFAULT_BASELINE_IR,
        finetuned_ir=DEFAULT_FINETUNED_IR,
        threshold=DEFAULT_THRESHOLD,
        measure_cascade=False,
    )
    # Sanity: baseline section must exist and produce a real rank1.
    assert report["baseline"] is not None
    assert 0.0 <= report["baseline"]["rank1"] <= 1.0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
