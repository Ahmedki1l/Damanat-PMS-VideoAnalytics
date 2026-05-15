"""
tools/calibrate_thresholds_top20.py — Phase 0.5 threshold calibration.

Sweeps the cosine cut-point against the labelled eval pairs produced by T-2
and recommends production threshold values for ``MatchingConfig``. Unlike
``tools/calibrate_thresholds.py`` (which mines production ``MATCH_EVENT``
logs + the parking-sessions DB), this tool only needs the curated eval
split, so it can run today against any new ReID IR.

What it does
------------
1. Loads ``data/facility_top20/eval/`` and ``pairs_eval.csv``.
2. Builds embeddings with ``VehicleReIDMatcher`` pointed at the supplied IR
   (defaults to the fine-tuned ``models/osnet_facility_int8_256x128``).
3. Computes a cosine score per pair, then sweeps thresholds in
   ``[0.30, 0.85]`` in 0.01 steps. For each step it reports precision,
   recall, F1, F0.5 (precision-leaning) and F2 (recall-leaning).
4. Picks an operating point optimised on **F0.5** (we care more about
   precision: false confirms bind a plate to a wrong slot, which is the
   loud failure mode that gets escalated).
5. Derives the per-context ``MatchingConfig`` thresholds by applying the
   same fixed offsets the production config already uses. The offsets
   encode operational intent (ANPR crops are higher-quality → lower bar;
   cross-camera handoffs lose appearance → lower bar; solo confirm needs
   to be very confident → higher bar).
6. Emits a paste-ready ``matching:`` block alongside the existing values
   so an operator can compare and apply manually.

Offsets used (relative to the central ``b1_zone`` value)
--------------------------------------------------------
    b1_anpr:            -0.08
    b1_cross_camera:    -0.12
    global_default:      0.00
    global_with_plate:  -0.09
    global_cross_camera:-0.12
    reattach_default:   -0.03
    reattach_cross_camera: -0.12
    reid_solo_confirm:  +0.15
    ocr_marginal_low:   -0.13   (b1_zone - 0.13)
    ocr_marginal_high:  +0.02   (b1_zone + 0.02)

These match the operational gaps the current ``config.yaml`` already
encodes between thresholds; we keep the gaps and slide the whole envelope
to the new IR's cosine distribution.

Usage
-----
    python tools/calibrate_thresholds_top20.py
    python tools/calibrate_thresholds_top20.py --ir-dir models/osnet_facility_int8_256x128
    python tools/calibrate_thresholds_top20.py --metric f1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Repo-root sys.path so ``import src.*`` works when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("calibrate_thresholds_top20")


DEFAULT_EVAL_DIR = Path("data/facility_top20/eval")
DEFAULT_PAIRS_CSV = Path("data/facility_top20/pairs_eval.csv")
DEFAULT_REPORT_PATH = Path("data/facility_top20/threshold_calibration.json")
DEFAULT_IR_DIR = Path("models/osnet_facility_int8_256x128")

# Fixed offsets from the central b1_zone threshold. Keep the production
# gaps so operational meaning is preserved across calibration runs.
THRESHOLD_OFFSETS: Dict[str, float] = {
    "b1_anpr":              -0.08,
    "b1_cross_camera":      -0.12,
    "global_default":        0.00,
    "global_with_plate":    -0.09,
    "global_cross_camera":  -0.12,
    "reattach_default":     -0.03,
    "reattach_cross_camera":-0.12,
    "reid_solo_confirm":    +0.15,
    "ocr_marginal_low":     -0.13,
    "ocr_marginal_high":    +0.02,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calibrate_thresholds_top20",
        description=(
            "Sweep cosine thresholds on the curated eval pairs and emit a "
            "paste-ready MatchingConfig block."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    p.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    p.add_argument("--ir-dir", type=Path, default=DEFAULT_IR_DIR)
    p.add_argument("--metric", type=str, default="f0_5",
                   choices=["f1", "f0_5", "f2", "youden"],
                   help="Optimisation target. f0_5 favours precision, "
                        "f2 favours recall, youden = TPR-FPR.")
    p.add_argument("--threshold-min", type=float, default=0.30)
    p.add_argument("--threshold-max", type=float, default=0.85)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Pair scoring
# --------------------------------------------------------------------------- #


def _build_matcher(ir_dir: Path):
    from src.config import MatchingConfig, ReIDPreprocessingConfig
    from src.reid_matcher.reid_matcher import VehicleReIDMatcher

    cfg = MatchingConfig()
    cfg.use_openvino_reid = True
    cfg.reid_openvino_model_dir = str(ir_dir)
    cfg.reid_input_size = (256, 128)
    return VehicleReIDMatcher(
        matching_config=cfg,
        preprocessing_config=ReIDPreprocessingConfig(),
        backend="openvino",
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return v.astype(np.float32) / n
    return v.astype(np.float32)


def _score_pairs(
    matcher,
    pairs_csv: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) arrays from pairs_eval.csv."""
    root = pairs_csv.parent
    rows: List[Tuple[Path, Path, int]] = []
    with pairs_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((
                root / r["query_path"],
                root / r["gallery_path"],
                int(r["label"]),
            ))

    # Cache embeddings per file (each path appears many times in the CSV).
    cache: Dict[str, np.ndarray] = {}

    def _embed(p: Path) -> np.ndarray:
        s = str(p)
        if s in cache:
            return cache[s]
        img = cv2.imread(s)
        if img is None:
            cache[s] = np.zeros(512, dtype=np.float32)
            return cache[s]
        vec = matcher.extract_feature(img)
        if vec is None:
            cache[s] = np.zeros(512, dtype=np.float32)
        else:
            cache[s] = _normalize(vec)
        return cache[s]

    scores: List[float] = []
    labels: List[int] = []
    for qp, gp, label in rows:
        q = _embed(qp)
        g = _embed(gp)
        scores.append(float(q @ g))
        labels.append(label)
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Sweep + metric helpers
# --------------------------------------------------------------------------- #


def _confusion_at(threshold: float, scores: np.ndarray, labels: np.ndarray):
    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn


def _metrics_at(threshold: float, scores: np.ndarray, labels: np.ndarray) -> Dict:
    tp, fp, fn, tn = _confusion_at(threshold, scores, labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    youden = tpr - fpr

    def _f_beta(beta_sq: float) -> float:
        denom = beta_sq * prec + rec
        if denom == 0:
            return 0.0
        return (1.0 + beta_sq) * prec * rec / denom

    f1 = _f_beta(1.0)
    f0_5 = _f_beta(0.25)   # beta=0.5 -> beta^2=0.25, precision-leaning
    f2 = _f_beta(4.0)      # beta=2.0 -> beta^2=4.0, recall-leaning
    return {
        "threshold": round(threshold, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "youden": round(youden, 4),
        "f1": round(f1, 4),
        "f0_5": round(f0_5, 4),
        "f2": round(f2, 4),
    }


def _sweep(
    scores: np.ndarray, labels: np.ndarray,
    lo: float, hi: float, step: float,
) -> List[Dict]:
    out: List[Dict] = []
    t = lo
    while t <= hi + 1e-9:
        out.append(_metrics_at(t, scores, labels))
        t += step
    return out


def _pick_best(sweep: List[Dict], metric_key: str) -> Dict:
    return max(sweep, key=lambda row: row[metric_key])


# --------------------------------------------------------------------------- #
# Threshold derivation + report
# --------------------------------------------------------------------------- #


def _derive_thresholds(b1_zone: float) -> Dict[str, float]:
    """Apply fixed offsets to derive the full MatchingConfig block."""
    out = {"b1_zone": round(b1_zone, 3)}
    for name, offset in THRESHOLD_OFFSETS.items():
        v = b1_zone + offset
        # Clamp to a reasonable range so a degenerate b1_zone doesn't
        # produce a negative threshold.
        v = max(0.05, min(0.95, v))
        out[name] = round(v, 3)
    return out


def _emit_yaml_block(derived: Dict[str, float]) -> str:
    lines = ["matching:"]
    order = (
        "b1_anpr", "b1_zone", "b1_cross_camera",
        "global_default", "global_with_plate", "global_cross_camera",
        "reattach_default", "reattach_cross_camera",
        "reid_solo_confirm",
        "ocr_marginal_low", "ocr_marginal_high",
    )
    for k in order:
        if k in derived:
            lines.append(f"  {k}: {derived[k]}")
    return "\n".join(lines)


def _print_summary(report: Dict, derived: Dict[str, float], current: Dict[str, float]) -> None:
    print()
    print("=" * 78)
    print(f"Threshold calibration (IR: {report['ir_dir']})")
    print(f"Eval pairs: {report['n_pairs']} ({report['n_positives']} pos, "
          f"{report['n_negatives']} neg)")
    print("=" * 78)
    op = report["best"]
    print(f"Best operating point on metric={report['metric']}:")
    print(f"  threshold     = {op['threshold']:.3f}")
    print(f"  precision     = {op['precision']:.4f}")
    print(f"  recall        = {op['recall']:.4f}")
    print(f"  F1 / F0.5/ F2 = {op['f1']:.4f} / {op['f0_5']:.4f} / {op['f2']:.4f}")
    print(f"  TPR / FPR     = {op['tpr']:.4f} / {op['fpr']:.4f}")
    print()
    print("Recommended thresholds vs current production values:")
    header = f"{'Key':>22} | {'Recommended':>11} | {'Current':>9} | {'Delta':>7}"
    print(header)
    print("-" * len(header))
    for k in sorted(derived.keys()):
        cur = current.get(k, None)
        rec = derived[k]
        if cur is None:
            print(f"{k:>22} | {rec:>11.3f} |     n/a |    n/a")
        else:
            delta = rec - cur
            print(f"{k:>22} | {rec:>11.3f} | {cur:>9.3f} | {delta:>+7.3f}")
    print()
    print("Paste-ready YAML block:")
    print(_emit_yaml_block(derived))
    print()


def _load_current_thresholds() -> Dict[str, float]:
    """Pull current production threshold values out of MatchingConfig."""
    from src.config import MatchingConfig
    cfg = MatchingConfig()
    out: Dict[str, float] = {}
    for k in (
        "b1_anpr", "b1_zone", "b1_cross_camera",
        "global_default", "global_with_plate", "global_cross_camera",
        "reattach_default", "reattach_cross_camera",
        "reid_solo_confirm",
        "ocr_marginal_low", "ocr_marginal_high",
    ):
        out[k] = float(getattr(cfg, k))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not (args.ir_dir / "model.xml").exists():
        logger.error("IR missing: %s", args.ir_dir)
        return 1
    if not args.pairs_csv.exists():
        logger.error("Pairs CSV missing: %s — run T-2 first.", args.pairs_csv)
        return 1

    logger.info("Building matcher with IR %s", args.ir_dir)
    matcher = _build_matcher(args.ir_dir)

    logger.info("Scoring pairs from %s ...", args.pairs_csv)
    scores, labels = _score_pairs(matcher, args.pairs_csv)
    if scores.size == 0:
        logger.error("No scored pairs — check the eval set.")
        return 1
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    logger.info("Scored %d pairs (%d pos, %d neg)", scores.size, n_pos, n_neg)

    sweep = _sweep(
        scores, labels,
        lo=args.threshold_min, hi=args.threshold_max, step=args.threshold_step,
    )
    best = _pick_best(sweep, args.metric)
    derived = _derive_thresholds(best["threshold"])
    current = _load_current_thresholds()

    # Same-vs-different distribution stats for quick sanity check.
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    score_stats = {
        "positives": {
            "n": int(pos.size),
            "mean": round(float(pos.mean()), 4) if pos.size else 0.0,
            "std":  round(float(pos.std()), 4) if pos.size else 0.0,
            "min":  round(float(pos.min()), 4) if pos.size else 0.0,
            "p10":  round(float(np.percentile(pos, 10)), 4) if pos.size else 0.0,
            "p50":  round(float(np.percentile(pos, 50)), 4) if pos.size else 0.0,
            "p90":  round(float(np.percentile(pos, 90)), 4) if pos.size else 0.0,
            "max":  round(float(pos.max()), 4) if pos.size else 0.0,
        },
        "negatives": {
            "n": int(neg.size),
            "mean": round(float(neg.mean()), 4) if neg.size else 0.0,
            "std":  round(float(neg.std()), 4) if neg.size else 0.0,
            "min":  round(float(neg.min()), 4) if neg.size else 0.0,
            "p10":  round(float(np.percentile(neg, 10)), 4) if neg.size else 0.0,
            "p50":  round(float(np.percentile(neg, 50)), 4) if neg.size else 0.0,
            "p90":  round(float(np.percentile(neg, 90)), 4) if neg.size else 0.0,
            "max":  round(float(neg.max()), 4) if neg.size else 0.0,
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ir_dir": str(args.ir_dir),
        "pairs_csv": str(args.pairs_csv),
        "n_pairs": int(scores.size),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "metric": args.metric,
        "score_distribution": score_stats,
        "best": best,
        "recommended_thresholds": derived,
        "current_thresholds": current,
        "deltas": {
            k: round(derived[k] - current[k], 4)
            for k in derived if k in current
        },
        "offset_table": THRESHOLD_OFFSETS,
        "yaml_block": _emit_yaml_block(derived),
        "sweep": sweep,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s", args.report_path)

    _print_summary(report, derived, current)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
