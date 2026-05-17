"""
tools/verify_on_eval.py — Validate the fine-tuned ReID on the labelled eval split.

For N random images from ``data/facility_top20/eval/``:

1. Extract a 512-D embedding via the fine-tuned IR.
2. Score against per-plate gallery prototypes built from the **train** split
   (so the test never sees its own image in the gallery — clean
   train→eval generalisation check).
3. Report top-K predictions with cosines + the true plate.
4. Apply the production confirmation thresholds (default: post-Youden values
   ``b1_zone=0.66`` for "would commit?" and ``reid_solo_confirm=0.81`` for
   "would solo-confirm?") and show the verdict.
5. Stage source + crop + top-1 gallery exemplar in
   ``data/eval_verification/sample_NN/`` for visual diff.

Usage
-----
    python tools/verify_on_eval.py --num-samples 5
    python tools/verify_on_eval.py --num-samples 5 --seed 0 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("verify_on_eval")


DEFAULT_EVAL_DIR = Path("data/facility_top20/eval")
DEFAULT_TRAIN_DIR = Path("data/facility_top20/train")
DEFAULT_OUTPUT_ROOT = Path("data/eval_verification")
DEFAULT_IR = Path("models/osnet_facility_int8_256x128")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="verify_on_eval",
        description="Sanity-check fine-tuned ReID against the labelled eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--ir-dir", type=Path, default=DEFAULT_IR)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--b1-zone", type=float, default=0.66,
                   help="Ensemble confirmation threshold to apply.")
    p.add_argument("--reid-solo-confirm", type=float, default=0.81,
                   help="Solo-confirm threshold (ReID alone bypasses ensemble).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


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


def _build_gallery(
    matcher, gallery_root: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Path]]]:
    """Per-plate mean embedding from gallery_root, plus the raw image lists."""
    prototypes: Dict[str, np.ndarray] = {}
    images_by_plate: Dict[str, List[Path]] = {}
    for plate_dir in sorted(p for p in gallery_root.iterdir() if p.is_dir()):
        files = sorted(plate_dir.glob("*.jpg"))
        if not files:
            continue
        embeddings: List[np.ndarray] = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            v = matcher.extract_feature(img)
            if v is None:
                continue
            embeddings.append(_normalize(v))
        if embeddings:
            prototypes[plate_dir.name] = _normalize(
                np.mean(np.stack(embeddings, axis=0), axis=0)
            )
            images_by_plate[plate_dir.name] = files
    return prototypes, images_by_plate


def _verdict(top1_cosine: float, b1_zone: float, solo: float) -> str:
    if top1_cosine >= solo:
        return "would_solo_confirm"
    if top1_cosine >= b1_zone:
        return "would_ensemble_confirm"
    return "would_reject"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.eval_dir.exists():
        logger.error("Eval dir missing: %s", args.eval_dir)
        return 1
    if not args.train_dir.exists():
        logger.error("Train dir missing: %s", args.train_dir)
        return 1
    if not (args.ir_dir / "model.xml").exists():
        logger.error("IR missing: %s", args.ir_dir)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)

    # 1. Collect eval images (flat list with their true plate from parent dir)
    eval_pool: List[Tuple[Path, str]] = []
    for plate_dir in sorted(p for p in args.eval_dir.iterdir() if p.is_dir()):
        for f in sorted(plate_dir.glob("*.jpg")):
            eval_pool.append((f, plate_dir.name))
    if not eval_pool:
        logger.error("Empty eval pool.")
        return 1
    logger.info("Eval pool: %d images across %d plates.",
                len(eval_pool), len({lab for _, lab in eval_pool}))

    rng = random.Random(args.seed)
    samples = rng.sample(eval_pool, min(args.num_samples, len(eval_pool)))

    # 2. Build gallery from TRAIN split
    logger.info("Building matcher with IR %s", args.ir_dir)
    matcher = _build_matcher(args.ir_dir)
    logger.info("Building gallery from %s", args.train_dir)
    prototypes, images_by_plate = _build_gallery(matcher, args.train_dir)
    if not prototypes:
        logger.error("No gallery prototypes — aborting.")
        return 1
    plate_names = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[p] for p in plate_names], axis=0)

    # 3. Score each sample
    correct = 0
    total = 0
    report: List[Dict] = []
    for sample_idx, (src, true_plate) in enumerate(samples, start=1):
        img = cv2.imread(str(src))
        if img is None:
            logger.warning("Unreadable: %s", src)
            continue
        vec = matcher.extract_feature(img)
        if vec is None:
            logger.warning("Embedding failed: %s", src)
            continue
        vec_n = _normalize(vec)
        sims = proto_matrix @ vec_n
        order = np.argsort(-sims)

        top_k = []
        for r_i in order[: args.top_k]:
            plate = plate_names[int(r_i)]
            top_k.append({"plate": plate, "cosine": float(sims[int(r_i)])})

        top1 = top_k[0]
        is_correct = (top1["plate"] == true_plate)
        if is_correct:
            correct += 1
        total += 1
        margin = top_k[0]["cosine"] - top_k[1]["cosine"] if len(top_k) >= 2 else 0.0
        verdict = _verdict(top1["cosine"], args.b1_zone, args.reid_solo_confirm)

        # Stage artifacts
        sample_dir = args.output_root / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(sample_dir / f"00_query__truth_{true_plate}__{src.name}"))
        for rank_i, entry in enumerate(top_k, start=1):
            ex_list = images_by_plate.get(entry["plate"], [])
            if ex_list:
                ex = ex_list[0]
                tag = "CORRECT" if entry["plate"] == true_plate else "wrong"
                dst = sample_dir / f"02_rank{rank_i}_{tag}_{entry['plate']}_{ex.name}"
                shutil.copyfile(str(ex), str(dst))

        print(f"\nSample {sample_idx}: query={src.name}")
        print(f"  TRUE plate: {true_plate}")
        print(f"  Top-{args.top_k} predictions:")
        for rank_i, entry in enumerate(top_k, start=1):
            tag = " <-- correct" if entry["plate"] == true_plate else ""
            print(f"    #{rank_i}: {entry['plate']:>10}   cosine={entry['cosine']:+.4f}{tag}")
        print(f"  Top1 prediction:  {top1['plate']}  (cosine={top1['cosine']:.4f})")
        print(f"  Top1-top2 margin: {margin:+.4f}")
        print(f"  Rank-1 verdict:   {'CORRECT' if is_correct else 'WRONG'}")
        print(f"  Threshold action: {verdict}  "
              f"(b1_zone={args.b1_zone:.2f}, solo={args.reid_solo_confirm:.2f})")

        report.append({
            "sample_id": sample_idx,
            "query": src.name,
            "true_plate": true_plate,
            "top_k": top_k,
            "top1_minus_top2": round(margin, 4),
            "rank1_correct": is_correct,
            "threshold_action": verdict,
            "b1_zone": args.b1_zone,
            "reid_solo_confirm": args.reid_solo_confirm,
            "artifact_dir": str(sample_dir),
        })

    if total > 0:
        print(f"\nSummary: rank-1 correct on {correct}/{total} sampled "
              f"({100.0*correct/total:.1f}%)")

    out_report = {
        "sampled": total,
        "rank1_correct": correct,
        "rank1_accuracy": (correct / total) if total else 0.0,
        "b1_zone": args.b1_zone,
        "reid_solo_confirm": args.reid_solo_confirm,
        "seed": args.seed,
        "details": report,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
