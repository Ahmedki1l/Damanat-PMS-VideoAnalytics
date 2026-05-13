"""
tools/test_ocr_on_crops.py — Plate-OCR benchmark / monitoring CLI.

Runs ``PaddlePlateOCR.read()`` against a directory of cropped vehicle images
and reports:

  * Per-call latency stats: median, mean, P95, count.
  * (Optional, when a ground-truth CSV is supplied) character-level
    accuracy and Levenshtein-≤1 match rate.

The same tool serves two roles:

  1. **Acceptance** during Wave-1 (WS-D): demonstrates the ≥0.90 character
     accuracy and ≤40 ms latency bar against a labelled crop set
     (Damanat data-track D-4 carve-out).
  2. **Production monitoring**: a nightly cron that scrapes recent floor
     crops from ``snapshots/``, runs OCR, and trips an alert when accuracy
     degrades or latency creeps up.

The report is emitted as JSON on stdout so it can be piped into
``jq`` / pushed to a metrics backend.

Usage
-----
::

    python tools/test_ocr_on_crops.py --input-dir tests/data/ocr_eval \\
        --ground-truth-csv tests/data/ocr_eval/plates.csv \\
        --max-crops 100

If no ground-truth CSV is supplied the tool runs in latency-only mode,
which is the typical case for ``vehicle_images/`` / ``snapshots/`` smoke
runs where the plate label isn't available crop-side.

Ground-truth CSV schema
-----------------------
::

    crop_path,plate
    relative/path/to/crop1.jpg,ABC1234
    relative/path/to/crop2.jpg,XYZ9876

``crop_path`` is resolved relative to ``--input-dir`` if not absolute.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the repo root is importable when the tool is invoked directly
# (``python tools/test_ocr_on_crops.py``). Without this ``from src.ocr ...``
# fails because Python only adds the script's own directory to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(input_dir: Path, max_crops: Optional[int]) -> List[Path]:
    """Walk ``input_dir`` recursively for supported image files."""
    if not input_dir.exists():
        return []
    crops: List[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        crops.append(p)
        if max_crops is not None and len(crops) >= max_crops:
            break
    return crops


def _load_ground_truth(
    csv_path: Optional[Path],
    input_dir: Path,
) -> Dict[Path, str]:
    """Return a ``{absolute_crop_path: plate}`` lookup from the CSV."""
    if csv_path is None or not csv_path.exists():
        return {}
    out: Dict[Path, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            crop = (row.get("crop_path") or "").strip()
            plate = (row.get("plate") or "").strip()
            if not crop or not plate:
                continue
            crop_path = Path(crop)
            if not crop_path.is_absolute():
                crop_path = (input_dir / crop_path).resolve()
            else:
                crop_path = crop_path.resolve()
            out[crop_path] = plate
    return out


def _char_accuracy(predicted: str, expected: str) -> float:
    """Levenshtein-based character accuracy: 1 - dist/max(len)."""
    from src.ocr.plate_ocr import levenshtein_distance, normalise_plate_text

    p = normalise_plate_text(predicted)
    e = normalise_plate_text(expected)
    if not e:
        return 0.0
    dist = levenshtein_distance(p, e)
    longer = max(len(p), len(e))
    if longer == 0:
        return 0.0
    return max(0.0, 1.0 - dist / longer)


def _percentile(values: List[float], pct: float) -> float:
    """Pure-Python percentile (no numpy dep here)."""
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(sv) - 1)
    if f == c:
        return sv[f]
    return sv[f] + (sv[c] - sv[f]) * (k - f)


def _build_report(
    crops: List[Path],
    gt: Dict[Path, str],
    timings_ms: List[float],
    readings: List[Tuple[Path, str, float]],
    warmup_ms: float,
) -> dict:
    """Assemble the JSON report dict."""
    report: dict = {
        "summary": {
            "n_crops": len(crops),
            "n_with_ground_truth": sum(
                1 for c in crops if c.resolve() in gt
            ),
            "warmup_ms": round(warmup_ms, 2),
        },
        "latency": {
            "count": len(timings_ms),
            "median_ms": round(statistics.median(timings_ms), 2) if timings_ms else 0,
            "mean_ms": round(statistics.mean(timings_ms), 2) if timings_ms else 0,
            "p95_ms": round(_percentile(timings_ms, 95), 2),
            "min_ms": round(min(timings_ms), 2) if timings_ms else 0,
            "max_ms": round(max(timings_ms), 2) if timings_ms else 0,
        },
    }

    if gt:
        from src.ocr.plate_ocr import plates_match

        labelled = [(c, t, conf) for c, t, conf in readings if c.resolve() in gt]
        char_accs = [
            _char_accuracy(t, gt[c.resolve()]) for c, t, conf in labelled
        ]
        exact_matches = sum(
            1
            for c, t, conf in labelled
            if plates_match(t, gt[c.resolve()], max_edit_distance=0)
        )
        edit1_matches = sum(
            1
            for c, t, conf in labelled
            if plates_match(t, gt[c.resolve()], max_edit_distance=1)
        )
        report["accuracy"] = {
            "n_labelled": len(labelled),
            "char_level_accuracy_mean": round(
                statistics.mean(char_accs), 4
            ) if char_accs else 0,
            "exact_match_rate": round(
                exact_matches / len(labelled), 4
            ) if labelled else 0,
            "edit_distance_1_match_rate": round(
                edit1_matches / len(labelled), 4
            ) if labelled else 0,
        }

    # Compact per-sample log — useful for debugging the worst readings.
    samples = []
    for c, t, conf in readings:
        rel = c.relative_to(c.anchor) if c.is_absolute() else c
        row = {
            "crop": str(rel),
            "ocr_text": t,
            "ocr_conf": round(conf, 4),
        }
        if c.resolve() in gt:
            row["expected"] = gt[c.resolve()]
            row["char_accuracy"] = round(_char_accuracy(t, gt[c.resolve()]), 4)
        samples.append(row)
    report["samples"] = samples

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and / or grade PaddlePlateOCR on a directory of "
            "vehicle crops. Emits a JSON report on stdout."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of cropped vehicle images (recursively scanned).",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with columns 'crop_path,plate'. "
            "When provided, the report includes character-level accuracy "
            "and Levenshtein-distance-1 match rate."
        ),
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=None,
        help="Stop after this many crops (useful for smoke runs).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="PaddleOCR recognition language (default: en).",
    )
    parser.add_argument(
        "--no-angle-cls",
        action="store_true",
        help="Disable the textline-orientation classifier (small speedup).",
    )
    parser.add_argument(
        "--rec-score-thresh",
        type=float,
        default=0.5,
        help="Drop per-line OCR detections below this confidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write the JSON report to this path as well as stdout.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    crops = _iter_images(input_dir, args.max_crops)
    if not crops:
        print(
            json.dumps(
                {
                    "error": f"No images found under {input_dir}",
                    "input_dir": str(input_dir),
                }
            )
        )
        return 1

    gt = _load_ground_truth(args.ground_truth_csv, input_dir)

    # Lazy import so --help works even when paddleocr isn't installed.
    try:
        from src.ocr.plate_ocr import PaddlePlateOCR
    except Exception as exc:  # pragma: no cover - exercised by missing dep
        print(json.dumps({"error": f"src.ocr import failed: {exc!r}"}))
        return 2

    try:
        ocr = PaddlePlateOCR(
            lang=args.lang,
            use_angle_cls=not args.no_angle_cls,
            rec_score_thresh=args.rec_score_thresh,
        )
    except ImportError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv always present
        print(json.dumps({"error": f"opencv-python required: {exc!r}"}))
        return 2

    # Warmup pass — first call pays the model-load cost (PaddleOCR lazy-loads
    # weights inside .predict). We do not include this in the latency stats.
    warmup_start = time.perf_counter()
    first_img = cv2.imread(str(crops[0]))
    if first_img is not None:
        ocr.read(first_img)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    timings_ms: List[float] = []
    readings: List[Tuple[Path, str, float]] = []

    for crop in crops:
        img = cv2.imread(str(crop))
        if img is None:
            continue
        t0 = time.perf_counter()
        text, conf = ocr.read(img)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        readings.append((crop, text, conf))

    report = _build_report(crops, gt, timings_ms, readings, warmup_ms)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")

    # Exit non-zero if a ground-truth pass falls short of the WS-D bar.
    if "accuracy" in report:
        acc = report["accuracy"]["char_level_accuracy_mean"]
        if acc < 0.90:
            print(
                f"\n[warn] char accuracy {acc:.3f} < 0.90 — see plan §WS-D",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
