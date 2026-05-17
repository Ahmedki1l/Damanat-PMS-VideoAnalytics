"""
Generate a tiny synthetic test set for the color-classifier accuracy test.

Owned by WS-B. Produces ~10 swatches per class under
``tests/data/color_classifier/synthetic/<label>/`` plus a manifest.csv with
every row marked as ``split=test``. Tracked in git so a fresh checkout's
``tests/test_color_classifier.py::TestColorClassifierAccuracy`` has data
to run against without first running the training pipeline.

Re-run only when the class set changes:

    python tests/data/color_classifier/generate_synthetic_test_set.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# Make the repo importable when the script is run as ``python <path>``.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from tools.train_color_classifier import (  # noqa: E402
    COLOR_CLASS_LABELS,
    _make_synthetic_swatch,
)


def main(out_root: Path = Path(__file__).resolve().parent / "synthetic", per_class: int = 10) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    rows = []
    for label in COLOR_CLASS_LABELS:
        cls_dir = out_root / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            h = rng.randint(80, 160)
            w = int(h * rng.uniform(1.3, 1.8))
            img = _make_synthetic_swatch(label, (h, w), rng)
            fname = f"{label}_{i:02d}.png"
            p = cls_dir / fname
            cv2.imwrite(str(p), img)
            rows.append(
                {
                    "crop_path": str(p).replace("\\", "/"),
                    "split": "test",
                    "confirmed_color": label,
                    "source": "synthetic_e2e",
                }
            )
    manifest = out_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["crop_path", "split", "confirmed_color", "source"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows -> {manifest}")


if __name__ == "__main__":
    main()
