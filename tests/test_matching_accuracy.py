"""
tests/test_matching_accuracy.py — Phase 2 / T2.4 end-to-end accuracy bench.

When a labelled pairs.csv exists at ``tests/data/matching_eval/pairs.csv``
this test runs the matching cascade against each pair, asserts the
acceptance bar (>=0.90 accuracy, >=0.92 precision, >=0.88 recall,
<=150 ms median per-match latency), and reports the full confusion matrix
on success.

CSV schema
~~~~~~~~~~

The CSV must have a header row with at least these columns:

  * ``entry_id``  — unique id of the entry/gate crop (filename without ext)
  * ``floor_id``  — unique id of the floor/in-zone crop
  * ``plate``     — labelled plate (may be empty for negative pairs)
  * ``is_match``  — 1 if (entry_id, floor_id) are the same vehicle, else 0

Optional columns ``entry_crop`` and ``floor_crop`` may override the default
file resolution (``entry_crops/{entry_id}.jpg`` and
``floor_crops/{floor_id}.jpg`` relative to the matching_eval directory).

When ``pairs.csv`` has only a header or is missing, the test skips with a
clear message pointing at ``tools/build_test_set.py`` (the labelling
helper produced in Phase 0 / T0.6).
"""

from __future__ import annotations

import csv
import os
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

# Standard pytest-style skip semantics aren't available without pytest, so
# unittest is used here. The acceptance harness is invoked through pytest
# at the top level which respects unittest.skipTest.

DATA_DIR = Path(__file__).resolve().parent / "data" / "matching_eval"
PAIRS_CSV = DATA_DIR / "pairs.csv"


def _load_pairs() -> List[dict]:
    """Return a list of dict rows from pairs.csv (empty list if no data)."""
    if not PAIRS_CSV.exists():
        return []
    with PAIRS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    # Drop rows missing the `is_match` label (still awaiting human review).
    return [row for row in rows if (row.get("is_match") or "").strip() != ""]


def _resolve_crop_path(row: dict, kind: str) -> Optional[Path]:
    """Resolve the path to the entry/floor crop for a row.

    ``kind`` is ``"entry"`` or ``"floor"``. Honours an explicit
    ``entry_crop`` / ``floor_crop`` column if present; otherwise falls back
    to ``<DATA_DIR>/<kind>_crops/<id>.jpg``.
    """
    explicit = (row.get(f"{kind}_crop") or "").strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = DATA_DIR / p
        return p if p.exists() else None
    crop_id = (row.get(f"{kind}_id") or "").strip()
    if not crop_id:
        return None
    for ext in (".jpg", ".png", ".jpeg"):
        p = DATA_DIR / f"{kind}_crops" / f"{crop_id}{ext}"
        if p.exists():
            return p
    return None


def _load_bgr(path: Path):
    import cv2  # imported lazily so the skip path stays light

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return img


class TestMatchingAccuracy(unittest.TestCase):
    """Run the real cascade over the labelled pair set and assert the bar."""

    def setUp(self):
        self._pairs = _load_pairs()

    def test_accuracy_bench(self):
        if not self._pairs:
            self.skipTest(
                "Awaiting labeled test set — see tools/build_test_set.py to "
                "produce candidate pairs, then add an `is_match` column to "
                "tests/data/matching_eval/pairs.csv (0/1)."
            )

        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"Required heavy deps unavailable: {exc!r}")

        # Build the cascade — VehicleRegistry's plugin auto-instantiation
        # (Phase 2 / T2.1) will pick up real classifiers when their IRs
        # exist, otherwise fall back to Noop plugins.
        try:
            from src.config import MatchingConfig
            from src.matching import MatchDecision
            from src.vehicle_registry.vehicle_registry import VehicleRegistry
            from src.reid_matcher import VehicleReIDMatcher
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"Failed to import cascade: {exc!r}")
            return

        try:
            reid = VehicleReIDMatcher()
        except Exception as exc:
            self.skipTest(f"ReID matcher unavailable: {exc!r}")
            return

        registry = VehicleRegistry.__new__(VehicleRegistry)
        registry._matching_config = MatchingConfig()
        registry._match_decision = VehicleRegistry._build_match_decision(
            registry._matching_config,
            lambda: None,
        )
        decision: MatchDecision = registry._match_decision

        tp = fp = tn = fn = 0
        latencies_ms: List[float] = []

        for row in self._pairs:
            expected = (row["is_match"] or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            entry_path = _resolve_crop_path(row, "entry")
            floor_path = _resolve_crop_path(row, "floor")
            if entry_path is None or floor_path is None:
                # Skip rows whose crops are not on disk — the labeller may
                # still be downloading them.
                continue

            entry_img = _load_bgr(entry_path)
            floor_img = _load_bgr(floor_path)

            t0 = time.perf_counter()
            entry_feat = reid.extract_feature(entry_img)
            floor_feat = reid.extract_feature(floor_img)
            score = (
                reid.compute_similarity(entry_feat, floor_feat)
                if entry_feat is not None and floor_feat is not None
                else 0.0
            )
            modalities = decision.check_modalities(
                query_crop=floor_img,
                candidate_crop=entry_img,
                candidate=None,
                pending_plate=(row.get("plate") or "").strip() or None,
                score_reid=score,
            )
            verdict = decision.decide_global(
                score,
                has_plate=bool((row.get("plate") or "").strip()),
                cross_camera=False,
                modalities=modalities,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            predicted = verdict.verdict == "confirm"

            if expected and predicted:
                tp += 1
            elif expected and not predicted:
                fn += 1
            elif not expected and predicted:
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        self.assertGreater(total, 0, "No pairs ran — check crop paths.")
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        median_latency = (
            sorted(latencies_ms)[len(latencies_ms) // 2] if latencies_ms else 0.0
        )

        print(
            f"\n[accuracy] total={total} tp={tp} fp={fp} tn={tn} fn={fn} "
            f"acc={accuracy:.3f} prec={precision:.3f} recall={recall:.3f} "
            f"median_lat={median_latency:.1f}ms"
        )

        self.assertGreaterEqual(
            accuracy,
            0.90,
            msg=f"accuracy={accuracy:.3f} below the 0.90 bar.",
        )
        self.assertGreaterEqual(
            precision,
            0.92,
            msg=f"precision={precision:.3f} below the 0.92 bar.",
        )
        self.assertGreaterEqual(
            recall,
            0.88,
            msg=f"recall={recall:.3f} below the 0.88 bar.",
        )
        self.assertLessEqual(
            median_latency,
            150.0,
            msg=f"median latency {median_latency:.1f} ms > 150 ms cap.",
        )


if __name__ == "__main__":
    unittest.main()
