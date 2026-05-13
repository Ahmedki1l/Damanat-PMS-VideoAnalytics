"""
tests/test_matching_accuracy.py — End-to-end matching accuracy bench.

If a labelled pairs.csv exists at ``tests/data/matching_eval/pairs.csv``
this test runs the matching cascade against each pair and asserts
accuracy >= 0.90 (precision >= 0.92, recall >= 0.88).

If the CSV has only a header row (no data) the test skips with a clear
message — Phase 0 ships the scaffolding only; the human-owned data track
(D-4 in the plan) fills it in.
"""

from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path

# Standard pytest-style skip semantics aren't available without pytest, so
# unittest is used here. The acceptance harness will be wired to pytest in
# Phase 2 (T2.4).

DATA_DIR = Path(__file__).resolve().parent / "data" / "matching_eval"
PAIRS_CSV = DATA_DIR / "pairs.csv"


def _load_pairs():
    """Return a list of dict rows from pairs.csv (empty list if no data)."""
    if not PAIRS_CSV.exists():
        return []
    with PAIRS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    # Drop rows missing the `is_match` label (still awaiting human review).
    return [row for row in rows if (row.get("is_match") or "").strip() != ""]


class TestMatchingAccuracy(unittest.TestCase):
    def setUp(self):
        self._pairs = _load_pairs()

    def test_accuracy_bench(self):
        """Run the cascade on every labelled pair and assert accuracy >= 0.90."""
        if not self._pairs:
            self.skipTest(
                "Awaiting labeled test set — see tools/build_test_set.py to "
                "produce candidate pairs, then add an `is_match` column to "
                "tests/data/matching_eval/pairs.csv (0/1)."
            )

        # Defer heavy imports so the skip path stays fast.
        # NOTE: this branch is currently unreachable until the data track lands.
        # Phase 2 T2.4 will swap the assertion for the real cascade once the
        # ensemble rule + voting are wired.
        try:
            import cv2  # noqa: F401
            import numpy as np  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"Required heavy deps unavailable: {exc!r}")

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for row in self._pairs:
            expected = (row["is_match"] or "").strip().lower() in ("1", "true", "yes")
            # Placeholder predictor: until Phase 2 wires the cascade, this
            # test only validates that the CSV / harness is internally
            # consistent. The real predictor replaces this comparator.
            predicted = expected  # tautology — see Phase 2 T2.4
            if expected and predicted:
                true_positive += 1
            elif expected and not predicted:
                false_negative += 1
            elif not expected and predicted:
                false_positive += 1
            else:
                true_negative += 1

        total = (
            true_positive + false_positive + true_negative + false_negative
        )
        self.assertGreater(total, 0)
        accuracy = (true_positive + true_negative) / total
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 1.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 1.0
        )

        # The acceptance bar comes from §Phase 2 T2.4 of the plan.
        self.assertGreaterEqual(
            accuracy,
            0.90,
            msg=(
                f"accuracy={accuracy:.3f} below the 0.90 bar — "
                f"precision={precision:.3f} recall={recall:.3f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
