"""
tests/test_matching_config_snapshot.py — Regression guard for Phase-0 thresholds.

Two snapshot tests live here:

  1. ``test_matching_config_defaults_snapshot``
     Pins ``MatchingConfig`` field defaults to a JSON fixture at
     ``tests/data/matching_config_snapshot.json``. Any future intentional
     change must update the fixture explicitly. This protects the values that
     were copied out of vehicle_registry_identity.py during the Phase 0
     consolidation.

  2. ``test_match_decision_verdict_snapshot``
     Feeds 18 representative ``(score, is_anpr, cross_camera, candidate_count)``
     tuples through ``MatchDecision.decide_b1/decide_global/decide_reattach``
     and snapshots the verdicts to
     ``tests/data/match_decision_verdict_snapshot.json``. This locks in the
     no-behaviour-change invariant promised by Phase 0 — if any future change
     shifts a verdict, the test fails and the developer must regenerate the
     snapshot deliberately.
"""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from pathlib import Path

from src.config import MatchingConfig
from src.matching import MatchDecision

DATA_DIR = Path(__file__).resolve().parent / "data"
CONFIG_SNAPSHOT_PATH = DATA_DIR / "matching_config_snapshot.json"
VERDICT_SNAPSHOT_PATH = DATA_DIR / "match_decision_verdict_snapshot.json"


def _matching_config_dict() -> dict:
    """Serialise default ``MatchingConfig`` to a JSON-stable dict.

    JSON has no tuple type, so we round-trip the dataclass dict through
    ``json.dumps``/``json.loads`` to coerce tuple fields (e.g.
    ``reid_input_size``) into the list form that ``json.load`` will
    re-hydrate from the snapshot file.
    """
    cfg = MatchingConfig()
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


# 18 representative inputs covering each fork of the three decide_* methods.
# The exact rows are chosen to exercise:
#   * b1: anpr/zone, single-candidate fallback (count==1), pool >1
#   * global: with-plate / without-plate, cross-camera / same-camera
#   * reattach: cross-camera / same-camera
SNAPSHOT_CASES = [
    # B1 — ANPR candidate, score above 0.47 -> confirm
    {"fn": "decide_b1", "args": {"score_reid": 0.50, "is_anpr_candidate": True, "candidate_count": 3}},
    # B1 — ANPR candidate, score equals threshold -> confirm
    {"fn": "decide_b1", "args": {"score_reid": 0.47, "is_anpr_candidate": True, "candidate_count": 3}},
    # B1 — ANPR candidate, score below threshold, pool >1 -> reject
    {"fn": "decide_b1", "args": {"score_reid": 0.30, "is_anpr_candidate": True, "candidate_count": 3}},
    # B1 — zone candidate, score above 0.55 -> confirm
    {"fn": "decide_b1", "args": {"score_reid": 0.60, "is_anpr_candidate": False, "candidate_count": 3}},
    # B1 — zone candidate, score equals threshold -> confirm
    {"fn": "decide_b1", "args": {"score_reid": 0.55, "is_anpr_candidate": False, "candidate_count": 3}},
    # B1 — zone candidate below threshold, pool >1 -> reject
    {"fn": "decide_b1", "args": {"score_reid": 0.50, "is_anpr_candidate": False, "candidate_count": 3}},
    # B1 — zone candidate below threshold, pool == 1 -> single-candidate fallback
    {"fn": "decide_b1", "args": {"score_reid": 0.00, "is_anpr_candidate": False, "candidate_count": 1}},
    # B1 — anpr candidate below threshold, pool == 1 -> single-candidate fallback
    {"fn": "decide_b1", "args": {"score_reid": 0.10, "is_anpr_candidate": True, "candidate_count": 1}},
    # Global — no plate, score above the default threshold (0.55) -> confirm
    {"fn": "decide_global", "args": {"score_reid": 0.60, "has_plate": False, "cross_camera": False}},
    # Global — no plate, below threshold -> reject
    {"fn": "decide_global", "args": {"score_reid": 0.50, "has_plate": False, "cross_camera": False}},
    # Global — plate, same-camera, threshold 0.46 -> confirm
    {"fn": "decide_global", "args": {"score_reid": 0.46, "has_plate": True, "cross_camera": False}},
    # Global — plate, same-camera, score 0.45 -> reject
    {"fn": "decide_global", "args": {"score_reid": 0.45, "has_plate": True, "cross_camera": False}},
    # Global — plate, cross-camera, threshold 0.43 -> confirm
    {"fn": "decide_global", "args": {"score_reid": 0.43, "has_plate": True, "cross_camera": True}},
    # Global — plate, cross-camera, 0.42 -> reject
    {"fn": "decide_global", "args": {"score_reid": 0.42, "has_plate": True, "cross_camera": True}},
    # Reattach — same-camera, default 0.52 -> confirm
    {"fn": "decide_reattach", "args": {"score_reid": 0.52, "cross_camera": False}},
    # Reattach — same-camera, below -> reject
    {"fn": "decide_reattach", "args": {"score_reid": 0.51, "cross_camera": False}},
    # Reattach — cross-camera, 0.43 -> confirm
    {"fn": "decide_reattach", "args": {"score_reid": 0.43, "cross_camera": True}},
    # Reattach — cross-camera, below 0.43 -> reject
    {"fn": "decide_reattach", "args": {"score_reid": 0.42, "cross_camera": True}},
]


def _compute_verdict_snapshot() -> list:
    """Run each case through ``MatchDecision`` and return the serialisable
    snapshot record."""
    decision = MatchDecision(MatchingConfig())
    records = []
    for case in SNAPSHOT_CASES:
        fn = getattr(decision, case["fn"])
        verdict = fn(**case["args"])
        records.append(
            {
                "case": case,
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                # Stable rounding so identical floats don't trip the snapshot
                # over JSON formatter quirks.
                "scores": {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in verdict.scores.items()
                },
            }
        )
    return records


class TestMatchingConfigSnapshot(unittest.TestCase):
    def test_matching_config_defaults_snapshot(self):
        """The MatchingConfig() default values must match the locked snapshot."""
        current = _matching_config_dict()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CONFIG_SNAPSHOT_PATH.exists() or os.environ.get(
            "REGENERATE_MATCHING_SNAPSHOT"
        ):
            with CONFIG_SNAPSHOT_PATH.open("w", encoding="utf-8") as fh:
                json.dump(current, fh, indent=2, sort_keys=True)
            self.skipTest(
                f"Wrote initial snapshot to {CONFIG_SNAPSHOT_PATH}; re-run."
            )
        with CONFIG_SNAPSHOT_PATH.open(encoding="utf-8") as fh:
            expected = json.load(fh)
        self.assertEqual(
            current,
            expected,
            msg=(
                "MatchingConfig defaults drifted from snapshot. If this is "
                "intentional, re-run with REGENERATE_MATCHING_SNAPSHOT=1."
            ),
        )

    def test_match_decision_verdict_snapshot(self):
        """``MatchDecision`` verdicts on representative inputs must be stable."""
        current = _compute_verdict_snapshot()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not VERDICT_SNAPSHOT_PATH.exists() or os.environ.get(
            "REGENERATE_MATCHING_SNAPSHOT"
        ):
            with VERDICT_SNAPSHOT_PATH.open("w", encoding="utf-8") as fh:
                json.dump(current, fh, indent=2, sort_keys=True)
            self.skipTest(
                f"Wrote initial verdict snapshot to {VERDICT_SNAPSHOT_PATH}; "
                "re-run."
            )
        with VERDICT_SNAPSHOT_PATH.open(encoding="utf-8") as fh:
            expected = json.load(fh)
        self.assertEqual(
            current,
            expected,
            msg=(
                "MatchDecision verdicts drifted from snapshot. If this is "
                "intentional, re-run with REGENERATE_MATCHING_SNAPSHOT=1."
            ),
        )


if __name__ == "__main__":
    unittest.main()
