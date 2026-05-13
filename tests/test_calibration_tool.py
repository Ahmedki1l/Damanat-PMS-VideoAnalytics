"""
tests.test_calibration_tool — Phase 2 / T2.3 calibration tool tests.

Happy-path coverage with a synthetic ``MATCH_EVENT`` log file. Exercises:

  * The audit-log parser
  * Threshold sweep + precision/recall math
  * Cosine drift detection
  * The recommended YAML block rendering

These tests do not require SQLAlchemy or PyYAML — both are exercised by the
tool's main() path but the analytic core is dependency-free.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow ``python -m pytest tests/...`` to find the tools/ module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import calibrate_thresholds as ct


# Minimal but realistic MATCH_EVENT log fragment. Mirrors the exact schema
# emitted by match_logger.info at vehicle_registry_identity.py:~1198.
_SAMPLE_LOG = (
    "2026-05-13 12:00:00,000 - reid_match_perf - INFO - MATCH_EVENT | "
    "Plate: ABC-001 | Slot: SLOT-A | Time: 2026-05-13T12:00:00 | "
    "NewCost: 0.7100 | OldCost: 0.6800 | "
    "Flags: {'CLAHE': False, 'MULTISHOT': False, 'COLOR_FILTER': False} | "
    "GateSnapshots: ['g1.jpg'] | SlotSnapshot: snap1.jpg\n"
    "2026-05-13 12:00:01,000 - reid_match_perf - INFO - MATCH_EVENT | "
    "Plate: DEF-002 | Slot: SLOT-B | Time: 2026-05-13T12:00:01 | "
    "NewCost: 0.4500 | OldCost: 0.4000 | "
    "Flags: {'CLAHE': False, 'MULTISHOT': False, 'COLOR_FILTER': False} | "
    "GateSnapshots: ['g2.jpg'] | SlotSnapshot: snap2.jpg\n"
    "2026-05-13 12:00:02,000 - reid_match_perf - INFO - MATCH_EVENT | "
    "Plate: ABC-001 | Slot: SLOT-A | Time: 2026-05-13T12:00:02 | "
    "NewCost: 0.6700 | OldCost: 0.6500 | "
    "Flags: {'CLAHE': True, 'MULTISHOT': False, 'COLOR_FILTER': False} | "
    "GateSnapshots: [] | SlotSnapshot: snap3.jpg\n"
    "noise line — not a MATCH_EVENT\n"
    "2026-05-13 12:00:03,000 - reid_match_perf - INFO - MATCH_EVENT | "
    "Plate: GHI-003 | Slot: SLOT-C | Time: 2026-05-13T12:00:03 | "
    "NewCost: 0.3200 | OldCost: 0.3100 | "
    "Flags: {} | GateSnapshots: [] | SlotSnapshot: snap4.jpg\n"
)


class TestCalibrationParser(unittest.TestCase):
    """The MATCH_EVENT parser tolerates noise lines and pulls out 4 events."""

    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        )
        self.tmp.write(_SAMPLE_LOG)
        self.tmp.flush()
        self.tmp.close()
        self.addCleanup(lambda: Path(self.tmp.name).unlink(missing_ok=True))

    def test_parse_log_files_extracts_match_events(self) -> None:
        events = ct.parse_log_files([Path(self.tmp.name)])
        plates = [ev.plate for ev in events]
        self.assertEqual(plates, ["ABC-001", "DEF-002", "ABC-001", "GHI-003"])
        # The "noise line" was dropped silently.
        self.assertEqual(len(events), 4)

    def test_parser_ignores_invalid_score_rows(self) -> None:
        bad_log = (
            "MATCH_EVENT | Plate: X | Slot: Y | Time: Z | NewCost: nan | "
            "OldCost: 0.1 | Flags: {} | GateSnapshots: [] | SlotSnapshot: a\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(bad_log)
            bad_path = fh.name
        try:
            events = ct.parse_log_files([Path(bad_path)])
            self.assertEqual(events, [])
        finally:
            Path(bad_path).unlink(missing_ok=True)


class TestCalibrationSweep(unittest.TestCase):
    """The threshold sweep produces a monotone curve and the recommender
    picks a sensible value."""

    def setUp(self) -> None:
        # Construct 100 synthetic events: 60 "true matches" with cosine in
        # [0.55, 0.85], 40 "non-matches" with cosine in [0.20, 0.45].
        self.events = []
        for i in range(60):
            score = 0.55 + (i % 30) * 0.01
            self.events.append(
                ct.MatchEvent(
                    plate=f"MATCH-{i:03d}",
                    slot=f"SLOT-{i:03d}",
                    time="t",
                    new_cost=score,
                    old_cost=score - 0.05,
                    flags="{}",
                    gates="[]",
                    slot_snapshot="snap.jpg",
                )
            )
        for i in range(40):
            score = 0.20 + (i % 25) * 0.01
            self.events.append(
                ct.MatchEvent(
                    plate=f"FALSE-{i:03d}",
                    slot=f"SLOT-F{i:03d}",
                    time="t",
                    new_cost=score,
                    old_cost=score + 0.05,
                    flags="{}",
                    gates="[]",
                    slot_snapshot="snap.jpg",
                )
            )

        # Ground truth — MATCH-* are positives, FALSE-* are negatives.
        self.truth = {
            ev.plate: ev.plate.startswith("MATCH-") for ev in self.events
        }

    def test_sweep_returns_one_point_per_threshold(self) -> None:
        points = ct.sweep_threshold(self.events, self.truth, low=0.30, high=0.80, step=0.05)
        self.assertEqual(len(points), 11)  # 0.30, 0.35, ..., 0.80
        # Confirm thresholds are strictly increasing.
        thresholds = [p.threshold for p in points]
        self.assertEqual(thresholds, sorted(thresholds))

    def test_recommender_picks_above_acceptance_bar(self) -> None:
        points = ct.sweep_threshold(self.events, self.truth)
        chosen = ct.recommend_threshold(points)
        # With the synthetic spread (positives 0.55-0.85, negatives <0.45)
        # the recommender should pick something in the separation band
        # — anything from ~0.44 (one sweep step below the highest negative)
        # up to ~0.55 (the lowest positive) classifies all events correctly.
        self.assertGreaterEqual(chosen, 0.42)
        self.assertLess(chosen, 0.60)

    def test_recommender_returns_default_on_empty_input(self) -> None:
        from src.config import MatchingConfig

        chosen = ct.recommend_threshold([])
        self.assertEqual(chosen, MatchingConfig.global_default)


class TestCalibrationDriftDetection(unittest.TestCase):
    """The cosine drift detector surfaces shifts >= COSINE_DRIFT_TRIGGER."""

    def _events_with_mean(self, mean: float, n: int = 20) -> list:
        return [
            ct.MatchEvent(
                plate=f"P{i}",
                slot="S",
                time="t",
                new_cost=mean,
                old_cost=mean,
                flags="{}",
                gates="[]",
                slot_snapshot="snap.jpg",
            )
            for i in range(n)
        ]

    def test_no_drift_returns_none(self) -> None:
        # Baseline is midpoint(b1_zone, b1_anpr) = 0.51
        events = self._events_with_mean(0.51)
        self.assertIsNone(ct.detect_cosine_drift(events))

    def test_drift_surfaces_recommendation(self) -> None:
        events = self._events_with_mean(0.45)  # -0.06 from 0.51
        drift = ct.detect_cosine_drift(events)
        self.assertIsNotNone(drift)
        self.assertLess(drift["delta"], 0)
        self.assertIn("uniform threshold shift", drift["recommendation"].lower())

    def test_empty_events_returns_none(self) -> None:
        self.assertIsNone(ct.detect_cosine_drift([]))


class TestCalibrationReport(unittest.TestCase):
    """End-to-end build_report happy path."""

    def test_build_report_packs_recommended_block_per_field(self) -> None:
        events = [
            ct.MatchEvent(
                plate=f"P{i}",
                slot="S",
                time="t",
                new_cost=0.55 + i * 0.001,
                old_cost=0.5,
                flags="{}",
                gates="[]",
                slot_snapshot="snap.jpg",
            )
            for i in range(20)
        ]
        truth = {ev.plate: True for ev in events}
        report = ct.build_report(events, truth)
        self.assertEqual(report["events_total"], 20)
        self.assertIn("recommended_block", report)
        self.assertEqual(
            set(report["recommended_block"].keys()),
            set(ct.CALIBRATABLE_FIELDS),
        )

    def test_render_yaml_block_is_well_formed(self) -> None:
        block = {name: 0.5 for name in ct.CALIBRATABLE_FIELDS}
        yaml = ct.render_yaml_block(block)
        self.assertTrue(yaml.startswith("matching:"))
        for name in ct.CALIBRATABLE_FIELDS:
            self.assertIn(f"{name}: 0.50", yaml)


class TestCalibrationCLI(unittest.TestCase):
    """`--help` and the no-log fallback both behave."""

    def test_help_works(self) -> None:
        parser = ct.build_parser()
        # argparse raises SystemExit(0) on --help. We just want to confirm
        # the parser is well-formed (no construction error).
        self.assertIsNotNone(parser.description)

    def test_main_with_empty_log_glob_emits_zero_event_report(self) -> None:
        # No log files match this glob → tool emits an empty report rather
        # than crashing. The output goes to stdout (no --output passed).
        from io import StringIO
        from contextlib import redirect_stdout, redirect_stderr

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = ct.main(
                [
                    "--log-glob",
                    "tests/.no-such-pattern-*.log",
                    "--quiet",
                ]
            )
        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["events_total"], 0)
        self.assertIn("recommended_block", payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
