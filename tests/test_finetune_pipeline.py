"""
tests/test_finetune_pipeline.py — Phase 3 / T3.1 mining pipeline tests.

Unit tests for ``tools/finetune_osnet_facility.py``:

* Filename parsing matches the registry's persistence convention.
* Positive-pair mining honours the same-burst skip threshold.
* Hard-negative mining picks up ``ocr_contradiction`` and
  ``reattach_below`` log entries.
* End-to-end ``mine_pairs`` integration on a tiny synthetic dataset.
* Graceful skip of the training step when torchreid is not installed.
* Smoke test that exercises the training entrypoint with a 1-epoch
  budget on a synthetic dataset when torchreid is available; otherwise
  the test ``skipTest`` so CI remains green.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Allow ``python -m pytest`` and ``python tests/...`` to find tools/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

finetune = importlib.import_module("tools.finetune_osnet_facility")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_blank_jpg(path: Path) -> None:
    """Write a 2-byte placeholder so ``_path_exists`` returns True.

    We don't need a real image for the mining tests — only the file's
    existence matters. The training smoke test creates real images
    separately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8")  # JPEG SOI marker; enough for existence.


def _make_real_jpg(path: Path, color: tuple = (127, 127, 127)) -> None:
    """Create a 32x32 BGR JPG so cv2.imread succeeds."""
    try:
        import cv2
        import numpy as np
    except Exception:
        _make_blank_jpg(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((32, 32, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #


class TestFilenameParsing(unittest.TestCase):
    def test_parse_standard_filename(self):
        result = finetune.parse_vehicle_image_filename(
            "ABC-123_CAM-03_20260505_122323.jpg"
        )
        self.assertIsNotNone(result)
        plate, camera, ts = result
        self.assertEqual(plate, "ABC-123")
        self.assertEqual(camera, "CAM-03")
        self.assertEqual(ts, _dt.datetime(2026, 5, 5, 12, 23, 23))

    def test_parse_rejects_unknown_pattern(self):
        # Non-conforming files (e.g. tooling-generated artifacts) must be
        # ignored without raising.
        self.assertIsNone(
            finetune.parse_vehicle_image_filename("random_blob.jpg")
        )

    def test_parse_rejects_invalid_timestamp(self):
        self.assertIsNone(
            finetune.parse_vehicle_image_filename(
                "ABC_CAM-03_20269999_999999.jpg"
            )
        )


# --------------------------------------------------------------------------- #
# index_vehicle_images
# --------------------------------------------------------------------------- #


class TestIndexVehicleImages(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vehicle_images_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_groups_by_plate_sorted_by_ts(self):
        _make_blank_jpg(self.tmp / "ABC-123_CAM-03_20260505_122323.jpg")
        _make_blank_jpg(self.tmp / "ABC-123_CAM-07_20260505_115634.jpg")
        _make_blank_jpg(self.tmp / "OTHER_CAM-12_20260505_125100.jpg")
        index = finetune.index_vehicle_images(self.tmp)
        self.assertIn("ABC-123", index)
        self.assertEqual(len(index["ABC-123"]), 2)
        # Sorted by timestamp ascending
        self.assertEqual(index["ABC-123"][0][1], "CAM-07")
        self.assertEqual(index["ABC-123"][1][1], "CAM-03")

    def test_skips_non_matching_files(self):
        _make_blank_jpg(self.tmp / "noise.jpg")
        _make_blank_jpg(self.tmp / "ABC_CAM-03_20260505_122323.jpg")
        index = finetune.index_vehicle_images(self.tmp)
        self.assertEqual(set(index.keys()), {"ABC"})

    def test_empty_dir_returns_empty(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        self.assertEqual(finetune.index_vehicle_images(empty), {})


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #


class TestLogParsing(unittest.TestCase):
    def test_parse_match_event_line(self):
        line = (
            "2026-05-13 12:00:00,000 - reid_match_perf - INFO - "
            "MATCH_EVENT | Plate: ABC-123 | Slot: SLOT-A | "
            "Time: 2026-05-13T12:00:00 | NewCost: 0.7345 | OldCost: 0.6201 | "
            "Flags: {'CLAHE': False} | GateSnapshots: ['snap_a.jpg'] | "
            "SlotSnapshot: snap_b.jpg"
        )
        rows = finetune.parse_match_event_lines(line)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.plate, "ABC-123")
        self.assertAlmostEqual(row.new_cost, 0.7345, places=4)
        self.assertEqual(row.gate_snapshots, ["snap_a.jpg"])
        self.assertEqual(row.slot_snapshot, "snap_b.jpg")

    def test_parse_rejection_ocr_contradiction(self):
        text = (
            "2026-05-13 ocr_contradiction | Plate: XYZ-999 | Score: 0.78\n"
            "2026-05-13 reattach_below | Plate: AAA-111 | Score: 0.45\n"
            "ignore me, not a rejection line\n"
        )
        rows = finetune.parse_reject_lines(text)
        self.assertEqual(len(rows), 2)
        kinds = {r[0] for r in rows}
        self.assertEqual(kinds, {"ocr_contradiction", "reattach_below"})

    def test_gate_snapshots_literal_eval_fallback(self):
        # The fallback splitter handles broken literal_eval cases.
        self.assertEqual(finetune._parse_gate_snapshots("[]"), [])
        self.assertEqual(
            finetune._parse_gate_snapshots("['a.jpg', 'b.jpg']"),
            ["a.jpg", "b.jpg"],
        )


# --------------------------------------------------------------------------- #
# Positive pair mining
# --------------------------------------------------------------------------- #


class TestMinePositivePairs(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="finetune_pos_"))
        self.img_dir = self.tmp / "images"
        self.img_dir.mkdir()
        # Two cameras, two timestamps apart by 5 minutes -> valid positive.
        self.early = self.img_dir / "ABC_CAM-03_20260505_120000.jpg"
        self.late = self.img_dir / "ABC_CAM-07_20260505_120500.jpg"
        # Same camera within 10s -> should be skipped (same burst).
        self.burst_a = self.img_dir / "ABC_CAM-03_20260505_130000.jpg"
        self.burst_b = self.img_dir / "ABC_CAM-03_20260505_130003.jpg"
        for p in (self.early, self.late, self.burst_a, self.burst_b):
            _make_blank_jpg(p)
        self.index = finetune.index_vehicle_images(self.img_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cross_camera_pair_is_positive(self):
        report = finetune.MiningReport()
        pairs = finetune.mine_positive_pairs(
            match_events=[],
            vehicle_image_index=self.index,
            db_truth={},
            report=report,
        )
        labels = [p.label for p in pairs]
        self.assertIn(True, labels)
        # The same-burst pair must have been skipped.
        self.assertGreater(report.skipped_same_burst, 0)
        # Cross-camera pair must be in the result.
        anchors = {p.anchor_path for p in pairs if p.label}
        self.assertTrue(
            any("CAM-03" in a or "CAM-07" in a for a in anchors),
            msg=f"expected a cross-camera positive, got anchors={anchors}",
        )

    def test_db_truth_filters_negative_plates(self):
        report = finetune.MiningReport()
        # DB says plate is False -> drop all positives for this plate.
        pairs = finetune.mine_positive_pairs(
            match_events=[],
            vehicle_image_index=self.index,
            db_truth={"ABC": False},
            report=report,
        )
        self.assertEqual([p for p in pairs if p.label], [])

    def test_match_event_path_produces_positive(self):
        # Build a synthetic MatchEventRow whose slot_snapshot and gate
        # snapshot files exist on disk and are > 60 s apart.
        gate_path = self.img_dir / "ABC_CAM-03_20260505_120000.jpg"
        slot_path = self.img_dir / "ABC_CAM-07_20260505_120500.jpg"
        ev = finetune.MatchEventRow(
            plate="ABC",
            slot="SLOT-A",
            timestamp=_dt.datetime(2026, 5, 5, 12, 5, 0),
            new_cost=0.85,
            old_cost=0.7,
            gate_snapshots=[str(gate_path)],
            slot_snapshot=str(slot_path),
        )
        report = finetune.MiningReport()
        pairs = finetune.mine_positive_pairs(
            match_events=[ev],
            vehicle_image_index={},
            db_truth={},
            report=report,
        )
        self.assertTrue(any(p.source == "match_event" for p in pairs))


# --------------------------------------------------------------------------- #
# Hard negative mining
# --------------------------------------------------------------------------- #


class TestMineHardNegatives(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="finetune_neg_"))
        img_dir = self.tmp / "images"
        img_dir.mkdir()
        self.plate_a = img_dir / "AAA_CAM-03_20260505_120000.jpg"
        self.plate_b = img_dir / "BBB_CAM-03_20260505_120000.jpg"
        _make_blank_jpg(self.plate_a)
        _make_blank_jpg(self.plate_b)
        self.index = finetune.index_vehicle_images(img_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ocr_contradiction_creates_negative_pair(self):
        report = finetune.MiningReport()
        pairs = finetune.mine_hard_negatives(
            rejections=[("ocr_contradiction", "AAA", 0.78)],
            match_events=[],
            vehicle_image_index=self.index,
            report=report,
        )
        # Pair AAA with a non-AAA image (BBB)
        self.assertTrue(any(p.source == "ocr_contradiction" for p in pairs))
        self.assertTrue(all(p.label is False for p in pairs))

    def test_reattach_below_in_band(self):
        report = finetune.MiningReport()
        pairs = finetune.mine_hard_negatives(
            rejections=[("reattach_below", "AAA", 0.45)],
            match_events=[],
            vehicle_image_index=self.index,
            report=report,
        )
        self.assertTrue(any(p.source == "reattach_marginal" for p in pairs))

    def test_reattach_below_below_band_skipped(self):
        # Score 0.1 is below low (0.40) — should NOT mine.
        report = finetune.MiningReport()
        pairs = finetune.mine_hard_negatives(
            rejections=[("reattach_below", "AAA", 0.1)],
            match_events=[],
            vehicle_image_index=self.index,
            report=report,
        )
        # The low-score reattach should be filtered out. The cross-plate
        # synthetic step still runs, but with no match events it produces
        # zero rows.
        self.assertEqual(
            [p for p in pairs if p.source == "reattach_marginal"], []
        )


# --------------------------------------------------------------------------- #
# Integration / end-to-end CLI smoke
# --------------------------------------------------------------------------- #


class TestMinePairsEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="finetune_e2e_"))
        self.img_dir = self.tmp / "vehicle_images"
        self.img_dir.mkdir()
        for plate in ("ABC", "DEF"):
            for cam_idx, cam in enumerate(("CAM-03", "CAM-07")):
                ts = _dt.datetime(2026, 5, 5, 12, cam_idx * 5, 0)
                p = (
                    self.img_dir
                    / f"{plate}_{cam}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                _make_blank_jpg(p)
        # Synthetic log
        log_dir = self.tmp / "logs"
        log_dir.mkdir()
        self.log_path = log_dir / "reid_match_perf.log"
        self.log_path.write_text(
            "2026-05-13 ocr_contradiction | Plate: ABC | Score: 0.78\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mine_pairs_returns_positives_and_negatives(self):
        pairs, report = finetune.mine_pairs(
            match_log_glob=str(self.log_path),
            vehicle_images_dir=self.img_dir,
            db_url=None,
        )
        self.assertGreater(report.positives, 0)
        self.assertGreater(report.negatives, 0)
        self.assertGreater(len(pairs), 0)
        labels = {p.label for p in pairs}
        self.assertEqual(labels, {True, False})

    def test_main_with_no_data_returns_nonzero(self):
        empty_dir = self.tmp / "empty"
        empty_dir.mkdir()
        rc = finetune.main(
            [
                "--match-log-glob",
                str(empty_dir / "*.log"),
                "--vehicle-images-dir",
                str(empty_dir),
                "--output-dir",
                str(self.tmp / "out"),
            ]
        )
        self.assertNotEqual(rc, 0)
        # Placeholder report must exist.
        self.assertTrue((self.tmp / "out" / "finetune_report.json").exists())

    def test_main_dry_run_with_data_returns_zero(self):
        rc = finetune.main(
            [
                "--match-log-glob",
                str(self.log_path),
                "--vehicle-images-dir",
                str(self.img_dir),
                "--output-dir",
                str(self.tmp / "out"),
                "--dry-run",
            ]
        )
        self.assertEqual(rc, 0)
        report_path = self.tmp / "out" / "finetune_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "skipped")
        self.assertEqual(payload["reason"], "dry-run")


# --------------------------------------------------------------------------- #
# Training smoke — runs only when torchreid is importable.
# --------------------------------------------------------------------------- #


class TestTrainingSmoke(unittest.TestCase):
    """One-epoch fine-tune smoke test.

    Skipped when torchreid (or torch) is not installed in the test env.
    The test is intentionally tiny — 1 epoch, 2 plates, 4 images — to keep
    CI under 10 s on CPU. It only asserts that a checkpoint file is
    produced, NOT that accuracy improves.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            import torchreid  # noqa: F401
        except Exception:
            raise unittest.SkipTest(
                "torch / torchreid not available — skipping training smoke."
            )

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="finetune_smoke_"))
        self.img_dir = self.tmp / "vehicle_images"
        self.img_dir.mkdir()
        # Two plates, two cameras each = 4 images.
        for plate, color in (("ABC", (255, 0, 0)), ("DEF", (0, 255, 0))):
            for cam_idx, cam in enumerate(("CAM-03", "CAM-07")):
                ts = _dt.datetime(2026, 5, 5, 12, cam_idx * 5, 0)
                _make_real_jpg(
                    self.img_dir
                    / f"{plate}_{cam}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg",
                    color=color,
                )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_epoch_fine_tune_produces_checkpoint(self):
        pairs, _ = finetune.mine_pairs(
            match_log_glob=str(self.tmp / "no-logs.log"),
            vehicle_images_dir=self.img_dir,
            db_url=None,
        )
        self.assertGreater(len(pairs), 0)
        out_dir = self.tmp / "out"
        ckpt = finetune.fine_tune(
            pairs,
            output_dir=out_dir,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            margin=0.3,
            input_size=(64, 32),  # tiny to keep the smoke under budget
            val_fraction=0.5,
        )
        self.assertIsNotNone(ckpt)
        self.assertTrue(ckpt.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
