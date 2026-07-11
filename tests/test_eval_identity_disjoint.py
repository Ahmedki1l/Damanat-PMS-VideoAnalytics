"""Unit tests for tools/eval_identity_disjoint.py — the D15 honest-eval harness.

These cover the parts that make the number trustworthy: the training-manifest
parse, the separator-normalising identity match, the held-out partition, and the
contamination guard. No model is loaded (the metric math is exercised by
tests/test_facility_match_accuracy.py)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.eval_identity_disjoint import (
    ContaminationError,
    assert_identity_disjoint,
    build_index,
    load_train_identities_from_splits,
    normalize_identity,
    partition_identities,
    scan_identities,
)


class TestNormalizeIdentity(unittest.TestCase):
    def test_separator_and_case_folded(self):
        # EEB_80 (folder) and EEB-80 (manifest) are the SAME car.
        self.assertEqual(normalize_identity("EEB_80"), normalize_identity("EEB-80"))
        self.assertEqual(normalize_identity("hbr_4920"), "HBR-4920")


class TestLoadTrainFromSplits(unittest.TestCase):
    def test_identity_is_parent_folder_of_train_paths(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "splits.json"
            p.write_text(json.dumps({
                "train": [
                    ["Cropped_Vehicles\\AAD_2560\\a.jpg", 0, 4],
                    ["Cropped_Vehicles\\AAD_2560\\b.jpg", 0, 0],
                    ["Cropped_Vehicles\\NDD_4141\\c.jpg", 1, 2],
                ],
                # query/gallery identities are the held-out TEST set and must
                # NOT be pulled in as training identities.
                "query": [["Cropped_Vehicles\\EEB_80\\q.jpg", 5, 0]],
                "gallery": [["Cropped_Vehicles\\EEB_80\\g.jpg", 5, 1]],
            }), encoding="utf-8")
            ids = load_train_identities_from_splits(p)
        self.assertEqual(ids, {"AAD_2560", "NDD_4141"})
        self.assertNotIn("EEB_80", ids)


class TestPartitionIdentities(unittest.TestCase):
    def _src(self):
        # ident -> fake file list (only the count matters here)
        return {
            "AAD_2560": [Path("x1.jpg"), Path("x2.jpg")],   # in training
            "EEB_80": [Path("y1.jpg"), Path("y2.jpg")],     # training, other separator
            "DJS-7842": [Path("z1.jpg"), Path("z2.jpg")],   # held-out, enough imgs
            "AAB_4183": [Path("t1.jpg")],                   # held-out but too thin
        }

    def test_training_excluded_across_separator(self):
        # Train set uses hyphens; source folder EEB_80 uses underscore — it must
        # still be recognised as training and excluded.
        part = partition_identities(self._src(), {"AAD-2560", "EEB-80"}, min_images=2)
        self.assertEqual(part.dropped["AAD_2560"], "in_training")
        self.assertEqual(part.dropped["EEB_80"], "in_training")
        self.assertEqual(part.dropped["AAB_4183"], "too_few_images")
        self.assertIn("DJS-7842", part.held_out)
        self.assertEqual(len(part.held_out), 1)


class TestContaminationGuard(unittest.TestCase):
    def test_raises_on_separator_variant_overlap(self):
        with self.assertRaises(ContaminationError):
            assert_identity_disjoint({"EEB_80", "DJS-7842"}, {"EEB-80"})

    def test_raises_on_empty_eval_set(self):
        with self.assertRaises(ContaminationError):
            assert_identity_disjoint(set(), {"AAD-2560"})

    def test_passes_when_clean(self):
        assert_identity_disjoint({"DJS-7842", "DZD-9488"}, {"AAD-2560", "EEB-80"})


class TestScanAndIndex(unittest.TestCase):
    def test_scan_and_labels_aligned(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for ident, n in (("CAR-A", 2), ("CAR-B", 3)):
                d = root / ident
                d.mkdir()
                for i in range(n):
                    (d / f"{i}.jpg").write_bytes(b"x")
            scanned = scan_identities(root)
            self.assertEqual(set(scanned), {"CAR-A", "CAR-B"})
            images, labels, names = build_index(scanned)
        self.assertEqual(len(images), 5)
        self.assertEqual(len(labels), 5)
        # Labels index into names and are grouped per identity.
        self.assertEqual({names[l] for l in labels}, {"CAR-A", "CAR-B"})


if __name__ == "__main__":
    unittest.main()
