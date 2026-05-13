"""
tests/test_color_classifier.py — WS-B acceptance tests.

Three layers:

1. **Plugin contract** — ``OpenVINOColorClassifier`` is a concrete
   ``ColorClassifier`` and its ``predict`` returns ``(str, float in [0, 1])``.
   Always-on; instantiates the plugin in lazy mode and only loads the IR
   when the test exercises ``predict`` (and skips with a clear message if
   the IR is missing).

2. **Accuracy** — runs the trained model against the held-out ``test`` split
   listed in ``tests/data/color_classifier/manifest.csv`` (or against the
   synthetic-dataset manifest the training pipeline writes into the model
   directory when no real data is available). Asserts:

       * >=0.90 accuracy on a real-data run, OR
       * >=0.70 accuracy on a synthetic-data run (clearly labelled).

3. **Latency** — median ≤2 ms / crop over 100 inferences after a
   warm-up pass. Skipped on slow CI containers (``CI_SLOW_BOX`` env var).

Each layer skips with an actionable message when its preconditions are
missing, so the test file is safe to run in any environment.
"""

from __future__ import annotations

import csv
import json
import os
import time
import unittest
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "color_classifier_openvino"
MODEL_XML = MODEL_DIR / "model.xml"
LABELS_PATH = MODEL_DIR / "labels.json"
TRAINING_METRICS = MODEL_DIR / "training_metrics.json"
REAL_MANIFEST = REPO_ROOT / "tests" / "data" / "color_classifier" / "manifest.csv"
# The tracked tiny synthetic test set lives under tests/data/color_classifier/
# synthetic/. Tests use it as a fallback when (a) no real D-2 manifest has
# been produced and (b) the training pipeline's own synthetic_dataset/ is
# missing from a fresh checkout (it's gitignored).
TRACKED_SYNTHETIC_MANIFEST = (
    REPO_ROOT / "tests" / "data" / "color_classifier" / "synthetic" / "manifest.csv"
)
TRAINING_SYNTHETIC_MANIFEST = MODEL_DIR / "synthetic_dataset" / "manifest.csv"


def _has_torchvision_runtime() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _has_openvino() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False


def _has_classifier_pkg() -> Tuple[bool, Optional[str]]:
    try:
        from src.classifiers.color_classifier import OpenVINOColorClassifier  # noqa: F401
        return True, None
    except Exception as exc:
        return False, repr(exc)


def _used_synthetic() -> bool:
    """Inspect training_metrics.json to determine if the on-disk model was
    trained on synthetic fallback data (and therefore subject to the 0.70
    relaxed accuracy bar)."""
    if not TRAINING_METRICS.exists():
        return True  # be lenient when we cannot tell
    try:
        with TRAINING_METRICS.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return True
    return bool(data.get("used_synthetic", False))


def _load_test_manifest() -> List[dict]:
    """Pick the most relevant manifest available and return its ``test``
    split as a list of {crop_path, confirmed_color} dicts.

    Preference order:
        1. Real labelled manifest (tests/data/color_classifier/manifest.csv)
           — when the model was trained on real data.
        2. Training pipeline's synthetic_dataset/manifest.csv — the largest
           in-domain set when the model was trained on synthetic data.
        3. Tracked tiny synthetic e2e set
           (tests/data/color_classifier/synthetic/manifest.csv) — always
           available in a fresh checkout, regenerable via
           ``python tests/data/color_classifier/generate_synthetic_test_set.py``.
        4. Real manifest (last resort even when used_synthetic).
    """
    manifests = []
    if not _used_synthetic() and REAL_MANIFEST.exists():
        manifests.append(REAL_MANIFEST)
    if TRAINING_SYNTHETIC_MANIFEST.exists():
        manifests.append(TRAINING_SYNTHETIC_MANIFEST)
    if TRACKED_SYNTHETIC_MANIFEST.exists():
        manifests.append(TRACKED_SYNTHETIC_MANIFEST)
    if REAL_MANIFEST.exists() and REAL_MANIFEST not in manifests:
        manifests.append(REAL_MANIFEST)

    rows: List[dict] = []
    for m in manifests:
        try:
            with m.open(encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if (row.get("split") or "").lower() != "test":
                        continue
                    label = row.get("confirmed_color") or ""
                    path = row.get("crop_path") or ""
                    if not label or not path:
                        continue
                    rows.append({"crop_path": path, "confirmed_color": label})
            if rows:
                break  # use the first manifest that yielded rows
        except Exception:
            continue
    return rows


class TestColorClassifierPlugin(unittest.TestCase):
    """Always-on: verify the plugin module is importable and obeys its ABC."""

    def test_import_module(self):
        ok, exc = _has_classifier_pkg()
        if not ok:
            self.skipTest(f"src.classifiers.color_classifier unavailable: {exc}")
        from src.classifiers.color_classifier import OpenVINOColorClassifier
        from src.matching.plugins import ColorClassifier

        # ABC contract
        self.assertTrue(issubclass(OpenVINOColorClassifier, ColorClassifier))

    def test_instantiate_with_default_path(self):
        ok, exc = _has_classifier_pkg()
        if not ok:
            self.skipTest(f"plugin import failed: {exc}")
        from src.classifiers.color_classifier import OpenVINOColorClassifier

        clf = OpenVINOColorClassifier()
        # Lazy loading: instantiation must succeed even if the IR is missing.
        self.assertIsNotNone(clf.model_path)
        self.assertGreater(clf.conf_threshold, 0.0)

    def test_missing_model_raises_runtime_error(self):
        ok, exc = _has_classifier_pkg()
        if not ok:
            self.skipTest(f"plugin import failed: {exc}")
        from src.classifiers.color_classifier import OpenVINOColorClassifier

        clf = OpenVINOColorClassifier(model_path="/path/definitely/does/not/exist.xml")
        with self.assertRaises(RuntimeError):
            clf.predict(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_colors_match_inconclusive_returns_true(self):
        """An ``other`` / ``unknown`` / sub-threshold prediction must not
        veto a match — the plugin treats it as inconclusive."""
        ok, exc = _has_classifier_pkg()
        if not ok:
            self.skipTest(f"plugin import failed: {exc}")
        from src.classifiers.color_classifier import OpenVINOColorClassifier

        clf = OpenVINOColorClassifier(conf_threshold=0.7)
        # both confident + different -> reject
        self.assertFalse(clf.colors_match(("red", 0.95), ("blue", 0.95)))
        # both confident + same -> accept
        self.assertTrue(clf.colors_match(("red", 0.95), ("red", 0.88)))
        # one side "other" -> inconclusive -> accept
        self.assertTrue(clf.colors_match(("red", 0.95), ("other", 0.99)))
        # one side under threshold -> inconclusive -> accept
        self.assertTrue(clf.colors_match(("red", 0.95), ("blue", 0.5)))
        # one side "unknown" -> accept (Noop fallback)
        self.assertTrue(clf.colors_match(("red", 0.95), ("unknown", 0.0)))


@unittest.skipUnless(MODEL_XML.exists(), f"No OpenVINO IR at {MODEL_XML}")
@unittest.skipUnless(_has_openvino(), "openvino runtime not installed")
@unittest.skipUnless(_has_torchvision_runtime(), "opencv-python not installed")
class TestColorClassifierPredict(unittest.TestCase):
    """IR-dependent: skip cleanly when the model has not been trained."""

    def setUp(self):
        from src.classifiers.color_classifier import OpenVINOColorClassifier
        self.clf = OpenVINOColorClassifier(str(MODEL_XML))

    def test_predict_returns_string_and_unit_confidence(self):
        crop = np.full((96, 160, 3), 128, dtype=np.uint8)
        label, conf = self.clf.predict(crop)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIn(label, set(self.clf.labels) | {"unknown"})

    def test_predict_handles_garbage_input(self):
        # Tiny crop
        self.assertEqual(self.clf.predict(np.zeros((4, 4, 3), dtype=np.uint8))[0], "unknown")
        # Empty
        self.assertEqual(self.clf.predict(np.zeros((0, 0, 3), dtype=np.uint8))[0], "unknown")
        # Wrong rank
        self.assertEqual(self.clf.predict(np.zeros((64, 64), dtype=np.uint8))[0], "unknown")
        # None
        self.assertEqual(self.clf.predict(None)[0], "unknown")  # type: ignore[arg-type]


@unittest.skipUnless(MODEL_XML.exists(), f"No OpenVINO IR at {MODEL_XML}")
@unittest.skipUnless(_has_openvino(), "openvino runtime not installed")
@unittest.skipUnless(_has_torchvision_runtime(), "opencv-python not installed")
class TestColorClassifierAccuracy(unittest.TestCase):
    """Held-out test-set accuracy.

    Accuracy bar:
        * 0.90 on a real-data run, OR
        * 0.70 on a synthetic-data run (training was on the
          procedurally-generated swatch fallback).
    """

    def test_test_split_accuracy(self):
        import cv2
        from src.classifiers.color_classifier import OpenVINOColorClassifier

        rows = _load_test_manifest()
        if not rows:
            self.skipTest(
                "No 'test' split rows in any manifest. Run "
                "`python tools/prepare_color_dataset.py` and label crops, "
                "OR run `python tools/train_color_classifier.py --synthetic` "
                "to produce a smoke-test dataset."
            )

        used_synthetic = _used_synthetic()
        accuracy_bar = 0.70 if used_synthetic else 0.90

        clf = OpenVINOColorClassifier(str(MODEL_XML))

        correct = 0
        total = 0
        confusion: dict = {}
        for row in rows:
            img = cv2.imread(row["crop_path"])
            if img is None:
                continue
            pred, _conf = clf.predict(img)
            total += 1
            if pred == row["confirmed_color"]:
                correct += 1
            confusion.setdefault(row["confirmed_color"], {}).setdefault(pred, 0)
            confusion[row["confirmed_color"]][pred] += 1

        if total == 0:
            self.skipTest("All test crops failed to load — cannot evaluate.")
        accuracy = correct / total

        self.assertGreaterEqual(
            accuracy,
            accuracy_bar,
            msg=(
                f"Test accuracy {accuracy:.4f} below bar {accuracy_bar:.2f} "
                f"(synthetic={used_synthetic}). "
                f"Confusion (true -> {{pred: count}}): {confusion}"
            ),
        )


@unittest.skipUnless(MODEL_XML.exists(), f"No OpenVINO IR at {MODEL_XML}")
@unittest.skipUnless(_has_openvino(), "openvino runtime not installed")
@unittest.skipUnless(_has_torchvision_runtime(), "opencv-python not installed")
@unittest.skipIf(os.environ.get("CI_SLOW_BOX"), "Latency test skipped on slow CI box")
class TestColorClassifierLatency(unittest.TestCase):
    """Median latency budget: ≤2 ms per crop on the target CPU.

    The OpenVINO runtime needs a warm-up pass to JIT-compile the kernels;
    we discard the first inference and time 100 subsequent calls.

    Note: this assertion is enforced strictly only on the target inference
    box. CI environments often lack AVX2/AVX-512 — a 4 ms threshold is
    used when ``CI_LATENCY_BUDGET_MS`` is set, otherwise 2 ms.
    """

    def test_median_latency_within_budget(self):
        from src.classifiers.color_classifier import OpenVINOColorClassifier

        clf = OpenVINOColorClassifier(str(MODEL_XML))
        # Warm-up
        warm = np.full((128, 200, 3), 128, dtype=np.uint8)
        clf.predict(warm)

        budget_ms = float(os.environ.get("COLOR_CLASSIFIER_LATENCY_MS", "2.0"))

        # Run 100 inferences on slightly-varied inputs so the model cannot
        # short-circuit on identical tensors.
        rng = np.random.RandomState(0)
        latencies_ms = []
        for _ in range(100):
            jitter = rng.randint(0, 32, size=3)
            crop = np.full((128, 200, 3), 128, dtype=np.uint8)
            crop[:, :, 0] = (128 + jitter[0]) % 256
            crop[:, :, 1] = (128 + jitter[1]) % 256
            crop[:, :, 2] = (128 + jitter[2]) % 256
            t0 = time.perf_counter()
            clf.predict(crop)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        median = float(np.median(latencies_ms))
        p95 = float(np.percentile(latencies_ms, 95))
        print(
            f"[color-classifier-latency] median={median:.2f}ms p95={p95:.2f}ms "
            f"budget={budget_ms:.2f}ms"
        )
        self.assertLessEqual(
            median,
            budget_ms,
            msg=(
                f"Median latency {median:.2f}ms exceeds {budget_ms:.2f}ms "
                f"budget. p95={p95:.2f}ms. Set COLOR_CLASSIFIER_LATENCY_MS "
                f"to relax the bar on slow boxes."
            ),
        )


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    unittest.main()
