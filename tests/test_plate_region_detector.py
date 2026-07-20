"""Plate-region detector + the LPD branch of read_slot_plate.

The contract this guards:

  * the LPD is OPTIONAL and NEVER FATAL — every failure path degrades to the
    historical full-vehicle-crop read rather than losing the read;
  * a tight plate crop is read with ``apply_hud_mask=False``. The HUD ratios are
    fractions of the input HEIGHT, so on a ~130x35 plate box 0.08+0.08 blanks the
    glyph rows themselves. Getting this wrong silently degrades exactly the reads
    the LPD exists to improve, which is why it is asserted explicitly.
"""

import os
import unittest
from typing import Optional

import numpy as np

from src.config import MatchingConfig
from src.matching.match_decision import MatchDecision
from src.matching.plugins import NoopPlateRegionDetector, PlateRegionDetector
from src.vehicle_registry.vehicle_registry_identity import VehicleRegistryIdentityMixin

MODEL_DIR = "models/yolo11n_lpd_openvino_model"


class _Shim(VehicleRegistryIdentityMixin):
    """Only what read_slot_plate touches."""

    def __init__(self, md):
        self._match_decision = md


class _RecordingOcr:
    """Captures the kwargs read_slot_plate passes, and the crop it was given."""

    def __init__(self, text="ABC1234", conf=0.9):
        self.calls = []
        self._text, self._conf = text, conf

    def read(self, crop_bgr, **kwargs):
        self.calls.append({"shape": crop_bgr.shape, **kwargs})
        return (self._text, self._conf)


class _StubLpd(PlateRegionDetector):
    def __init__(self, boxes=None, raises=False):
        self._boxes = boxes or []
        self._raises = raises

    def detect(self, bgr):
        if self._raises:
            raise RuntimeError("boom")
        return list(self._boxes)

    def crop_plate(self, bgr, *, pad_ratio: float = 0.15) -> Optional[np.ndarray]:
        if not self._boxes:
            return None
        x1, y1, x2, y2, _ = self._boxes[0]
        return bgr[int(y1):int(y2), int(x1):int(x2)]


def _shim(*, enabled, fallback=True, detector=None, ocr=None):
    cfg = MatchingConfig()
    cfg.slot_lpd_enabled = enabled
    cfg.slot_lpd_fallback_enabled = fallback
    ocr = ocr or _RecordingOcr()
    md = MatchDecision(cfg, plate_ocr=ocr, plate_detector=detector)
    return _Shim(md), ocr


def _frame(w=600, h=400):
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestReadSlotPlateLpdBranch(unittest.TestCase):
    def test_disabled_reads_full_crop_with_hud_masking(self):
        """Default path must be byte-identical to the historical behaviour."""
        shim, ocr = _shim(enabled=False)
        img = _frame()

        text, conf = shim.read_slot_plate(img, True, slot_id="B12 CCO")

        self.assertEqual(text, "ABC1234")
        self.assertEqual(len(ocr.calls), 1)
        call = ocr.calls[0]
        self.assertEqual(call["shape"], img.shape)          # full crop
        self.assertFalse(call["apply_plate_roi"])           # slot geometry
        self.assertTrue(call["apply_hud_mask"])             # recorder overlay masked

    def test_hit_feeds_plate_crop_and_disables_hud_mask(self):
        """THE regression this file exists for — see module docstring."""
        shim, ocr = _shim(
            enabled=True, detector=_StubLpd([(100.0, 200.0, 230.0, 235.0, 0.9)])
        )
        img = _frame()

        shim.read_slot_plate(img, True, slot_id="B12 CCO")

        call = ocr.calls[0]
        self.assertLess(call["shape"][0], img.shape[0])     # a crop, not the frame
        self.assertLess(call["shape"][1], img.shape[1])
        self.assertFalse(call["apply_hud_mask"])            # <-- the assertion
        self.assertFalse(call["apply_plate_roi"])

    def test_miss_falls_back_to_full_crop_with_hud_masking(self):
        shim, ocr = _shim(enabled=True, detector=_StubLpd([]))
        img = _frame()

        text, _ = shim.read_slot_plate(img, True, slot_id="B13 COO")

        self.assertEqual(text, "ABC1234")
        call = ocr.calls[0]
        self.assertEqual(call["shape"], img.shape)
        self.assertTrue(call["apply_hud_mask"])

    def test_miss_without_fallback_skips_ocr_entirely(self):
        shim, ocr = _shim(enabled=True, fallback=False, detector=_StubLpd([]))

        text, conf = shim.read_slot_plate(_frame(), True, slot_id="B13 COO")

        self.assertEqual((text, conf), ("", 0.0))
        self.assertEqual(ocr.calls, [])                     # OCR never invoked

    def test_detector_exception_degrades_to_full_crop(self):
        """A broken LPD must never cost us the read."""
        shim, ocr = _shim(enabled=True, detector=_StubLpd(raises=True))
        img = _frame()

        text, _ = shim.read_slot_plate(img, True, slot_id="B7_CHRO")

        self.assertEqual(text, "ABC1234")
        self.assertEqual(ocr.calls[0]["shape"], img.shape)

    def test_degenerate_box_degrades_to_full_crop(self):
        """A zero-area / inverted box must not produce an empty OCR input."""
        shim, ocr = _shim(
            enabled=True, detector=_StubLpd([(300.0, 200.0, 300.0, 200.0, 0.9)])
        )
        img = _frame()

        shim.read_slot_plate(img, True, slot_id="B7_CHRO")

        self.assertEqual(ocr.calls[0]["shape"], img.shape)
        self.assertTrue(ocr.calls[0]["apply_hud_mask"])

    def test_noop_detector_is_a_miss_not_a_crash(self):
        shim, ocr = _shim(enabled=True, detector=NoopPlateRegionDetector())
        img = _frame()

        text, _ = shim.read_slot_plate(img, True, slot_id="B12 CCO")

        self.assertEqual(text, "ABC1234")
        self.assertEqual(ocr.calls[0]["shape"], img.shape)

    def test_legacy_ocr_without_new_kwargs_still_works(self):
        """PlateOCR implementations predating apply_hud_mask must not break."""

        class _LegacyOcr:
            def __init__(self):
                self.calls = 0

            def read(self, crop_bgr, allow_retry=True, apply_plate_roi=True):
                self.calls += 1
                return ("LEGACY1", 0.5)

        legacy = _LegacyOcr()
        shim, _ = _shim(enabled=True, detector=_StubLpd([]), ocr=legacy)

        text, _ = shim.read_slot_plate(_frame(), True, slot_id="B9")

        self.assertEqual(text, "LEGACY1")
        self.assertGreaterEqual(legacy.calls, 1)


class TestNoopPlateRegionDetector(unittest.TestCase):
    def test_returns_nothing(self):
        d = NoopPlateRegionDetector()
        self.assertEqual(d.detect(_frame()), [])
        self.assertIsNone(d.crop_plate(_frame()))


@unittest.skipUnless(
    os.path.isdir(MODEL_DIR), f"{MODEL_DIR} not present in this checkout"
)
class TestOpenVINOPlateRegionDetector(unittest.TestCase):
    """Exercises the real IR when it is committed alongside the code."""

    @classmethod
    def setUpClass(cls):
        from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector

        cls.det = OpenVINOPlateRegionDetector(MODEL_DIR, num_threads=1)

    def test_bad_input_returns_empty_never_raises(self):
        self.assertEqual(self.det.detect(None), [])
        self.assertEqual(self.det.detect(np.zeros((0, 0, 3), np.uint8)), [])
        self.assertIsNone(self.det.crop_plate(None))

    def test_blank_frame_finds_no_plate(self):
        self.assertEqual(self.det.detect(_frame()), [])

    def test_boxes_are_inside_the_input(self):
        img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        for x1, y1, x2, y2, score in self.det.detect(img):
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, img.shape[1])
            self.assertLessEqual(y2, img.shape[0])
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(score, self.det.confidence)

    def test_missing_model_dir_raises_at_construction(self):
        from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector

        with self.assertRaises(FileNotFoundError):
            OpenVINOPlateRegionDetector("models/__does_not_exist__")


if __name__ == "__main__":
    unittest.main()
