import unittest
from unittest.mock import patch

import numpy as np

from src.reid_matcher import reid_matcher
from src.reid_matcher import VehicleReIDMatcher
from src.reid_matcher.reid_preprocessing import normalize_for_reid


class TestReIDPreprocessing(unittest.TestCase):
    def test_normalize_none_returns_none(self):
        self.assertIsNone(normalize_for_reid(None))

    def test_normalize_empty_returns_same_empty_image(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)

        result = normalize_for_reid(empty)

        self.assertIs(result, empty)
        self.assertEqual(result.size, 0)

    def test_normalize_valid_bgr_preserves_shape_and_dtype(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 30
        image[:, :, 1] = 80
        image[:, :, 2] = 160

        result = normalize_for_reid(image)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)

    def test_normalize_does_not_modify_input_in_place(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 30
        image[:, :, 1] = 80
        image[:, :, 2] = 160
        original = image.copy()

        normalize_for_reid(image)

        np.testing.assert_array_equal(image, original)

    def test_preprocess_skips_clahe_when_flag_disabled(self):
        matcher = self._make_lightweight_matcher()
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        with patch.object(reid_matcher, "USE_LAB_CLAHE", False), patch.object(
            reid_matcher,
            "normalize_for_reid",
            wraps=normalize_for_reid,
        ) as normalize_mock:
            tensor = matcher._preprocess(image)

        normalize_mock.assert_not_called()
        self.assertEqual(tuple(tensor.shape), (3, 128, 256))

    def test_preprocess_applies_clahe_when_flag_enabled(self):
        matcher = self._make_lightweight_matcher()
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        with patch.object(reid_matcher, "USE_LAB_CLAHE", True), patch.object(
            reid_matcher,
            "normalize_for_reid",
            wraps=normalize_for_reid,
        ) as normalize_mock:
            tensor = matcher._preprocess(image)

        normalize_mock.assert_called_once()
        self.assertEqual(tuple(tensor.shape), (3, 128, 256))

    @staticmethod
    def _make_lightweight_matcher():
        matcher = VehicleReIDMatcher.__new__(VehicleReIDMatcher)
        matcher.input_size = (128, 256)
        matcher.norm_mean = [0.485, 0.456, 0.406]
        matcher.norm_std = [0.229, 0.224, 0.225]
        return matcher


if __name__ == "__main__":
    unittest.main()
