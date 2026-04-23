import unittest

import cv2
import numpy as np

from src.reid_matcher.reid_attributes import color_compatible, dominant_color_hsv


class TestReIDAttributes(unittest.TestCase):
    def test_empty_and_none_hsv_extraction_returns_none(self):
        self.assertIsNone(dominant_color_hsv(None))
        self.assertIsNone(dominant_color_hsv(np.empty((0, 0, 3), dtype=np.uint8)))

    def test_similar_colors_pass(self):
        red_a = dominant_color_hsv(self._bgr(0, 0, 220))
        red_b = dominant_color_hsv(self._bgr(0, 0, 180))

        self.assertTrue(color_compatible(red_a, red_b))

    def test_clearly_different_colors_fail(self):
        red = dominant_color_hsv(self._bgr(0, 0, 220))
        blue = dominant_color_hsv(self._bgr(220, 0, 0))

        self.assertFalse(color_compatible(red, blue))

    def test_hue_wraparound_passes(self):
        self.assertTrue(color_compatible((2.0, 200.0, 200.0), (178.0, 200.0, 200.0)))

    def test_missing_hsv_inputs_are_compatible(self):
        self.assertTrue(color_compatible(None, (20.0, 100.0, 100.0)))
        self.assertTrue(color_compatible((20.0, 100.0, 100.0), None))

    @staticmethod
    def _bgr(blue, green, red):
        return np.full((32, 32, 3), (blue, green, red), dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
