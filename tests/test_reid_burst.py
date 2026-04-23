import unittest

import cv2
import numpy as np

from src.reid_matcher.reid_burst import is_overexposed, select_best_frames


class TestReIDBurst(unittest.TestCase):
    def test_empty_and_invalid_crops_return_empty(self):
        self.assertEqual(select_best_frames([], top_k=3), [])
        self.assertEqual(select_best_frames([None, np.empty((0, 0, 3), dtype=np.uint8)]), [])

    def test_non_overexposed_sharp_crops_are_preferred(self):
        blurry = self._solid(80)
        sharp = self._checkerboard()

        selected = select_best_frames([blurry, sharp], top_k=1)

        self.assertIs(selected[0], sharp)

    def test_overexposed_crops_rejected_when_usable_exists(self):
        overexposed = self._solid(255)
        usable = self._checkerboard()

        selected = select_best_frames([overexposed, usable], top_k=2)

        self.assertEqual(selected, [usable])

    def test_all_overexposed_falls_back_to_sharpest_valid_crop(self):
        flat_overexposed = self._solid(255)
        sharp_overexposed = self._checkerboard(low=246, high=255)

        self.assertTrue(is_overexposed(flat_overexposed))
        self.assertTrue(is_overexposed(sharp_overexposed))

        selected = select_best_frames([flat_overexposed, sharp_overexposed], top_k=1)

        self.assertIs(selected[0], sharp_overexposed)

    @staticmethod
    def _solid(value):
        return np.full((32, 32, 3), value, dtype=np.uint8)

    @staticmethod
    def _checkerboard(low=0, high=220):
        gray = np.indices((32, 32)).sum(axis=0) % 2
        gray = np.where(gray == 0, low, high).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    unittest.main()
