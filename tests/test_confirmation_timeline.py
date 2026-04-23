import unittest

import numpy as np

from src.core.engine.engine_tracking import ParkingEngineTrackingMixin


class _FakeImageMatcher:
    @staticmethod
    def _compare_dominant_colors(image_a, image_b):
        mean_a = float(np.mean(image_a))
        mean_b = float(np.mean(image_b))
        delta = abs(mean_a - mean_b)
        return max(0.0, 1.0 - (delta / 255.0))


class _FakeRegistry:
    def __init__(self):
        self.matcher = _FakeImageMatcher()


class _TrackingHarness(ParkingEngineTrackingMixin):
    def __init__(self):
        self.vehicle_registry = _FakeRegistry()


class TestConfirmationTimeline(unittest.TestCase):
    def test_inconsistent_exit_view_is_dropped_from_gallery(self):
        tracker = _TrackingHarness()
        entry = np.full((20, 20, 3), 30, dtype=np.uint8)
        deep = np.full((20, 20, 3), 40, dtype=np.uint8)
        wrong_exit = np.full((20, 20, 3), 240, dtype=np.uint8)

        burst = {
            "entry_view": {"crop": entry},
            "deep_view": {"crop": deep},
            "exit_view": {"crop": wrong_exit},
        }

        images, primary_index = tracker._build_confirmation_timeline_gallery(
            burst,
            include_exit_view=True,
        )

        self.assertEqual(len(images), 2)
        self.assertEqual(primary_index, 1)
        np.testing.assert_array_equal(images[0], entry)
        np.testing.assert_array_equal(images[1], deep)

    def test_consistent_exit_view_is_kept_in_gallery(self):
        tracker = _TrackingHarness()
        entry = np.full((20, 20, 3), 30, dtype=np.uint8)
        deep = np.full((20, 20, 3), 45, dtype=np.uint8)
        exit_view = np.full((20, 20, 3), 35, dtype=np.uint8)

        burst = {
            "entry_view": {"crop": entry},
            "deep_view": {"crop": deep},
            "exit_view": {"crop": exit_view},
        }

        images, primary_index = tracker._build_confirmation_timeline_gallery(
            burst,
            include_exit_view=True,
        )

        self.assertEqual(len(images), 3)
        self.assertEqual(primary_index, 1)
        np.testing.assert_array_equal(images[0], entry)
        np.testing.assert_array_equal(images[1], deep)
        np.testing.assert_array_equal(images[2], exit_view)


if __name__ == "__main__":
    unittest.main()
