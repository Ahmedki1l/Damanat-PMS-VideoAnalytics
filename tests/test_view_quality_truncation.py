"""
A partial car must not become a ReID reference.

The old view-quality gate could not see truncation. Its "edge" term only ever docked
the width of a 1% margin, so a car running off the top of frame scored:

    bbox 707x222 at y1=0, frame 1280x720
    my   = 0.01 * 720                = 7.2px
    edge = (222 - 7.2) / 222         = 0.968
    quality                          ~ 0.97   -> sails through the 0.9 gallery gate

That is exactly how CAM-04 wrote a 707x222 sliver of a car into DJS-7842's gallery.
The detector had clamped the box to the visible fragment; the missing two-thirds of the
car is simply invisible to the geometry, so "how much of the BOX is inside the frame"
can never detect it. You have to ask whether the box TOUCHES the border at all.

Aspect is a second, overlapping net: a car cut off at the BOTTOM of frame stays
plausibly shaped, and a badly-merged box can be a sliver without touching any border.
"""

import unittest

import numpy as np

from src.core.engine.engine_tracking import (
    ParkingEngineTrackingMixin,
    _vq_size_thresholds,
    _VQ_MAX_ASPECT,
)


class _Det:
    def __init__(self, bbox, track_id=1):
        self.bbox = bbox
        self.track_id = track_id


class TestTruncationAndAspect(unittest.TestCase):
    class _Harness(ParkingEngineTrackingMixin):
        pass

    def setUp(self):
        self.h = self._Harness()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # the real stream size

    def _q(self, bbox):
        return self.h._bbox_view_quality(self.frame, _Det(bbox))

    # ---- the crop that actually poisoned the gallery ----------------------------

    def test_the_cam04_sliver_that_reached_the_gallery_now_scores_zero(self):
        """THE REGRESSION. 707x222 hard against the top of frame. Under the old gate
        this scored ~0.97 and cleared gallery_min_view_quality (0.9)."""
        bbox = (300.0, 0.0, 1007.0, 222.0)  # w=707 h=222, y1=0 -> truncated AND 3.2 aspect
        self.assertEqual(self._q(bbox), 0.0)

    # ---- truncation --------------------------------------------------------------

    def test_car_running_off_the_top_is_a_fragment(self):
        self.assertEqual(self._q((500.0, 0.0, 800.0, 400.0)), 0.0)

    def test_car_running_off_the_left_is_a_fragment(self):
        self.assertEqual(self._q((0.0, 200.0, 300.0, 600.0)), 0.0)

    def test_car_running_off_the_right_is_a_fragment(self):
        self.assertEqual(self._q((1000.0, 200.0, 1280.0, 600.0)), 0.0)

    def test_car_running_off_the_bottom_is_a_fragment(self):
        """Note this one is plausibly SHAPED (aspect 0.75) — only the truncation test
        catches it, which is why aspect alone would not be enough."""
        self.assertEqual(self._q((500.0, 400.0, 800.0, 720.0)), 0.0)

    # ---- aspect ------------------------------------------------------------------

    def test_a_sliver_away_from_every_border_is_still_not_a_car(self):
        """A badly-merged box can be sliver-shaped without touching any edge — so
        truncation alone would not be enough either. w/h = 700/250 = 2.8."""
        bbox = (300.0, 300.0, 1000.0, 550.0)
        self.assertGreater(700.0 / 250.0, _VQ_MAX_ASPECT)
        self.assertEqual(self._q(bbox), 0.0)

    # ---- and the gate must still let a GOOD crop through -------------------------

    def test_a_clean_well_framed_car_still_scores_full_marks(self):
        """The real parked view of DJS-7842 in B2: 314px tall, ~1.06 aspect, clear of
        every border. It must still score 1.0 — the gate has to reject fragments
        WITHOUT rejecting the parked-pose reference the whole system depends on."""
        bbox = (400.0, 300.0, 733.0, 614.0)  # 333x314, aspect 1.06
        _, good_h = _vq_size_thresholds(720)   # this harness frame is 720p
        self.assertGreaterEqual(314.0, good_h)
        self.assertEqual(self._q(bbox), 1.0)

    def test_a_small_but_clean_car_is_penalised_by_SIZE_not_zeroed(self):
        """Far cars are down-weighted, not rejected outright — that is the size ramp's
        job, and it must keep working alongside the new hard rejections."""
        bbox = (600.0, 300.0, 700.0, 448.0)  # 100x148 -> mid-ramp
        q = self._q(bbox)
        self.assertGreater(q, 0.0)
        self.assertLess(q, 1.0)


if __name__ == "__main__":
    unittest.main()
