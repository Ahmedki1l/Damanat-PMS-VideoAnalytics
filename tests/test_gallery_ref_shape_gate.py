"""A gallery reference must be ONE WHOLE CAR.

On 2026-07-12 two crops that are not cars were sitting in the durable gallery:

    EEB-80    CAM-03   1527x519   aspect 2.94   (written TWICE, byte-identical)
    DJS-7842  CAM-03    722x258   aspect 2.80

Both carried ``quality: 999.0`` — the "authoritative gate seed" sentinel. That is the
whole bug: ``_seed_gallery`` and ``save_parked_reference`` call ``store.save_ref``
DIRECTLY with a hardcoded quality, bypassing the view-quality path where the engine's
aspect cap lives. So the only two paths that write references we call AUTHORITATIVE
were the only two that never checked the crop's shape.

An authoritative identity says the crop belongs to THIS PLATE. It says nothing about
the crop containing one car — a box spanning a car and its neighbour still reads the
right plate, and still poisons the gallery with half a stranger.
"""

import unittest

import numpy as np

from src.vehicle_registry.vehicle_registry_identity import is_plausible_car_crop


def _crop(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestGalleryRefShapeGate(unittest.TestCase):
    def test_the_two_slivers_that_actually_got_in_are_now_refused(self):
        self.assertFalse(is_plausible_car_crop(_crop(1527, 519)))  # EEB-80, aspect 2.94
        self.assertFalse(is_plausible_car_crop(_crop(722, 258)))   # DJS-7842, aspect 2.80

    def test_the_real_gate_crops_still_pass(self):
        """Every genuine ANPR / CAM-23 / CAM-03 reference from the live gallery."""
        for w, h in [
            (1007, 705), (1539, 1349), (1118, 926), (1108, 756),  # ANPR
            (264, 293), (566, 585), (432, 523), (274, 283),       # CAM-23
            (681, 537), (1074, 547), (739, 585), (921, 547),      # CAM-03
        ]:
            self.assertTrue(is_plausible_car_crop(_crop(w, h)), f"{w}x{h}")

    def test_a_car_seen_side_on_is_still_a_car(self):
        """The cap must not reject legitimately wide views — a side profile is the
        widest a single car gets, and those are exactly the slot references we need."""
        self.assertTrue(is_plausible_car_crop(_crop(440, 200)))  # aspect 2.20, at the cap

    def test_a_degenerate_or_absent_crop_is_refused(self):
        self.assertFalse(is_plausible_car_crop(None))
        self.assertFalse(is_plausible_car_crop(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertFalse(is_plausible_car_crop(_crop(20, 15)))  # too small to be useful


if __name__ == "__main__":
    unittest.main()
