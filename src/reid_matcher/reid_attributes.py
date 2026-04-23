from typing import Optional, Tuple

import cv2
import numpy as np


def dominant_color_hsv(bgr: Optional[np.ndarray]) -> Optional[Tuple[float, float, float]]:
    """Return mean HSV of the center crop to avoid background/window bias."""
    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]
    center = bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.size == 0:
        return None

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    return tuple(float(v) for v in np.mean(hsv.reshape(-1, 3), axis=0))


def color_compatible(hsv_a, hsv_b, h_tol=25, s_tol=80, v_tol=80) -> bool:
    """Loose compatibility check that only rejects obvious color mismatches."""
    if hsv_a is None or hsv_b is None:
        return True

    dh = min(abs(hsv_a[0] - hsv_b[0]), 180 - abs(hsv_a[0] - hsv_b[0]))
    ds = abs(hsv_a[1] - hsv_b[1])
    dv = abs(hsv_a[2] - hsv_b[2])
    return dh < h_tol and ds < s_tol and dv < v_tol
