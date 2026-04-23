"""
Lighting normalization helpers for vehicle ReID crops.
"""

from typing import Optional

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def normalize_for_reid(bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Normalize lighting for cross-camera ReID.

    Applies CLAHE only to the LAB luminance channel so contrast improves without
    intentionally shifting color channels.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _CLAHE.apply(l_channel)
    normalized_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
