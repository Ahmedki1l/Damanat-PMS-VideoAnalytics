"""
Lighting normalization helpers for vehicle ReID crops.

Delegates to the shared preprocessing utility so the detector and ReID
paths use the same algorithm (luminance-only CLAHE in LAB) with
independently tuned parameters.
"""

from typing import Optional

import numpy as np

from src.preprocessing import luminance_normalize


def normalize_for_reid(
    bgr: Optional[np.ndarray],
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8),
) -> Optional[np.ndarray]:
    """
    Normalize lighting for cross-camera ReID.

    Applies CLAHE only to the LAB luminance channel so contrast improves without
    intentionally shifting color channels.

    Parameters
    ----------
    bgr : np.ndarray | None
        BGR image crop (uint8).
    clip_limit : float
        CLAHE clip limit.  Default 2.0 matches the original behaviour.
    grid_size : tuple
        CLAHE tile grid.  Default (8, 8) matches the original behaviour.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    return luminance_normalize(bgr, clip_limit=clip_limit, grid_size=grid_size)
