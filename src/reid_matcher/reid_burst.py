import cv2
import numpy as np
from typing import List

def sharpness_score(bgr: np.ndarray) -> float:
    """Variance of Laplacian — higher = sharper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_overexposed(bgr: np.ndarray, threshold: float = 0.35) -> bool:
    """Reject crops where >35% of pixels are saturated."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    saturated_ratio = np.mean(gray > 245)
    return saturated_ratio > threshold

def select_best_frames(crops: List[np.ndarray], top_k: int = 3) -> List[np.ndarray]:
    """
    From a burst of vehicle crops, return the top_k sharpest non-overexposed ones.
    Falls back gracefully if fewer than top_k usable frames exist.
    """
    usable = [c for c in crops if c is not None and c.size > 0 and not is_overexposed(c)]
    if not usable:
        # Fallback: return sharpest overall even if overexposed — better than nothing
        usable = [c for c in crops if c is not None and c.size > 0]
    if not usable:
        return []
    ranked = sorted(usable, key=sharpness_score, reverse=True)
    return ranked[:top_k]