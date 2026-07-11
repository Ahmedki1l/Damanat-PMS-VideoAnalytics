"""
normalize.py — Luminance-safe image normalization for vehicle analytics.

Provides CLAHE on the LAB luminance channel and optional gamma correction
for dark frames.  Hue and saturation are never touched so downstream
color-based matching (image_matcher, ReID) stays stable.

Design notes
------------
* Parameterized so the detector path (full 640px frame) and the ReID path
  (128×256 crop) can each use their own tuned settings while sharing the
  same algorithm.
* Gamma correction uses the **median** of the L channel — not the mean —
  to stay robust against bright/dark outlier regions (e.g. bright floor
  with a dark vehicle in shadow).
* Gamma is capped to [0.7, 1.0] to prevent overcorrection.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Module-level CLAHE cache — avoids re-creating identical CLAHE objects
# every frame.  Keyed by (clip_limit, grid_w, grid_h).
# ---------------------------------------------------------------------------
_clahe_cache: dict = {}


def _get_clahe(clip_limit: float, grid_size: Tuple[int, int]) -> cv2.CLAHE:
    """Return a cached CLAHE instance for the given parameters."""
    key = (clip_limit, grid_size[0], grid_size[1])
    if key not in _clahe_cache:
        _clahe_cache[key] = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=grid_size
        )
    return _clahe_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _gamma_from_l(
    l_channel: np.ndarray,
    dark_threshold: int,
    target_l: int,
) -> Optional[float]:
    """Gamma in [0.7, 1.0] for an already-extracted L channel, or None if bright."""
    median_l = float(np.median(l_channel))
    if median_l >= dark_threshold:
        return None  # Frame is bright enough

    # gamma = log(target / 255) / log(median / 255)  — standard formula,
    # but we cap it to [0.7, 1.0] so we never over-brighten.
    if median_l < 1:
        median_l = 1.0  # Avoid log(0)

    gamma = np.log(target_l / 255.0) / np.log(median_l / 255.0)
    return float(max(0.7, min(1.0, gamma)))


def _apply_gamma(l_channel: np.ndarray, gamma: float) -> np.ndarray:
    """``pixel = 255 * (pixel / 255) ** gamma`` on a uint8 L channel."""
    l_float = l_channel.astype(np.float32) / 255.0
    l_float = np.power(l_float, gamma)
    return np.clip(l_float * 255.0, 0, 255).astype(np.uint8)


def auto_gamma(
    bgr: np.ndarray,
    dark_threshold: int = 65,
    target_l: int = 90,
) -> Optional[float]:
    """
    Compute a gamma correction value for dark frames.

    Returns a gamma value in [0.7, 1.0] if the frame is dark enough to
    benefit from a brightness lift.  Returns ``None`` if the frame is
    already bright enough (median L >= *dark_threshold*).

    NOTE: this does its own BGR->LAB conversion. If you are about to call
    ``luminance_normalize`` on the same image, use ``luminance_normalize_auto``
    instead — it shares the one conversion rather than paying for two.

    Parameters
    ----------
    bgr : np.ndarray
        BGR image (uint8).
    dark_threshold : int
        Median L-channel value below which gamma correction is triggered.
    target_l : int
        Target median luminance used to calculate the correction factor.
    """
    if bgr is None or bgr.size == 0:
        return None

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return _gamma_from_l(lab[:, :, 0], dark_threshold, target_l)


def luminance_normalize_auto(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
    gamma_correction: bool = True,
    dark_threshold: int = 65,
    target_l: int = 90,
) -> np.ndarray:
    """
    ``auto_gamma`` + ``luminance_normalize`` sharing a SINGLE BGR<->LAB round trip.

    Identical output to calling the two in sequence, but the colour-space
    conversion — which dominates the cost at frame size — happens once instead
    of twice. Prefer this anywhere both steps are wanted.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    if gamma_correction:
        gamma = _gamma_from_l(l_channel, dark_threshold, target_l)
        if gamma is not None and gamma != 1.0:
            l_channel = _apply_gamma(l_channel, gamma)

    l_channel = _get_clahe(clip_limit, grid_size).apply(l_channel)

    normalized_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def luminance_normalize(
    bgr: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Apply luminance-only CLAHE and optional gamma correction.

    Only the L channel of the LAB colour space is modified.  The A and B
    (chrominance) channels pass through unchanged, so hue and saturation
    are preserved.

    Parameters
    ----------
    bgr : np.ndarray
        BGR image (uint8).
    clip_limit : float
        CLAHE contrast clip limit.  2.0 is mild; higher = more contrast.
    grid_size : tuple[int, int]
        CLAHE tile grid size.
    gamma : float | None
        If provided, apply ``pixel = 255 * (pixel / 255) ** gamma`` to the
        L channel *before* CLAHE.  Values < 1.0 brighten dark regions.
        ``None`` skips gamma entirely.

    Returns
    -------
    np.ndarray
        Normalised BGR image with same shape and dtype as input.
    """
    if bgr is None or bgr.size == 0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Optional gamma lift on luminance only
    if gamma is not None and gamma != 1.0:
        l_channel = _apply_gamma(l_channel, gamma)

    # CLAHE on luminance only
    clahe = _get_clahe(clip_limit, grid_size)
    l_channel = clahe.apply(l_channel)

    normalized_lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
