"""
Shared preprocessing utilities for lighting normalization.

Used by both the detector (full-frame) and ReID (crop) paths.
"""

from .normalize import luminance_normalize, luminance_normalize_auto, auto_gamma

__all__ = ["luminance_normalize", "luminance_normalize_auto", "auto_gamma"]
