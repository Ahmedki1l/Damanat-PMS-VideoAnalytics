"""
Shared preprocessing utilities for lighting normalization.

Used by both the detector (full-frame) and ReID (crop) paths.
"""

from .normalize import luminance_normalize, auto_gamma

__all__ = ["luminance_normalize", "auto_gamma"]
