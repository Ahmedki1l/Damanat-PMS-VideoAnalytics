"""
src.classifiers — Real (non-Noop) plugin implementations for the matching cascade.

Wave 1 / Phase 1 ships two learned heads here:

  * ``OpenVINOColorClassifier``  (WS-B, src/classifiers/color_classifier.py)
  * ``OpenVINOTypeClassifier``   (WS-C, src/classifiers/type_classifier.py)

Both replace the Phase-0 Noop plugins in ``src.matching.plugins``. The OpenVINO
runtime is loaded lazily — importing this package never forces the heavy
``openvino`` import, so environments without it (e.g. CI sanity, doc builds)
still work for the rest of the codebase.

This file is shared between WS-B and WS-C — both workstreams add their export
ADDITIVELY. The merge step reconciles a single ``__all__`` listing both heads.
"""

from __future__ import annotations

# WS-C export — Phase 1 body-type classifier.
from src.classifiers.type_classifier import OpenVINOTypeClassifier

__all__ = [
    "OpenVINOTypeClassifier",
]
