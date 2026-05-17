"""
src.classifiers — Learned classification heads for the matching cascade.

Wave 1 / Phase 1 introduces two small CPU-friendly classifiers that plug into
``src.matching.MatchDecision`` via the ABCs in ``src.matching.plugins``:

* :class:`OpenVINOColorClassifier` — 11-class vehicle color head
  (black / white / grey / silver / red / blue / green / yellow / brown /
  beige / other). Owned by WS-B.
* :class:`OpenVINOTypeClassifier` — 6-class body-type head
  (sedan / SUV / hatchback / pickup / van / bus). Owned by WS-C.

Both replace the Phase-0 Noop plugins in ``src.matching.plugins``. OpenVINO
runtime is loaded lazily — importing this package never forces the heavy
``openvino`` import, so environments without it (CI sanity, doc builds)
still work for the rest of the codebase.

Each classifier ships its OpenVINO IR under ``models/<name>_openvino/`` with a
``labels.json`` describing the class ordering and a ``README.md`` documenting
training provenance.
"""

from __future__ import annotations

from .color_classifier import (
    COLOR_CLASS_LABELS,
    OpenVINOColorClassifier,
)
from .type_classifier import OpenVINOTypeClassifier

__all__ = [
    "COLOR_CLASS_LABELS",
    "OpenVINOColorClassifier",
    "OpenVINOTypeClassifier",
]
