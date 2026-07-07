"""
src.classifiers.type_classifier — Learned 6-class body-type classifier.

Implements the ``TypeClassifier`` ABC from ``src.matching.plugins`` with an
OpenVINO IR backed MobileNetV3-Small head. The 6 output classes are:

    sedan, SUV, hatchback, pickup, van, bus

The class loads the IR lazily (first ``predict`` call) so importing this module
does not force ``openvino`` to be installed. Missing-model handling is graceful:
when the IR cannot be found or loaded, ``predict`` returns ``("unknown", 0.0)``
and ``types_match`` returns ``False`` — i.e. the type modality is treated as
"no signal" and the rest of the matching cascade falls through to ReID alone.

Public surface:

    OpenVINOTypeClassifier(model_path=..., labels_path=..., device="CPU")
        .predict(crop_bgr)         -> (label: str, confidence: float)
        .types_match(a, b)         -> bool   # ensemble helper

The ensemble rule (T2.1) calls ``types_match`` from
``MatchDecision.type_check`` — see the plan §Phase-2 T2.1.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.matching.plugins import TypeClassifier

logger = logging.getLogger(__name__)


# Canonical class list — must match the head's output ordering. Persisted to
# ``models/type_classifier_openvino/labels.json`` by the training pipeline; we
# keep an in-source fallback so the plugin still names labels correctly when
# the labels file is missing.
DEFAULT_CLASSES: Tuple[str, ...] = (
    "sedan",
    "suv",
    "hatchback",
    "pickup",
    "van",
    "bus",
)

# MobileNetV3 standard input. Kept here (not pushed into MatchingConfig)
# because changing it requires re-exporting the IR — it is a model-shape
# constant, not an operator-tunable knob.
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)  # (H, W)

# ImageNet mean / std — MobileNetV3 was pretrained on ImageNet, our fine-tune
# preserves this normalisation. Order is RGB.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default location of the OpenVINO IR — populated by
# ``tools/train_type_classifier.py``. The plan asks for
# ``models/type_classifier_openvino/model.xml`` as the canonical path.
DEFAULT_MODEL_PATH = "models/type_classifier_openvino/model.xml"
DEFAULT_LABELS_PATH = "models/type_classifier_openvino/labels.json"

# Both predictions must clear this confidence floor for ``types_match`` to
# return True. The same floor (0.70) is used by WS-B's colour classifier so
# the ensemble rule has consistent semantics across modalities.
TYPES_MATCH_CONFIDENCE_FLOOR: float = 0.70


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class OpenVINOTypeClassifier(TypeClassifier):
    """
    OpenVINO-backed body-type head, 6 classes.

    Args:
        model_path: Path to the OpenVINO IR ``.xml``. Defaults to
            ``models/type_classifier_openvino/model.xml``.
        labels_path: Optional path to a ``labels.json`` mapping class index
            to label string. Falls back to ``DEFAULT_CLASSES`` if missing.
        device: OpenVINO inference device ("CPU", "GPU", "AUTO"). Default
            "CPU" — the whole matching cascade is CPU-targeted.
        input_size: ``(H, W)`` input shape — must match the exported IR.
        confidence_floor: Per-prediction floor below which ``types_match``
            refuses to agree two predictions. Defaults to 0.70.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        device: str = "CPU",
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        confidence_floor: float = TYPES_MATCH_CONFIDENCE_FLOOR,
    ) -> None:
        self._model_path = str(model_path) if model_path else DEFAULT_MODEL_PATH
        self._labels_path = (
            str(labels_path) if labels_path else DEFAULT_LABELS_PATH
        )
        self._device = device
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self._confidence_floor = float(confidence_floor)

        # Lazy state — populated on first predict() call so importing this
        # module does not require openvino to be installed.
        self._core = None
        self._compiled = None
        self._input_layer = None
        self._output_layer = None
        self._labels: List[str] = list(DEFAULT_CLASSES)
        self._load_attempted: bool = False
        self._load_succeeded: bool = False

    # ----------------------------------------------------------------- #
    # Lazy model loading
    # ----------------------------------------------------------------- #

    def _ensure_loaded(self) -> bool:
        """Load the IR + labels on first call. Cached on subsequent calls.

        Returns True iff the model is ready to infer. False is the
        graceful "missing model" signal — callers fall through to
        ``("unknown", 0.0)``.
        """
        if self._load_attempted:
            return self._load_succeeded
        self._load_attempted = True

        # Read labels first — these are independent of the model itself
        # and useful even if the model is missing (e.g. test harness).
        labels_path = Path(self._labels_path)
        if labels_path.exists():
            try:
                with labels_path.open(encoding="utf-8") as fh:
                    payload = json.load(fh)
                if isinstance(payload, dict) and "classes" in payload:
                    self._labels = [str(c) for c in payload["classes"]]
                elif isinstance(payload, list):
                    self._labels = [str(c) for c in payload]
                else:
                    logger.warning(
                        "[OpenVINOTypeClassifier] labels.json has unexpected "
                        "shape (%s); falling back to defaults.",
                        type(payload).__name__,
                    )
            except Exception as exc:
                logger.warning(
                    "[OpenVINOTypeClassifier] failed to read labels at %s: %r",
                    labels_path,
                    exc,
                )

        model_path = Path(self._model_path)
        if not model_path.exists():
            logger.info(
                "[OpenVINOTypeClassifier] model not found at %s; "
                "predict() will return ('unknown', 0.0)",
                model_path,
            )
            return False

        try:
            # Lazy import — openvino is a heavy dep, only required when an
            # IR actually exists on disk. OpenVINO 2024+ keeps Core on
            # ``openvino.runtime``; 2026+ canonicalises it on the top-level
            # package (``openvino.runtime`` is gone), so try that first —
            # matching OpenVINOColorClassifier. Without this the type modality
            # silently drops out of the ensemble under OpenVINO 2026.
            try:
                from openvino import Core  # type: ignore
            except ImportError:
                from openvino.runtime import Core  # type: ignore
        except Exception as exc:  # pragma: no cover - environment guard
            logger.warning(
                "[OpenVINOTypeClassifier] OpenVINO runtime unavailable: %r",
                exc,
            )
            return False

        try:
            self._core = Core()
            model = self._core.read_model(str(model_path))
            self._compiled = self._core.compile_model(model, self._device)
            self._input_layer = self._compiled.inputs[0]
            self._output_layer = self._compiled.outputs[0]
            self._load_succeeded = True
            logger.info(
                "[OpenVINOTypeClassifier] loaded %s on %s (input=%s, output=%s)",
                model_path,
                self._device,
                tuple(self._input_layer.shape),
                tuple(self._output_layer.shape),
            )
        except Exception as exc:
            logger.warning(
                "[OpenVINOTypeClassifier] failed to compile model at %s: %r",
                model_path,
                exc,
            )
            self._load_succeeded = False
        return self._load_succeeded

    # ----------------------------------------------------------------- #
    # Preprocessing
    # ----------------------------------------------------------------- #

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Resize + BGR->RGB + ImageNet normalise + NCHW float32.

        ``crop_bgr`` is an OpenCV-shaped HxWx3 BGR uint8 array. We do not
        import cv2 at module top so the package stays importable in
        cv2-less environments; cv2 is imported inside this method only.
        """
        if crop_bgr is None:
            raise ValueError("crop_bgr is None")
        if not isinstance(crop_bgr, np.ndarray):
            raise TypeError(
                f"crop_bgr must be a numpy ndarray, got {type(crop_bgr).__name__}"
            )
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError(
                f"crop_bgr must be HxWx3 (BGR), got shape {crop_bgr.shape}"
            )

        h, w = self._input_size
        try:
            import cv2  # type: ignore

            resized = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        except Exception:
            # cv2 missing — fall back to a pure-numpy nearest-neighbour
            # resize. Lower quality but keeps the predict path alive for
            # smoke tests in cv2-less envs.
            sh, sw = crop_bgr.shape[:2]
            y_idx = (np.linspace(0, sh - 1, h)).astype(np.int64)
            x_idx = (np.linspace(0, sw - 1, w)).astype(np.int64)
            resized = crop_bgr[np.ix_(y_idx, x_idx)]
            rgb = resized[..., ::-1]  # BGR -> RGB

        # Normalise to [0, 1], ImageNet mean/std, NCHW.
        f = rgb.astype(np.float32) / 255.0
        f = (f - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
            IMAGENET_STD, dtype=np.float32
        )
        # HWC -> CHW -> NCHW
        f = np.transpose(f, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(f, dtype=np.float32)

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Return ``(label, confidence)`` for a single BGR crop.

        Graceful missing-model path: when the IR is not available or fails
        to load, returns ``("unknown", 0.0)`` so the ensemble rule reads
        "no signal" rather than a hard False.
        """
        if not self._ensure_loaded():
            return ("unknown", 0.0)

        try:
            tensor = self._preprocess(crop_bgr)
        except Exception as exc:
            logger.debug(
                "[OpenVINOTypeClassifier] preprocess failed: %r", exc
            )
            return ("unknown", 0.0)

        try:
            outputs = self._compiled([tensor])
            logits = outputs[self._output_layer]
        except Exception as exc:
            logger.warning(
                "[OpenVINOTypeClassifier] inference failed: %r", exc
            )
            return ("unknown", 0.0)

        logits = np.asarray(logits)
        # Some IRs already produce softmax probs — detect by checking
        # whether the output sums to ~1 along the last axis.
        last_axis_sum = float(np.sum(logits[..., :]))
        if logits.ndim == 2:
            row = logits[0]
        else:
            row = logits.reshape(-1)

        if abs(last_axis_sum - 1.0) > 1e-2 and abs(
            float(np.sum(row)) - 1.0
        ) > 1e-2:
            probs = _softmax(row)
        else:
            probs = row.astype(np.float32)

        idx = int(np.argmax(probs))
        if idx < 0 or idx >= len(self._labels):
            return ("unknown", float(probs[idx]) if idx < len(probs) else 0.0)
        return (self._labels[idx], float(probs[idx]))

    def types_match(
        self,
        label_a: Tuple[str, float] | str,
        label_b: Tuple[str, float] | str,
    ) -> bool:
        """Ensemble helper.

        Accepts either bare label strings or full ``(label, confidence)``
        tuples (the form returned by ``predict``). Returns True iff:

          * neither side is unknown / empty
          * both confidences (if provided) clear the floor
          * the two labels are equal (after case-folding)

        Callers that only have raw labels (e.g. backfilled session
        metadata) can pass bare strings; in that case the confidence
        floor is bypassed (we assume the label was committed only after
        passing the floor at write time).
        """

        def _split(item) -> Tuple[Optional[str], float]:
            if item is None:
                return None, 0.0
            if isinstance(item, str):
                return (item or None), 1.0
            if isinstance(item, tuple) and len(item) == 2:
                label, conf = item
                if not label:
                    return None, 0.0
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.0
                return str(label), conf
            return None, 0.0

        a_label, a_conf = _split(label_a)
        b_label, b_conf = _split(label_b)
        if not a_label or not b_label:
            return False
        if a_label.lower() == "unknown" or b_label.lower() == "unknown":
            return False
        if a_conf < self._confidence_floor or b_conf < self._confidence_floor:
            return False
        return a_label.strip().lower() == b_label.strip().lower()

    # ----------------------------------------------------------------- #
    # Introspection helpers (used by tests + tools)
    # ----------------------------------------------------------------- #

    @property
    def labels(self) -> List[str]:
        """Read-only copy of the class-index → label table."""
        return list(self._labels)

    @property
    def is_loaded(self) -> bool:
        """True iff the IR is loaded and ready to infer."""
        return self._load_succeeded

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def confidence_floor(self) -> float:
        return self._confidence_floor
