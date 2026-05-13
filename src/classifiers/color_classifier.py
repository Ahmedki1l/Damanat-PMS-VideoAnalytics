"""
src.classifiers.color_classifier — Learned 11-class vehicle color head.

Replaces the loose HSV pre-filter that was the only color modality before
Phase 1 (see ``NoopColorClassifier`` in ``src/matching/plugins.py``). The
classifier is a MobileNetV3-Small backbone (trained by
``tools/train_color_classifier.py``) exported to OpenVINO IR.

Class set (locked, mirrors §WS-B of the master plan):

    0: black
    1: white
    2: grey
    3: silver
    4: red
    5: blue
    6: green
    7: yellow
    8: brown
    9: beige
   10: other

Runtime contract (see :class:`src.matching.plugins.ColorClassifier`):

    predict(crop_bgr) -> (label: str, confidence: float)
    colors_match(a, b) -> bool

Design notes
~~~~~~~~~~~~

* **Lazy loading.** The IR is loaded on first ``predict()`` call so importing
  this module is cheap and instantiating it does not fail when the IR is
  missing — the failure surfaces with a clear ``RuntimeError`` only when the
  classifier is actually used. This keeps unit-test bootstrap fast.

* **Confidence-aware matching.** ``colors_match`` only enforces equality when
  *both* predictions clear ``conf_threshold`` (default 0.7). When either side
  is below threshold or labelled ``other`` / ``unknown``, we treat the
  comparison as inconclusive and return ``True`` so this modality cannot
  *veto* a match it did not understand. The cascade ensemble (Phase 2) is
  what actually decides; this plugin must only refuse confirmations it is
  *confident* are wrong.

* **Input pipeline.** Crops are letterboxed (preserve aspect) to
  ``224 x 224``, BGR -> RGB, ImageNet mean / std normalised, CHW float32.
  The training pipeline writes the same constants out so a single source of
  truth (``labels.json``) defines both the class index map and the
  preprocessing dims.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Re-export the ABC so callers can ``isinstance``-check without reaching into
# the matching package directly.
from src.matching.plugins import ColorClassifier

# --------------------------------------------------------------------------- #
# Class set — locked by §WS-B of the master plan. Order matters: training,
# inference, labels.json and tests all read this constant.
# --------------------------------------------------------------------------- #

COLOR_CLASS_LABELS: List[str] = [
    "black",
    "white",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "brown",
    "beige",
    "other",
]

NUM_COLOR_CLASSES = len(COLOR_CLASS_LABELS)

# Inputs the training pipeline ships out. Keep aligned with
# ``tools/train_color_classifier.py``.
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)  # (H, W)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default location the training pipeline writes to.
DEFAULT_MODEL_PATH = "models/color_classifier_openvino/model.xml"


# --------------------------------------------------------------------------- #
# Preprocessing helpers (importable so tests can exercise them without an IR)
# --------------------------------------------------------------------------- #


def letterbox(image: np.ndarray, size: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """
    Resize ``image`` (BGR) to ``size`` (h, w) preserving aspect via zero-pad.

    Padding colour is mid-grey (114) so the network sees a neutral background
    rather than skewing its colour prediction toward black borders.
    """
    target_h, target_w = size
    if image.size == 0:
        return np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def preprocess_crop(
    crop_bgr: np.ndarray,
    size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """
    Convert a single BGR crop to the NCHW float32 tensor the model expects.

    Returns a contiguous ``np.ndarray`` of shape ``(1, 3, H, W)``.

    Must match ``tools/train_color_classifier.py``'s torchvision pipeline:
    ``transforms.ToPILImage() -> Resize((H, W)) -> ToTensor -> Normalize``.
    PIL's ``Resize((H, W))`` is a *squish* (no aspect preservation), so the
    plugin uses a straight ``cv2.resize`` here. Earlier this code used
    ``letterbox`` with grey padding — that matched typical detector
    preprocessing but caused a ~17 pp train/runtime accuracy gap because
    the model was trained on squished/distorted crops, not letterboxed
    ones. If you change one, change the other.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    target_h, target_w = size
    # cv2.resize takes (w, h), torchvision Resize takes (h, w) — same effect.
    resized = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb -= IMAGENET_MEAN
    rgb /= IMAGENET_STD
    # HWC -> CHW, add batch dim
    chw = np.transpose(rgb, (2, 0, 1))
    return np.ascontiguousarray(chw[None, ...], dtype=np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax across ``axis``."""
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# --------------------------------------------------------------------------- #
# OpenVINO-backed plugin
# --------------------------------------------------------------------------- #


class OpenVINOColorClassifier(ColorClassifier):
    """
    OpenVINO Runtime backed implementation of the 11-class color head.

    Args:
        model_path: path to the ``.xml`` IR file. When ``None`` (the default)
            the loader falls back to ``MatchingConfig.color_classifier_model``
            if a ``MatchingConfig`` instance is supplied, otherwise to
            :data:`DEFAULT_MODEL_PATH`.
        labels_path: path to the ``labels.json`` companion file. When None,
            assumed to be ``labels.json`` next to the IR.
        conf_threshold: minimum softmax confidence required for a prediction
            to be treated as "trustworthy" inside :meth:`colors_match`.
            Defaults to 0.7 per §WS-B's marginal-band policy.
        device: OpenVINO device string. ``"CPU"`` by default.
        config: optional ``MatchingConfig`` whose
            ``color_classifier_model`` field overrides
            :data:`DEFAULT_MODEL_PATH` when ``model_path`` is None.
    """

    # Class-level fallback labels in case labels.json is missing.
    DEFAULT_LABELS: List[str] = COLOR_CLASS_LABELS

    UNCONFIDENT_LABELS = {"other", "unknown", ""}

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        conf_threshold: float = 0.7,
        device: str = "CPU",
        config=None,
    ):
        # Resolve model path: explicit arg > config field > default.
        if model_path is None and config is not None:
            cfg_path = getattr(config, "color_classifier_model", "") or ""
            if cfg_path:
                model_path = cfg_path
        self._model_path = str(model_path or DEFAULT_MODEL_PATH)
        self._labels_path = (
            str(labels_path)
            if labels_path is not None
            else str(Path(self._model_path).with_name("labels.json"))
        )
        self._conf_threshold = float(conf_threshold)
        self._device = str(device)

        # Lazy-init state — populated on first predict() call.
        self._compiled = None
        self._input_layer = None
        self._output_layer = None
        self._labels: List[str] = list(self.DEFAULT_LABELS)
        self._input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
        self._initialised = False

    # ------------------------------------------------------------------ #
    # Public properties
    # ------------------------------------------------------------------ #

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold

    @property
    def labels(self) -> List[str]:
        # Returned as a defensive copy so callers cannot mutate the cached list.
        return list(self._labels)

    @property
    def is_loaded(self) -> bool:
        return self._compiled is not None

    # ------------------------------------------------------------------ #
    # Lazy loader
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._initialised:
            return

        if not os.path.exists(self._model_path):
            raise RuntimeError(
                "Color classifier model not found at "
                f"'{self._model_path}'. Train and export it first via "
                "`python tools/train_color_classifier.py` (see plan §WS-B)."
            )

        Core = None
        try:
            # OpenVINO 2024+ keeps Core on ``openvino.runtime`` for backward-
            # compat; 2026+ canonicalises it on the top-level package.
            try:
                from openvino import Core  # type: ignore
            except ImportError:
                from openvino.runtime import Core  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "OpenVINO Runtime is required for OpenVINOColorClassifier "
                "but could not be imported. Install `openvino>=2024.0.0`."
            ) from exc

        core = Core()
        try:
            model = core.read_model(self._model_path)
            self._compiled = core.compile_model(model, self._device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load OpenVINO IR at '{self._model_path}': {exc!r}"
            ) from exc

        self._input_layer = self._compiled.input(0)
        self._output_layer = self._compiled.output(0)

        # Derive expected input size from the IR shape (N, C, H, W). When the
        # model declares dynamic dims we fall back to the training default.
        shape = list(self._input_layer.partial_shape)
        try:
            h_dim = shape[2]
            w_dim = shape[3]
            h = int(h_dim.get_length()) if hasattr(h_dim, "get_length") else int(h_dim)
            w = int(w_dim.get_length()) if hasattr(w_dim, "get_length") else int(w_dim)
            if h > 0 and w > 0:
                self._input_size = (h, w)
        except Exception:
            # Dynamic shape → keep DEFAULT_INPUT_SIZE.
            pass

        # Load labels mapping
        labels = self._load_labels()
        if labels:
            self._labels = labels

        self._initialised = True
        logger.info(
            "[OpenVINOColorClassifier] loaded '%s' on %s (input=%s, labels=%s)",
            self._model_path,
            self._device,
            self._input_size,
            self._labels,
        )

    def _load_labels(self) -> List[str]:
        path = self._labels_path
        if not os.path.exists(path):
            logger.warning(
                "[OpenVINOColorClassifier] labels.json missing at '%s'; "
                "falling back to compiled-in label set.",
                path,
            )
            return list(self.DEFAULT_LABELS)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning(
                "[OpenVINOColorClassifier] failed to read '%s': %r — using defaults",
                path,
                exc,
            )
            return list(self.DEFAULT_LABELS)

        # Accept two formats: a plain list of strings OR a dict
        # {"labels": [...], "input_size": [H, W]} (the format written by the
        # training pipeline so the IR can ship its own preprocessing dims).
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            labels = data.get("labels") or data.get("classes")
            input_size = data.get("input_size") or data.get("input_shape")
            if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
                try:
                    h, w = int(input_size[0]), int(input_size[1])
                    if h > 0 and w > 0:
                        self._input_size = (h, w)
                except (TypeError, ValueError):
                    pass
            if isinstance(labels, list):
                return [str(x) for x in labels]
        return list(self.DEFAULT_LABELS)

    # ------------------------------------------------------------------ #
    # ABC implementation
    # ------------------------------------------------------------------ #

    def predict(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Run inference on a single BGR crop.

        Returns a ``(label, confidence)`` tuple where confidence is the
        softmax probability of the top-1 class in ``[0.0, 1.0]``.

        Empty / tiny crops return ``("unknown", 0.0)`` without invoking
        the network so callers don't pay inference cost on garbage input.
        """
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return ("unknown", 0.0)
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            return ("unknown", 0.0)
        if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            return ("unknown", 0.0)

        self._ensure_loaded()

        try:
            tensor = preprocess_crop(crop_bgr, self._input_size)
            result = self._compiled([tensor])
            logits = result[self._output_layer]
        except Exception as exc:
            logger.warning(
                "[OpenVINOColorClassifier] inference failed (%r) — "
                "returning 'unknown'.",
                exc,
            )
            return ("unknown", 0.0)

        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits.size == 0:
            return ("unknown", 0.0)

        # Truncate to known label count to be defensive against IR/label
        # mismatch (e.g. labels.json got out of sync with retraining).
        n = min(logits.size, len(self._labels))
        probs = softmax(logits[:n])
        idx = int(np.argmax(probs))
        return (self._labels[idx], float(probs[idx]))

    def colors_match(
        self,
        a,
        b,
    ) -> bool:
        """
        Two predictions match when:

        1. Both clear ``conf_threshold`` AND share the same label, OR
        2. At least one side is "unconfident" (label in
           :attr:`UNCONFIDENT_LABELS` or confidence below threshold).
           In this case we *cannot* refute the match, so we return ``True``
           and let the ensemble cascade decide on other modalities (this
           plugin must not veto a match it does not understand).
        """
        if not a or not b:
            return True  # nothing to refute on
        label_a, conf_a = self._unpack_prediction(a)
        label_b, conf_b = self._unpack_prediction(b)

        if self._is_unconfident(label_a, conf_a) or self._is_unconfident(label_b, conf_b):
            return True
        return label_a == label_b

    def _unpack_prediction(self, pred) -> Tuple[str, float]:
        # Tolerate both tuples and dict-like predictions for forward-compat.
        if isinstance(pred, tuple) and len(pred) == 2:
            label, conf = pred
        elif isinstance(pred, dict):
            label = pred.get("label", "unknown")
            conf = pred.get("confidence", 0.0)
        elif isinstance(pred, str):
            label, conf = pred, 1.0
        else:
            label, conf = "unknown", 0.0
        return (str(label), float(conf))

    def _is_unconfident(self, label: str, conf: float) -> bool:
        if label in self.UNCONFIDENT_LABELS:
            return True
        return conf < self._conf_threshold


__all__ = [
    "COLOR_CLASS_LABELS",
    "NUM_COLOR_CLASSES",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_MODEL_PATH",
    "OpenVINOColorClassifier",
    "letterbox",
    "preprocess_crop",
    "softmax",
]
