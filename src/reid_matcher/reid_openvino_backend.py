"""
src.reid_matcher.reid_openvino_backend — OpenVINO Runtime ReID backend.

CPU-fast replacement for the torchreid OSNet inference path used by
``VehicleReIDMatcher``. Target: ≤ 40 ms / image on AVX2 CPUs with the
INT8-quantised IR shipped in ``models/osnet_openvino_int8/``.

The public surface is intentionally minimal so the matcher can swap
backends at runtime without callers (``vehicle_registry_identity.py``,
``tracking_manager.py``) noticing:

    OpenVINOReIDBackend(model_dir: str | Path,
                         input_size: tuple[int, int] = (192, 96),
                         device: str = "CPU")
    .input_size                                  # property
    .extract(image_bgr: np.ndarray) -> np.ndarray
    .extract_batch(images_bgr: list[np.ndarray]) -> list[np.ndarray | None]
    .metadata: dict   # populated from metadata.yaml when available

All preprocessing must match ``tools/export_osnet_openvino.py`` exactly:
BGR → RGB letterbox to (H, W) with 127-grey padding, /255.0,
ImageNet mean/std normalisation, NCHW float32.

The class is thread-safe: the underlying ``CompiledModel`` is shared and a
fresh ``InferRequest`` is acquired per call, but per-call inference is
serialised behind ``self._lock`` to avoid OpenVINO's compiled-model state
being mutated concurrently.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Normalisation constants — must match tools/export_osnet_openvino.py and
# src/reid_matcher/reid_matcher.py legacy path.
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _load_metadata(model_dir: Path) -> dict:
    """Parse metadata.yaml emitted by the exporter (no pyyaml dep)."""
    meta_path = model_dir / "metadata.yaml"
    out: dict = {}
    if not meta_path.exists():
        return out
    try:
        import yaml
        with meta_path.open("r", encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh) or {}
        if isinstance(parsed, dict):
            return parsed
    except ImportError:
        pass
    # Fallback: tiny YAML-subset parser for the known schema.
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                out[key.strip()] = val.strip().strip('"')
    except OSError:
        pass
    return out


class OpenVINOReIDBackend:
    """Thin OpenVINO Runtime wrapper around the OSNet IR."""

    BACKEND_NAME = "openvino"

    def __init__(
        self,
        model_dir: Union[str, Path],
        input_size: Tuple[int, int] = (192, 96),
        device: str = "CPU",
    ):
        try:
            import openvino as ov  # local import — avoid hard dep at module load
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "openvino runtime is required for OpenVINOReIDBackend; "
                "install with 'pip install openvino'."
            ) from exc

        self.model_dir = Path(model_dir)
        self.input_size = tuple(input_size)  # (H, W)
        self.device = device

        xml_path = self.model_dir / "model.xml"
        if not xml_path.exists():
            raise FileNotFoundError(
                f"OpenVINO IR not found at {xml_path}. Run "
                "tools/export_osnet_openvino.py to generate it."
            )

        logger.info(
            "[REID/OV] Loading IR from %s on %s (input %dx%d)...",
            xml_path,
            device,
            self.input_size[0],
            self.input_size[1],
        )
        core = ov.Core()
        self._ov_core = core
        self._model = core.read_model(str(xml_path))
        # Compile for throughput: many short ReID calls benefit from
        # AUTO/CPU's default LATENCY hint on CPU. We keep this minimal and
        # let OpenVINO pick the right hint per device.
        self._compiled = core.compile_model(self._model, device)
        # Acquire one infer request up-front; the lock below ensures we
        # never call .infer() concurrently against the same request.
        self._request = self._compiled.create_infer_request()
        self._lock = threading.Lock()
        self.metadata = _load_metadata(self.model_dir)
        if self.metadata:
            meta_size = self.metadata.get("input_size")
            if isinstance(meta_size, (list, tuple)) and len(meta_size) == 2:
                # Trust metadata.yaml if present so the runtime matches the
                # exporter even when the user supplied a stale --input-size.
                self.input_size = (int(meta_size[0]), int(meta_size[1]))

        logger.info(
            "[REID/OV] Backend ready (quantisation=%s, calibration=%s).",
            self.metadata.get("quantisation", "unknown"),
            self.metadata.get("calibration_image_count", "?"),
        )

    # --------------------------------------------------------------------- #
    # Preprocessing
    # --------------------------------------------------------------------- #

    def _preprocess(
        self,
        image_bgr: np.ndarray,
        *,
        normalise_luminance: Optional[bool] = False,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """BGR uint8 -> NCHW float32 (1, 3, H, W) using letterbox + mean/std.

        Matches `_letterbox_preprocess` in tools/export_osnet_openvino.py
        (which is the distribution NNCF calibrated against).
        """
        if normalise_luminance:
            from src.reid_matcher.reid_preprocessing import normalize_for_reid
            image_bgr = normalize_for_reid(
                image_bgr, clip_limit=clip_limit, grid_size=grid_size
            )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
        dw = (target_w - nw) // 2
        dh = (target_h - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw, :] = resized
        chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = (chw - NORM_MEAN) / NORM_STD
        return chw[np.newaxis, ...].astype(np.float32)

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a feature vector in-place semantics (returns new array)."""
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            return vec / norm
        return vec

    def extract(
        self,
        image_bgr: np.ndarray,
        *,
        normalise_luminance: Optional[bool] = False,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> Optional[np.ndarray]:
        """Extract a single normalised 512-D feature for a BGR crop."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        try:
            nchw = self._preprocess(
                image_bgr,
                normalise_luminance=normalise_luminance,
                clip_limit=clip_limit,
                grid_size=grid_size,
            )
            with self._lock:
                output = self._request.infer({"images": nchw})
            features = list(output.values())[0]
            vec = np.asarray(features, dtype=np.float32)[0]
            return self._normalise(vec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[REID/OV] extract failed: %r", exc)
            return None

    def extract_batch(
        self,
        images_bgr: List[np.ndarray],
        *,
        normalise_luminance: Optional[bool] = False,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> List[Optional[np.ndarray]]:
        """Extract features for all valid crops in a single forward pass.

        Returns a list of the same length as ``images_bgr`` with ``None``
        for entries that were empty / unreadable. Order is preserved.
        """
        if not images_bgr:
            return []

        valid_idx: List[int] = []
        tensors: List[np.ndarray] = []
        for i, img in enumerate(images_bgr):
            if img is None or img.size == 0:
                continue
            try:
                tensors.append(
                    self._preprocess(
                        img,
                        normalise_luminance=normalise_luminance,
                        clip_limit=clip_limit,
                        grid_size=grid_size,
                    )
                )
                valid_idx.append(i)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[REID/OV] preprocess failed for idx %d: %r", i, exc
                )

        results: List[Optional[np.ndarray]] = [None] * len(images_bgr)
        if not tensors:
            return results

        batch = np.concatenate(tensors, axis=0)
        try:
            with self._lock:
                output = self._request.infer({"images": batch})
            features = np.asarray(list(output.values())[0], dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[REID/OV] batch infer failed: %r", exc)
            return results

        for k, global_idx in enumerate(valid_idx):
            results[global_idx] = self._normalise(features[k])
        return results
