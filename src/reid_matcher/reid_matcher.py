"""
reid_matcher.py — Deep Learning ReID feature extraction for vehicles.

Two interchangeable backends are supported:

* ``openvino`` (Phase 1 / WS-A) — OpenVINO Runtime loading the INT8 IR
  exported by ``tools/export_osnet_openvino.py``. Target ≤ 40 ms / image
  on CPU; mandatory for production CPU deployments.
* ``torchreid`` (legacy) — torch + torchreid running ``osnet_ain_x1_0``.
  Used as a fallback so developers without the exported IR can still run
  the code (~1 s / image on CPU). Phase 1 callers should never see this
  in production.

The public API of :class:`VehicleReIDMatcher` (``extract_feature``,
``extract_features_batch``, ``compute_similarity``) is byte-compatible
with the pre-WS-A implementation — callers in
``vehicle_registry_identity.py``, ``vehicle_registry_core.py`` and
``detection/tracking_manager.py`` do not need to change.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import MatchingConfig, ReIDPreprocessingConfig
from src.reid_matcher.reid_preprocessing import normalize_for_reid

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backend selection helpers
# --------------------------------------------------------------------------- #


def _detect_torchreid():
    """Lazy import of torchreid. Returns ``build_model`` or ``None``."""
    try:
        import torchreid  # noqa: WPS433 - intentional lazy import
        return torchreid.models.build_model
    except ImportError as exc:
        logger.warning("[REID] torchreid not importable: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[REID] torchreid import failed: %s", exc)
    return None


def _ir_path(model_dir: Path) -> Path:
    return model_dir / "model.xml"


def _resolve_model_dir(model_dir: Optional[Path]) -> Path:
    """Resolve the OpenVINO model directory to an existing absolute path.

    Tries (in order):
      1. The supplied path as-is (absolute or cwd-relative).
      2. Repository-root relative ``models/osnet_openvino_int8``.
      3. The path resolved against the package parent (helps when the
         working dir is set by tests or installed packages).
    """
    candidates = []
    if model_dir is not None:
        candidates.append(Path(model_dir))
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    candidates.append(repo_root / "models" / "osnet_openvino_int8")

    seen = set()
    for cand in candidates:
        c = cand.expanduser()
        try:
            c = c.resolve()
        except OSError:
            pass
        if c in seen:
            continue
        seen.add(c)
        if _ir_path(c).exists():
            return c
    # Last resort — return the first non-None candidate even if it is
    # missing; the caller will see FileNotFoundError when constructing
    # the backend.
    return candidates[0]


# --------------------------------------------------------------------------- #
# torchreid fallback (legacy path)
# --------------------------------------------------------------------------- #


class _TorchreidBackend:
    """The original OSNet-AIN inference path, used as a fallback.

    Kept identical to pre-WS-A behaviour (input 128x256, ImageNet
    normalisation, letterbox) so historical features stay binary-compatible.
    """

    BACKEND_NAME = "torchreid"

    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        use_gpu: bool = True,
        batch_size: int = 16,
        preprocessing_config: Optional[ReIDPreprocessingConfig] = None,
    ):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - tested via skip path
            raise ImportError(
                "torch is required for the torchreid ReID backend"
            ) from exc

        build_model = _detect_torchreid()
        if build_model is None:
            raise ImportError(
                "torchreid not found. Install with 'pip install torchreid' "
                "or generate the OpenVINO IR via "
                "'tools/export_osnet_openvino.py' to use the fast backend."
            )

        self._torch = torch
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.preprocessing_config = preprocessing_config or ReIDPreprocessingConfig()

        logger.info("[REID/torchreid] Initialising %s on %s...", model_name, self.device)
        self.model = build_model(name=model_name, num_classes=1000, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Legacy input pipeline (preserved verbatim).
        self.input_size: Tuple[int, int] = (128, 256)  # (H, W)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def _preprocess(self, image: np.ndarray):
        import torch
        pp = self.preprocessing_config
        if pp.enabled:
            image = normalize_for_reid(
                image, clip_limit=pp.clip_limit, grid_size=pp.grid_size
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
        dw = (target_w - nw) // 2
        dh = (target_h - nh) // 2
        canvas[dh:dh + nh, dw:dw + nw, :] = image_resized
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.norm_mean).view(3, 1, 1)
        std = torch.tensor(self.norm_std).view(3, 1, 1)
        return (tensor - mean) / std

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        try:
            import torch
            with torch.no_grad():
                tensor = self._preprocess(image).unsqueeze(0).to(self.device)
                features = self.model(tensor).cpu().numpy()[0]
                norm = float(np.linalg.norm(features))
                if norm > 0:
                    features = features / norm
                return features
        except Exception as exc:
            logger.error("[REID/torchreid] extract failed: %r", exc)
            return None

    def extract_batch(
        self, images: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        if not images:
            return []
        try:
            import torch
        except ImportError:  # pragma: no cover
            return [None] * len(images)

        results: List[Optional[np.ndarray]] = [None] * len(images)
        valid_idx = [
            i for i, img in enumerate(images) if img is not None and img.size > 0
        ]
        if not valid_idx:
            return results
        valid_images = [images[i] for i in valid_idx]
        for i in range(0, len(valid_images), self.batch_size):
            batch_imgs = valid_images[i:i + self.batch_size]
            batch_idx = valid_idx[i:i + self.batch_size]
            try:
                with torch.no_grad():
                    tensors = [self._preprocess(img) for img in batch_imgs]
                    inp = torch.stack(tensors).to(self.device)
                    feats = self.model(inp).cpu().numpy()
                    for k, gi in enumerate(batch_idx):
                        vec = feats[k]
                        norm = float(np.linalg.norm(vec))
                        if norm > 1e-6:
                            vec = vec / norm
                        results[gi] = vec
            except Exception as exc:
                logger.error(
                    "[REID/torchreid] batch extract failed: %r", exc
                )
        return results


# --------------------------------------------------------------------------- #
# Public matcher
# --------------------------------------------------------------------------- #


class VehicleReIDMatcher:
    """Extracts deep ReID features from vehicle crops.

    Backend selection precedence:

      1. Explicit ``backend="openvino"`` or ``backend="torchreid"`` kwarg.
      2. When ``matching_config.use_openvino_reid`` is true AND the IR file
         at ``matching_config.reid_openvino_model_dir/model.xml`` exists
         the OpenVINO backend is used.
      3. Otherwise fall back to torchreid (legacy path, slow on CPU).

    The choice is recorded on ``self.backend`` (``"openvino"`` or
    ``"torchreid"``) so callers / tests can inspect the active path.
    """

    DEFAULT_OV_MODEL_DIR = Path("models/osnet_openvino_int8")

    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        use_gpu: bool = True,
        batch_size: int = 16,
        preprocessing_config: Optional[ReIDPreprocessingConfig] = None,
        *,
        matching_config: Optional[MatchingConfig] = None,
        backend: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.preprocessing_config = preprocessing_config or ReIDPreprocessingConfig()
        self._matching_config = matching_config or MatchingConfig()

        env_backend = (
            os.environ.get("REID_BACKEND", "").strip().lower() or None
        )
        requested = (backend or env_backend or "").lower() or None

        backend_impl, backend_name = self._select_backend(requested, use_gpu)
        self._backend = backend_impl
        self.backend: str = backend_name

        # Expose attributes some tests rely on so the public surface
        # stays stable irrespective of backend choice.
        self.input_size: Tuple[int, int] = getattr(
            backend_impl, "input_size", (192, 96)
        )
        self.norm_mean = list(getattr(backend_impl, "norm_mean", [0.485, 0.456, 0.406]))
        self.norm_std = list(getattr(backend_impl, "norm_std", [0.229, 0.224, 0.225]))

        # For backwards compatibility (``self.model`` previously referred to
        # the torch.nn module — some legacy code peeks at it). We surface
        # the underlying backend object so naive ``hasattr`` checks still
        # see it, but it is no longer a torch model when OpenVINO is active.
        self.model = backend_impl

        # Expose the device attribute torchreid users expect. For OpenVINO
        # the "device" is the OV plugin string ("CPU"/"GPU") and there is
        # no torch.device — we keep a stub for compatibility.
        try:
            import torch
            self.device = getattr(backend_impl, "device", torch.device("cpu"))
        except ImportError:
            self.device = getattr(backend_impl, "device", "cpu")

        if self.backend == "openvino":
            # The OpenVINO path deliberately applies NO CLAHE: the production
            # IR (e.g. PS_carMatching) is trained and calibrated on raw
            # squish-resized crops, so luminance normalisation here would
            # shift embeddings off the trained distribution.
            # ReIDPreprocessingConfig only affects the torchreid fallback.
            pp_status = "OFF (model contract: squish, no CLAHE)"
            if self.preprocessing_config.enabled:
                logger.info(
                    "[REID] Note: ReID preprocessing (CLAHE) is configured ON "
                    "but is NOT applied by the OpenVINO backend — the IR's "
                    "trained contract is squish-resize without CLAHE. The "
                    "flag only affects the legacy torchreid fallback."
                )
        else:
            pp_status = "ON" if self.preprocessing_config.enabled else "OFF"
        logger.info(
            "[REID] Active backend: %s (preprocessing=%s, input=%dx%d)",
            self.backend, pp_status,
            self.input_size[0], self.input_size[1],
        )

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #

    def _select_backend(self, requested: Optional[str], use_gpu: bool):
        cfg = self._matching_config
        ov_dir = _resolve_model_dir(
            Path(cfg.reid_openvino_model_dir) if getattr(cfg, "reid_openvino_model_dir", None)
            else self.DEFAULT_OV_MODEL_DIR
        )
        ov_xml_exists = _ir_path(ov_dir).exists()

        prefer_ov = (
            requested == "openvino"
            or (
                requested is None
                and getattr(cfg, "use_openvino_reid", True)
                and ov_xml_exists
            )
        )

        if prefer_ov:
            try:
                from src.reid_matcher.reid_openvino_backend import (
                    OpenVINOReIDBackend,
                )
                input_size = tuple(getattr(cfg, "reid_input_size", (192, 96)))
                impl = OpenVINOReIDBackend(
                    model_dir=ov_dir,
                    input_size=input_size,
                    device="CPU",
                )
                return impl, "openvino"
            except Exception as exc:
                if requested == "openvino":
                    raise
                logger.warning(
                    "[REID] OpenVINO backend unavailable (%r); falling back "
                    "to torchreid.",
                    exc,
                )

        # torchreid fallback (legacy)
        impl = _TorchreidBackend(
            model_name=self.model_name,
            use_gpu=use_gpu,
            batch_size=self.batch_size,
            preprocessing_config=self.preprocessing_config,
        )
        return impl, "torchreid"

    # ------------------------------------------------------------------ #
    # Public API — byte-compatible with the pre-WS-A implementation.
    # ------------------------------------------------------------------ #

    def extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract a single normalised 512-D feature vector.

        Returns ``None`` for empty/invalid input. Output dtype is
        ``float32`` (matches the pre-WS-A behaviour).
        """
        if hasattr(self._backend, "extract"):
            return _ensure_f32(self._backend.extract(image))
        return None

    # Some legacy callers expect ``extract_features`` (plural).
    # Provide it as a thin alias.
    extract_features = extract_feature

    def extract_features_batch(
        self, images: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """Extract features for a list of crops.

        For the OpenVINO backend, ALL valid crops go through one inference
        call (true batched forward pass). The torchreid backend chunks at
        ``self.batch_size`` to preserve historical memory behaviour.

        Returns a list of the same length as ``images`` with ``None`` for
        failed entries. Order is preserved.
        """
        if not images:
            return []
        if hasattr(self._backend, "extract_batch"):
            return [_ensure_f32(v) for v in self._backend.extract_batch(images)]
        return [self.extract_feature(img) for img in images]

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))


def _ensure_f32(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalise the dtype/contiguity of feature vectors for callers."""
    if vec is None:
        return None
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return np.ascontiguousarray(vec)


# --------------------------------------------------------------------------- #
# Thread-safe singleton
# --------------------------------------------------------------------------- #


_matcher_instance: Optional[VehicleReIDMatcher] = None
_matcher_lock = threading.Lock()


def get_reid_matcher(
    preprocessing_config: Optional[ReIDPreprocessingConfig] = None,
    matching_config: Optional[MatchingConfig] = None,
) -> VehicleReIDMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = VehicleReIDMatcher(
                    preprocessing_config=preprocessing_config,
                    matching_config=matching_config,
                )
    return _matcher_instance


def _reset_matcher_for_tests() -> None:
    """Drop the cached singleton — used by tests to force backend swap."""
    global _matcher_instance
    with _matcher_lock:
        _matcher_instance = None
