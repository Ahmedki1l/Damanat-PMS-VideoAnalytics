"""
tracker.py — Lightweight tracker integration using Ultralytics built-in tracking.

Ultralytics YOLO has built-in support for ByteTrack and BoT-SORT.
This wrapper combines detection + tracking into a single call via model.track(),
which keeps stable IDs across frames with persist=True.

Why not a standalone tracker?
  - Ultralytics handles the tracker lifecycle internally.
  - No extra dependency needed (no separate `deep_sort` or `norfair`).
  - ByteTrack is fast and CPU-friendly.
  - Using model.track() is simpler and avoids common integration bugs.
"""

import inspect
from typing import Dict, List

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from src.config import DetectorConfig, TrackerConfig, DetectorPreprocessingConfig
from src.detection.detector import Detection
from src.preprocessing import luminance_normalize, auto_gamma
from src import perf_trace

# Ultralytics tracker classes by their `tracker_type` config value.
_TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}

# Fallback key for callers that don't pass a camera_id (single-camera mode).
_DEFAULT_CAMERA = "__default__"


class TrackedDetector:
    """
    Combines YOLO detection with built-in tracking.

    Detection weights are shared (one model load), but ByteTrack *state* is
    kept per camera: ``detect_and_track`` swaps the active tracker in/out of
    the Ultralytics predictor keyed by ``camera_id`` before each call. This
    decouples detection (shared → flat RAM at many cameras) from tracking
    (per-camera → stable IDs), so round-robin frames from different cameras
    never corrupt each other's track state the way a single shared tracker did.
    """

    def __init__(
        self,
        detector_config: DetectorConfig,
        tracker_config: TrackerConfig,
        preprocessing_config: DetectorPreprocessingConfig = None,
    ):
        """
        Args:
            detector_config: Model path, confidence, classes, imgsz.
            tracker_config: Tracker type (bytetrack or botsort).
            preprocessing_config: Optional luminance normalization settings.
        """
        self.detector_config = detector_config
        self.tracker_config = tracker_config
        self.preprocessing_config = preprocessing_config or DetectorPreprocessingConfig()

        # One ByteTrack/BoT-SORT instance per camera_id (decoupled tracking).
        self._camera_trackers: Dict[str, object] = {}

        print(f"[INFO] Loading YOLO model from '{detector_config.model_path}'...")
        self.model = YOLO(detector_config.model_path, task="detect")

        # Resolve device: "auto" picks CUDA if available, else CPU
        if detector_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = detector_config.device

        pp_status = "ON" if self.preprocessing_config.enabled else "OFF"
        print(f"[INFO] Model loaded. Tracker: {tracker_config.type} | Device: {self.device} | Preprocessing: {pp_status}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply luminance-safe preprocessing if enabled.

        Only modifies the L channel in LAB space — hue and saturation
        are preserved so downstream color matching stays stable.
        """
        pp = self.preprocessing_config
        if not pp.enabled:
            return frame

        # Optional gamma correction for dark frames
        gamma = None
        if pp.gamma_correction:
            gamma = auto_gamma(frame, dark_threshold=pp.dark_threshold)

        return luminance_normalize(
            frame,
            clip_limit=pp.clip_limit,
            grid_size=pp.grid_size,
            gamma=gamma,
        )

    def _new_tracker(self):
        """Build a fresh tracker instance the way Ultralytics' track() does.

        Reads the same ``<type>.yaml`` (e.g. ``bytetrack.yaml``) the predictor
        would load, so per-camera trackers behave identically to the built-in
        one — only their *state* is isolated.
        """
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{self.tracker_config.type}.yaml")))
        tracker_class = _TRACKER_MAP.get(cfg.tracker_type)
        if tracker_class is None:
            raise AssertionError(
                f"Only {set(_TRACKER_MAP)} are supported, but got '{cfg.tracker_type}'"
            )
        # ``frame_rate`` is accepted by most but not all Ultralytics versions —
        # only pass it when the constructor actually takes it, so the same code
        # works across versions. When absent, the tracker uses its own default.
        kwargs = {}
        if "frame_rate" in inspect.signature(tracker_class.__init__).parameters:
            kwargs["frame_rate"] = 30
        return tracker_class(cfg, **kwargs)

    def _tracker_for(self, camera_id: str):
        """Return (lazily creating) this camera's dedicated tracker."""
        tracker = self._camera_trackers.get(camera_id)
        if tracker is None:
            tracker = self._new_tracker()
            self._camera_trackers[camera_id] = tracker
        return tracker

    def detect_and_track(
        self,
        frame: np.ndarray,
        camera_id: str = _DEFAULT_CAMERA,
    ) -> List[Detection]:
        """
        Run detection + tracking on a single frame.

        Detection uses the shared model; tracking uses this camera's own
        ByteTrack state (swapped into the predictor before the call), so IDs
        stay stable per camera even when cameras are interleaved round-robin.

        Args:
            frame: BGR image as numpy array (from cv2).
            camera_id: Identifies which per-camera tracker state to use.

        Returns:
            List of Detection objects with stable track_id values.
        """
        # Preprocess for better detection in hard lighting
        with perf_trace.stage("clahe"):
            inference_frame = self._preprocess_frame(frame)

        # Build tracker config filename — Ultralytics expects e.g. "bytetrack.yaml"
        tracker_cfg = f"{self.tracker_config.type}.yaml"

        # Swap in this camera's tracker before inference. On the very first
        # call the predictor doesn't exist yet; Ultralytics creates trackers[0]
        # during track(), and we adopt that instance for this camera afterwards.
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            predictor.trackers[0] = self._tracker_for(camera_id)

        with perf_trace.stage("infer"):
            results = self.model.track(
                inference_frame,
                conf=self.detector_config.confidence,
                classes=self.detector_config.classes,
                imgsz=self.detector_config.imgsz,
                device=self.device,
                persist=True,               # Maintain tracker state across frames
                tracker=tracker_cfg,         # e.g., "bytetrack.yaml"
                verbose=False,
            )

        # First-call bootstrap: adopt the predictor's auto-created tracker for
        # this camera so subsequent calls swap the correct instance back in.
        predictor = getattr(self.model, "predictor", None)
        if (
            camera_id not in self._camera_trackers
            and predictor is not None
            and getattr(predictor, "trackers", None)
        ):
            self._camera_trackers[camera_id] = predictor.trackers[0]

        return self._parse_results(results)

    def _parse_results(self, results) -> List[Detection]:
        """Parse YOLO tracking results into Detection objects."""
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())

            # Track ID — may be None if tracker hasn't assigned one yet
            tid = -1
            if boxes.id is not None:
                tid = int(boxes.id[i].cpu().numpy())

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                confidence=confidence,
                track_id=tid,
            ))

        return detections
