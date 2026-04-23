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

from typing import List

import torch
import numpy as np
from ultralytics import YOLO

from src.config import DetectorConfig, TrackerConfig, DetectorPreprocessingConfig
from src.detection.detector import Detection
from src.preprocessing import luminance_normalize, auto_gamma


class TrackedDetector:
    """
    Combines YOLO detection with built-in tracking.

    Uses model.track() with persist=True so that the tracker
    maintains state across consecutive frames.
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

    def detect_and_track(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection + tracking on a single frame.

        persist=True tells Ultralytics to maintain tracker state
        across calls, so the same vehicle gets the same ID.

        Args:
            frame: BGR image as numpy array (from cv2).

        Returns:
            List of Detection objects with stable track_id values.
        """
        # Preprocess for better detection in hard lighting
        inference_frame = self._preprocess_frame(frame)

        # Build tracker config filename — Ultralytics expects e.g. "bytetrack.yaml"
        tracker_cfg = f"{self.tracker_config.type}.yaml"

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
