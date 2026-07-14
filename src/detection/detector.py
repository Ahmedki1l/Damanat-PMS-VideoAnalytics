"""
detector.py — YOLO inference wrapper for vehicle detection.

Wraps Ultralytics YOLO to:
  - Load a model from .pt, .onnx, or OpenVINO format.
  - Run inference on a single frame.
  - Filter results to car class only (COCO class 2).
  - Return structured Detection objects.

Usage:
    detector = VehicleDetector(config.detector)
    detections = detector.detect(frame)
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from src.config import DetectorConfig, DetectorPreprocessingConfig
from src.preprocessing import luminance_normalize_auto


@dataclass
class Detection:
    """
    A single detected vehicle.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        class_id: COCO class ID (2 = car).
        confidence: Detection confidence score [0, 1].
        track_id: Stable tracker ID (set later by tracker, -1 if untracked).
    """
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    confidence: float
    track_id: int = -1

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """
        Vehicle ground-contact point: horizontal center of the bbox at its
        bottom edge ``(x_mid, y2)``.

        This is the point tested against slot polygons (``slot_assigner``) and
        boundary/zone polygons, all of which are authored on the ground plane.
        It must match ``zoning.boundary_detector._bottom_center`` exactly so
        slot membership and boundary crossings reference the same point.

        (Historically this returned ``cy = (y1 + y2) / 1.5`` — neither the
        center nor the bottom; for boxes low in the frame it fell *below* the
        box entirely and mis-attributed slot occupancy.)
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        cy = ( y1 + y2) / 1.5
        return (cx, cy)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class VehicleDetector:
    """
    YOLO-based vehicle detector.

    Loads the model once and provides a simple detect() interface.
    Supports .pt (PyTorch), .onnx, and OpenVINO model formats.
    """

    def __init__(
        self,
        config: DetectorConfig,
        preprocessing_config: DetectorPreprocessingConfig = None,
    ):
        """
        Initialize the detector.

        Args:
            config: DetectorConfig with model_path, confidence, classes, imgsz.
            preprocessing_config: Optional luminance normalization settings.
        """
        self.config = config
        self.preprocessing_config = preprocessing_config or DetectorPreprocessingConfig()
        print(f"[INFO] Loading YOLO model from '{config.model_path}'...")
        self.model = YOLO(config.model_path)
        print(f"[INFO] Model loaded. Classes to detect: {config.classes}")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply luminance-safe preprocessing if enabled.

        Only modifies the L channel in LAB space — hue and saturation
        are preserved so downstream color matching stays stable.
        """
        pp = self.preprocessing_config
        if not pp.enabled:
            return frame

        return luminance_normalize_auto(
            frame,
            clip_limit=pp.clip_limit,
            grid_size=pp.grid_size,
            gamma_correction=pp.gamma_correction,
            dark_threshold=pp.dark_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame (without tracking).

        This is used for standalone detection. For detection + tracking,
        use the TrackedDetector wrapper instead.

        Args:
            frame: BGR image as numpy array (from cv2).

        Returns:
            List of Detection objects for vehicles found in the frame.
        """
        # Preprocess for better detection in hard lighting
        inference_frame = self._preprocess_frame(frame)

        # Run inference — verbose=False suppresses per-frame logging
        results = self.model(
            inference_frame,
            conf=self.config.confidence,
            classes=self.config.classes,
            imgsz=self.config.imgsz,
            verbose=False,
        )

        return self._parse_results(results)

    def _parse_results(
        self,
        results,
        track_ids: List[int] = None,
    ) -> List[Detection]:
        """
        Parse YOLO results into Detection objects.

        Args:
            results: Ultralytics Results object.
            track_ids: Optional list of track IDs (from tracker).

        Returns:
            List of Detection objects.
        """
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]  # Single image, so take first result

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            # Bounding box coordinates
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            # Class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())

            # Track ID (if available from tracker)
            tid = -1
            if boxes.id is not None:
                tid = int(boxes.id[i].cpu().numpy())
            elif track_ids and i < len(track_ids):
                tid = track_ids[i]

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                confidence=confidence,
                track_id=tid,
            ))

        return detections
