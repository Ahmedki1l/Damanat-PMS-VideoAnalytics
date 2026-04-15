"""
tracking_manager.py — Manages track state, feature smoothing, and identity linkage.

This component sits between the low-level tracker (ByteTrack) and the 
vehicle registry. It handles the temporal aspects of vehicle ReID, 
such as feature smoothing (EMA) and deciding when to re-identify.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.detection.detector import Detection
from src.reid_matcher import VehicleReIDMatcher, get_reid_matcher

logger = logging.getLogger(__name__)

class TrackState:
    """
    Per-track state including smoothed features and smoothing history.
    """
    def __init__(self, track_id: int, camera_id: str, alpha: float = 0.9):
        self.track_id = track_id
        self.camera_id = camera_id
        self.alpha = alpha  # Smoothing factor (higher = more weight to history)
        
        self.smoothed_feature: Optional[np.ndarray] = None
        self.last_updated = time.time()
        self.update_count = 0
        
        # Meta-data for fusion
        self.plate: Optional[str] = None
        self.session_id: Optional[str] = None

    def update_feature(self, current_feature: np.ndarray):
        """
        Update the track's feature vector using Exponential Moving Average (EMA).
        
        new_feature = alpha * old_feature + (1 - alpha) * current_feature
        """
        if current_feature is None:
            return

        if self.smoothed_feature is None:
            self.smoothed_feature = current_feature.copy()
        else:
            # EMA formula
            self.smoothed_feature = (self.alpha * self.smoothed_feature) + ((1.0 - self.alpha) * current_feature)
            
            # Re-normalize after smoothing to keep it on the unit hypersphere
            norm = np.linalg.norm(self.smoothed_feature)
            if norm > 1e-6:
                self.smoothed_feature /= norm

        self.last_updated = time.time()
        self.update_count += 1

class TrackingManager:
    """
    Orchestrates tracking and ReID feature management for a single camera stream.
    """
    def __init__(self, camera_id: str, reid_matcher: Optional[VehicleReIDMatcher] = None):
        self.camera_id = camera_id
        self.matcher = reid_matcher or get_reid_matcher()
        self.tracks: Dict[int, TrackState] = {}
        
        # Configurable smoothing
        self.ema_alpha = 0.9
        
    def process_detections(self, frame: np.ndarray, detections: List[Detection]):
        """
        Update track features for all active detections in a frame.
        
        Optimized to avoid extracting features for every vehicle in every frame.
        Usually, we extract periodically or when the vehicle first appears.
        """
        active_tids = set()
        to_extract_indices = []
        to_extract_crops = []

        for i, det in enumerate(detections):
            if det.track_id == -1:
                continue
            
            active_tids.add(det.track_id)
            
            if det.track_id not in self.tracks:
                self.tracks[det.track_id] = TrackState(det.track_id, self.camera_id, alpha=self.ema_alpha)
            
            track = self.tracks[det.track_id]
            
            # Decision Logic: When to extract ReID?
            # 1. New track
            # 2. Periodically (e.g. every 10 frames or 2 seconds) to handle perspective change
            # 3. If quality is much better than before (handled by engine burst logic optionally)
            
            # Extract densely while the track is still young so cross-camera
            # handoff gets several viewpoints, then back off to a lighter cadence.
            should_extract = (
                track.update_count == 0
                or track.update_count < 10
                or track.update_count % 10 == 0
            )
            if should_extract:
                crop = self._crop_vehicle(frame, det)
                if crop is not None:
                    to_extract_indices.append(i)
                    to_extract_crops.append(crop)

        # Batch extraction
        if to_extract_crops:
            features = self.matcher.extract_features_batch(to_extract_crops)
            for i, feat in enumerate(features):
                if feat is not None:
                    tid = detections[to_extract_indices[i]].track_id
                    self.tracks[tid].update_feature(feat)

        # Cleanup lost tracks (not seen in this frame)
        # Note: In a real system, we might keep them longer if they might reappear, 
        # but the YOLO tracker already handles short-term occlusion.
        self._cleanup_stale_tracks(active_tids)

    def _crop_vehicle(self, frame: np.ndarray, det: Detection) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _cleanup_stale_tracks(self, active_tids: set):
        # Remove tracks not seen for 1 minute
        now = time.time()
        tids_to_remove = [
            tid for tid, state in self.tracks.items()
            if tid not in active_tids and (now - state.last_updated) > 60
        ]
        for tid in tids_to_remove:
            del self.tracks[tid]

    def get_track_feature(self, track_id: int) -> Optional[np.ndarray]:
        """Return the smoothed feature for a track."""
        if track_id in self.tracks:
            return self.tracks[track_id].smoothed_feature
        return None

    def bind_plate(self, track_id: int, plate: str):
        """Link a license plate to a track."""
        if track_id in self.tracks:
            self.tracks[track_id].plate = plate
