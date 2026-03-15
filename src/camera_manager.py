"""
camera_manager.py — Multi-camera RTSP stream manager.

Manages 12 RTSP camera streams with round-robin scheduling.
Each camera gets ~1 frame per second (sufficient for parking detection).

Features:
  - Round-robin frame reading (one camera at a time)
  - Automatic stream reconnection on failure
  - Sub-stream (720p) by default for CPU efficiency
  - Per-camera slot polygon loading
  - Shares a single YOLO model instance across all cameras
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: str
    name: str
    floor: str
    ip: str
    user: str
    password: str
    slots_file: str
    rtsp_url: str = ""  # Built from ip/user/password

    def build_rtsp_url(self, channel: int = 102) -> str:
        """
        Build RTSP URL from credentials.

        Channel 101 = main stream (4K), 102 = sub stream (720p).
        """
        self.rtsp_url = (
            f"rtsp://{self.user}:{self.password}@{self.ip}:554"
            f"/Streaming/Channels/{channel}"
        )
        return self.rtsp_url


class CameraStream:
    """Manages a single RTSP stream with reconnection logic."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self._last_reconnect = 0.0
        self._reconnect_interval = 10.0  # seconds between reconnect attempts

    def open(self) -> bool:
        """Open the RTSP stream."""
        try:
            self.cap = cv2.VideoCapture(self.config.rtsp_url)
            # Set buffer size to 1 to always get the latest frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_open = self.cap.isOpened()
            if self.is_open:
                print(f"  [OK] {self.config.id} ({self.config.name}) — connected")
            else:
                print(f"  [FAIL] {self.config.id} ({self.config.name}) — cannot open stream")
            return self.is_open
        except Exception as e:
            print(f"  [FAIL] {self.config.id} ({self.config.name}) — {e}")
            self.is_open = False
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame. Returns (success, frame).

        If the stream has dropped, attempts reconnection with rate limiting.
        """
        if not self.is_open or self.cap is None:
            return self._try_reconnect()

        ret, frame = self.cap.read()
        if not ret:
            self.is_open = False
            return self._try_reconnect()

        return True, frame

    def _try_reconnect(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Rate-limited reconnection attempt."""
        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            return False, None

        self._last_reconnect = now
        print(f"[WARN] {self.config.id} — reconnecting...")
        self.close()
        if self.open():
            return self.read()
        return False, None

    def close(self):
        """Release the stream."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_open = False


class CameraManager:
    """
    Manages multiple camera streams with round-robin scheduling.

    Cycles through all cameras, reading one frame at a time.
    This keeps CPU usage low while covering all cameras.
    """

    def __init__(self, camera_configs: List[CameraConfig]):
        """
        Args:
            camera_configs: List of CameraConfig for each camera.
        """
        self.cameras: Dict[str, CameraStream] = {}
        self.camera_ids: List[str] = []
        self._current_index: int = 0

        for config in camera_configs:
            stream = CameraStream(config)
            self.cameras[config.id] = stream
            self.camera_ids.append(config.id)

    def open_all(self) -> int:
        """
        Open all camera streams.

        Returns:
            Number of successfully opened streams.
        """
        print(f"\n[INFO] Opening {len(self.cameras)} camera streams...")
        success_count = 0
        for cam_id in self.camera_ids:
            if self.cameras[cam_id].open():
                success_count += 1

        print(f"[INFO] {success_count}/{len(self.cameras)} cameras connected.\n")
        return success_count

    def next_frame(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Read the next frame in round-robin order.

        Returns:
            (camera_id, frame) if successful, (None, None) if all cameras failed.
        """
        # Try each camera in sequence until we get a frame
        attempts = 0
        while attempts < len(self.camera_ids):
            cam_id = self.camera_ids[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.camera_ids)

            success, frame = self.cameras[cam_id].read()
            if success and frame is not None:
                return cam_id, frame

            attempts += 1

        return None, None

    def read_camera(self, camera_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a specific camera's frame (for single-camera mode)."""
        if camera_id not in self.cameras:
            return False, None
        return self.cameras[camera_id].read()

    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get config for a specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].config
        return None

    def close_all(self):
        """Release all streams."""
        for stream in self.cameras.values():
            stream.close()
        print("[INFO] All camera streams released.")

    @property
    def active_count(self) -> int:
        """Number of currently active streams."""
        return sum(1 for s in self.cameras.values() if s.is_open)

    @property
    def total_count(self) -> int:
        """Total number of cameras."""
        return len(self.cameras)
