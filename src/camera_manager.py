"""
camera_manager.py — Multi-camera RTSP stream manager with threaded grabbing.

Each camera runs a background thread that continuously reads frames,
keeping only the latest one. This prevents RTSP buffer overflow and
ensures all cameras stay connected even when processing is slow.

Features:
  - Per-camera background threads for continuous frame grabbing
  - Round-robin or all-at-once frame retrieval
  - Automatic stream reconnection on failure
  - Sub-stream (720p) by default for CPU efficiency
  - Thread-safe latest-frame access
"""

import os
import time
import threading
from dataclasses import dataclass
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
    rtsp_port: int = 554  # From the DB cameras table; 554 keeps the legacy default
    rtsp_url: str = ""  # Built from ip/user/password

    def build_rtsp_url(self, channel: int = 102) -> str:
        """
        Build RTSP URL from credentials.

        Channel 101 = main stream (4K), 102 = sub stream (720p).
        """
        self.rtsp_url = (
            f"rtsp://{self.user}:{self.password}@{self.ip}:{self.rtsp_port}"
            f"/Streaming/Channels/{channel}"
        )
        return self.rtsp_url


class CameraStream:
    """
    Manages a single RTSP stream with a background grabber thread.

    The grabber thread continuously reads frames from the RTSP stream,
    keeping only the latest one. This prevents buffer overflow and
    ensures the stream stays alive.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self.frame_width: int = 0
        self.frame_height: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_interval = 10.0
        self._last_reconnect = 0.0

    def open(self) -> bool:
        """Open the RTSP stream and start the background grabber."""
        try:
            # Set FFmpeg to use TCP before opening
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            self.cap = cv2.VideoCapture(
                self.config.rtsp_url,
                cv2.CAP_FFMPEG,
            )
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            self.is_open = self.cap.isOpened()
            if self.is_open:
                # Read actual frame dimensions from the stream
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"  [OK] {self.config.id} ({self.config.name}) "
                    f"— connected [{self.frame_width}×{self.frame_height}]"
                )
                # Start the background grabber thread
                self._stop_event.clear()
                self._grab_thread = threading.Thread(
                    target=self._grabber_loop,
                    name=f"grab-{self.config.id}",
                    daemon=True,
                )
                self._grab_thread.start()
            else:
                print(f"  [FAIL] {self.config.id} ({self.config.name}) — cannot open stream")
            return self.is_open
        except Exception as e:
            print(f"  [FAIL] {self.config.id} ({self.config.name}) — {e}")
            self.is_open = False
            return False

    def _grabber_loop(self):
        """
        Background thread: continuously grabs frames to keep stream alive.

        Only keeps the latest frame — old frames are discarded.
        This is the key to preventing RTSP buffer overflow.
        """
        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                # Stream died — try reconnecting
                time.sleep(self._reconnect_interval)
                self._reconnect()
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            else:
                # Read failed — reconnect
                self.is_open = False
                self._reconnect()

    def _reconnect(self):
        """Reconnect the stream (called from grabber thread)."""
        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            time.sleep(1)
            return

        self._last_reconnect = now
        print(f"[WARN] {self.config.id} — reconnecting...")

        if self.cap is not None:
            self.cap.release()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(
            self.config.rtsp_url,
            cv2.CAP_FFMPEG,
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        self.is_open = self.cap.isOpened()
        if self.is_open:
            print(f"  [OK] {self.config.id} ({self.config.name}) — reconnected")
        else:
            print(f"  [FAIL] {self.config.id} — reconnect failed")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame (non-blocking).

        Returns the most recent frame grabbed by the background thread.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                return True, frame
        return False, None

    def close(self):
        """Stop the grabber thread and release the stream."""
        self._stop_event.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3)
            self._grab_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_open = False
        self._latest_frame = None


class CameraManager:
    """
    Manages multiple camera streams with threaded frame grabbing.

    Each camera has its own thread that continuously reads frames.
    The main processing loop can then grab the latest frame from
    any camera instantly without blocking.
    """

    def __init__(self, camera_configs: List[CameraConfig]):
        self.cameras: Dict[str, CameraStream] = {}
        self.camera_ids: List[str] = []
        self._current_index: int = 0

        for config in camera_configs:
            stream = CameraStream(config)
            self.cameras[config.id] = stream
            self.camera_ids.append(config.id)

    def open_all(self) -> int:
        """Open all camera streams (starts background grabber threads)."""
        print(f"\n[INFO] Opening {len(self.cameras)} camera streams...")
        success_count = 0
        for cam_id in self.camera_ids:
            if self.cameras[cam_id].open():
                success_count += 1

        # Give grabber threads a moment to get first frames
        time.sleep(1)
        print(f"[INFO] {success_count}/{len(self.cameras)} cameras connected.\n")
        return success_count

    def next_frame(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Read the next frame in round-robin order.

        Since frames are grabbed in background threads, this is
        essentially instant — just picks up the latest frame.
        """
        attempts = 0
        while attempts < len(self.camera_ids):
            cam_id = self.camera_ids[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.camera_ids)

            success, frame = self.cameras[cam_id].read()
            if success and frame is not None:
                return cam_id, frame

            attempts += 1

        return None, None

    def get_resolution(self, camera_id: str) -> Tuple[int, int]:
        """
        Return the (width, height) of a camera's stream.

        Returns (0, 0) if the camera is not found or not yet open.
        Used by the engine to scale slot polygons from their reference
        resolution to the actual stream resolution.
        """
        stream = self.cameras.get(camera_id)
        if stream and stream.is_open:
            return (stream.frame_width, stream.frame_height)
        return (0, 0)

    def read_camera(self, camera_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a specific camera's latest frame."""
        if camera_id not in self.cameras:
            return False, None
        return self.cameras[camera_id].read()

    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get config for a specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].config
        return None

    def close_all(self):
        """Stop all grabber threads and release all streams."""
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
