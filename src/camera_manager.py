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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import perf_trace


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

    # FFMPEG per-decoder options. `threads;2` is load-bearing: without it FFMPEG
    # sizes EVERY capture's HEVC frame-thread pool from the visible CPUs — which
    # under the unpinned/quota regime is the whole 20-core node, i.e. 26 cameras
    # x ~20 threads ≈ 500 decoder threads fleet-wide (measured 2026-07-16: the
    # 12-camera b2 group carried ~240 of them in one process, wall-stretched
    # decode to 442-844ms/frame vs gate's 10-25ms, and starved its serial
    # inference loop to ~1 frame per 2 MINUTES per camera). Pinned/cpuset mode
    # never hit this because the affinity mask capped what FFMPEG saw. Two
    # threads drain 1080p HEVC comfortably; the total decode WORK is unchanged,
    # only the pointless parallelism is gone.
    _FFMPEG_OPTIONS = "rtsp_transport;tcp|threads;2"

    def __init__(self, config: CameraConfig, max_grab_fps: float = 0.0):
        self.config = config
        # Cap how often a drained frame is materialized and published. The
        # compressed RTSP stream itself must still be consumed continuously;
        # sleeping between reads lets FFmpeg queue old frames behind our back.
        self._max_grab_fps = max_grab_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self.frame_width: int = 0
        self.frame_height: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        # Provenance of _latest_frame, stamped at grab time so any downstream
        # stage can compute frame_age_ms and reject stale evidence, and so the
        # async pipeline can order per-camera completions by capture sequence.
        # capture_ts is a wall clock (time.time); seq is a per-camera monotonic
        # counter that increments on every successfully grabbed frame.
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0
        self._stream_epoch: int = 0
        self._stream_interrupted = True
        self._frame_lock = threading.Lock()
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_interval = 10.0
        self._last_reconnect = 0.0

    def open(self) -> bool:
        """Open the RTSP stream and start the background grabber."""
        try:
            # TCP transport + capped decoder threads (see _FFMPEG_OPTIONS).
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._FFMPEG_OPTIONS

            self.cap = cv2.VideoCapture(
                self.config.rtsp_url,
                cv2.CAP_FFMPEG,
            )
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            self.is_open = self.cap.isOpened()
            if self.is_open:
                self._mark_connected()
                # Read actual frame dimensions from the stream
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(
                    f"  [OK] {self.config.id} ({self.config.name}) "
                    f"— connected [{self.frame_width}×{self.frame_height}]"
                )
            else:
                self._mark_disconnected()
                print(f"  [FAIL] {self.config.id} ({self.config.name}) — cannot open stream")
            # A failed initial connection must keep retrying; otherwise one
            # transient startup outage leaves this camera dead until restart.
            self._start_grabber()
            return self.is_open
        except Exception as e:
            print(f"  [FAIL] {self.config.id} ({self.config.name}) — {e}")
            self.is_open = False
            self._mark_disconnected()
            self._start_grabber()
            return False

    def _start_grabber(self) -> None:
        if self._grab_thread is not None and self._grab_thread.is_alive():
            return
        self._stop_event.clear()
        self._grab_thread = threading.Thread(
            target=self._grabber_loop,
            name=f"grab-{self.config.id}",
            daemon=True,
        )
        self._grab_thread.start()

    def _grabber_loop(self):
        """
        Background thread: continuously grabs frames to keep stream alive.

        Only keeps the latest frame — old frames are discarded.
        This is the key to preventing RTSP buffer overflow.
        """
        # ``CAP_PROP_BUFFERSIZE=1`` is ignored by OpenCV's FFmpeg backend on the
        # production cameras. Sleeping between cap.read() calls therefore makes
        # a 20+ fps RTSP stream accumulate behind an 8 fps reader. Drain every
        # compressed frame with grab(), but only retrieve/materialize BGR frames
        # at the configured rate. This keeps the published frame near-live
        # without paying full-rate colorspace/copy cost.
        publish_interval = (
            1.0 / self._max_grab_fps if self._max_grab_fps and self._max_grab_fps > 0 else 0.0
        )
        next_publish_at = 0.0
        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                # Stream died — try reconnecting
                self._mark_disconnected()
                if perf_trace.enabled():
                    perf_trace.record_rtsp_grab(self.config.id, False, 0.0)
                if self._stop_event.wait(self._reconnect_interval):
                    break
                self._reconnect()
                continue

            if publish_interval > 0.0:
                ret = self._grab_frame()
                frame = None
                if ret:
                    # Evaluate the deadline AFTER the blocking grab. The frame
                    # that crosses the deadline is the one we must publish.
                    now = time.perf_counter()
                    if now < next_publish_at:
                        continue
                    ret, frame = self._retrieve_frame()
            else:
                ret, frame = self._read_frame()

            if ret and frame is not None:
                self._publish_frame(frame)
                if publish_interval > 0.0:
                    now = time.perf_counter()
                    if next_publish_at <= 0.0:
                        next_publish_at = now + publish_interval
                    else:
                        while next_publish_at <= now:
                            next_publish_at += publish_interval
            else:
                # Read failed — reconnect
                self.is_open = False
                self._mark_disconnected()
                self._reconnect()
                continue

    def _grab_frame(self) -> bool:
        """Consume one compressed frame without materializing a BGR image."""
        if self.cap is None:
            return False
        t0 = time.perf_counter()
        ok = self.cap.grab()
        if perf_trace.enabled():
            perf_trace.record_rtsp_grab(
                self.config.id, ok, (time.perf_counter() - t0) * 1000.0
            )
        return ok

    def _retrieve_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        t0 = time.perf_counter()
        result = self.cap.retrieve()
        if perf_trace.enabled():
            perf_trace.record_decode((time.perf_counter() - t0) * 1000.0)
        return result

    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        t0 = time.perf_counter()
        result = self.cap.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if perf_trace.enabled():
            perf_trace.record_rtsp_grab(self.config.id, result[0], elapsed_ms)
            perf_trace.record_decode(elapsed_ms)
        return result

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Atomically mark a resumed stream connected and publish its frame."""
        with self._frame_lock:
            if self._stream_interrupted:
                self._stream_epoch += 1
                self._stream_interrupted = False
            self.is_open = True
            self._latest_frame = frame
            self._latest_ts = time.time()
            self._latest_seq += 1

    def _mark_connected(self) -> None:
        """Start a new provenance epoch after a successful stream connection."""
        with self._frame_lock:
            if not self._stream_interrupted:
                return
            self._stream_epoch += 1
            self._stream_interrupted = False

    def _mark_disconnected(self) -> None:
        """Atomically invalidate frame and epoch for one stream interruption."""
        with self._frame_lock:
            if not self._stream_interrupted:
                self._stream_epoch += 1
                self._stream_interrupted = True
            self._latest_frame = None
            self._latest_ts = 0.0

    def _reconnect(self):
        """Reconnect the stream (called from grabber thread)."""
        now = time.time()
        if now - self._last_reconnect < self._reconnect_interval:
            self._stop_event.wait(1)
            return

        self._last_reconnect = now
        print(f"[WARN] {self.config.id} — reconnecting...")
        self._mark_disconnected()

        if self.cap is not None:
            self.cap.release()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self._FFMPEG_OPTIONS
        self.cap = cv2.VideoCapture(
            self.config.rtsp_url,
            cv2.CAP_FFMPEG,
        )
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        self.is_open = self.cap.isOpened()
        if self.is_open:
            self._mark_connected()
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [OK] {self.config.id} ({self.config.name}) — reconnected")
        else:
            self._mark_disconnected()
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

    def read_stamped(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Latest frame plus its grab-time provenance ``(ok, frame, capture_ts, seq)``.

        Same non-blocking latest-frame semantics as ``read()``, but also returns
        the wall-clock capture timestamp and the per-camera monotonic sequence
        number stamped by the grabber. The async pipeline uses ``seq`` to apply
        per-camera completions in order and ``capture_ts`` to drop stale frames.
        Returns ``(False, None, 0.0, 0)`` when no frame has been grabbed yet.
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy(), self._latest_ts, self._latest_seq
        return False, None, 0.0, 0

    def read_stamped_with_epoch(
        self,
    ) -> Tuple[bool, Optional[np.ndarray], float, int, int]:
        """Stamped frame plus the continuous-stream epoch."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return (
                    True,
                    self._latest_frame.copy(),
                    self._latest_ts,
                    self._latest_seq,
                    self._stream_epoch,
                )
            return False, None, 0.0, 0, self._stream_epoch

    @property
    def stream_epoch(self) -> int:
        with self._frame_lock:
            return self._stream_epoch

    @property
    def seconds_since_frame(self) -> float:
        """Age of the newest grabbed frame, in seconds (``inf`` if none yet).

        The grabber thread runs independently of the processing loop, so this tracks
        RTSP-stream liveness, not how fast frames are consumed. A dead stream, a stalled
        grabber, or a reconnect loop all let this climb; it is the signal ``is_open``
        misses, because ``is_open`` only flips on an outright read FAILURE. (It does NOT
        catch a stream that keeps re-sending an identical frame — ``grab()`` still
        succeeds there and ``_latest_ts`` keeps advancing; that duplicate-frame freeze is
        what the VA_OCC_TRACE ``sig=`` proof is for.)
        """
        with self._frame_lock:
            ts = self._latest_ts
        if not ts:
            return float("inf")
        return max(0.0, time.time() - ts)

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
        self._mark_disconnected()


class CameraManager:
    """
    Manages multiple camera streams with threaded frame grabbing.

    Each camera has its own thread that continuously reads frames.
    The main processing loop can then grab the latest frame from
    any camera instantly without blocking.
    """

    def __init__(self, camera_configs: List[CameraConfig], max_grab_fps: float = 0.0):
        self.cameras: Dict[str, CameraStream] = {}
        self.camera_ids: List[str] = []
        self._current_index: int = 0

        for config in camera_configs:
            stream = CameraStream(config, max_grab_fps=max_grab_fps)
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
        camera_id, frame, _, _ = self.next_frame_stamped()
        return camera_id, frame

    def next_frame_stamped(
        self,
    ) -> Tuple[Optional[str], Optional[np.ndarray], float, int]:
        """Round-robin frame with its host grab timestamp and sequence."""
        attempts = 0
        while attempts < len(self.camera_ids):
            cam_id = self.camera_ids[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.camera_ids)

            success, frame, grabbed_at, sequence = self.cameras[cam_id].read_stamped()
            if success and frame is not None:
                return cam_id, frame, grabbed_at, sequence

            attempts += 1

        return None, None, 0.0, 0

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

    def read_camera_stamped(
        self, camera_id: str
    ) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """Read a specific camera's latest frame with grab-time provenance.

        See ``CameraStream.read_stamped``. Used by the single-process async
        scheduler to pick fresh frames per camera and order completions by seq.
        """
        if camera_id not in self.cameras:
            return False, None, 0.0, 0
        return self.cameras[camera_id].read_stamped()

    def read_camera_stamped_with_epoch(
        self, camera_id: str
    ) -> Tuple[bool, Optional[np.ndarray], float, int, int]:
        if camera_id not in self.cameras:
            return False, None, 0.0, 0, 0
        return self.cameras[camera_id].read_stamped_with_epoch()

    def get_stream_epoch(self, camera_id: str) -> int:
        stream = self.cameras.get(camera_id)
        return stream.stream_epoch if stream is not None else 0

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

    def stream_health(self, max_age_s: float = 15.0) -> Dict[str, Any]:
        """Split streams into delivering / stale / down — the truth ``active_count`` hides.

        ``active_count`` counts streams whose socket is open; this counts streams actually
        producing fresh frames. A stream that opened but has since frozen or entered a
        reconnect loop is ``is_open`` True yet has a stale (or infinite) frame age — it
        lands in ``stale``, not ``delivering``.

        Returns counts plus the ids of the non-delivering streams so the health payload can
        name the offenders.
        """
        delivering, stale, down = 0, [], []
        for cam_id, s in self.cameras.items():
            if not s.is_open:
                down.append(cam_id)
            elif s.seconds_since_frame > max_age_s:
                stale.append(cam_id)
            else:
                delivering += 1
        return {
            "delivering": delivering,
            "stale": stale,
            "down": down,
            "total": len(self.cameras),
        }
