"""Fair per-camera completion handoff for asynchronous inference."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceOutput:
    """One detector completion, including the RTSP stream it belongs to."""

    camera_id: str
    frame: object
    detections: object
    sequence: int
    capture_ts: float
    stream_epoch: int
    submitted_at: float
    inferred_at: float
    unknown_reason: str = ""


class InferenceCompletionGate:
    """Accept each camera's completions once and only in its live stream epoch."""

    def __init__(self) -> None:
        self._last_applied: dict[str, tuple[int, int]] = {}

    def rejection_reason(
        self,
        completion: InferenceOutput,
        current_stream_epoch: int,
    ) -> str:
        provenance = (completion.stream_epoch, completion.sequence)
        if completion.stream_epoch != current_stream_epoch:
            return "stale_epoch"
        if provenance <= self._last_applied.get(completion.camera_id, (-1, -1)):
            return "stale"
        return ""

    def record_acceptance(self, completion: InferenceOutput) -> None:
        self._last_applied[completion.camera_id] = (
            completion.stream_epoch,
            completion.sequence,
        )


class FairLatestOutputQueue:
    """Keep one latest completion per camera and dequeue ready cameras FIFO.

    A fast camera may replace only its own pending completion. It cannot consume
    queue capacity belonging to another camera, and after it is consumed its
    next ready token joins behind every camera that was already waiting.
    """

    def __init__(self) -> None:
        self._ready_camera_ids: "queue.Queue[str]" = queue.Queue()
        self._latest_by_camera: dict[str, InferenceOutput] = {}
        self._lock = threading.Lock()
        self.replacements = 0

    def put_latest(self, item: InferenceOutput) -> None:
        with self._lock:
            existing = self._latest_by_camera.get(item.camera_id)
            if existing is not None:
                existing_provenance = (existing.stream_epoch, existing.sequence)
                item_provenance = (item.stream_epoch, item.sequence)
                if item_provenance >= existing_provenance:
                    self._latest_by_camera[item.camera_id] = item
                    self.replacements += 1
                return
            self._latest_by_camera[item.camera_id] = item
            self._ready_camera_ids.put_nowait(item.camera_id)

    def get(self, timeout: float) -> InferenceOutput:
        while True:
            camera_id = self._ready_camera_ids.get(timeout=timeout)
            with self._lock:
                item = self._latest_by_camera.pop(camera_id, None)
            if item is not None:
                return item

    def qsize(self) -> int:
        with self._lock:
            return len(self._latest_by_camera)
