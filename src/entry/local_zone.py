"""In-process bridge from VA RTSP entry zones to Entry V2 crossings.

The camera webhooks still terminate at PMS-AI.  This module consumes only the
vehicle detections that VA already produces from RTSP frames and submits a
metadata-plus-crop crossing to the shared :class:`EntryCoordinator`.  Images
remain in memory and are released after inference; nothing is written here.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .coordinator import EntryCoordinator
from .domain import CrossingInput, CrossingRole, norm_camera_id


logger = logging.getLogger(__name__)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
VA_HOST_GRAB_TIMESTAMP_SOURCE = "va_host_grab"
VA_HOST_DIRECT_READ_TIMESTAMP_SOURCE = "va_host_direct_read"


@dataclass(frozen=True)
class LocalZonePolicy:
    """One explicitly calibrated, one-way VA entry zone."""

    camera_id: str
    canonical_zone_id: str
    aliases: frozenset[str]
    direction: str
    role: CrossingRole

    @property
    def line_id(self) -> str:
        # The canonical zone identifier is also the synthetic line identifier.
        # This keeps the existing Entry V2 role/camera/line/direction policy as
        # the single deployment allow-list for both HTTP and local crossings.
        return self.canonical_zone_id


@dataclass(frozen=True)
class LocalVehicleCrop:
    """A tracked vehicle currently inside a configured entry polygon."""

    track_id: int
    image: np.ndarray
    quality: float


@dataclass
class _Visit:
    sequence: int
    track_id: int
    captured_at: datetime
    timestamp_source: str
    frames_seen: int = 0
    missed_frames: int = 0
    outside_frames: int = 0
    ambiguous: bool = False
    submitted: bool = False
    crops: list[Tuple[float, int, np.ndarray]] = field(default_factory=list)


@dataclass
class _ZoneState:
    initialized: bool = False
    armed: bool = False
    blocked_until_empty: bool = False
    empty_frames: int = 0
    next_sequence: int = 0
    active: Optional[_Visit] = None
    last_observation_token: object = None


_POLICIES: Tuple[LocalZonePolicy, ...] = (
    LocalZonePolicy(
        camera_id="CAM-23",
        canonical_zone_id="Park_Entry",
        aliases=frozenset({"parkentry"}),
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
    ),
    LocalZonePolicy(
        camera_id="CAM-03",
        # Preserve the deployed identifier's historical spelling while also
        # accepting correctly-spelled/hyphenated aliases from the DB.
        canonical_zone_id="B1_Entrence",
        aliases=frozenset({"b1entry", "b1entrance", "b1entrence"}),
        direction="b-entry",
        role=CrossingRole.FALLBACK,
    ),
)


def normalized_zone_id(value: str) -> str:
    return _NON_ALNUM.sub("", str(value or "").lower())


def local_zone_policy(camera_id: str, zone_id: str) -> Optional[LocalZonePolicy]:
    camera_key = norm_camera_id(camera_id)
    zone_key = normalized_zone_id(zone_id)
    return next(
        (
            policy
            for policy in _POLICIES
            if norm_camera_id(policy.camera_id) == camera_key
            and zone_key in policy.aliases
        ),
        None,
    )


class LocalZoneCrossingBridge:
    """Debounce zone visits and submit bounded, non-blocking Entry V2 work.

    ``observe`` is called only by the engine's single post-processing thread.
    Coordinator inference/callback delivery runs on one dedicated worker so a
    slow OCR or PMS request cannot stall slot occupancy updates.
    """

    def __init__(
        self,
        coordinator: EntryCoordinator,
        *,
        stable_frames: int = 2,
        initial_empty_frames_to_arm: int = 2,
        outside_frames_to_rearm: int = 2,
        empty_frames_to_rearm: int = 8,
        max_ingest_retries: int = 1,
        max_queued: Optional[int] = None,
    ):
        if stable_frames < 2:
            raise ValueError("stable_frames must be at least 2")
        if initial_empty_frames_to_arm < 2:
            raise ValueError("initial_empty_frames_to_arm must be at least 2")
        if outside_frames_to_rearm < 2:
            raise ValueError("outside_frames_to_rearm must be at least 2")
        if empty_frames_to_rearm < 2:
            raise ValueError("empty_frames_to_rearm must be at least 2")
        if max_ingest_retries < 0:
            raise ValueError("max_ingest_retries cannot be negative")

        self._coordinator = coordinator
        self._stable_frames = stable_frames
        self._initial_empty_frames_to_arm = initial_empty_frames_to_arm
        self._outside_frames_to_rearm = outside_frames_to_rearm
        self._empty_frames_to_rearm = empty_frames_to_rearm
        self._max_ingest_retries = int(max_ingest_retries)
        queue_capacity = (
            max_queued or coordinator.settings.max_concurrent_ingest_requests
        )
        self._queue_capacity = max(1, int(queue_capacity))
        self._capacity = threading.BoundedSemaphore(self._queue_capacity)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._states: Dict[Tuple[str, str], _ZoneState] = {}
        self._futures: set[Future] = set()
        self._lock = threading.Lock()
        self._closed = False
        self._last_error = ""
        self._capacity_rejected_crossings: set[str] = set()
        self._invalid_timestamp_cameras: set[str] = set()
        self._metrics = {
            "visits_started": 0,
            "visits_ambiguous": 0,
            "submissions_queued": 0,
            "submissions_completed": 0,
            "submissions_failed": 0,
            "submissions_retried": 0,
            "submissions_capacity_rejected": 0,
            "source_timestamp_rejected": 0,
            "confirmations": 0,
            "duplicate_frame_observations": 0,
        }

    def configured_policy(
        self, camera_id: str, zone_id: str
    ) -> Optional[LocalZonePolicy]:
        """Return a policy only when the existing Entry V2 allow-list enables it."""
        policy = local_zone_policy(camera_id, zone_id)
        if policy is None or not self._coordinator.available:
            return None
        cameras, lines, directions = self._coordinator.settings.crossing_policy(
            policy.role
        )
        if norm_camera_id(policy.camera_id) not in cameras:
            return None
        if policy.line_id.upper() not in lines:
            return None
        if policy.direction.lower() not in directions:
            return None
        return policy

    def observe(
        self,
        *,
        camera_id: str,
        zone_id: str,
        inside_track_ids: Iterable[int],
        outside_track_ids: Iterable[int] = (),
        crops: Sequence[LocalVehicleCrop],
        captured_at: datetime,
        timestamp_source: str,
        observation_token: object = None,
    ) -> None:
        """Advance one zone's transition state from a processed RTSP frame.

        A zone must first be observed empty, then contain exactly one stable
        track.  Startup-with-car, tracker flicker, mixed tracks, and simultaneous
        vehicles all abstain rather than fabricating an inward crossing.
        """
        policy = self.configured_policy(camera_id, zone_id)
        if policy is None or self._closed:
            return
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise ValueError("captured_at must include a timezone")
        timestamp_source = str(timestamp_source or "").strip()
        if not timestamp_source:
            raise ValueError("timestamp_source is required")

        tracks = {int(track_id) for track_id in inside_track_ids}
        outside_tracks = {int(track_id) for track_id in outside_track_ids}
        usable = {
            int(item.track_id): item
            for item in crops
            if item.image is not None and getattr(item.image, "size", 0) > 0
        }
        state_key = (
            norm_camera_id(policy.camera_id),
            normalized_zone_id(policy.canonical_zone_id),
        )
        state = self._states.setdefault(state_key, _ZoneState())
        if (
            observation_token is not None
            and observation_token == state.last_observation_token
        ):
            # Duplicate aliases for one canonical DB zone are iterated in the
            # same engine frame. They must not satisfy a two-frame transition.
            self._increment("duplicate_frame_observations")
            return
        state.last_observation_token = observation_token

        if not state.initialized:
            state.initialized = True
            if tracks:
                # We did not witness the outside->inside edge. Require a clean
                # empty re-arm before accepting a later physical visit.
                state.blocked_until_empty = True
                return
            self._observe_empty(state, self._initial_empty_frames_to_arm)
            return

        visit = state.active
        if visit is not None:
            if not tracks:
                if visit.track_id in outside_tracks:
                    # Seeing the same stable track beyond the polygon is much
                    # stronger exit evidence than YOLO simply omitting it. Two
                    # consecutive outside observations re-arm promptly.
                    visit.outside_frames += 1
                    visit.missed_frames = 0
                    if visit.outside_frames >= self._outside_frames_to_rearm:
                        self._rearm_after_visit(state)
                else:
                    # A missing detection is UNKNOWN, not proof that a waiting
                    # car moved. Hold across short detector flicker and use a
                    # longer consecutive-empty escape hatch for tracker loss.
                    visit.outside_frames = 0
                    visit.missed_frames += 1
                    if visit.missed_frames >= self._empty_frames_to_rearm:
                        self._rearm_after_visit(state)
                return

            visit.missed_frames = 0
            visit.outside_frames = 0
            if len(tracks) != 1 or visit.track_id not in tracks:
                self._mark_ambiguous(state, visit)
                return
            visit.frames_seen += 1
            observation = usable.get(visit.track_id)
            if observation is not None and not visit.ambiguous and not visit.submitted:
                self._retain_crop(visit, observation)
            self._submit_if_ready(policy, zone_id, visit)
            return

        if not tracks:
            threshold = (
                self._initial_empty_frames_to_arm
                if state.next_sequence == 0 and not state.blocked_until_empty
                else self._empty_frames_to_rearm
            )
            self._observe_empty(state, threshold)
            return

        if state.blocked_until_empty or not state.armed:
            # Empty evidence must be consecutive. A vehicle reappearing while
            # the episode is blocked restarts the fail-closed re-arm count.
            state.empty_frames = 0
            return
        if len(tracks) != 1:
            state.blocked_until_empty = True
            state.armed = False
            state.empty_frames = 0
            self._increment("visits_ambiguous")
            return

        track_id = next(iter(tracks))
        state.next_sequence += 1
        visit = _Visit(
            sequence=state.next_sequence,
            track_id=track_id,
            captured_at=captured_at.astimezone(timezone.utc),
            timestamp_source=timestamp_source,
            frames_seen=1,
        )
        state.active = visit
        state.armed = False
        state.empty_frames = 0
        self._increment("visits_started")
        observation = usable.get(track_id)
        if observation is not None:
            self._retain_crop(visit, observation)
        self._submit_if_ready(policy, zone_id, visit)

    def _observe_empty(self, state: _ZoneState, threshold: int) -> None:
        state.empty_frames += 1
        if state.empty_frames >= threshold:
            state.armed = True
            state.blocked_until_empty = False

    @staticmethod
    def _rearm_after_visit(state: _ZoneState) -> None:
        state.active = None
        state.empty_frames = 0
        state.armed = True
        state.blocked_until_empty = False

    def _mark_ambiguous(self, state: _ZoneState, visit: _Visit) -> None:
        if not visit.ambiguous:
            self._increment("visits_ambiguous")
        visit.ambiguous = True
        visit.crops.clear()
        state.blocked_until_empty = True

    def _retain_crop(self, visit: _Visit, observation: LocalVehicleCrop) -> None:
        visit.crops.append(
            (
                float(observation.quality),
                visit.frames_seen,
                observation.image.copy(),
            )
        )
        # Keep a small quality reservoir while waiting for the stable edge.
        reservoir = max(4, self._coordinator.settings.max_images_per_event * 2)
        if len(visit.crops) > reservoir:
            visit.crops.sort(key=lambda item: (item[0], item[1]), reverse=True)
            del visit.crops[reservoir:]

    def _submit_if_ready(
        self,
        policy: LocalZonePolicy,
        observed_zone_id: str,
        visit: _Visit,
    ) -> None:
        if (
            visit.submitted
            or visit.ambiguous
            or visit.frames_seen < self._stable_frames
            or len(visit.crops) < self._stable_frames
        ):
            return

        selected = sorted(
            visit.crops,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[: self._coordinator.settings.max_images_per_event]
        selected.sort(key=lambda item: item[1])
        encoded = self._encode_crops(item[2] for item in selected)
        if len(encoded) < self._stable_frames:
            self._increment("submissions_failed")
            logger.warning(
                "[EntryV2Local] insufficient encodable crops camera=%s zone=%s "
                "track=%s",
                policy.camera_id,
                observed_zone_id,
                visit.track_id,
            )
            return

        seed = "|".join(
            (
                norm_camera_id(policy.camera_id),
                normalized_zone_id(policy.canonical_zone_id),
                str(visit.sequence),
                str(visit.track_id),
                visit.captured_at.isoformat(),
            )
        )
        event_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
        crossing_id = f"va-zone-{event_hash}"
        request = CrossingInput(
            crossing_id=crossing_id,
            source_event_id=crossing_id,
            camera_id=policy.camera_id,
            captured_at=visit.captured_at,
            line_id=policy.line_id,
            direction=policy.direction,
            role=policy.role,
            metadata={
                "source": "va_local_zone",
                "crossing_source": "va_local_zone",
                "zone_id": observed_zone_id,
                "canonical_zone_id": policy.canonical_zone_id,
                "track_id": visit.track_id,
                "visit_sequence": visit.sequence,
                "timestamp_source": visit.timestamp_source,
                "direction_evidence": "deployment_calibrated_one_way_polygon",
            },
        )
        if not self._queue(request, encoded, retry_attempt=0):
            return
        visit.submitted = True
        visit.crops.clear()

    def _encode_crops(self, crops: Iterable[np.ndarray]) -> Tuple[bytes, ...]:
        encoded = []
        for crop in crops:
            try:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
            except cv2.error:
                logger.warning(
                    "[EntryV2Local] rejected an unencodable vehicle crop",
                    exc_info=True,
                )
                continue
            if not ok:
                continue
            payload = buffer.tobytes()
            if len(payload) <= self._coordinator.settings.max_image_bytes:
                encoded.append(payload)
        return tuple(encoded)

    def _queue(
        self,
        request: CrossingInput,
        images: Tuple[bytes, ...],
        *,
        retry_attempt: int,
    ) -> bool:
        if self._closed:
            return False
        if not self._capacity.acquire(blocking=False):
            self._mark_capacity_rejected(request)
            return False
        self._clear_capacity_saturation(request)

        with self._lock:
            if self._closed:
                self._capacity.release()
                return False
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="entry-v2-local",
                )
            try:
                future = self._executor.submit(
                    self._coordinator.ingest_crossing,
                    request,
                    images,
                )
            except Exception:
                self._capacity.release()
                self._metrics["submissions_failed"] += 1
                self._last_error = "queue_submit_failed"
                logger.exception("[EntryV2Local] could not queue local crossing")
                return False
            self._futures.add(future)
        self._increment("submissions_queued")
        future.add_done_callback(
            lambda completed: self._submission_done(
                completed,
                request,
                images,
                retry_attempt=retry_attempt,
            )
        )
        return True

    def _submission_done(
        self,
        future: Future,
        request: CrossingInput,
        images: Tuple[bytes, ...],
        *,
        retry_attempt: int,
    ) -> None:
        retry = False
        try:
            result = future.result()
        except Exception:
            self._increment("submissions_failed")
            with self._lock:
                self._last_error = "crossing_ingest_failed"
                retry = not self._closed and retry_attempt < self._max_ingest_retries
            logger.exception("[EntryV2Local] local zone crossing failed")
        else:
            self._increment("submissions_completed")
            if result.decision_status == "confirmed":
                self._increment("confirmations")
            logger.info(
                "[EntryV2Local] crossing=%s accepted=%s duplicate=%s "
                "decision=%s callback_delivered=%s",
                result.resource_id,
                result.accepted,
                result.duplicate,
                result.decision_status,
                result.callback_delivered,
            )
        finally:
            self._capacity.release()
        if retry:
            self._increment("submissions_retried")
            logger.warning(
                "[EntryV2Local] retrying crossing=%s attempt=%s/%s",
                request.crossing_id,
                retry_attempt + 1,
                self._max_ingest_retries,
            )
            self._queue(
                request,
                images,
                retry_attempt=retry_attempt + 1,
            )
        # Keep the completed future visible until any retry has been installed;
        # otherwise wait_for_idle could observe a false empty gap between them.
        with self._lock:
            self._futures.discard(future)

    def metrics(self) -> Dict[str, object]:
        with self._lock:
            result: Dict[str, object] = dict(self._metrics)
            result["inflight"] = len(self._futures)
            result["queue_capacity"] = self._queue_capacity
            result["queue_saturated"] = bool(self._capacity_rejected_crossings)
            result["queue_rejected_crossing_count"] = len(
                self._capacity_rejected_crossings
            )
            result["source_timestamp_invalid"] = bool(
                self._invalid_timestamp_cameras
            )
            result["invalid_timestamp_cameras"] = sorted(
                self._invalid_timestamp_cameras
            )
            result["enabled"] = self._coordinator.available
            result["healthy"] = not (
                self._last_error
                or self._capacity_rejected_crossings
                or self._invalid_timestamp_cameras
            )
            result["last_error"] = (
                self._last_error
                or (
                    "local_zone_source_timestamp_invalid"
                    if self._invalid_timestamp_cameras
                    else ""
                )
                or (
                    "local_zone_queue_capacity_exceeded"
                    if self._capacity_rejected_crossings
                    else ""
                )
                or None
            )
        return result

    def mark_source_timestamp_invalid(self, camera_id: str) -> None:
        """Expose and latch an active missing/invalid frame timestamp."""
        normalized_camera = norm_camera_id(camera_id)
        with self._lock:
            self._metrics["source_timestamp_rejected"] += 1
            first_rejection = normalized_camera not in self._invalid_timestamp_cameras
            self._invalid_timestamp_cameras.add(normalized_camera)
        if first_rejection:
            logger.warning(
                "[EntryV2Local] rejected local-zone frames without a valid host "
                "grab timestamp camera=%s",
                camera_id,
            )

    def mark_source_timestamp_valid(self, camera_id: str) -> None:
        """Clear the transient timestamp fault after a valid frame arrives."""
        normalized_camera = norm_camera_id(camera_id)
        with self._lock:
            recovered = normalized_camera in self._invalid_timestamp_cameras
            self._invalid_timestamp_cameras.discard(normalized_camera)
        if recovered:
            logger.info(
                "[EntryV2Local] local-zone source timestamps recovered camera=%s",
                camera_id,
            )

    def _mark_capacity_rejected(self, request: CrossingInput) -> None:
        with self._lock:
            self._metrics["submissions_capacity_rejected"] += 1
            first_rejection = (
                request.crossing_id not in self._capacity_rejected_crossings
            )
            self._capacity_rejected_crossings.add(request.crossing_id)
        if first_rejection:
            logger.warning(
                "[EntryV2Local] queue full; crossing remains unsubmitted camera=%s "
                "zone=%s",
                request.camera_id,
                request.line_id,
            )

    def _clear_capacity_saturation(self, request: CrossingInput) -> None:
        with self._lock:
            recovered = request.crossing_id in self._capacity_rejected_crossings
            self._capacity_rejected_crossings.discard(request.crossing_id)
        if recovered:
            logger.info(
                "[EntryV2Local] previously rejected crossing queued crossing=%s",
                request.crossing_id,
            )

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for worker completion in tests/controlled shutdowns, never in-frame."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                futures = tuple(self._futures)
            if not futures:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            try:
                futures[0].result(timeout=remaining)
            except Exception:
                # Completion callbacks own error accounting. A failed task is
                # still idle once removed from the in-flight set.
                pass

    def _increment(self, name: str) -> None:
        with self._lock:
            self._metrics[name] += 1

    def close(self, *, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)
