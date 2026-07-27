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
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
VA_HOST_GRAB_TIMESTAMP_SOURCE = "va_host_grab"
VA_HOST_DIRECT_READ_TIMESTAMP_SOURCE = "va_host_direct_read"

# Zone crops are RAM-only by contract, and a visit that never reaches the
# coordinator (tracker_loss, ambiguity, queue saturation) is freed without ever
# passing through the analyzer — so nothing lands in entry_plate_crops and there
# is no artefact to inspect. That makes "did we actually capture the car?"
# unanswerable from disk, which is exactly what field calibration needs.
#
# Setting ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR writes the entry frame as it was
# captured, before any submission decision. Deliberately OFF by default: this
# persists transit imagery, so it is a diagnostic you switch on for a drive and
# switch off again, not a production default.
LOCAL_CAPTURE_DEBUG_DIR_ENV = "ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR"
ZONE_CAPTURE_SUBDIRECTORY = "entry_zone_captures"


def _safe_part(value, *, max_length: int = 32) -> str:
    cleaned = _UNSAFE_FILENAME_CHARS.sub("-", str(value or "")).strip("._-")
    return (cleaned or "unknown")[:max_length]


def _save_zone_capture_for_debug(
    image,
    *,
    camera_id: str,
    zone_id: str,
    track_id,
    sequence,
    quality: float,
) -> Optional[str]:
    """Persist one entry-zone crop when the debug dir is configured.

    Returns the written path, or None when disabled or on any failure. Never
    raises: this runs on the engine's single post-processing thread, and a full
    disk must not stall slot occupancy.
    """
    import os
    import tempfile

    base = os.environ.get(LOCAL_CAPTURE_DEBUG_DIR_ENV, "").strip()
    if not base or image is None or getattr(image, "size", 0) == 0:
        return None

    target_dir = os.path.join(base, ZONE_CAPTURE_SUBDIRECTORY)
    shape = getattr(image, "shape", None)
    dimensions = (
        f"{shape[1]}x{shape[0]}" if shape and len(shape) >= 2 else "unknown"
    )
    filename = (
        f"{time.strftime('%Y%m%d_%H%M%S')}"
        f"__camera-{_safe_part(camera_id)}"
        f"__zone-{_safe_part(zone_id)}"
        f"__track-{_safe_part(track_id, max_length=16)}"
        f"__seq-{_safe_part(sequence, max_length=8)}"
        f"__{dimensions}"
        f"__quality-{quality:.3f}.jpg"
    )
    target = os.path.join(target_dir, filename)
    temporary_path = None
    try:
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not ok:
            raise OSError("OpenCV could not encode the zone capture")
        os.makedirs(target_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=target_dir
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded.tobytes())
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    except Exception:
        logger.warning(
            "[EntryV2Local][zone-capture] save failed dir=%s; capture continues",
            target_dir,
            exc_info=True,
        )
        return None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


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
    # A SECOND, wider view of the same vehicle: the union of the car box and the
    # zone bounds, which reaches past the grille to include the bumper and number
    # plate. `image` stays tight to the car because that is what ReID wants —
    # background dilutes an appearance embedding — while plate reading needs the
    # part of the car that the detector's box excludes.
    #
    # Not submitted as evidence yet. The analyzer runs ReID on EVERY image, and
    # coordinator._require_single_vehicle_event compares all embeddings pairwise
    # against event_consistency_min_score (0.82); two zoom levels of one car
    # could fall under that and be rejected as mixed_vehicle_evidence, killing
    # the whole event. Using it for real needs a per-image role so LPD reads it
    # while ReID ignores it. Until then it is captured for inspection so plate
    # legibility can be judged before that plumbing is written.
    plate_image: Optional[np.ndarray] = None


@dataclass
class _Visit:
    sequence: int
    track_id: int
    captured_at: datetime
    timestamp_source: str
    frames_seen: int = 0
    crops_seen: int = 0
    missed_frames: int = 0
    outside_frames: int = 0
    exit_confirmed: bool = False
    ambiguous: bool = False
    submitted: bool = False
    # (quality, frame_no, reid_crop, plate_view) — the plate view is the
    # zone-union crop; None when the union path is disabled.
    crops: list[Tuple[float, int, np.ndarray, Optional[np.ndarray]]] = field(
        default_factory=list
    )
    prepared_request: Optional[CrossingInput] = None
    prepared_images: Tuple[bytes, ...] = ()
    selected_count: int = 0
    best_selected_quality: float = 0.0


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
            "visits_exit_confirmed": 0,
            "visits_finalized": 0,
            "visits_discarded_ambiguous": 0,
            "visits_discarded_insufficient": 0,
            "visits_discarded_tracker_loss": 0,
            "candidate_crops_observed": 0,
            "snapshots_selected": 0,
            "submissions_queued": 0,
            "submissions_completed": 0,
            "submissions_failed": 0,
            "submissions_retried": 0,
            "submissions_capacity_rejected": 0,
            "source_timestamp_rejected": 0,
            "confirmations": 0,
            "duplicate_frame_observations": 0,
            "best_plate_view_plate_found": 0,
            "best_plate_view_no_plate_anywhere": 0,
            "best_plate_view_rescued_from_area_pick": 0,
        }
        # Built lazily on the first flush that needs it, never in the hot path.
        self._plate_view_detector = None
        self._plate_view_detector_failed = False

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
        usable: Dict[int, LocalVehicleCrop] = {}
        for item in crops:
            if item.image is None or getattr(item.image, "size", 0) <= 0:
                continue
            track_id = int(item.track_id)
            current = usable.get(track_id)
            if current is None or float(item.quality) > float(current.quality):
                # Defensive against duplicate detector rows for one track in a
                # frame: keep the best crop, not whichever row was seen last.
                usable[track_id] = item
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
            if visit.exit_confirmed:
                # A confirmed physical exit owns this state until its prepared
                # evidence is queued. In particular, queue saturation must not
                # let a following vehicle overwrite the completed visit.
                if self._finalize_exited_visit(policy, zone_id, visit):
                    self._rearm_after_visit(state)
                return

            if not tracks:
                if visit.track_id in outside_tracks:
                    # Seeing the same stable track beyond the polygon is much
                    # stronger exit evidence than YOLO simply omitting it.
                    # Collect throughout the zone and finalize only after two
                    # consecutive explicit same-track outside observations.
                    visit.outside_frames += 1
                    visit.missed_frames = 0
                    if visit.outside_frames >= self._outside_frames_to_rearm:
                        visit.exit_confirmed = True
                        self._increment("visits_exit_confirmed")
                        if self._finalize_exited_visit(policy, zone_id, visit):
                            self._rearm_after_visit(state)
                else:
                    # A missing detection is UNKNOWN, not proof that a waiting
                    # car moved. Hold across short detector flicker and use a
                    # longer consecutive-empty escape hatch to discard a visit
                    # whose track was lost. Never turn that loss into an entry.
                    visit.outside_frames = 0
                    visit.missed_frames += 1
                    if visit.missed_frames >= self._empty_frames_to_rearm:
                        self._discard_visit(
                            policy,
                            zone_id,
                            visit,
                            reason=(
                                "ambiguous"
                                if visit.ambiguous
                                else "tracker_loss"
                            ),
                        )
                        self._rearm_after_visit(state)
                return

            visit.missed_frames = 0
            # Re-entering after one outside observation proves there was no
            # completed exit; the two outside observations must be consecutive.
            visit.outside_frames = 0
            if len(tracks) != 1 or visit.track_id not in tracks:
                self._mark_ambiguous(state, visit)
                return
            visit.frames_seen += 1
            observation = usable.get(visit.track_id)
            if observation is not None and not visit.ambiguous and not visit.submitted:
                self._retain_crop(visit, observation)
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

        # The snapshot IS taken the instant the car is first seen in the zone —
        # what waits is the submission, which needs a confirmed spatial exit and
        # >= stable_frames crops. On a transit ramp at ~0.1 fps that second frame
        # never arrives, so the entry frame was being captured and then thrown
        # away with no trace. Log the capture itself so "did we get a snapshot"
        # stops depending on whether the visit later completed.
        # Save the WIDER zone-union view when there is one: the tight ReID crop
        # stops at the grille, so it cannot answer whether the plate was captured
        # or is legible — which is the whole reason the file is written.
        image = getattr(observation, "image", None) if observation else None
        wide = getattr(observation, "plate_image", None) if observation else None
        if wide is not None and getattr(wide, "size", 0) > 0:
            image = wide
        shape = getattr(image, "shape", None)
        quality = (
            float(getattr(observation, "quality", 0.0) or 0.0)
            if observation
            else 0.0
        )
        saved = _save_zone_capture_for_debug(
            image,
            camera_id=policy.camera_id,
            zone_id=zone_id,
            track_id=track_id,
            sequence=visit.sequence,
            quality=quality,
        )
        logger.info(
            "[EntryV2Local][entry-capture] camera=%s zone=%s track=%s seq=%s "
            "captured=%s crop=%s quality=%.3f saved=%s",
            policy.camera_id,
            zone_id,
            track_id,
            visit.sequence,
            observation is not None,
            f"{shape[1]}x{shape[0]}" if shape and len(shape) >= 2 else "none",
            quality,
            saved or "-",
        )

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
        visit.crops_seen += 1
        self._increment("candidate_crops_observed")

        candidate = (
            float(observation.quality),
            visit.frames_seen,
            observation.image,
        )
        # Every usable frame is ranked, but only a bounded top-quality reservoir
        # owns image copies. This stays memory-bounded even if a car waits in the
        # zone for minutes before the barrier opens.
        reservoir = max(4, self._coordinator.settings.max_images_per_event * 2)
        if len(visit.crops) >= reservoir:
            weakest_index = min(
                range(len(visit.crops)),
                key=lambda index: (
                    visit.crops[index][0],
                    visit.crops[index][1],
                ),
            )
            weakest = visit.crops[weakest_index]
            if (candidate[0], candidate[1]) <= (weakest[0], weakest[1]):
                return
            del visit.crops[weakest_index]
        # Carry the wider plate view alongside the tight ReID crop. The reservoir
        # is what survives to flush, so a plate view that is not retained here
        # cannot be inspected later — and the entry frame, which is the one we
        # were saving, is by definition the FARTHEST the car ever is.
        wide = getattr(observation, "plate_image", None)
        visit.crops.append(
            (
                candidate[0],
                candidate[1],
                candidate[2].copy(),
                wide.copy() if wide is not None and getattr(wide, "size", 0) else None,
            )
        )

    def _finalize_exited_visit(
        self,
        policy: LocalZonePolicy,
        observed_zone_id: str,
        visit: _Visit,
    ) -> bool:
        """Queue one completed visit.

        Returns ``False`` only for a transient queue failure, in which case the
        confirmed-exit visit and its encoded evidence remain available for the
        next observation to retry. All deterministic rejection paths return
        ``True`` so the zone can safely re-arm.
        """
        if visit.submitted:
            return True
        if visit.ambiguous:
            self._discard_visit(
                policy,
                observed_zone_id,
                visit,
                reason="ambiguous",
            )
            return True

        if visit.prepared_request is None:
            if (
                visit.frames_seen < self._stable_frames
                or visit.crops_seen < self._stable_frames
                or len(visit.crops) < self._stable_frames
            ):
                self._discard_visit(
                    policy,
                    observed_zone_id,
                    visit,
                    reason="insufficient",
                )
                return True

            # Rank every retained candidate by quality. If a top crop cannot be
            # encoded, continue down the reservoir instead of losing a useful
            # lower-ranked view. The chosen payloads are restored to capture
            # order before they enter the multi-image evidence pipeline.
            selected = []
            for quality, frame_no, crop, _wide in sorted(
                visit.crops,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            ):
                encoded = self._encode_crops((crop,))
                if not encoded:
                    continue
                selected.append((frame_no, quality, encoded[0]))
                if (
                    len(selected)
                    >= self._coordinator.settings.max_images_per_event
                ):
                    break
            selected.sort(key=lambda item: item[0])

            # Encoding is complete. Release the substantially larger raw arrays
            # even if queue capacity is temporarily unavailable.
            self._save_best_plate_view(policy, observed_zone_id, visit)
            visit.crops.clear()
            if len(selected) < self._stable_frames:
                self._increment("submissions_failed")
                self._discard_visit(
                    policy,
                    observed_zone_id,
                    visit,
                    reason="insufficient",
                    detail="insufficient_encodable_crops",
                )
                return True

            visit.prepared_images = tuple(item[2] for item in selected)
            visit.selected_count = len(selected)
            visit.best_selected_quality = max(item[1] for item in selected)
            visit.prepared_request = self._build_request(
                policy,
                observed_zone_id,
                visit,
            )

        if not self._queue(
            visit.prepared_request,
            visit.prepared_images,
            retry_attempt=0,
        ):
            return False

        visit.submitted = True
        self._increment("visits_finalized")
        with self._lock:
            self._metrics["snapshots_selected"] += visit.selected_count
        logger.info(
            "[EntryV2Local] finalized crossing=%s camera=%s zone=%s track=%s "
            "frames_seen=%s candidates_seen=%s selected=%s best_quality=%.3f "
            "captured_at=%s evidence=tracked_outside_zone",
            visit.prepared_request.crossing_id,
            policy.camera_id,
            observed_zone_id,
            visit.track_id,
            visit.frames_seen,
            visit.crops_seen,
            visit.selected_count,
            visit.best_selected_quality,
            visit.captured_at.isoformat(),
        )
        visit.prepared_images = ()
        return True

    def _build_request(
        self,
        policy: LocalZonePolicy,
        observed_zone_id: str,
        visit: _Visit,
    ) -> CrossingInput:
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
        return CrossingInput(
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
                "finalization_evidence": "tracked_outside_zone",
                "frames_seen": visit.frames_seen,
                "candidate_crops_seen": visit.crops_seen,
                "selected_images": visit.selected_count,
                "best_crop_quality": visit.best_selected_quality,
            },
        )

    def _plate_view_lpd(self):
        """Lazily build the plate detector used to rank plate views.

        Only ever called at flush, once per visit, over a bounded reservoir — so
        the ~19 ms per detect never touches the per-frame path. Any failure here
        is non-fatal: ranking falls back to vehicle area, which is what this
        method replaced.
        """
        if self._plate_view_detector is not None or self._plate_view_detector_failed:
            return self._plate_view_detector
        try:
            from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector

            settings = self._coordinator.settings
            self._plate_view_detector = OpenVINOPlateRegionDetector(
                model_dir=settings.lpd_model_dir,
                confidence=settings.lpd_confidence,
                iou=settings.lpd_iou,
                num_threads=settings.lpd_threads,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self._plate_view_detector_failed = True
            logger.warning(
                "[EntryV2Local][best-plate-view] plate detector unavailable, "
                "falling back to area ranking: %s",
                exc,
            )
        return self._plate_view_detector

    @staticmethod
    def _plate_view_score(detector, image) -> Tuple[float, str]:
        """Rank a candidate by how legible its plate actually is.

        Vehicle area — what _score_snapshot_quality returns — is the wrong
        signal here. It is `area * (1 + 0.5 * sharpness)` where area reaches
        ~10^6 and the sharpness term only ever scales it by 1.0-1.5, so area
        decides, and the largest frame always wins. On a steep ramp the largest
        frame is the one where the car sits almost under the lens and the plate
        has rotated out of view: measured over 29 CAM-23 captures, every frame
        this picked with no plate in it was one of the four largest, and on one
        track it discarded a frame that read `1198 SHR` at 0.980 confidence.

        So rank on the plate instead: a detected box first, then the sharpness
        of the plate crop itself, which separated readable from unreadable
        cleanly on that sample (84-233 unreadable vs 1405-12180 readable).
        """
        if detector is None:
            return 0.0, "no-detector"
        try:
            boxes = detector.detect(image)
        except Exception:
            return 0.0, "detect-failed"
        if not boxes:
            return 0.0, "no-plate"
        x1, y1, x2, y2, conf = max(boxes, key=lambda b: b[4])
        h, w = image.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 6:
            return 0.0, "plate-degenerate"
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharpness, f"plate {x2 - x1}x{y2 - y1} conf {float(conf):.2f}"

    def _save_best_plate_view(
        self,
        policy: LocalZonePolicy,
        observed_zone_id: str,
        visit: _Visit,
    ) -> None:
        """Write the most *legible* plate view the visit collected.

        This used to write the highest-quality view, where quality means
        vehicle area — see _plate_view_score for why that reliably picked the
        one frame with no plate in it.

        Ranking is by detected-plate sharpness, falling back to the old area
        pick when no candidate yields a plate box at all (nothing better is
        available then, and writing something still beats writing nothing for
        field calibration).

        Called from BOTH flush paths: visits on a transit ramp end in
        tracker_loss far more often than they finalize, so saving only on
        success would almost never produce a file.
        """
        candidates = [
            (entry[0], entry[1], entry[3])
            for entry in visit.crops
            if len(entry) > 3
            and entry[3] is not None
            and getattr(entry[3], "size", 0)
        ]
        if not candidates:
            return

        area_pick = max(candidates, key=lambda item: item[0])

        detector = self._plate_view_lpd()
        scored = []
        for quality, frame_no, image in candidates:
            score, detail = self._plate_view_score(detector, image)
            if score > 0.0:
                scored.append((score, quality, frame_no, image, detail))

        if scored:
            score, quality, frame_no, image, detail = max(
                scored, key=lambda item: item[0]
            )
            reason = f"plate-sharpness {score:.0f} ({detail})"
            self._increment("best_plate_view_plate_found")
            if frame_no != area_pick[1]:
                self._increment("best_plate_view_rescued_from_area_pick")
                logger.info(
                    "[EntryV2Local][best-plate-view] camera=%s zone=%s track=%s "
                    "plate ranking chose frame=%s over area pick frame=%s",
                    policy.camera_id,
                    observed_zone_id,
                    visit.track_id,
                    frame_no,
                    area_pick[1],
                )
        else:
            quality, frame_no, image = area_pick
            reason = "no plate in any candidate, fell back to area"
            self._increment("best_plate_view_no_plate_anywhere")

        saved = _save_zone_capture_for_debug(
            image,
            camera_id=policy.camera_id,
            zone_id=observed_zone_id,
            track_id=visit.track_id,
            sequence=f"{visit.sequence}best{frame_no}",
            quality=quality,
        )
        if saved:
            shape = getattr(image, "shape", None)
            logger.info(
                "[EntryV2Local][best-plate-view] camera=%s zone=%s track=%s "
                "frame=%s of %s crop=%s quality=%.3f picked_by=%s saved=%s",
                policy.camera_id,
                observed_zone_id,
                visit.track_id,
                frame_no,
                visit.frames_seen,
                f"{shape[1]}x{shape[0]}" if shape and len(shape) >= 2 else "?",
                quality,
                reason,
                saved,
            )

    def _discard_visit(
        self,
        policy: LocalZonePolicy,
        observed_zone_id: str,
        visit: _Visit,
        *,
        reason: str,
        detail: str = "",
    ) -> None:
        metric = {
            "ambiguous": "visits_discarded_ambiguous",
            "insufficient": "visits_discarded_insufficient",
            "tracker_loss": "visits_discarded_tracker_loss",
        }[reason]
        self._increment(metric)
        logger.warning(
            "[EntryV2Local] discarded visit camera=%s zone=%s track=%s "
            "reason=%s detail=%s frames_seen=%s candidates_seen=%s",
            policy.camera_id,
            observed_zone_id,
            visit.track_id,
            reason,
            detail or "-",
            visit.frames_seen,
            visit.crops_seen,
        )
        # Before the reservoir is freed: this is the only chance to see the
        # closest view of the plate the visit ever held.
        self._save_best_plate_view(policy, observed_zone_id, visit)
        visit.crops.clear()
        visit.prepared_images = ()

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
        if wait and not self._closed:
            # A completed visit can be holding prepared evidence only because
            # the single ingest slot was busy. Drain that work, then give each
            # spatially confirmed visit one final chance to queue before the
            # executor is shut down. Active visits without a proven exit remain
            # unsubmitted; shutdown must never fabricate movement.
            self.wait_for_idle()
            for state_key, state in tuple(self._states.items()):
                visit = state.active
                if visit is None or not visit.exit_confirmed or visit.submitted:
                    continue
                policy = next(
                    (
                        candidate
                        for candidate in _POLICIES
                        if (
                            norm_camera_id(candidate.camera_id),
                            normalized_zone_id(candidate.canonical_zone_id),
                        )
                        == state_key
                    ),
                    None,
                )
                if policy is not None and self._finalize_exited_visit(
                    policy,
                    policy.canonical_zone_id,
                    visit,
                ):
                    self._rearm_after_visit(state)
            self.wait_for_idle()

        with self._lock:
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)
