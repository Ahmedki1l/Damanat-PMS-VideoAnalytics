import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from src.vehicle_registry.vehicle_registry_models import ParkEntryCandidate, PendingANPREvent

logger = logging.getLogger(__name__)


class VehicleRegistryCoreMixin:
    def _gc_loop(self) -> None:
        """Runs in a daemon thread; periodically purges stale state."""
        while True:
            time.sleep(self._GC_INTERVAL_SECONDS)
            try:
                self._cleanup_stale_data(datetime.now())
            except Exception:
                logger.exception("Unhandled error in VehicleRegistry GC loop")

    def _mark_track_seen(
        self,
        camera_id: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Refresh the last-seen time for a camera-local track binding."""
        self._track_last_seen[(camera_id, track_id)] = timestamp or datetime.now()

    def _drop_other_track_mappings_for_session(self, session_id: str, keep=None) -> None:
        """
        Enforce a single active live-track binding for a session.

        Old camera/track keys are removed so recycled tracker IDs do not inherit
        the same confirmed plate on unrelated cars later.
        """
        keys_to_remove = [
            key
            for key, sid in self._track_session_map.items()
            if sid == session_id and key != keep
        ]
        for key in keys_to_remove:
            self._track_session_map.pop(key, None)
            self._track_last_seen.pop(key, None)

    @property
    def matcher(self):
        """
        Lazy-load the image matcher exactly once, even under concurrent access.
        """
        if self._matcher is None:
            with self._matcher_lock:
                if self._matcher is None:
                    from src.image_matcher import VehicleImageMatcher

                    self._matcher = VehicleImageMatcher()
                    logger.debug("VehicleImageMatcher instantiated")
        return self._matcher

    @property
    def reid_matcher(self):
        """Lazy-load the deep ReID matcher."""
        if self._reid_matcher is None:
            with self._matcher_lock:
                if self._reid_matcher is None:
                    from src.reid_matcher import get_reid_matcher

                    self._reid_matcher = get_reid_matcher()
                    logger.debug("VehicleReIDMatcher (OSNet) instantiated")
        return self._reid_matcher

    def _enhance_snapshot_for_storage(self, image: np.ndarray) -> np.ndarray:
        """
        Apply light enhancement before persisting a snapshot for UI/API usage.
        """
        if image is None or image.size == 0:
            return image

        output = image.copy()
        h, w = output.shape[:2]
        min_edge = min(h, w)

        if min_edge > 0 and min_edge < 240:
            scale = min(2.0, 240.0 / float(min_edge))
            output = cv2.resize(
                output,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        output = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

        blur = cv2.GaussianBlur(output, (0, 0), 1.0)
        output = cv2.addWeighted(output, 1.15, blur, -0.15, 0)
        return output

    def _write_snapshot_file(
        self,
        prefix: str,
        image: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Persist an enhanced snapshot image and return its path."""
        if image is None or image.size == 0:
            return None

        now = timestamp or datetime.now()
        enhanced = self._enhance_snapshot_for_storage(image)
        filename = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self._image_dir, filename)
        cv2.imwrite(filepath, enhanced)
        return filepath

    def _store_session_reference_snapshots(
        self,
        images: List[np.ndarray],
        timestamp: Optional[datetime] = None,
    ) -> List[str]:
        """Persist one or more CAM_03 reference snapshots for a confirmed session."""
        now = timestamp or datetime.now()
        token = uuid.uuid4().hex[:12]
        paths: List[str] = []
        for idx, image in enumerate(images):
            if image is None or image.size == 0:
                continue
            suffix = "" if idx == 0 else f"_ref{idx + 1}"
            path = self._write_snapshot_file(
                f"session_{token}{suffix}",
                image,
                timestamp=now,
            )
            if path:
                paths.append(path)
        return paths

    def register_anpr_event(
        self,
        plate: str,
        direction: str,
        timestamp: Optional[datetime] = None,
        camera_id: Optional[str] = None,
    ) -> PendingANPREvent:
        """
        Register a new ANPR event.

        Entry events become pending ANPR records.
        Exit events close an existing confirmed session if found.
        """
        now = timestamp or datetime.now()

        event = PendingANPREvent(
            event_id=f"anpr_{uuid.uuid4().hex[:12]}",
            plate=plate,
            direction=direction,
            timestamp=now,
            camera_id=camera_id,
        )

        if direction == "entry":
            with self._lock:
                self._pending_events[event.event_id] = event
                self._pending_event_order.append(event.event_id)
                logger.info("[ANPR] Entry: plate=%s", plate)
        elif direction == "exit":
            self._handle_exit(plate, now)
            logger.info("[ANPR] Exit: plate=%s", plate)
        else:
            raise ValueError(f"Unsupported ANPR direction: {direction}")

        return event

    def _cleanup_stale_data(self, now: datetime):
        """Garbage collection for expired candidates and pending events."""
        with self._lock:
            stale_track_keys = [
                key
                for key, seen_at in self._track_last_seen.items()
                if (now - seen_at).total_seconds() > self.TRACK_MAPPING_EXPIRY_SECONDS
            ]
            for key in stale_track_keys:
                self._track_last_seen.pop(key, None)
                self._track_session_map.pop(key, None)

            active_orders = []
            for event_id in self._pending_event_order:
                event = self._pending_events.get(event_id)
                if not event:
                    continue
                if event.status in ("pending", "provisional"):
                    age = (now - event.timestamp).total_seconds()
                    if age > self.PENDING_ANPR_EXPIRY_SECONDS:
                        event.status = "expired"
                        if event.candidate_id:
                            candidate = self._park_entry_candidates.get(event.candidate_id)
                            if candidate is not None:
                                candidate.status = "expired"
                                candidate.bound_event_id = None
                                candidate.snapshot_image = None
                                candidate.feature_vector = None
                            event.candidate_id = None
                    else:
                        active_orders.append(event_id)
            self._pending_event_order = active_orders

            candidates_to_delete = []
            for candidate_id, candidate in self._park_entry_candidates.items():
                if candidate.status in ("open", "provisional"):
                    age = (now - candidate.last_seen_at).total_seconds()
                    if age > self.CANDIDATE_EXPIRY_SECONDS:
                        candidate.status = "expired"
                        if candidate.bound_event_id:
                            event = self._pending_events.get(candidate.bound_event_id)
                            if event is not None and event.status in (
                                "pending",
                                "provisional",
                            ):
                                event.status = "expired"
                                event.candidate_id = None
                        candidate.bound_event_id = None
                        candidate.snapshot_image = None
                        candidate.feature_vector = None
                        candidates_to_delete.append(candidate_id)
                elif candidate.status in ("confirmed", "dropped", "expired"):
                    candidates_to_delete.append(candidate_id)

            for candidate_id in candidates_to_delete:
                self._park_entry_candidates.pop(candidate_id, None)

    def open_park_entry_candidate(
        self,
        camera_id: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
    ) -> ParkEntryCandidate:
        """Create a candidate for a car entering the Park_Entry zone."""
        now = timestamp or datetime.now()
        candidate = ParkEntryCandidate(
            candidate_id=f"cand_{uuid.uuid4().hex[:12]}",
            camera_id=camera_id,
            track_id=track_id,
            entered_at=now,
            last_seen_at=now,
        )

        with self._lock:
            self._park_entry_candidates[candidate.candidate_id] = candidate

        return candidate

    def update_park_entry_candidate_snapshot(
        self,
        candidate_id: str,
        image: np.ndarray,
        quality_score: float,
        feature_vector: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Keep the best Park_Entry snapshot for this candidate.
        Replace only when the new snapshot is better.
        """
        now = timestamp or datetime.now()

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if not candidate or quality_score <= candidate.quality_score:
                if candidate:
                    candidate.last_seen_at = now
                return

        if feature_vector is None:
            feature_vector = self.reid_matcher.extract_feature(image)
            if feature_vector is not None:
                logger.debug("[REID] Extracted 512-D vector for candidate %s", candidate_id)

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if not candidate:
                return

            candidate.last_seen_at = now
            if quality_score <= candidate.quality_score:
                return

            candidate.quality_score = quality_score
            candidate.snapshot_image = image.copy()
            candidate.feature_vector = feature_vector
            filepath = self._write_snapshot_file(candidate.candidate_id, image, timestamp=now)
            candidate.snapshot_path = filepath

            logger.debug(
                "[PARK_ENTRY] Updated snapshot for %s (score=%.3f)",
                candidate.candidate_id,
                quality_score,
            )
            return filepath
