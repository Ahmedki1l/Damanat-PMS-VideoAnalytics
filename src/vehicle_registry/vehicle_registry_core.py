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
REID_USE_MULTISHOT = os.getenv("REID_USE_MULTISHOT", "false").lower() == "true"
REID_USE_COLOR_FILTER = os.getenv("REID_USE_COLOR_FILTER", "false").lower() == "true"


class VehicleRegistryCoreMixin:
    def _gc_loop(self) -> None:
        """Runs in a daemon thread; periodically purges stale state."""
        while True:
            time.sleep(self._GC_INTERVAL_SECONDS)
            try:
                self._cleanup_stale_data(self._clock())
            except Exception:
                logger.exception("Unhandled error in VehicleRegistry GC loop")

    def _mark_track_seen(
        self,
        camera_id: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Refresh the last-seen time for a camera-local track binding."""
        self._track_last_seen[(camera_id, track_id)] = timestamp or self._clock()

    def _drop_other_track_mappings_for_session(
        self,
        session_id: str,
        keep=None,
        same_camera_only: bool = True,
    ) -> None:
        """
        Clean up stale track bindings for a session.

        By default (same_camera_only=True), only removes old track IDs on the
        SAME camera — protecting against tracker-ID recycling — while leaving
        other cameras' bindings intact so a session can be observed by multiple
        cameras simultaneously.

        Set same_camera_only=False to revert to the old behavior (remove ALL
        bindings except *keep*).
        """
        if keep is None:
            keep_camera = None
            keep_track = None
        else:
            keep_camera, keep_track = keep

        keys_to_remove = []
        for key, sid in self._track_session_map.items():
            if sid != session_id or key == keep:
                continue
            if same_camera_only and keep_camera is not None and key[0] != keep_camera:
                # Different camera — leave it alone
                continue
            keys_to_remove.append(key)

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

        now = timestamp or self._clock()
        enhanced = self._enhance_snapshot_for_storage(image)
        filename = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(self._image_dir, filename)
        cv2.imwrite(filepath, enhanced)
        return filename

    def _store_session_reference_snapshots(
        self,
        images: List[np.ndarray],
        timestamp: Optional[datetime] = None,
    ) -> List[str]:
        """Persist one or more CAM_03 reference snapshots for a confirmed session."""
        now = timestamp or self._clock()
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

    def add_session_snapshot(
        self,
        session_id: str,
        image: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Append a new snapshot image to an existing session's reference gallery.
        Also extracts and adds a new ReID feature vector to improve matching accuracy.
        """
        if image is None or image.size == 0:
            return None

        now = timestamp or self._clock()
        
        # Extract feature vector to enrich the session's ReID profile
        feature_vector = self.reid_matcher.extract_feature(image)

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            
            # Persist the file
            path = self._write_snapshot_file(
                f"extra_{session_id[-8:]}",
                image,
                timestamp=now,
            )
            
            if path:
                session.reference_snapshot_paths.append(path)
                if feature_vector is not None:
                    session.reference_feature_vectors.append(feature_vector)
                
                logger.info(
                    "[REGISTRY] Added extra snapshot to session %s (total=%d)",
                    session_id,
                    len(session.reference_snapshot_paths),
                )
                return path
        return None

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
        now = timestamp or self._clock()

        event = PendingANPREvent(
            event_id=f"anpr_{uuid.uuid4().hex[:12]}",
            plate=plate,
            direction=direction,
            timestamp=now,
            camera_id=camera_id,
        )

        if direction == "entry":
            with self._lock:
                for event_id in reversed(self._pending_event_order):
                    existing = self._pending_events.get(event_id)
                    if not existing:
                        continue
                    if existing.direction != "entry" or existing.plate != plate:
                        continue
                    if existing.status not in ("pending", "provisional"):
                        continue
                    age = (now - existing.timestamp).total_seconds()
                    if age <= 600.0:
                        logger.info(
                            "[ANPR] Duplicate entry ignored for plate=%s (age=%.1fs, status=%s)",
                            plate,
                            age,
                            existing.status,
                        )
                        return existing

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
                session_id = self._track_session_map.pop(key, None)
                # Also clean up the session's observing_tracks entry
                if session_id:
                    session = self._sessions.get(session_id)
                    if session and key[0] in session.observing_tracks:
                        del session.observing_tracks[key[0]]

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
                                self._clear_candidate_references(candidate)
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
                        self._clear_candidate_references(candidate)
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
        now = timestamp or self._clock()
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

    @staticmethod
    def _clear_candidate_references(candidate: ParkEntryCandidate) -> None:
        candidate.snapshot_image = None
        candidate.snapshot_path = None
        candidate.feature_vector = None
        candidate.quality_score = 0.0
        candidate.snapshot_images.clear()
        candidate.snapshot_paths.clear()
        candidate.feature_vectors.clear()
        candidate.color_hsv = None
        candidate.color_hsv_values.clear()

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
        now = timestamp or self._clock()

        if REID_USE_MULTISHOT:
            return self._update_park_entry_candidate_multishot(
                candidate_id,
                image,
                quality_score,
                feature_vector=feature_vector,
                timestamp=now,
            )

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
            if REID_USE_COLOR_FILTER:
                from src.reid_matcher import dominant_color_hsv

                candidate.color_hsv = dominant_color_hsv(image)
                candidate.color_hsv_values = (
                    [candidate.color_hsv] if candidate.color_hsv is not None else []
                )

            logger.debug(
                "[PARK_ENTRY] Updated snapshot for %s (score=%.3f)",
                candidate.candidate_id,
                quality_score,
            )
            return filepath

    def _update_park_entry_candidate_multishot(
        self,
        candidate_id: str,
        image: np.ndarray,
        quality_score: float,
        feature_vector: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Keep up to three sharp Park_Entry references for this candidate.
        """
        if image is None or image.size == 0:
            return None

        now = timestamp or self._clock()
        from src.reid_matcher import select_best_frames
        from src.reid_matcher.reid_burst import sharpness_score

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if not candidate:
                return None
            candidate.last_seen_at = now
            existing_images = [img for img in candidate.snapshot_images if img is not None and img.size > 0]
            candidate_images = existing_images + [image.copy()]
            camera_id = candidate.camera_id

        top_k = 3 if camera_id == "CAM-03" else 1
        selected_images = select_best_frames(candidate_images, top_k=top_k)
        if not selected_images:
            return None

        if feature_vector is not None and len(selected_images) == 1 and selected_images[0] is image:
            feature_vectors = [feature_vector]
        else:
            feature_vectors = self.reid_matcher.extract_features_batch(selected_images)

        token = uuid.uuid4().hex[:12]
        snapshot_paths: List[str] = []
        for idx, selected_image in enumerate(selected_images):
            suffix = "" if idx == 0 else f"_ref{idx + 1}"
            path = self._write_snapshot_file(
                f"{candidate_id}_{token}{suffix}",
                selected_image,
                timestamp=now,
            )
            if path:
                snapshot_paths.append(path)

        primary_image = selected_images[0]
        primary_vector = feature_vectors[0] if feature_vectors else None
        primary_path = snapshot_paths[0] if snapshot_paths else None
        primary_quality = max(float(quality_score), float(sharpness_score(primary_image)))
        color_hsv_values = []
        if REID_USE_COLOR_FILTER:
            from src.reid_matcher import dominant_color_hsv

            color_hsv_values = [
                hsv
                for hsv in (dominant_color_hsv(img) for img in selected_images)
                if hsv is not None
            ]

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if not candidate:
                return None

            candidate.last_seen_at = now
            candidate.snapshot_images = [img.copy() for img in selected_images]
            candidate.snapshot_paths = snapshot_paths
            candidate.feature_vectors = [
                vec for vec in feature_vectors if vec is not None
            ]
            candidate.snapshot_image = primary_image.copy()
            candidate.snapshot_path = primary_path
            candidate.feature_vector = primary_vector
            candidate.quality_score = primary_quality
            if REID_USE_COLOR_FILTER:
                candidate.color_hsv_values = color_hsv_values
                candidate.color_hsv = (
                    color_hsv_values[0] if color_hsv_values else None
                )

            logger.debug(
                "[PARK_ENTRY] Updated multishot snapshot set for %s (refs=%d)",
                candidate.candidate_id,
                len(candidate.snapshot_images),
            )
            return primary_path
