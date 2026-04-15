import logging
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np

from src.vehicle_registry.vehicle_registry_models import VehicleSession

logger = logging.getLogger(__name__)


class VehicleRegistryIdentityMixin:
    def bind_next_pending_anpr_to_candidate(
        self,
        candidate_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        FIFO rule:
        the first pending ANPR entry is provisionally bound to the first Park_Entry candidate.
        """
        now = timestamp or datetime.now()

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None or candidate.status != "open":
                return None

            event = None
            for event_id in self._pending_event_order:
                pending = self._pending_events.get(event_id)
                if pending and pending.direction == "entry" and pending.status == "pending":
                    age = (now - pending.timestamp).total_seconds()
                    if age <= self.PENDING_ANPR_EXPIRY_SECONDS:
                        event = pending
                        break
                    pending.status = "expired"

            if event is None:
                return None

            event.status = "provisional"
            event.candidate_id = candidate.candidate_id

            candidate.status = "provisional"
            candidate.bound_event_id = event.event_id
            candidate.last_seen_at = now

            logger.info(
                "[PARK_ENTRY] Bound ANPR event %s (plate=%s) to candidate %s",
                event.event_id,
                event.plate,
                candidate_id,
            )
            return event.plate

    def confirm_at_b1_entrance(
        self,
        camera_id: str,
        track_id: int,
        image: np.ndarray,
        reference_images: Optional[List[np.ndarray]] = None,
        similarity_threshold: float = 0.35,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Confirm that a B1_Entrance car is the same car that was provisionally
        captured at Park_Entry.
        """
        now = timestamp or datetime.now()
        session_reference_images = [image]
        if reference_images:
            for extra in reference_images:
                if extra is None or extra.size == 0:
                    continue
                if extra.shape == image.shape and np.array_equal(extra, image):
                    continue
                session_reference_images.append(extra)

        with self._lock:
            provisional_pairs = []
            for candidate in self._park_entry_candidates.values():
                if candidate.status != "provisional" or candidate.snapshot_image is None:
                    continue
                if not candidate.bound_event_id:
                    continue
                event = self._pending_events.get(candidate.bound_event_id)
                if event is None or event.status != "provisional":
                    continue
                provisional_pairs.append((event.timestamp, candidate))

        provisional_pairs.sort(key=lambda item: item[0])

        session_reference_features = self.reid_matcher.extract_features_batch(
            session_reference_images
        )
        current_reid_feat = (
            session_reference_features[0] if session_reference_features else None
        )

        best_candidate = None
        best_score = 0.0

        for _, candidate in provisional_pairs:
            snapshot = candidate.snapshot_image
            if snapshot is None:
                continue

            color_similarity = self.matcher._compare_dominant_colors(image, snapshot)
            if color_similarity < 0.45:
                logger.debug(
                    "[REID] Candidate %s rejected by color filter (score=%.2f)",
                    candidate.candidate_id,
                    color_similarity,
                )
                continue

            if current_reid_feat is not None and candidate.feature_vector is not None:
                score = self.reid_matcher.compute_similarity(
                    current_reid_feat,
                    candidate.feature_vector,
                )
                logger.debug(
                    "[REID] Candidate %s vector similarity: %.3f",
                    candidate.candidate_id,
                    score,
                )
                match_threshold = 0.55
            else:
                score = self.matcher.compare(image, snapshot)
                logger.debug(
                    "[REID] Candidate %s using legacy fallback: %.3f",
                    candidate.candidate_id,
                    score,
                )
                match_threshold = similarity_threshold

            if score > best_score and score >= match_threshold:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            if len(provisional_pairs) == 1:
                best_candidate = provisional_pairs[0][1]
                logger.warning(
                    "[B1] Falling back to the only provisional candidate %s "
                    "for track (%s, %d); strict visual threshold was not met",
                    best_candidate.candidate_id,
                    camera_id,
                    track_id,
                )
            else:
                return None

        session_feature_vector = (
            current_reid_feat
            if current_reid_feat is not None
            else best_candidate.feature_vector
        )
        reference_feature_vectors = [
            feature for feature in session_reference_features if feature is not None
        ]

        with self._lock:
            live_candidate = self._park_entry_candidates.get(best_candidate.candidate_id)
            if live_candidate is None or live_candidate.status != "provisional":
                logger.debug(
                    "[B1] Candidate %s is no longer provisional; discarding match",
                    best_candidate.candidate_id,
                )
                return None

            event = self._pending_events.get(live_candidate.bound_event_id)
            if event is None or event.status != "provisional":
                logger.debug(
                    "[B1] Bound event for candidate %s is no longer provisional; discarding match",
                    live_candidate.candidate_id,
                )
                return None

            stored_paths = self._store_session_reference_snapshots(
                session_reference_images,
                timestamp=now,
            )
            cam03_path = stored_paths[0] if stored_paths else None

            existing_sid = self._track_session_map.get((camera_id, track_id))
            session = self._sessions.get(existing_sid) if existing_sid else None

            if session:
                logger.info(
                    "[B1] Upgrading appearance session %s with plate %s",
                    session.session_id,
                    event.plate,
                )
                session.plate = event.plate
                session.event_id = event.event_id
                session.candidate_id = live_candidate.candidate_id
                session.last_seen_at = now
                session.last_seen_camera = camera_id
                session.last_seen_track_id = track_id
                session.snapshot_path = cam03_path
                session.reference_snapshot_paths = stored_paths
                if session_feature_vector is not None:
                    session.feature_vector = session_feature_vector
                session.reference_feature_vectors = reference_feature_vectors
                self._drop_other_track_mappings_for_session(
                    session.session_id,
                    keep=(camera_id, track_id),
                )
                self._track_session_map[(camera_id, track_id)] = session.session_id
                self._mark_track_seen(camera_id, track_id, now)
            else:
                session = VehicleSession(
                    session_id=f"sess_{uuid.uuid4().hex[:12]}",
                    plate=event.plate,
                    feature_vector=session_feature_vector,
                    first_seen_at=now,
                    last_seen_at=now,
                    last_seen_camera=camera_id,
                    last_seen_track_id=track_id,
                    event_id=event.event_id,
                    candidate_id=live_candidate.candidate_id,
                    status="confirmed",
                    snapshot_path=cam03_path,
                    reference_snapshot_paths=stored_paths,
                    reference_feature_vectors=reference_feature_vectors,
                )
                self._sessions[session.session_id] = session
                self._drop_other_track_mappings_for_session(
                    session.session_id,
                    keep=(camera_id, track_id),
                )
                self._track_session_map[(camera_id, track_id)] = session.session_id
                self._mark_track_seen(camera_id, track_id, now)

            event.status = "confirmed"
            event.session_id = session.session_id
            event.candidate_id = live_candidate.candidate_id

            live_candidate.status = "confirmed"
            live_candidate.last_seen_at = now
            live_candidate.snapshot_image = None
            live_candidate.feature_vector = None

            logger.info(
                "[B1] Confirmed plate=%s for track (%s, %d) - session %s (score=%.3f)",
                session.plate,
                camera_id,
                track_id,
                session.session_id,
                best_score,
            )
            return session.plate

    def drop_provisional_binding(self, candidate_id: str) -> None:
        """Remove a failed provisional match."""
        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None:
                return

            if candidate.bound_event_id:
                event = self._pending_events.get(candidate.bound_event_id)
                if event:
                    event.status = "dropped"
                    event.candidate_id = None

            candidate.status = "dropped"
            candidate.bound_event_id = None
            candidate.snapshot_image = None
            candidate.feature_vector = None

    def attach_session_to_track(
        self,
        camera_id: str,
        track_id: int,
        session_id: str,
    ) -> None:
        """Attach a confirmed session to a new camera/track."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            now = datetime.now()
            session.last_seen_at = now
            session.last_seen_camera = camera_id
            session.last_seen_track_id = track_id
            self._drop_other_track_mappings_for_session(
                session_id,
                keep=(camera_id, track_id),
            )
            self._track_session_map[(camera_id, track_id)] = session_id
            self._mark_track_seen(camera_id, track_id, now)

    def get_plate_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[str]:
        """Resolve plate through confirmed session mapping."""
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id is None:
                return None
            session = self._sessions.get(session_id)
            return session.plate if session else None

    def get_session_id_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[str]:
        """
        Resolve the session_id for a (camera_id, track_id) pair.
        """
        with self._lock:
            return self._track_session_map.get((camera_id, track_id))

    def get_display_id_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> str:
        """
        Returns a stable, human-readable ID for a track.
        """
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    return session.display_id
        return f"T:{track_id}"

    def create_appearance_session(
        self,
        camera_id: str,
        track_id: int,
        feature_vector: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Create a new session for a vehicle based only on its appearance (ReID).
        """
        now = timestamp or datetime.now()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"

        with self._lock:
            session = VehicleSession(
                session_id=session_id,
                plate=None,
                feature_vector=feature_vector,
                first_seen_at=now,
                last_seen_at=now,
                last_seen_camera=camera_id,
                last_seen_track_id=track_id,
                status="confirmed",
            )
            self._sessions[session.session_id] = session
            self._drop_other_track_mappings_for_session(
                session.session_id,
                keep=(camera_id, track_id),
            )
            self._track_session_map[(camera_id, track_id)] = session.session_id
            self._mark_track_seen(camera_id, track_id, now)

            logger.info(
                "[REGISTRY] Created appearance-only session %s for track (%s, %d)",
                session_id,
                camera_id,
                track_id,
            )
            return session_id

    def match_global_session(
        self,
        query_vector: Optional[np.ndarray],
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        max_time_gap_seconds: float = 600.0,
        similarity_threshold: float = 0.55,
    ) -> Optional[str]:
        """
        Search for an existing confirmed session that matches this query vector.
        """
        if query_vector is None:
            return None

        now = datetime.now()
        with self._lock:
            potential_sessions = [
                session
                for session in self._sessions.values()
                if (
                    session.status in ("confirmed", "parked")
                    and session.feature_vector is not None
                    and (now - session.last_seen_at).total_seconds()
                    <= max_time_gap_seconds
                    and (
                        track_id is None
                        or (session.last_seen_camera, session.last_seen_track_id)
                        == (camera_id, track_id)
                        or (now - session.last_seen_at).total_seconds()
                        >= self.SESSION_HANDOFF_GUARD_SECONDS
                    )
                )
            ]

        if not potential_sessions:
            return None

        best_sid = None
        best_score = -1.0
        for session in potential_sessions:
            session_vectors = [session.feature_vector] + list(
                session.reference_feature_vectors
            )
            score = max(
                (
                    self.reid_matcher.compute_similarity(query_vector, ref_vec)
                    for ref_vec in session_vectors
                    if ref_vec is not None
                ),
                default=0.0,
            )
            effective_threshold = 0.52 if session.plate else similarity_threshold
            if score >= effective_threshold and score > best_score:
                best_score = score
                best_sid = session.session_id

        if best_sid:
            with self._lock:
                final_session = self._sessions.get(best_sid)
                if final_session and final_session.status in ("confirmed", "parked"):
                    logger.info(
                        "[GLOBAL] cam=%s matched session %s (score=%.3f)",
                        camera_id,
                        best_sid,
                        best_score,
                    )
                    return best_sid

        return None

    def link_plate_to_session(
        self,
        session_id: str,
        plate: str,
        feature_vector: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Plate + ReID Fusion: Bind a newly recognized plate to an existing ReID session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            if session.plate and session.plate != plate:
                logger.warning(
                    "[FUSION] Session %s already has plate=%s; ignoring new plate=%s",
                    session_id,
                    session.plate,
                    plate,
                )
                return False

            session.plate = plate
            if feature_vector is not None:
                session.feature_vector = feature_vector

            logger.info(
                "[FUSION] Linked plate=%s to session %s (appearance-only -> plate-confirmed)",
                plate,
                session_id,
            )
            return True

    def update_session_feature(
        self,
        session_id: str,
        smoothed_feature: np.ndarray,
    ) -> None:
        """
        Update the stored feature vector of a session with a smoothed EMA vector.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and smoothed_feature is not None:
                session.feature_vector = smoothed_feature
                if session.last_seen_camera and session.last_seen_track_id is not None:
                    self._mark_track_seen(
                        session.last_seen_camera,
                        session.last_seen_track_id,
                    )

    def reattach_track_to_confirmed_session(
        self,
        camera_id: str,
        track_id: int,
        query_vector: Optional[np.ndarray],
        similarity_threshold: float = 0.52,
    ) -> Optional[str]:
        """
        Upgrade an anonymous appearance-only track to an already confirmed session.
        """
        if query_vector is None:
            return None

        with self._lock:
            current_sid = self._track_session_map.get((camera_id, track_id))
            current_session = self._sessions.get(current_sid) if current_sid else None
            if current_session is None or current_session.plate:
                return current_sid

            candidates = [
                session
                for sid, session in self._sessions.items()
                if (
                    sid != current_sid
                    and session.status in ("confirmed", "parked")
                    and session.plate
                    and session.feature_vector is not None
                    and (
                        (session.last_seen_camera, session.last_seen_track_id)
                        == (camera_id, track_id)
                        or (datetime.now() - session.last_seen_at).total_seconds()
                        >= self.SESSION_HANDOFF_GUARD_SECONDS
                    )
                )
            ]

        best_sid = None
        best_score = -1.0
        for session in candidates:
            session_vectors = [session.feature_vector] + list(
                session.reference_feature_vectors
            )
            score = max(
                (
                    self.reid_matcher.compute_similarity(query_vector, ref_vec)
                    for ref_vec in session_vectors
                    if ref_vec is not None
                ),
                default=0.0,
            )
            if score >= similarity_threshold and score > best_score:
                best_score = score
                best_sid = session.session_id

        if best_sid is None:
            return None

        with self._lock:
            current_sid = self._track_session_map.get((camera_id, track_id))
            current_session = self._sessions.get(current_sid) if current_sid else None
            target_session = self._sessions.get(best_sid)
            if current_session is None or target_session is None:
                return None
            if current_session.plate:
                return current_sid

            target_session.last_seen_at = datetime.now()
            target_session.last_seen_camera = camera_id
            target_session.last_seen_track_id = track_id
            if query_vector is not None:
                target_session.feature_vector = query_vector
            self._drop_other_track_mappings_for_session(
                best_sid,
                keep=(camera_id, track_id),
            )
            self._track_session_map[(camera_id, track_id)] = best_sid
            self._mark_track_seen(camera_id, track_id)

            orphan_sid = current_sid
            if orphan_sid != best_sid:
                still_used = orphan_sid in self._track_session_map.values()
                if (
                    not still_used
                    and current_session.status == "confirmed"
                    and current_session.linked_slot is None
                ):
                    self._sessions.pop(orphan_sid, None)

            logger.info(
                "[GLOBAL] Reattached track (%s, %d) from anonymous session %s to confirmed session %s (score=%.3f)",
                camera_id,
                track_id,
                current_sid,
                best_sid,
                best_score,
            )
            return best_sid

    def try_link_to_slot(
        self,
        slot_id: str,
        slot_name: str,
        zone_id: Optional[str],
        zone_name: Optional[str],
        camera_id: str,
        floor: str,
        track_id: Optional[int],
        timestamp: datetime,
    ) -> Optional[str]:
        """
        Link a slot only from a confirmed session.
        No blind fallback to recent ANPR events.
        """
        if track_id is None:
            return None

        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id is None:
                return None

            session = self._sessions.get(session_id)
            if session is None:
                return None

            existing = self._parked.get(slot_id)
            if existing is not None:
                if existing.session_id == session.session_id:
                    return existing.plate
                return existing.plate

            session.linked_slot = slot_id
            session.linked_slot_name = slot_name
            session.linked_camera = camera_id
            session.linked_floor = floor
            session.linked_zone_id = zone_id
            session.linked_zone_name = zone_name
            session.linked_at = timestamp
            session.status = "parked"
            self._parked[slot_id] = session
            return session.plate

    def unlink_slot(self, slot_id: str) -> Optional[str]:
        """
        Detach the currently parked session from a slot after vacancy is confirmed.
        """
        with self._lock:
            session = self._parked.pop(slot_id, None)
            if session is None:
                return None

            plate = session.plate
            session.linked_slot = None
            session.linked_slot_name = None
            session.linked_camera = None
            session.linked_floor = None
            session.linked_zone_id = None
            session.linked_zone_name = None
            session.linked_at = None
            if session.status == "parked":
                session.status = "confirmed"
            return plate

    def _handle_exit(self, plate: str, timestamp: datetime) -> None:
        """Close a parked session when ANPR sends an exit event."""
        with self._lock:
            slot_to_remove = None
            for slot_id, session in self._parked.items():
                if session.plate == plate:
                    slot_to_remove = slot_id
                    break

            if slot_to_remove is not None:
                session = self._parked.pop(slot_to_remove)
                session.status = "exited"
                self._sessions.pop(session.session_id, None)
                self._history.append(session)

                if session.last_seen_camera and session.last_seen_track_id is not None:
                    self._track_session_map.pop(
                        (session.last_seen_camera, session.last_seen_track_id),
                        None,
                    )
