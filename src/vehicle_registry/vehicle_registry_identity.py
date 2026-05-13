import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np

from src.matching.match_decision import Decision
from src.vehicle_registry.vehicle_registry_models import ParkEntryCandidate, VehicleSession

logger = logging.getLogger(__name__)

# Legacy module-level flag retained for backward compatibility — Phase 2
# cleanup drops ``REID_USE_LAB_CLAHE`` and ``REID_USE_MULTISHOT`` from this
# module (they were never read from here in the post-Phase-0 code; the
# cascade now reads ``self.matching_config.use_*``). ``REID_USE_COLOR_FILTER``
# remains as a shim because ``tests/test_multishot_registry.py`` still
# monkey-patches it via ``patch.object(vehicle_registry_identity, ...)``.
# Drop this once that test is rewritten to mutate the MatchingConfig
# directly.
REID_USE_COLOR_FILTER = os.getenv("REID_USE_COLOR_FILTER", "false").lower() == "true"
match_logger = logging.getLogger("reid_match_perf")


class VehicleRegistryIdentityMixin:
    def is_plate_inside(self, plate: Optional[str]) -> bool:
        """Returns True iff `plate` has an open `parking_sessions` row in the
        shared DB. Single source of truth — avoids in-memory drift between VA
        and PMS-AI when the gate's exit-ANPR event was missed.

        Returns True if VA has no DB binding (legacy/test env) or no plate is
        passed; the guard fails open so unrelated logic still runs.

        Tests can inject an alternate predicate via the ``db_checker`` DI
        seam on ``VehicleRegistry.__init__``; when set, it is consulted
        instead of the SQL probe.
        """
        if not plate:
            return True

        db_checker = getattr(self, "_db_checker", None)
        if db_checker is not None:
            try:
                return bool(db_checker(plate))
            except Exception as exc:
                logger.warning(
                    "[is_plate_inside] injected db_checker raised for %s: %r",
                    plate,
                    exc,
                )
                return True  # fail-open

        db_manager = getattr(self, "db_manager", None)
        if db_manager is None:
            return True
        try:
            from sqlalchemy import text as _text

            session = db_manager.SessionLocal()
            try:
                row = session.execute(
                    _text(
                        "SELECT TOP 1 status FROM dbo.parking_sessions "
                        "WHERE plate_number = :p ORDER BY entry_time DESC"
                    ),
                    {"p": plate},
                ).first()
            finally:
                session.close()
        except Exception as exc:
            logger.warning("[is_plate_inside] DB probe failed for %s: %r", plate, exc)
            return True  # fail-open so a transient DB blip doesn't block re-id

        if row is None:
            # No session ever recorded — could be fresh-arrival before ANPR
            # has fired, so allow the link to proceed (registry-side gates
            # cover the "never entered" case).
            return True
        return row[0] == "open"

    @staticmethod
    def _dedupe_valid_images(images: List[np.ndarray]) -> List[np.ndarray]:
        deduped: List[np.ndarray] = []
        for image in images:
            if image is None or image.size == 0:
                continue
            if any(
                existing.shape == image.shape and np.array_equal(existing, image)
                for existing in deduped
            ):
                continue
            deduped.append(image)
        return deduped

    @staticmethod
    def _candidate_snapshot_paths(candidate) -> List[str]:
        if list(candidate.snapshot_paths):
            return list(candidate.snapshot_paths)
        if candidate.snapshot_path:
            return [candidate.snapshot_path]
        return []

    def _bootstrap_b1_candidate_from_pending_entry(
        self,
        image: np.ndarray,
        now: datetime,
        feature_vector: Optional[np.ndarray] = None,
    ) -> Optional[ParkEntryCandidate]:
        """
        If B1 sees the car before any Park_Entry candidate exists, create a
        provisional candidate from the single pending ANPR entry.
        """
        if image is None or image.size == 0:
            return None

        with self._lock:
            pending_events = []
            for event_id in self._pending_event_order:
                event = self._pending_events.get(event_id)
                if not event or event.direction != "entry" or event.status != "pending":
                    continue
                age = (now - event.timestamp).total_seconds()
                if age <= self.PENDING_ANPR_EXPIRY_SECONDS:
                    pending_events.append(event)
                else:
                    event.status = "expired"

            if len(pending_events) != 1:
                return None

            event = pending_events[0]

        snapshot_path = self._write_snapshot_file(
            f"b1_bootstrap_{uuid.uuid4().hex[:12]}",
            image,
            timestamp=now,
        )

        if feature_vector is None:
            feature_vector = self.reid_matcher.extract_feature(image)

        candidate = ParkEntryCandidate(
            candidate_id=f"cand_{uuid.uuid4().hex[:12]}",
            camera_id="CAM-03",
            track_id=-1,
            entered_at=event.timestamp,
            last_seen_at=now,
            snapshot_path=snapshot_path,
            snapshot_image=image.copy(),
            feature_vector=feature_vector,
            quality_score=999.0,
            snapshot_paths=[snapshot_path] if snapshot_path else [],
            feature_vectors=[feature_vector] if feature_vector is not None else [],
            status="provisional",
            bound_event_id=event.event_id,
        )

        with self._lock:
            live_event = self._pending_events.get(event.event_id)
            if live_event is None or live_event.status != "pending":
                return None

            live_event.status = "provisional"
            live_event.candidate_id = candidate.candidate_id
            self._park_entry_candidates[candidate.candidate_id] = candidate

        logger.info(
            "[B1] Bootstrapped provisional candidate %s from pending ANPR plate=%s",
            candidate.candidate_id,
            event.plate,
        )
        return candidate

    def _persist_session_gallery(
        self,
        session: VehicleSession,
        images: List[np.ndarray],
        now: datetime,
        primary_snapshot_index: int = 0,
    ) -> bool:
        ordered_images = self._dedupe_valid_images(images)
        if not ordered_images:
            return False

        primary_snapshot_index = max(
            0,
            min(primary_snapshot_index, len(ordered_images) - 1),
        )
        feature_vectors = self.reid_matcher.extract_features_batch(ordered_images)
        stored_paths = self._store_session_reference_snapshots(
            ordered_images,
            timestamp=now,
        )

        session.reference_snapshot_paths = stored_paths
        session.reference_feature_vectors = [
            feature for feature in feature_vectors if feature is not None
        ]
        if stored_paths:
            session.snapshot_path = stored_paths[primary_snapshot_index]
        if feature_vectors:
            primary_feature = feature_vectors[primary_snapshot_index]
            if primary_feature is not None:
                session.feature_vector = primary_feature
        return True

    def confirm_anpr_session_directly(
        self,
        plate: str,
        image: np.ndarray,
        event_id: str,
        candidate_id: str,
        gate_snapshot_paths: List[str],
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Create a confirmed VehicleSession directly from an ANPR image, without
        waiting for the car to pass through the B1_Entrence confirmation zone.

        This session is immediately discoverable by match_global_session on any
        camera, so the plate label propagates to CAM_07 (and others) as soon as
        the ReID engine sees the parked car — even if the car bypassed CAM_03.

        Returns the new session_id, or None if the feature vector could not be
        extracted from the image.
        """
        now = timestamp or self._clock()

        feature_vector = self.reid_matcher.extract_feature(image)
        if feature_vector is None:
            logger.warning(
                "[ANPR] Could not extract feature vector from ANPR image for plate=%s; "
                "session will not be searchable via global ReID",
                plate,
            )
            return None

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = VehicleSession(
            session_id=session_id,
            plate=plate,
            feature_vector=feature_vector,
            first_seen_at=now,
            last_seen_at=now,
            last_seen_camera="ANPR",
            last_seen_track_id=None,
            event_id=event_id,
            candidate_id=candidate_id,
            status="confirmed",
            gate_snapshot_paths=gate_snapshot_paths,
        )

        # Persist the ANPR image as the primary gallery reference so it appears
        # in reference_snapshot_paths (the UI gallery) alongside any future
        # CAM_03 confirmation images that get added later.
        self._persist_session_gallery(session, [image], now, primary_snapshot_index=0)

        with self._lock:
            # Guard: if a session for this plate was already confirmed (e.g. the
            # car DID pass through B1_Entrence in a race), skip creating a duplicate.
            for existing in self._sessions.values():
                if (
                    existing.plate == plate
                    and existing.status in ("confirmed", "parked")
                    and (now - existing.first_seen_at).total_seconds() < 120
                ):
                    logger.info(
                        "[ANPR] Session for plate=%s already confirmed (%s); "
                        "skipping duplicate ANPR direct-session",
                        plate,
                        existing.session_id,
                    )
                    return existing.session_id

            self._sessions[session_id] = session
            # Phase 3 / T3.2 — track the new confirmed session in the FAISS
            # gallery (no-op when use_faiss_index is False).
            self._gallery_index_upsert(session)

        logger.info(
            "[ANPR] Direct session %s created for plate=%s — "
            "immediately searchable via global ReID on all cameras",
            session_id,
            plate,
        )
        return session_id


    def bind_next_pending_anpr_to_candidate(
        self,
        candidate_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        FIFO rule:
        the first pending ANPR entry is provisionally bound to the first Park_Entry candidate.
        """
        now = timestamp or self._clock()

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
        ordered_images: Optional[List[np.ndarray]] = None,
        primary_snapshot_index: int = 0,
    ) -> Optional[str]:
        """
        Confirm that a B1_Entrance car is the same car that was provisionally
        captured at Park_Entry.
        """
        now = timestamp or self._clock()
        if ordered_images:
            session_reference_images = self._dedupe_valid_images(ordered_images)
        else:
            session_reference_images = [image]
            if reference_images:
                for extra in reference_images:
                    if extra is None or extra.size == 0:
                        continue
                    if extra.shape == image.shape and np.array_equal(extra, image):
                        continue
                    session_reference_images.append(extra)
            session_reference_images = self._dedupe_valid_images(session_reference_images)

        if not session_reference_images:
            return None

        primary_snapshot_index = max(
            0,
            min(primary_snapshot_index, len(session_reference_images) - 1),
        )

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
            session_reference_features[primary_snapshot_index]
            if session_reference_features
            else None
        )

        if not provisional_pairs:
            bootstrapped_candidate = self._bootstrap_b1_candidate_from_pending_entry(
                image=image,
                now=now,
                feature_vector=current_reid_feat,
            )
            if bootstrapped_candidate is not None:
                provisional_pairs = [(now, bootstrapped_candidate)]

        cfg = self.matching_config
        decision = self.match_decision

        # Honour both the new MatchingConfig flag AND the legacy module-level
        # global so existing tests that ``patch.object(vehicle_registry_identity,
        # "REID_USE_COLOR_FILTER", True)`` keep working unchanged. The module
        # global will be removed once the rest of the codebase / tests stop
        # patching it (tracked separately).
        use_color_filter = cfg.use_color_filter or REID_USE_COLOR_FILTER

        query_hsv = None
        if use_color_filter:
            from src.reid_matcher import dominant_color_hsv
            query_hsv = dominant_color_hsv(image)

        best_candidate = None
        best_score = 0.0
        best_decision = None
        hard_rejected_candidate_ids = set()

        for _, candidate in provisional_pairs:
            snapshot = candidate.snapshot_image
            if snapshot is None:
                continue

            # --- Color predicate ---------------------------------------- #
            # Folds the historical _compare_dominant_colors + color_compatible
            # blocks (identity:389-413). dominant-color failure causes a plain
            # ``continue`` (candidate stays eligible for the single-candidate
            # fallback); HSV failure HARD-REJECTS the candidate so the
            # fallback also discards it.
            candidate_hsv_values = [
                hsv for hsv in list(candidate.color_hsv_values) if hsv is not None
            ]
            if not candidate_hsv_values and candidate.color_hsv is not None:
                candidate_hsv_values = [candidate.color_hsv]

            color = decision.color_check(
                image,
                snapshot,
                query_hsv=query_hsv,
                candidate_hsvs=candidate_hsv_values,
                use_color_filter=use_color_filter,
            )
            if not color.passes_dominant:
                logger.debug(
                    "[REID] Candidate %s rejected by color filter (score=%.2f)",
                    candidate.candidate_id,
                    color.dominant_score,
                )
                continue
            if color.hard_reject:
                hard_rejected_candidate_ids.add(candidate.candidate_id)
                logger.debug(
                    "[REID] Candidate %s rejected by HSV color filter",
                    candidate.candidate_id,
                )
                continue

            candidate_vectors = [
                feature for feature in list(candidate.feature_vectors) if feature is not None
            ]
            if not candidate_vectors and candidate.feature_vector is not None:
                candidate_vectors = [candidate.feature_vector]

            if current_reid_feat is not None and candidate_vectors:
                score = max(
                    self.reid_matcher.compute_similarity(
                        current_reid_feat,
                        candidate_vector,
                    )
                    for candidate_vector in candidate_vectors
                )
                # ANPR-image candidates were captured by the dedicated ANPR
                # camera and are a higher-quality reference — give them the
                # lower bar via MatchingConfig.b1_anpr; zone crops use b1_zone.
                is_anpr_candidate = (
                    getattr(candidate, "source", "zone_crop") == "anpr_image"
                )

                # Phase 2 T2.1 — populate modality inputs for the ensemble.
                # Pull the pending ANPR plate from the bound event so OCR
                # has something to cross-check against. Skip the modality
                # pass for the legacy fallback path (no current_reid_feat).
                pending_plate = None
                if candidate.bound_event_id:
                    bound_event = self._pending_events.get(candidate.bound_event_id)
                    if bound_event is not None:
                        pending_plate = bound_event.plate
                modalities = decision.check_modalities(
                    query_crop=image,
                    candidate_crop=snapshot,
                    candidate=candidate,
                    pending_plate=pending_plate,
                    score_reid=score,
                )

                verdict = decision.decide_b1(
                    score,
                    is_anpr_candidate=is_anpr_candidate,
                    candidate_count=0,  # per-candidate pass; fallback handled below
                    modalities=modalities,
                )
                match_threshold = verdict.scores["threshold"]
                logger.debug(
                    "[REID] Candidate %s vector similarity: %.3f (refs=%d, source=%s, threshold=%.2f)",
                    candidate.candidate_id,
                    score,
                    len(candidate_vectors),
                    getattr(candidate, "source", "zone_crop"),
                    match_threshold,
                )
            else:
                score = self.matcher.compare(image, snapshot)
                logger.debug(
                    "[REID] Candidate %s using legacy fallback: %.3f",
                    candidate.candidate_id,
                    score,
                )
                match_threshold = similarity_threshold
                verdict = None

            # Phase 2 T2.1: the ensemble rule can produce a 'confirm' even
            # when the ReID score does not clear ``match_threshold`` (e.g.
            # OCR plate-match in the marginal band). We accept any verdict
            # that the decision chokepoint says is a 'confirm'; the legacy
            # threshold gate is only consulted when we have no verdict
            # (the legacy image-matcher fallback path).
            ensemble_confirm = bool(
                verdict is not None
                and verdict.verdict == "confirm"
                and verdict.reason.startswith("ensemble_")
            )
            score_pass = score >= match_threshold
            if score > best_score and (score_pass or ensemble_confirm):
                best_score = score
                best_candidate = candidate
                best_decision = verdict

                # Compute old score in parallel for logging (single-shot ReID)
                if current_reid_feat is not None and candidate.feature_vector is not None:
                    candidate.old_pipeline_score = self.reid_matcher.compute_similarity(
                        current_reid_feat,
                        candidate.feature_vector
                    )
                else:
                    candidate.old_pipeline_score = score # fallback

        if best_candidate is None:
            fallback_pairs = [
                pair
                for pair in provisional_pairs
                if pair[1].candidate_id not in hard_rejected_candidate_ids
            ]
            # Re-ask MatchDecision with the surviving candidate count so the
            # "single-candidate fallback" branch fires identically to the
            # historical implementation. We feed it a 0.0 score so only the
            # candidate_count==1 fallback path can produce a 'confirm'.
            fallback_verdict = decision.decide_b1(
                0.0,
                is_anpr_candidate=False,
                candidate_count=len(fallback_pairs),
            )
            if (
                fallback_verdict.verdict == "confirm"
                and fallback_verdict.reason == "single_candidate_fallback"
            ):
                best_candidate = fallback_pairs[0][1]
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

            # Refuse to confirm a plate whose latest parking_sessions row is
            # closed — the car has already exited per PMS-AI. Without this,
            # a re-id match against a stale gate snapshot can resurrect a
            # ghost session for a car that's no longer in the garage.
            if not self.is_plate_inside(event.plate):
                logger.warning(
                    "[B1] refused: plate=%s already exited "
                    "(no open parking_sessions row); discarding confirmation",
                    event.plate,
                )
                return None

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
                if session_feature_vector is not None:
                    session.feature_vector = session_feature_vector
                self._persist_session_gallery(
                    session,
                    session_reference_images,
                    now,
                    primary_snapshot_index=primary_snapshot_index,
                )
                session.new_pipeline_score = best_score
                session.old_pipeline_score = getattr(best_candidate, "old_pipeline_score", best_score)
                session.gate_snapshot_paths = self._candidate_snapshot_paths(live_candidate)
                
                self._drop_other_track_mappings_for_session(
                    session.session_id,
                    keep=(camera_id, track_id),
                )
                self._track_session_map[(camera_id, track_id)] = session.session_id
                session.observing_tracks[camera_id] = track_id
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
                    new_pipeline_score=best_score,
                    old_pipeline_score=getattr(best_candidate, "old_pipeline_score", best_score),
                    gate_snapshot_paths=self._candidate_snapshot_paths(live_candidate),
                )
                self._persist_session_gallery(
                    session,
                    session_reference_images,
                    now,
                    primary_snapshot_index=primary_snapshot_index,
                )
                self._sessions[session.session_id] = session
                self._drop_other_track_mappings_for_session(
                    session.session_id,
                    keep=(camera_id, track_id),
                )
                self._track_session_map[(camera_id, track_id)] = session.session_id
                session.observing_tracks[camera_id] = track_id
                self._mark_track_seen(camera_id, track_id, now)
                # Phase 3 / T3.2 — track the new B1-confirmed session in
                # the FAISS gallery (no-op when use_faiss_index is False).
                self._gallery_index_upsert(session)

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

    def update_confirmed_session_gallery(
        self,
        camera_id: str,
        track_id: int,
        ordered_images: List[np.ndarray],
        primary_snapshot_index: int = 0,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Enrich an already confirmed session with the final ordered CAM_03 gallery.
        """
        now = timestamp or self._clock()
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            session = self._sessions.get(session_id) if session_id else None
            if session is None:
                return False

            session.last_seen_at = now
            session.last_seen_camera = camera_id
            session.last_seen_track_id = track_id
            return self._persist_session_gallery(
                session,
                ordered_images,
                now,
                primary_snapshot_index=primary_snapshot_index,
            )

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
        """Attach a confirmed session to a new camera/track.

        Only cleans up old track IDs on the SAME camera (track-ID recycling
        protection). Other cameras keep their own bindings so the session
        is visible on all cameras that can see the car.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            now = self._clock()
            session.last_seen_at = now
            session.last_seen_camera = camera_id
            session.last_seen_track_id = track_id
            self._drop_other_track_mappings_for_session(
                session_id,
                keep=(camera_id, track_id),
            )
            self._track_session_map[(camera_id, track_id)] = session_id
            session.observing_tracks[camera_id] = track_id
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
        return "0"

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
        now = timestamp or self._clock()
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
            session.observing_tracks[camera_id] = track_id
            self._mark_track_seen(camera_id, track_id, now)
            # Phase 3 / T3.2 — appearance-only sessions are also "confirmed"
            # status so the gallery indexer upserts them too.
            self._gallery_index_upsert(session)

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

        Guard: if a session is already being actively observed by a LIVE track
        on another camera, it is excluded — the car we are looking at cannot be
        that session because the real car is already accounted for elsewhere.

        Phase 3 / T3.2: when ``MatchingConfig.use_faiss_index`` is True the
        candidate pool is pre-narrowed to the top-K nearest neighbours via
        :class:`GalleryIndex` for an O(log n) lookup; the active-track
        guard, ``check_modalities`` cascade and ``MatchDecision.decide_global``
        verdict still apply on the surviving candidates.
        """
        if query_vector is None:
            return None

        ACTIVE_TRACK_STALENESS_SECONDS = 3.0
        FAISS_TOPK = 10

        now = self._clock()
        use_faiss = bool(getattr(self._matching_config, "use_faiss_index", False))
        faiss_topk_ids = None
        if use_faiss and len(self._gallery_index) > 0:
            try:
                hits = self._gallery_index.search(query_vector, k=FAISS_TOPK)
                faiss_topk_ids = {sid for sid, _ in hits}
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "[GLOBAL] gallery_index.search raised %r — falling back to linear scan.",
                    exc,
                )
                faiss_topk_ids = None

        with self._lock:
            if faiss_topk_ids is not None:
                # Pre-narrow to the FAISS top-K. Preserve all the existing
                # invariants on the pool (confirmed/parked, feature present,
                # within max_time_gap_seconds).
                potential_sessions = [
                    session
                    for sid, session in self._sessions.items()
                    if (
                        sid in faiss_topk_ids
                        and session.status in ("confirmed", "parked")
                        and session.feature_vector is not None
                        and (now - session.last_seen_at).total_seconds()
                        <= max_time_gap_seconds
                    )
                ]
            else:
                potential_sessions = [
                    session
                    for session in self._sessions.values()
                    if (
                        session.status in ("confirmed", "parked")
                        and session.feature_vector is not None
                        and (now - session.last_seen_at).total_seconds()
                        <= max_time_gap_seconds
                    )
                ]

            # Active track guard: skip sessions that have a live track on a
            # DIFFERENT camera (the car we see can't be that one — it's already
            # being tracked elsewhere).
            guarded_sessions = []
            for session in potential_sessions:
                has_live_track_elsewhere = False
                for obs_cam, obs_tid in session.observing_tracks.items():
                    if obs_cam == camera_id:
                        continue  # Same camera — could be our own track
                    last_seen = self._track_last_seen.get((obs_cam, obs_tid))
                    if last_seen and (now - last_seen).total_seconds() < ACTIVE_TRACK_STALENESS_SECONDS:
                        has_live_track_elsewhere = True
                        break
                if not has_live_track_elsewhere:
                    guarded_sessions.append(session)

        if not guarded_sessions:
            return None

        best_sid = None
        best_score = -1.0
        decision = self.match_decision
        for session in guarded_sessions:
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
            cross_camera = bool(
                session.plate
                and session.last_seen_camera
                and camera_id
                and session.last_seen_camera != camera_id
            )
            # Phase 2 T2.1 — global-match has no live query crop available
            # at this callsite (we receive a feature vector, not an image),
            # so we pass only the cached metadata on the session and let
            # MatchDecision treat the rest as "no signal". When the caller
            # extends this signature to pass a crop in a future revision the
            # plumbing here is unchanged.
            modalities = decision.check_modalities(
                query_crop=None,
                candidate_crop=None,
                candidate=session,
                pending_plate=session.plate,
                score_reid=score,
            )
            verdict = decision.decide_global(
                score,
                has_plate=bool(session.plate),
                cross_camera=cross_camera,
                similarity_threshold=similarity_threshold,
                modalities=modalities,
            )
            if verdict.verdict == "confirm" and score > best_score:
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
            # Phase 3 / T3.2 — the session may have been added to the index
            # without a plate; refresh so the index sees the latest vector.
            self._gallery_index_upsert(session)

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
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> None:
        """
        Update the stored feature vector of a session with a smoothed EMA vector.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and smoothed_feature is not None:
                session.feature_vector = smoothed_feature
                # Refresh the specific camera's binding if provided
                if camera_id and track_id is not None:
                    session.observing_tracks[camera_id] = track_id
                    self._mark_track_seen(camera_id, track_id)
                elif session.last_seen_camera and session.last_seen_track_id is not None:
                    self._mark_track_seen(
                        session.last_seen_camera,
                        session.last_seen_track_id,
                    )
                # Phase 3 / T3.2 — EMA-smoothed feature update; sync the
                # gallery index so the next FAISS search reflects the new
                # vector. No-op when the feature flag is False.
                self._gallery_index_upsert(session)

    def refresh_track_binding(
        self,
        camera_id: str,
        track_id: int,
        session_id: str,
    ) -> None:
        """
        Lightweight refresh: update last_seen timestamps and observing_tracks
        without the full reattach logic.  Used by _process_global_tracking
        when a track already has a session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            now = self._clock()
            session.last_seen_at = now
            session.observing_tracks[camera_id] = track_id
            self._mark_track_seen(camera_id, track_id, now)

    def reattach_track_to_confirmed_session(
        self,
        camera_id: str,
        track_id: int,
        query_vector: Optional[np.ndarray],
        similarity_threshold: float = 0.52,
        reattach_dry_run: bool = False,
    ) -> Optional[str]:
        """
        Upgrade an anonymous appearance-only track to an already confirmed session.

        Args:
            reattach_dry_run: when True, skip the destructive orphan-session
                cleanup at the end of the function. Used by Phase 2's
                ``MatchVoter`` so a not-yet-committed vote does not delete the
                anonymous session before the vote is in.
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
                        or (
                            session.last_seen_camera
                            and session.last_seen_camera != camera_id
                        )
                        or (self._clock() - session.last_seen_at).total_seconds()
                        >= self.SESSION_HANDOFF_GUARD_SECONDS
                    )
                )
            ]

        best_sid = None
        best_score = -1.0
        decision = self.match_decision
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
            cross_camera = bool(
                session.last_seen_camera
                and camera_id
                and session.last_seen_camera != camera_id
            )
            # Phase 2 T2.1 — reattach has no live crop here (the caller
            # only supplies a feature vector). Re-use cached session
            # metadata so the ensemble can still see whatever the cascade
            # has previously committed about this session.
            modalities = decision.check_modalities(
                query_crop=None,
                candidate_crop=None,
                candidate=session,
                pending_plate=session.plate,
                score_reid=score,
            )
            verdict = decision.decide_reattach(
                score,
                cross_camera=cross_camera,
                similarity_threshold=similarity_threshold,
                modalities=modalities,
            )
            if verdict.verdict == "confirm" and score > best_score:
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

            target_session.last_seen_at = self._clock()
            target_session.last_seen_camera = camera_id
            target_session.last_seen_track_id = track_id
            if query_vector is not None:
                target_session.feature_vector = query_vector
                # Phase 3 / T3.2 — refresh the gallery index with the new
                # query vector so subsequent FAISS searches see this view.
                self._gallery_index_upsert(target_session)
            self._drop_other_track_mappings_for_session(
                best_sid,
                keep=(camera_id, track_id),
            )
            self._track_session_map[(camera_id, track_id)] = best_sid
            target_session.observing_tracks[camera_id] = track_id
            self._mark_track_seen(camera_id, track_id)

            orphan_sid = current_sid
            if orphan_sid != best_sid and not reattach_dry_run:
                # Phase 2's MatchVoter passes reattach_dry_run=True so a
                # not-yet-committed vote does not delete the anonymous session
                # before the vote is in. Default behaviour (False) removes the
                # orphan immediately, matching pre-refactor semantics.
                still_used = orphan_sid in self._track_session_map.values()
                if (
                    not still_used
                    and current_session.status == "confirmed"
                    and current_session.linked_slot is None
                ):
                    self._sessions.pop(orphan_sid, None)
                    # Phase 3 / T3.2 — drop orphan from the FAISS gallery.
                    self._gallery_index_remove(orphan_sid)

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
        snapshot_path: Optional[str] = None,
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

            # Refuse to bind a plate whose latest parking_sessions row is
            # closed — the car has exited the garage (per PMS-AI) and any
            # further re-id match would be ghost activity. The check fails
            # open if the DB is unreachable so a blip doesn't strand traffic.
            if not self.is_plate_inside(session.plate):
                logger.warning(
                    "[try_link_to_slot] refused: plate=%s already exited "
                    "(no open parking_sessions row); skipping bind to %s",
                    session.plate, slot_id,
                )
                return None

            existing = self._parked.get(slot_id)
            if existing is not None:
                if existing.session_id == session.session_id:
                    return existing.plate
                return existing.plate

            # Phase 2 / T2.2 — temporal voting. When enabled, the commit
            # is gated by ``MatchVoter`` so a single noisy frame cannot
            # parked-flip a session on its own. The voter returns ``None``
            # while the buffer is still filling or no plate has won the
            # K-of-N vote yet; callers in engine_runtime.py:678 / :715
            # already tolerate ``None`` (rate-gated retry next frame).
            voter = getattr(self, "_match_voter", None)
            if voter is not None and self.matching_config.voting_enabled:
                vote_input = Decision(
                    verdict="confirm",
                    reason="try_link_to_slot",
                    scores={
                        "plate": session.plate,
                        "session_id": session.session_id,
                        "slot_id": slot_id,
                        "camera_id": camera_id,
                        "track_id": int(track_id) if track_id is not None else None,
                    },
                )
                commit = voter.submit(camera_id, track_id, vote_input)
                if commit is None:
                    logger.debug(
                        "[try_link_to_slot] voter deferred commit for plate=%s "
                        "session=%s slot=%s (window not yet decisive)",
                        session.plate,
                        session.session_id,
                        slot_id,
                    )
                    return None

            # Enforce one-slot-per-vehicle rule: if this session is already
            # linked to a different slot, remove that old linkage first.
            if session.linked_slot and session.linked_slot != slot_id:
                old_slot_id = session.linked_slot
                logger.info(
                    "[REGISTRY] Vehicle session %s (plate %s) moving from slot %s to %s",
                    session.session_id,
                    session.plate,
                    old_slot_id,
                    slot_id,
                )
                self._parked.pop(old_slot_id, None)

            session.linked_slot = slot_id
            session.linked_slot_name = slot_name
            session.linked_camera = camera_id
            session.linked_floor = floor
            session.linked_zone_id = zone_id
            session.linked_zone_name = zone_name
            session.linked_at = timestamp
            session.status = "parked"
            self._parked[slot_id] = session

            # Detailed Match Performance Logging
            cfg = self.matching_config
            flags = {
                "CLAHE": cfg.use_lab_clahe,
                "MULTISHOT": cfg.use_multishot,
                "COLOR_FILTER": cfg.use_color_filter,
            }
            match_logger.info(
                "MATCH_EVENT | Plate: %s | Slot: %s | Time: %s | "
                "NewCost: %.4f | OldCost: %.4f | Flags: %s | "
                "GateSnapshots: %s | SlotSnapshot: %s",
                session.plate,
                slot_id,
                timestamp.isoformat(),
                session.new_pipeline_score,
                session.old_pipeline_score,
                flags,
                session.gate_snapshot_paths,
                snapshot_path or "N/A"
            )

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
        """
        Close a parked session and purge all record of a plate when it exits.
        Ensures the system 'forgets' the vehicle completely upon exit.
        """
        with self._lock:
            # 1. Find and close any active sessions for this plate (parked or driving)
            sessions_to_remove = [
                s for s in self._sessions.values() if s.plate == plate
            ]
            
            # Also check _parked dict specifically
            parked_slots_to_clear = [
                slot_id for slot_id, sess in self._parked.items() if sess.plate == plate
            ]
            for slot_id in parked_slots_to_clear:
                sess = self._parked.pop(slot_id)
                if sess not in sessions_to_remove:
                    sessions_to_remove.append(sess)

            for session in sessions_to_remove:
                session.status = "exited"
                # Clean up track bindings across all cameras
                for obs_cam, obs_tid in list(session.observing_tracks.items()):
                    self._track_session_map.pop((obs_cam, obs_tid), None)
                    self._track_last_seen.pop((obs_cam, obs_tid), None)
                session.observing_tracks.clear()
                
                # Legacy cleanup
                if session.last_seen_camera and session.last_seen_track_id is not None:
                    self._track_session_map.pop(
                        (session.last_seen_camera, session.last_seen_track_id),
                        None,
                    )
                
                self._sessions.pop(session.session_id, None)
                # Phase 3 / T3.2 — drop exited session from the FAISS
                # gallery so a future query never matches against a car
                # that has already left the facility.
                self._gallery_index_remove(session.session_id)
                self._history.append(session)
                logger.info("[REGISTRY] Closed session %s for plate %s (Exit)", session.session_id, plate)

            # 2. Purge any PENDING/PROVISIONAL ANPR events for this plate
            # This prevents an old entry record from matching a future car
            events_to_remove = []
            for event_id, event in self._pending_events.items():
                if event.plate == plate:
                    events_to_remove.append(event_id)
                    # If this event was bound to a candidate, kill the candidate too
                    if event.candidate_id:
                        candidate = self._park_entry_candidates.pop(event.candidate_id, None)
                        if candidate:
                            candidate.status = "dropped"
                            self._clear_candidate_references(candidate)

            for event_id in events_to_remove:
                self._pending_events.pop(event_id, None)
                if event_id in self._pending_event_order:
                    try:
                        self._pending_event_order.remove(event_id)
                    except ValueError:
                        pass
            
            if sessions_to_remove or events_to_remove:
                logger.info("[REGISTRY] Purged %d sessions and %d events for plate %s", 
                            len(sessions_to_remove), len(events_to_remove), plate)
