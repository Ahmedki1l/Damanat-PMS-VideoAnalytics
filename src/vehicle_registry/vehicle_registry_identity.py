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

# Floors that run YOLO occupancy only — no ReID embedding compute and no
# plate/identity matching. Currently the ground floor. These cameras host real
# slots and occupancy state machines, but do NOT participate in ReID feature
# extraction, B1 confirmation, cross-camera reattach, or plate→slot linking;
# identity for them is established via the external ANPR gate plates, not
# appearance. The engine layer (`_process_special_zones` / `_update_slot_state`
# in `core.engine.engine_runtime`) reads this predicate directly to skip ReID
# and plate calls proactively; the three registry entry points below also guard
# against it as defense-in-depth (via the injected camera→floor map).
#
# Matched on the normalized floor label so the DB spelling ("Ground") and the
# test spelling ("Ground Floor") both resolve. Replaces the former hardcoded
# ``IDENTITY_MATCHING_DISABLED_CAMERAS = {"CAM-01", "CAM-02"}`` camera-id set so
# any ground-floor camera is covered automatically, regardless of id.
def is_reid_disabled_floor(floor: str) -> bool:
    return (floor or "").strip().lower() in {"ground", "ground floor"}

# Single-camera ownership: a session observed by several cameras at once is
# owned by exactly one — the live observer with the highest ReID score. A track
# counts as "live" only if seen within OWNER_STALENESS_SECONDS; the current
# owner is kept unless a challenger beats its score by OWNER_SWITCH_MARGIN
# (hysteresis to stop label flicker when scores are close).
OWNER_STALENESS_SECONDS = 3.0
OWNER_SWITCH_MARGIN = 0.05
# Score seeded for a camera that owns a session by definition (brand-new
# appearance session, or a B1/ANPR plate confirmation) rather than by a ReID
# similarity comparison.
OWNER_DEFINITIVE_SCORE = 1.0

# Rank-5 OCR disambiguation: minimum plate-OCR confidence before a read is
# allowed to pick a session among the ReID top-5. Combined with the hard
# requirement that the read match a candidate ALREADY in the ReID top-5 (and
# in the camera's area), so a stray misread cannot invent a match.
RANK5_OCR_MIN_CONF = 0.60
# How many nearest ReID neighbours to keep as the candidate pool ("rank-5").
GLOBAL_MATCH_RANK = 5

# Minimum view-quality for a camera to count as having a "full view" of a car
# in the ownership contest (see _resolve_owner_camera). view_quality is 1.0
# when the bbox is fully inside the frame, lower when clipped by an edge.
_FULL_VIEW_MIN = 0.9


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
                # Called from the per-frame identity gate — must not block on a
                # lock (NOLOCK + short LOCK_TIMEOUT; a stale read fails open
                # below, which is the safe default here).
                session.execute(_text("SET LOCK_TIMEOUT 3000"))
                row = session.execute(
                    _text(
                        "SELECT TOP 1 status FROM dbo.parking_sessions WITH (NOLOCK) "
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
        seed_gallery: bool = False,
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

        # Seed the durable per-plate gallery folder (vehicle_images/gallery/
        # <plate>/) so a folder exists on disk the moment a car is confirmed —
        # instead of only when live-tracking accumulation happens to fire (which
        # may never happen for a short-lived / low-FPS car). Reuses the
        # already-extracted feature. ONLY when the caller passes seed_gallery:
        # the authoritative CAM-03 B-entry reference (confirm_b1_entrance_by_plate)
        # is safe to persist, but the wide gate-only ANPR shot from the direct
        # session is NOT — it is kept out of matching by gate_reference_only, and
        # persisting it would reintroduce the false-match-against-a-parked-car
        # risk on warm-start (build_session_from_gallery loads it primary-first).
        if seed_gallery:
            self._seed_plate_gallery_reference(
                session, ordered_images, feature_vectors, primary_snapshot_index
            )
        return True

    def _seed_plate_gallery_reference(
        self,
        session,
        ordered_images: List[np.ndarray],
        feature_vectors: List[Optional[np.ndarray]],
        primary_snapshot_index: int,
    ) -> None:
        """Persist the primary confirmation crop + embedding to the plate's
        on-disk gallery. No-op when persistence is off, the plate is unknown, or
        the primary image/feature is missing. Best-effort — disk errors are
        swallowed so confirmation never fails on a storage hiccup."""
        store = self.gallery_store
        plate = getattr(session, "plate", None)
        if store is None or not plate:
            return
        idx = primary_snapshot_index
        if idx >= len(ordered_images) or idx >= len(feature_vectors):
            return
        image = ordered_images[idx]
        feature = feature_vectors[idx]
        if image is None or getattr(image, "size", 0) == 0 or feature is None:
            return
        try:
            store.save_ref(
                plate,
                image,
                feature,
                quality=999.0,  # authoritative reference — keep over live views
                camera_id=getattr(session, "last_seen_camera", "") or "ANPR",
            )
        except Exception as exc:  # pragma: no cover - disk best-effort
            logger.debug("[gallery] seed for %s failed: %r", plate, exc)

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
            # Gate image only — keep this session out of global ReID matching
            # until CAM-03 attaches a real B1 reference (confirm_b1_entrance_by_plate).
            # Prevents the wide gate shot from false-matching a car already parked
            # inside the garage the instant the entry event arrives.
            gate_reference_only=True,
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

    def bind_anpr_event_to_candidate(
        self,
        candidate_id: str,
        event_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Bind a SPECIFIC pending ANPR entry (by ``event_id``) to a candidate.

        Used by the ANPR-image entry path, where the candidate's snapshot IS the
        ANPR camera's shot of the very plate we just decoded — so the pairing is
        known by identity and must NOT go through the FIFO
        :meth:`bind_next_pending_anpr_to_candidate` (which pairs by arrival
        order and can cross-bind an image to an *older* pending entry for a
        different plate, swapping the two cars). The live-track path, where a
        CAM-03 crop has no plate yet, still uses the FIFO method.

        Returns the bound plate, or None when the event is not (any longer) a
        bindable pending entry — the caller then falls back to FIFO.
        """
        now = timestamp or self._clock()
        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None or candidate.status != "open":
                return None
            event = self._pending_events.get(event_id)
            if event is None or event.direction != "entry":
                return None
            if event.status not in ("pending", "provisional"):
                return None
            age = (now - event.timestamp).total_seconds()
            if age > self.PENDING_ANPR_EXPIRY_SECONDS:
                event.status = "expired"
                return None

            # If this event was already provisionally bound to a different
            # candidate (e.g. a burst re-read produced a fresh ANPR-image
            # candidate for the same plate), retire the stale candidate so the
            # "one event ↔ one candidate" invariant holds — otherwise both would
            # match the next B1 image. Newer image wins (burst last-wins).
            prev_cid = event.candidate_id
            if prev_cid and prev_cid != candidate_id:
                stale = self._park_entry_candidates.get(prev_cid)
                if stale is not None and stale.status != "confirmed":
                    stale.status = "dropped"
                    stale.bound_event_id = None

            event.status = "provisional"
            event.candidate_id = candidate.candidate_id

            candidate.status = "provisional"
            candidate.bound_event_id = event.event_id
            candidate.last_seen_at = now

            logger.info(
                "[PARK_ENTRY] Bound ANPR event %s (plate=%s) to candidate %s (by event)",
                event.event_id,
                event.plate,
                candidate_id,
            )
            return event.plate

    def _is_reid_disabled_camera(self, camera_id: str) -> bool:
        """Floor-based identity gate for a ``camera_id``, resolved through the
        ``camera_id → floor`` map injected at registry construction
        (``self._camera_floors``). Defaults to *enabled* (returns False) when
        the map lacks the camera, so a missing entry never silently drops
        identity matching."""
        floors = getattr(self, "_camera_floors", None) or {}
        return is_reid_disabled_floor(floors.get(camera_id, ""))

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
        if self._is_reid_disabled_camera(camera_id):
            return None
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
            track_session = self._sessions.get(existing_sid) if existing_sid else None

            # Reuse a session already created for THIS ANPR event — e.g. the
            # direct gate-entry session from confirm_anpr_session_directly (also
            # enriched by the CAM-03 B-entry API push). That path and this live
            # in-zone confirmation both confirm the same physical entry, so
            # creating a fresh session here would leave two sessions carrying the
            # same plate (the frozen-at-entrance duplicate seen in production).
            event_session = None
            for candidate_session in self._sessions.values():
                if (
                    candidate_session.event_id == event.event_id
                    and candidate_session.status in ("confirmed", "parked")
                ):
                    event_session = candidate_session
                    break

            # The event-bound session wins over a bare appearance session the
            # live track may have spun up first (ordering: appearance-before-gate).
            session = event_session or track_session

            # Collapse that appearance orphan onto the event session so the car
            # doesn't end up with two sessions. Only ever drop a plate-less
            # appearance session — never delete another confirmed identity.
            if (
                event_session is not None
                and track_session is not None
                and track_session.session_id != event_session.session_id
                and not track_session.plate
            ):
                self._track_session_map.pop((camera_id, track_id), None)
                self._sessions.pop(track_session.session_id, None)
                self._gallery_index_remove(track_session.session_id)
                logger.info(
                    "[B1] Collapsed appearance session %s into event session %s "
                    "(same car, plate=%s)",
                    track_session.session_id, event_session.session_id, event.plate,
                )

            if session:
                logger.info(
                    "[B1] Upgrading existing session %s with plate %s",
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
                # The primary reference is now the live CAM-03 crop, not the wide
                # gate image — this session is a trustworthy global-match target.
                session.gate_reference_only = False
                session.new_pipeline_score = best_score
                session.old_pipeline_score = getattr(best_candidate, "old_pipeline_score", best_score)
                session.gate_snapshot_paths = self._candidate_snapshot_paths(live_candidate)

                self._drop_other_track_mappings_for_session(
                    session.session_id,
                    keep=(camera_id, track_id),
                )
                self._track_session_map[(camera_id, track_id)] = session.session_id
                session.observing_tracks[camera_id] = track_id
                session.observing_scores[camera_id] = OWNER_DEFINITIVE_SCORE
                session.owner_camera = camera_id
                self._mark_track_seen(camera_id, track_id, now)
                # Re-sync the gallery index: the feature vector / references just
                # changed (and a reused direct session was already indexed).
                self._gallery_index_upsert(session)
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
                session.observing_scores[camera_id] = OWNER_DEFINITIVE_SCORE
                session.owner_camera = camera_id
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

    def confirm_b1_entrance_by_plate(
        self,
        plate: str,
        image: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Attach an externally-pushed B1 entrance snapshot to the plate's
        session and make it the primary identity reference at B1.

        This is the API-push counterpart of the in-engine CAM_03 B1_Entrence
        confirmation (:meth:`confirm_at_b1_entrance`): the external ANPR/vision
        integration delivers the CAM_03 image via the ANPR endpoint with
        direction ``B-entry`` (plate + image), instead of relying on the live
        YOLO track passing through the confirmation zone — which is unreliable
        when per-camera FPS is low. Keyed by plate, since there is no live track.

        Returns the session_id it enriched, or None when no session exists yet
        for the plate (e.g. the gate ``entry`` event was missed) or the image is
        unusable.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return None
        now = timestamp or self._clock()
        with self._lock:
            session = None
            for s in self._sessions.values():
                if s.plate != plate:
                    continue
                if s.status not in ("confirmed", "parked", "provisional"):
                    continue
                if session is None or s.last_seen_at > session.last_seen_at:
                    session = s
            if session is None:
                return None

            session.last_seen_at = now
            session.last_seen_camera = "CAM-03"
            # seed_gallery=True: the CAM-03 B-entry crop is the authoritative
            # primary identity reference — safe to persist to the durable gallery
            # (unlike the wide gate-only shot in confirm_anpr_session_directly).
            if not self._persist_session_gallery(
                session, [image], now, primary_snapshot_index=0, seed_gallery=True
            ):
                return None
            # The primary reference is now the real CAM-03 shot, not the wide gate
            # image — the session is a trustworthy match target again.
            session.gate_reference_only = False
            self._gallery_index_upsert(session)
            logger.info(
                "[B1] CAM-03 B-entry snapshot attached for plate=%s -> session %s "
                "(primary identity reference at B1)",
                plate,
                session.session_id,
            )
            return session.session_id

    def add_gallery_snapshot_by_plate(
        self,
        plate: str,
        image: np.ndarray,
        source_cam: str = "CAM-23",
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """Append an externally-pushed snapshot to the plate's session gallery as
        a SECONDARY appearance reference — without overriding the primary B1
        (CAM-03 ``B-entry``) identity reference.

        This is the plate-keyed counterpart of :meth:`add_session_snapshot` for
        the entry-ramp camera (CAM-23, ``ramp-entry``): the gate ANPR establishes
        the plate, CAM-03 sets the primary B1 reference, and CAM-23 contributes an
        extra viewpoint that enriches cross-camera ReID recall without replacing
        anything. Keyed by plate (no live track), mirroring
        :meth:`confirm_b1_entrance_by_plate`.

        Returns the session_id it enriched, or None when no session exists yet for
        the plate or the image is unusable.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return None
        now = timestamp or self._clock()
        with self._lock:
            session = None
            for s in self._sessions.values():
                if s.plate != plate:
                    continue
                if s.status not in ("confirmed", "parked", "provisional"):
                    continue
                if session is None or s.last_seen_at > session.last_seen_at:
                    session = s
            if session is None:
                return None

            # add_session_snapshot appends to the gallery (does NOT touch the
            # primary feature_vector/snapshot_path); self._lock is a reentrant
            # RLock so the nested acquire is safe.
            path = self.add_session_snapshot(session.session_id, image, timestamp=now)
            if path is None:
                return None
            self._gallery_index_upsert(session)
            logger.info(
                "[B1] %s snapshot added to plate=%s -> session %s "
                "(secondary ReID reference)",
                source_cam, plate, session.session_id,
            )
            return session.session_id

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

    def _resolve_owner_camera(
        self,
        session: VehicleSession,
        now: datetime,
    ) -> Optional[str]:
        """Pick the single camera that owns this session's identity.

        Assumes ``self._lock`` is held. A car seen by several cameras at once is
        owned by exactly one:

        * If the session is linked to a parking slot, the slot's camera owns it
          (a parked car belongs to the camera whose slot it physically occupies).
        * Otherwise the live observer (track seen within
          ``OWNER_STALENESS_SECONDS``) with the highest ReID score owns it, with
          ``OWNER_SWITCH_MARGIN`` hysteresis favouring the current owner so the
          label does not flicker between cameras with near-equal scores.

        Updates and returns ``session.owner_camera``. Returns ``None`` only when
        no live observer can be determined.
        """
        if session.linked_slot and session.linked_camera:
            session.owner_camera = session.linked_camera
            return session.owner_camera

        live = [
            cam
            for cam, tid in session.observing_tracks.items()
            if (
                (seen := self._track_last_seen.get((cam, tid))) is not None
                and (now - seen).total_seconds() <= OWNER_STALENESS_SECONDS
            )
        ]

        # Zoning: restrict the ownership contest to cameras of the car's settled
        # area (mirrors IntraAreaFusion.resolve_owner — the unit-tested spec of
        # this policy). A neighbouring area's camera can't steal ownership and
        # teleport the identity mid-transit. When no in-area camera is live, keep
        # the current owner unchanged rather than handing it to an out-of-area
        # camera. Fully gated: un-zoned / un-settled cars fall through unchanged.
        if (
            self._area_registry is not None
            and self._area_registry.enabled
            and session.current_area
        ):
            in_area = set(self._area_registry.cameras(session.current_area))
            restricted = [cam for cam in live if cam in in_area]
            if restricted:
                live = restricted
            else:
                # No in-area camera is live. Hand off to any live *un-zoned*
                # camera (NULL/empty area): it belongs to no competing area, so
                # it can't teleport the identity, and owning it won't move
                # current_area. Only keep the (stale) owner when there isn't even
                # an un-zoned observer — this stops a car drifting into un-zoned
                # space (e.g. a NULL-area ramp camera) from sticking on a dead
                # owner. A *different zoned area's* camera is still excluded.
                unzoned = [
                    cam
                    for cam in live
                    if not self._area_registry.area_for_camera(cam)
                ]
                if not unzoned:
                    return session.owner_camera
                live = unzoned

        # Priority 1 — slot-hosting cameras. A car straddling a slot camera and
        # a slotless aisle/transit camera should be owned by the slot camera, so
        # its identity (and plate) is attributed where it can actually be bound
        # to a slot and locked. Only narrows when a slot camera is live; empty
        # ``_cameras_with_slots`` (not configured) leaves ``live`` untouched.
        if self._cameras_with_slots:
            slotted = [cam for cam in live if cam in self._cameras_with_slots]
            if slotted:
                live = slotted

        # Priority 2 — cameras with a full view of the car. Among the surviving
        # candidates, one that sees the whole vehicle (bbox not clipped by a
        # frame edge) outranks one that only catches part of it, since the full
        # view yields the more trustworthy crop/identity. Only narrows when at
        # least one live camera has a full-view reading; no readings ⇒ unchanged.
        if len(live) > 1:
            full_view = [
                cam
                for cam in live
                if self._track_view_quality.get(
                    (cam, session.observing_tracks.get(cam)), 0.0
                )
                >= _FULL_VIEW_MIN
            ]
            if full_view:
                live = full_view

        if not live:
            # Nobody live right now — keep the last known owner if it is still
            # an observer, otherwise clear it.
            if session.owner_camera not in session.observing_tracks:
                session.owner_camera = None
            return session.owner_camera

        if len(live) == 1:
            session.owner_camera = live[0]
            return session.owner_camera

        def _score(cam: str) -> float:
            return session.observing_scores.get(cam, 0.0)

        best = max(live, key=_score)
        current = session.owner_camera
        if current in live and current != best:
            # Incumbent keeps ownership unless the challenger clears the margin.
            if _score(best) < _score(current) + OWNER_SWITCH_MARGIN:
                best = current
        session.owner_camera = best
        return best

    def get_plate_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[str]:
        """Resolve plate through confirmed session mapping.

        Single-camera ownership: the plate is only returned to the owning
        camera, so plate-driven data (slot linking, presence, alerts) is
        attributed to one camera even when several see the car.
        """
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id is None:
                return None
            session = self._sessions.get(session_id)
            if session is None:
                return None
            owner = self._resolve_owner_camera(session, self._clock())
            if owner is not None and owner != camera_id:
                return None
            return session.plate

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

        Single-camera ownership: when several cameras observe the same car at
        once, only the owning camera (highest ReID score among live observers)
        gets the identity label; the others receive the ``"0"`` fallback so the
        box is still drawn without the duplicated plate.
        """
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    owner = self._resolve_owner_camera(session, self._clock())
                    if owner is not None and owner != camera_id:
                        return "0"
                    return session.display_id
        return "0"

    def get_reid_score_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[float]:
        """Latest ReID similarity for a track on this camera (None if unknown).

        Reads the per-camera ``observing_scores`` recorded when the track was
        matched/confirmed; falls back to the session's ``new_pipeline_score``.
        Used only for the ``--show`` overlay — display, not decisions.
        """
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if not session_id:
                return None
            session = self._sessions.get(session_id)
            if session is None:
                return None
            score = session.observing_scores.get(camera_id)
            if score is None:
                score = session.new_pipeline_score
            return float(score) if score else None

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
            # The originating camera is the sole, definitive observer at birth.
            session.observing_scores[camera_id] = OWNER_DEFINITIVE_SCORE
            session.owner_camera = camera_id
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

    def record_reference_for_track(
        self,
        camera_id: str,
        track_id: int,
        crop_bgr: np.ndarray,
        view_quality: float,
    ) -> bool:
        """Engine-facing wrapper: resolve the (camera, track)'s session and add a
        gallery reference for it. No-op when the track has no session or the
        session has no plate yet. See :meth:`accumulate_reference`."""
        with self._lock:
            sid = self._track_session_map.get((camera_id, track_id))
            session = self._sessions.get(sid) if sid else None
        if session is None:
            return False
        return self.accumulate_reference(session, crop_bgr, camera_id, view_quality)

    def accumulate_reference(
        self,
        session,
        crop_bgr: np.ndarray,
        camera_id: str,
        view_quality: float,
    ) -> bool:
        """Add one quality-gated reference view to a plate's growing gallery.

        Grows ``session.reference_feature_vectors`` (which
        ``match_global_session`` already scores against) AND persists the crop +
        embedding to the plate's on-disk folder so it survives restart. Gated:
        persistence enabled, plate known, full view, sharp enough; throttled to
        one add per (plate, camera) per interval; deduped against existing refs;
        capped in-memory (disk cap handled by the store). No-op / False when any
        gate fails. Poor crops are rejected so the gallery is not poisoned.
        """
        store = self.gallery_store
        if store is None or session is None or not getattr(session, "plate", None):
            return False
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return False
        cfg = self._matching_config
        if float(view_quality) < cfg.gallery_min_view_quality:
            return False

        now_dt = self._clock()
        now_ts = now_dt.timestamp()
        key = (session.plate, camera_id or "")
        if now_ts - self._gallery_last_add.get(key, 0.0) < cfg.gallery_accumulate_interval_s:
            return False
        # Consume the interval now (before the heavy embed) so a blurry or
        # duplicate frame doesn't retrigger extract_feature every tick.
        self._gallery_last_add[key] = now_ts

        try:
            from src.reid_matcher.reid_burst import sharpness_score

            if sharpness_score(crop_bgr) < cfg.gallery_min_sharpness:
                return False
        except Exception:
            return False

        vec = self.reid_matcher.extract_feature(crop_bgr)
        if vec is None:
            return False

        with self._lock:
            live = self._sessions.get(session.session_id)
            if live is None or not live.plate:
                return False
            for ref in [live.feature_vector] + list(live.reference_feature_vectors):
                if ref is not None and self.reid_matcher.compute_similarity(vec, ref) > cfg.gallery_dedup_cosine:
                    return False  # near-duplicate — nothing new to learn
            live.reference_feature_vectors.append(vec)
            cap = max(1, int(cfg.gallery_max_refs_per_car))
            if len(live.reference_feature_vectors) > cap:
                # FIFO cap in memory; the disk store keeps the quality-best set.
                live.reference_feature_vectors = live.reference_feature_vectors[-cap:]
            self._gallery_index_upsert(live)
            plate = live.plate

        # Disk write off the lock.
        store.save_ref(plate, crop_bgr, vec, quality=float(view_quality), camera_id=camera_id or "")
        return True

    def build_session_from_gallery(
        self, plate: str, floor: Optional[str] = None
    ) -> Optional[str]:
        """Reload a plate's persisted gallery into a live confirmed session.

        Used at startup (cars still inside) and on ANPR re-entry (warm-start).
        Uses the cached vectors when the stored model tag matches; otherwise
        re-embeds from the retained crops. If a live session already exists for
        the plate (e.g. the vectorless one created by _restore_plate_locks), its
        appearance vectors are ENRICHED from the gallery instead of duplicating.
        Returns the session_id, or None when there is nothing to load."""
        store = self.gallery_store
        if store is None or not plate or not store.has(plate):
            return None

        vectors, tag = store.load_vectors(plate)
        if (not vectors) or tag != store._model_tag:
            crops = store.load_crops(plate)
            if crops:
                feats = self.reid_matcher.extract_features_batch(crops)
                vectors = [f for f in feats if f is not None]
        if not vectors:
            return None

        now = self._clock()
        with self._lock:
            existing = next(
                (
                    s
                    for s in self._sessions.values()
                    if s.plate == plate and s.status in ("confirmed", "parked")
                ),
                None,
            )
            if existing is not None:
                # Attach the persisted appearance to the live (often vectorless)
                # session so ReID can re-identify it — no duplicate session.
                if existing.feature_vector is None:
                    existing.feature_vector = vectors[0]
                    extra = vectors[1:]
                else:
                    extra = vectors
                existing.reference_feature_vectors.extend(extra)
                # Cap after enrich: a repeated warm-start (duplicate ANPR entry,
                # or a reload racing a live session) must not grow the ref list
                # without bound — more refs cost matching time AND widen the
                # false-match surface. Keep the most recent up to the cap.
                cap = max(1, int(self._matching_config.gallery_max_refs_per_car))
                if len(existing.reference_feature_vectors) > cap:
                    existing.reference_feature_vectors = (
                        existing.reference_feature_vectors[-cap:]
                    )
                self._gallery_index_upsert(existing)
                logger.info(
                    "[gallery] enriched existing session %s (plate=%s) with %d refs",
                    existing.session_id, plate, len(vectors),
                )
                return existing.session_id

            session_id = f"reload_{uuid.uuid4().hex[:12]}"
            session = VehicleSession(
                session_id=session_id,
                plate=plate,
                feature_vector=vectors[0],
                reference_feature_vectors=list(vectors[1:]),
                first_seen_at=now,
                last_seen_at=now,
                last_seen_camera="",
                status="confirmed",
                linked_floor=floor or "",
            )
            self._sessions[session_id] = session
            self._gallery_index_upsert(session)
        logger.info(
            "[gallery] reloaded plate=%s -> session %s (%d refs)",
            plate, session_id, len(vectors),
        )
        return session_id

    def match_global_session(
        self,
        query_vector: Optional[np.ndarray],
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        max_time_gap_seconds: float = 600.0,
        similarity_threshold: float = 0.55,
        area_id: Optional[str] = None,
        query_crop: Optional[np.ndarray] = None,
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

        Rank-5 selection: rather than committing to the single nearest neighbour
        (rank-1, noisy on oblique 720p views), the candidate pool is narrowed to
        the top-5 by ReID similarity. When ``query_crop`` is supplied and plate
        OCR is available, a confident read that matches one of those 5
        candidates' plate is chosen outright (ReID need only place the car in
        the top-5; OCR makes the precise pick). If OCR can't disambiguate, the
        method falls back to the rank-1 verdict gated by the
        abstain-on-ambiguity margin, returning None when the top-2 are too
        close — a missing label self-heals on a later frame; a wrong one does
        not.
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
            # Zoning: when an area is given (and the deployment is zoned), bound
            # the candidate pool to cars currently IN that area plus the
            # cross-area handoff pool (cars that recently departed an adjacent
            # area within transit time). area_id None/"" or un-zoned → the
            # legacy all-sessions pool (area_allowed_ids stays None).
            area_allowed_ids = None
            if (
                area_id
                and self._area_registry is not None
                and self._area_registry.enabled
            ):
                area_allowed_ids = set(self._area_sessions.get(area_id, set()))
                if self._handoff_matcher is not None:
                    area_allowed_ids |= self._handoff_matcher.candidate_session_ids(
                        area_id, list(self._sessions.values())
                    )

            if faiss_topk_ids is not None:
                # Pre-narrow to the FAISS top-K. Preserve all the existing
                # invariants on the pool (confirmed/parked, feature present,
                # within max_time_gap_seconds).
                potential_sessions = [
                    session
                    for sid, session in self._sessions.items()
                    if (
                        sid in faiss_topk_ids
                        and (area_allowed_ids is None or sid in area_allowed_ids)
                        and session.status in ("confirmed", "parked")
                        and session.feature_vector is not None
                        and not session.gate_reference_only
                        and (now - session.last_seen_at).total_seconds()
                        <= max_time_gap_seconds
                    )
                ]
            else:
                potential_sessions = [
                    session
                    for sid, session in self._sessions.items()
                    if (
                        (area_allowed_ids is None or sid in area_allowed_ids)
                        and session.status in ("confirmed", "parked")
                        and session.feature_vector is not None
                        and not session.gate_reference_only
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

        # Rank-5: score every guarded candidate once, then keep only the top-5
        # nearest by ReID similarity. The correct identity is far more reliably
        # within the 5 nearest than exactly rank-1 on these oblique views, so we
        # let the secondary signal decide among the 5 instead of trusting the
        # single argmax.
        scored = []
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
            scored.append((session, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_candidates = scored[:GLOBAL_MATCH_RANK]

        # --- Rank-5 OCR disambiguation ---------------------------------- #
        # With the live crop in hand, read the plate ONCE and pick the top-5
        # candidate whose plate it matches. Requires a confident read AND a
        # plate that matches a candidate already in the ReID top-5, so a stray
        # misread can't fabricate an identity. This is the reliable path: ReID
        # need only place the car in the top-5, OCR nails the exact plate.
        #
        # Cost gate: OCR (~50-100 ms) runs only when the ReID top-1 is NOT a
        # clear, confident winner — i.e. its score is below the solo-confirm bar
        # OR the runner-up is within the ambiguity margin. A decisive top-1
        # skips OCR entirely (the fallback confirms it), so easy matches stay
        # cheap and OCR is spent where it actually disambiguates.
        cfg = self._matching_config
        _best = top_candidates[0][1] if top_candidates else -1.0
        _runner = top_candidates[1][1] if len(top_candidates) > 1 else -1.0
        _solo = getattr(cfg, "reid_solo_confirm", 0.68) or 0.68
        _margin = getattr(cfg, "global_match_margin", 0.05) or 0.0
        _ocr_worth_it = (_best < _solo) or ((_best - _runner) < _margin)
        if query_crop is not None and top_candidates and _ocr_worth_it:
            ocr = getattr(self._match_decision, "plate_ocr", None)
            if ocr is not None and hasattr(ocr, "read"):
                try:
                    ocr_text, ocr_conf = ocr.read(query_crop)
                except Exception:
                    ocr_text, ocr_conf = "", 0.0
                if ocr_text and float(ocr_conf or 0.0) >= RANK5_OCR_MIN_CONF:
                    from src.ocr.plate_ocr import plates_match

                    for cand, cand_score in top_candidates:
                        if not cand.plate or not plates_match(ocr_text, cand.plate):
                            continue
                        with self._lock:
                            final = self._sessions.get(cand.session_id)
                            if final and final.status in ("confirmed", "parked"):
                                if camera_id is not None:
                                    final.observing_scores[camera_id] = cand_score
                                final.ocr_confirmed = True
                                logger.info(
                                    "[GLOBAL] cam=%s rank-5 OCR pick: session %s "
                                    "plate=%s (reid=%.3f ocr_conf=%.2f)",
                                    camera_id, cand.session_id, cand.plate,
                                    cand_score, ocr_conf,
                                )
                                return cand.session_id

        # --- Fallback: rank-1 verdict + abstain-on-ambiguity margin ------ #
        # OCR couldn't pick among the top-5 (unreadable, or no candidate plate
        # matched). Fall back to the highest-scoring candidate that the
        # ensemble confirms, but abstain when the top-2 are within the margin —
        # a near-tie is a coin flip and a wrong plate propagates into the DB.
        best_sid = None
        best_score = -1.0
        runner_up_score = -1.0
        decision = self.match_decision
        for session, score in top_candidates:
            cross_camera = bool(
                session.plate
                and session.last_seen_camera
                and camera_id
                and session.last_seen_camera != camera_id
            )
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
                if best_sid is not None:
                    # The demoted previous winner becomes a runner-up.
                    runner_up_score = max(runner_up_score, best_score)
                best_score = score
                best_sid = session.session_id
            else:
                runner_up_score = max(runner_up_score, score)

        if best_sid is not None and runner_up_score >= 0.0:
            margin = float(
                getattr(self._matching_config, "global_match_margin", 0.05) or 0.0
            )
            if margin > 0.0 and (best_score - runner_up_score) < margin:
                logger.info(
                    "[GLOBAL] cam=%s abstain: ambiguous match "
                    "(best=%.3f runner_up=%.3f margin=%.2f)",
                    camera_id,
                    best_score,
                    runner_up_score,
                    margin,
                )
                return None

        if best_sid:
            with self._lock:
                final_session = self._sessions.get(best_sid)
                if final_session and final_session.status in ("confirmed", "parked"):
                    # Record the matching score so single-camera ownership can
                    # pick the highest-confidence observer for this session.
                    if camera_id is not None:
                        final_session.observing_scores[camera_id] = best_score
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
        if self._is_reid_disabled_camera(camera_id):
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
            # Keep this camera's ownership score fresh as the car moves between
            # views, so the highest-confidence observer stays the owner.
            target_session.observing_scores[camera_id] = best_score
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
        if is_reid_disabled_floor(floor):
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

            # Plate-keyed defence: release any OTHER slot currently held by a
            # DIFFERENT session for the same plate. Catches the rare case
            # where two sessions for the same plate end up in _parked (the
            # 120 s duplicate-session guard at confirm_anpr_session_directly
            # missed because the prior session was created longer ago, or a
            # service restart rebuilt _parked from observation). Same-session
            # moves are handled by the block further down. The engine's slot
            # state machine will write the slot-free DB event when it next
            # observes the released slot as empty — same as the same-session
            # release path below.
            if session.plate:
                stale_slots = [
                    sid
                    for sid, s in self._parked.items()
                    if sid != slot_id
                    and s.session_id != session.session_id
                    and s.plate == session.plate
                    and sid not in self._locked_slots  # never release a frozen slot
                ]
                for stale_sid in stale_slots:
                    stale = self._parked.pop(stale_sid, None)
                    if stale is None:
                        continue
                    logger.warning(
                        "[REGISTRY] Plate %s was already linked to slot %s via "
                        "stale session %s; releasing it in favour of session %s "
                        "-> slot %s",
                        session.plate,
                        stale_sid,
                        stale.session_id,
                        session.session_id,
                        slot_id,
                    )
                    stale.linked_slot = None
                    stale.linked_slot_name = None
                    stale.linked_camera = None
                    stale.linked_floor = None
                    stale.linked_zone_id = None
                    stale.linked_zone_name = None
                    stale.linked_at = None
                    stale.status = "confirmed"

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
                # Plate-lock: a session frozen into a locked slot must not be
                # relocated. A mis-association at another (unlocked) slot would
                # otherwise pop the locked slot and move the plate — exactly the
                # drift this feature prevents. Refuse; keep the frozen binding.
                if old_slot_id in self._locked_slots:
                    logger.debug(
                        "[REGISTRY] Refusing to move session %s (plate %s) off "
                        "LOCKED slot %s to %s — binding frozen.",
                        session.session_id, session.plate, old_slot_id, slot_id,
                    )
                    return session.plate
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
            # Vacancy always releases any plate-lock on the slot — this is the
            # only place a lock is dropped (the "slot state changed" release).
            self._locked_slots.discard(slot_id)
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
            session.ocr_confirmed = False
            if session.status == "parked":
                session.status = "confirmed"
            return plate

    # ------------------------------------------------------------------ #
    # Plate-lock: freeze a confirmed plate onto a parked slot
    # ------------------------------------------------------------------ #
    def lock_slot(self, slot_id: str) -> None:
        """Freeze the current plate binding on ``slot_id``. Idempotent.

        Once locked, the relocate / stale-release / move paths in
        ``try_link_to_slot`` refuse to touch the slot until ``unlink_slot``
        (slot confirmed VACANT) drops the lock.
        """
        with self._lock:
            if slot_id in self._parked:
                self._locked_slots.add(slot_id)

    def is_slot_locked(self, slot_id: str) -> bool:
        """True if ``slot_id``'s plate binding is currently frozen."""
        with self._lock:
            return slot_id in self._locked_slots

    def get_slot_binding_confidence(self, slot_id: str) -> float:
        """ReID score of the session currently parked in ``slot_id`` (0.0 if none)."""
        with self._lock:
            session = self._parked.get(slot_id)
            return float(session.new_pipeline_score) if session is not None else 0.0

    def get_slot_ocr_confirmed(self, slot_id: str) -> bool:
        """True if the session parked in ``slot_id`` has had its plate OCR-confirmed."""
        with self._lock:
            session = self._parked.get(slot_id)
            return bool(session.ocr_confirmed) if session is not None else False

    def try_ocr_confirm_slot(self, slot_id: str, crop_bgr) -> bool:
        """Forced-OCR dead-zone pass (§3b): read the plate off the parked car
        crop and, if it agrees with the session's plate, mark the session
        ``ocr_confirmed`` so the lock gate's OCR arm fires.

        Reuses the already-loaded OCR plugin held by MatchDecision — no second
        PaddleOCR is constructed. Returns True once confirmed (also short-circuits
        True if already confirmed). The per-slot attempt cap lives on the engine
        side so this stays a single cheap read.
        """
        with self._lock:
            session = self._parked.get(slot_id)
            if session is None or not session.plate:
                return False
            if session.ocr_confirmed:
                return True
        # OCR is heavy — run it OUTSIDE the registry lock so the per-frame loop
        # for other cameras/slots isn't blocked on a PaddleOCR read.
        ocr = getattr(self._match_decision, "plate_ocr", None)
        if ocr is None or not hasattr(ocr, "read"):
            return False
        try:
            ocr_text, ocr_conf = ocr.read(crop_bgr)
        except Exception:
            return False
        if not ocr_text:
            return False
        from src.ocr.plate_ocr import plates_match
        with self._lock:
            session = self._parked.get(slot_id)
            if session is None or not session.plate:
                return False
            if plates_match(ocr_text, session.plate):
                session.ocr_confirmed = True
                return True

            # Corrective rebind: OCR read a DIFFERENT plate off the physically
            # parked car. ReID guessed wrong — OCR is ground truth when it
            # reads at high confidence. If the read matches another in-garage
            # session's plate, move that identity onto this slot (and mark it
            # OCR-confirmed so the engine's lock gate freezes the CORRECT
            # plate). Without this, a wrong provisional binding can never be
            # fixed: OCR only "agrees or nothing", and the upgrade-only rule
            # keeps the wrong plate indefinitely.
            if float(ocr_conf or 0.0) < 0.80:
                return False  # not confident enough to overturn ReID
            other = None
            for s in self._sessions.values():
                if s.session_id == session.session_id or not s.plate:
                    continue
                if s.status not in ("confirmed", "parked"):
                    continue
                if plates_match(ocr_text, s.plate):
                    other = s
                    break
            if other is None:
                logger.info(
                    "[OCR-REBIND] slot=%s OCR read %r (conf=%.2f) contradicts "
                    "bound plate %s but matches no in-garage session — leaving "
                    "binding provisional",
                    slot_id, ocr_text, ocr_conf, session.plate,
                )
                return False
            # Never steal an identity frozen onto another slot; a competing
            # LOCK means the conflict needs a human/exit event, not a swap.
            if other.linked_slot and other.linked_slot != slot_id:
                if other.linked_slot in self._locked_slots:
                    logger.warning(
                        "[OCR-REBIND] slot=%s OCR says plate %s, but that plate "
                        "is LOCKED to slot %s — refusing to rebind",
                        slot_id, other.plate, other.linked_slot,
                    )
                    return False
                # The car is physically HERE per OCR — release its stale
                # (provisional) binding elsewhere.
                self._parked.pop(other.linked_slot, None)

            now = self._clock()
            old_plate = session.plate
            # Carry the slot metadata from the evicted session onto the
            # corrected one, then detach the wrong session from this slot.
            other.linked_slot = slot_id
            other.linked_slot_name = session.linked_slot_name
            other.linked_camera = session.linked_camera
            other.linked_floor = session.linked_floor
            other.linked_zone_id = session.linked_zone_id
            other.linked_zone_name = session.linked_zone_name
            other.linked_at = now
            other.status = "parked"
            other.ocr_confirmed = True
            session.linked_slot = None
            session.linked_slot_name = None
            session.linked_camera = None
            session.linked_floor = None
            session.linked_at = None
            session.ocr_confirmed = False
            if session.status == "parked":
                session.status = "confirmed"
            self._parked[slot_id] = other
            logger.warning(
                "[OCR-REBIND] slot=%s corrected plate %s -> %s "
                "(OCR read %r, conf=%.2f)",
                slot_id, old_plate, other.plate, ocr_text, ocr_conf,
            )
            return True

    def restore_parked_binding(
        self,
        slot_id: str,
        slot_name: str,
        plate: str,
        confidence: float,
        camera_id: str,
        floor: str,
        locked: bool,
        timestamp: datetime,
    ) -> None:
        """Rebuild a parked+locked binding from persisted DB state on restart.

        The registry starts empty after a restart, but the DB remembers which
        slots held which plate (and whether frozen). This reinserts a minimal
        ``VehicleSession`` into ``_parked`` (and ``_locked_slots`` when locked)
        so the API reports the plate immediately and the first post-restart
        frame cannot overwrite a correct record with a fresh provisional one.
        See engine_runtime._load_camera_db_state.
        """
        with self._lock:
            if slot_id in self._parked:
                return
            session = VehicleSession(
                session_id=f"restored_{uuid.uuid4().hex[:12]}",
                plate=plate,
                first_seen_at=timestamp,
                last_seen_at=timestamp,
                last_seen_camera=camera_id,
                status="parked",
                new_pipeline_score=confidence,
                old_pipeline_score=confidence,
            )
            session.linked_slot = slot_id
            session.linked_slot_name = slot_name
            session.linked_camera = camera_id
            session.linked_floor = floor
            session.linked_at = timestamp
            self._sessions[session.session_id] = session
            self._parked[slot_id] = session
            if locked:
                self._locked_slots.add(slot_id)

    def _handle_exit(self, plate: str, timestamp: datetime) -> None:
        """
        Close a parked session and purge all record of a plate when it exits.
        Ensures the system 'forgets' the vehicle completely upon exit.
        """
        # Retention: keep the plate's on-disk gallery for a future return
        # (warm-start), just stamp the exit time so the TTL GC ages it from now.
        # The in-memory purge below still runs — a car outside the facility must
        # never match a car inside.
        store = self.gallery_store
        if store is not None and plate:
            try:
                store.stamp_exit(plate, timestamp)
            except Exception as exc:  # pragma: no cover - disk best-effort
                logger.debug("[gallery] stamp_exit failed for %s: %r", plate, exc)
        # Drop this plate's accumulation-throttle entries so the map doesn't
        # grow unbounded over a long-running process.
        for key in [k for k in self._gallery_last_add if k[0] == plate]:
            self._gallery_last_add.pop(key, None)

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
                # The car left the facility — drop any plate-lock too so we don't
                # leak a frozen entry pointing at a slot no longer in _parked.
                self._locked_slots.discard(slot_id)
                if sess not in sessions_to_remove:
                    sessions_to_remove.append(sess)

            for session in sessions_to_remove:
                session.status = "exited"
                # Clean up track bindings across all cameras
                for obs_cam, obs_tid in list(session.observing_tracks.items()):
                    self._track_session_map.pop((obs_cam, obs_tid), None)
                    self._track_last_seen.pop((obs_cam, obs_tid), None)
                session.observing_tracks.clear()
                session.observing_scores.clear()
                session.owner_camera = None

                # Legacy cleanup
                if session.last_seen_camera and session.last_seen_track_id is not None:
                    self._track_session_map.pop(
                        (session.last_seen_camera, session.last_seen_track_id),
                        None,
                    )
                
                self._sessions.pop(session.session_id, None)
                # Zoning — drop the exited car from its area bucket so the
                # bounded (per-area) candidate pool never matches a car that
                # has already left the facility.
                self._drop_session_from_area_index(session)
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
