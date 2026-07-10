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

# Phase 2 cleanup audit: these env-driven flags are kept here because
# ``tests/test_multishot_registry.py`` patches them via
# ``patch.object(vehicle_registry_core, "REID_USE_MULTISHOT", ...)`` etc.
# The matching cascade itself now reads ``self.matching_config.use_*`` and
# does not consult these globals — they are pure backward-compatibility
# shims. When the multishot tests migrate to mutating ``MatchingConfig``
# directly, these can be removed.
REID_USE_MULTISHOT = os.getenv("REID_USE_MULTISHOT", "false").lower() == "true"
REID_USE_COLOR_FILTER = os.getenv("REID_USE_COLOR_FILTER", "false").lower() == "true"

# ANPR burst coalescing: consecutive entry reads on the SAME ANPR camera that
# arrive within this many seconds of each other are treated as one passing car;
# the latest plate wins (overwrites the earlier read), even if the plate text
# differs (e.g. an OCR refinement). Each new read slides the window.
ANPR_COALESCE_SECONDS = 3.0

# Two burst reads only coalesce when their plate texts are within this many
# edits of each other. A genuine re-read of the same plate differs by an OCR
# slip (1-2 characters); a bigger distance means a SECOND car tailgated into
# the window, and collapsing the two (last-plate-wins) would rename the first
# car's session to the second car's plate — the burst-coalesce swap vector.
ANPR_COALESCE_MAX_PLATE_EDITS = 2


def _plate_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two plate strings (case-insensitive)."""
    a = (a or "").strip().upper()
    b = (b or "").strip().upper()
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


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
        ts = timestamp or self._clock()
        self._track_last_seen[(camera_id, track_id)] = ts
        # Stamp the first sighting once — the entry anchor for match_global_session.
        self._track_first_seen.setdefault((camera_id, track_id), ts)

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
            self._track_first_seen.pop(key, None)

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
        source_camera: str = "",
    ) -> Optional[str]:
        """
        Append a new snapshot image to an existing session's reference gallery.
        Also extracts and adds a new ReID feature vector to improve matching accuracy.

        ``source_camera`` tags the added reference so match-time trust weighting
        knows which camera produced it (ground-truth vs secondary).
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
                    self._sync_reference_cameras(session)
                    session.reference_source_cameras[-1] = source_camera or ""

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
            # Re-entry grace: record the latest ANPR entry time for this plate
            # BEFORE the coalesce/duplicate logic, so every entry sub-path (fresh
            # event, burst-coalesced re-read, duplicate-ignored) refreshes it.
            # Pairs with _last_anpr_exit_at (stamped in _handle_exit) so a genuine
            # exit->re-entry is honored while PMS-AI's open row is still lagging.
            self._last_anpr_entry_at[plate] = now
            # Set inside the lock: the pending event to return early (a coalesced
            # burst or an ignored duplicate), and any slots freed by a burst
            # rebind eviction. DB clear + early return happen off the lock below.
            coalesced_event = None
            evicted_slots: List[str] = []
            with self._lock:
                # Burst coalescing: a rapid re-read on the same ANPR camera
                # within ANPR_COALESCE_SECONDS is the same passing car — keep the
                # open burst event and let the latest plate win (overwrite now),
                # even if the plate text differs. The window slides on each read.
                if camera_id is not None:
                    burst_eid = self._last_anpr_entry.get(camera_id)
                    burst = self._pending_events.get(burst_eid) if burst_eid else None
                    if (
                        burst is not None
                        and burst.direction == "entry"
                        # "confirmed" is included so a corrected read still
                        # coalesces when the first read was already bound to a
                        # session inside the window (bind-now, overwrite-within-3s).
                        and burst.status in ("pending", "provisional", "confirmed")
                        and (now - burst.timestamp).total_seconds() <= ANPR_COALESCE_SECONDS
                        # Only a plausible re-read of the SAME plate coalesces.
                        # A very different plate inside the window is a second
                        # car tailgating — it must get its own event, or
                        # last-plate-wins renames car A's session to car B's
                        # plate and the two cars swap identities.
                        and _plate_edit_distance(burst.plate, plate)
                        <= ANPR_COALESCE_MAX_PLATE_EDITS
                    ):
                        old_plate = burst.plate
                        burst.timestamp = now  # slide the window
                        if old_plate != plate:
                            burst.plate = plate
                            evicted_slots = self._rebind_anpr_plate(
                                burst.event_id, plate, timestamp=now
                            )
                            logger.info(
                                "[ANPR] Burst re-read on %s: plate %s -> %s (last wins)",
                                camera_id,
                                old_plate,
                                plate,
                            )
                        coalesced_event = burst

                if coalesced_event is None:
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
                            coalesced_event = existing
                            break

                if coalesced_event is None:
                    self._pending_events[event.event_id] = event
                    self._pending_event_order.append(event.event_id)
                    if camera_id is not None:
                        self._last_anpr_entry[camera_id] = event.event_id
                    logger.info("[ANPR] Entry: plate=%s", plate)

            # DB clear off-lock — never hold self._lock across DB I/O. Then return
            # the coalesced/duplicate event early (skipping warm-start, as before).
            for slot_id in evicted_slots:
                self._clear_slot_db_binding(slot_id)
            if coalesced_event is not None:
                return coalesced_event
        elif direction == "exit":
            self._handle_exit(plate, now)
            logger.info("[ANPR] Exit: plate=%s", plate)
        elif direction == "B-entry":
            # The B1 (CAM-03) confirmation snapshot, pushed via the ANPR API as
            # plate + image. It does NOT open a new pending entry — it confirms
            # an existing gate read. The image is attached to the plate's session
            # (and used for ReID) by confirm_b1_entrance_by_plate in the API
            # layer, which has the decoded frame. Nothing to record here.
            logger.info("[ANPR] B1 confirmation (CAM-03) snapshot: plate=%s", plate)
        elif direction == "ramp-entry":
            # The CAM-23 ramp-entry snapshot, pushed via the ANPR API as a
            # SECONDARY appearance reference (not the primary). It does NOT open a
            # pending entry — it enriches an existing gate-created session. The image
            # is attached to the plate's session by add_gallery_snapshot_by_plate in
            # the API layer, which has the decoded frame. Nothing to record here.
            logger.info("[ANPR] CAM-23 ramp-entry snapshot: plate=%s", plate)
        else:
            # Be defensive: an unrecognised direction from the ANPR client must
            # not 500 the webhook (which would drop the event). Log and no-op.
            logger.warning("[ANPR] Ignoring unsupported direction %r (plate=%s)", direction, plate)

        # Warm-start: a returning car with a retained gallery is re-loaded now so
        # ReID can re-identify it immediately on this visit (no-op when disabled,
        # no folder, or already live). Off the lock — build_session_from_gallery
        # takes it itself.
        if direction == "entry" and plate and self.gallery_store is not None:
            try:
                self.build_session_from_gallery(plate)
            except Exception as exc:  # pragma: no cover - warm-start best-effort
                logger.debug("[gallery] warm-start failed for %s: %r", plate, exc)

        return event

    def _rebind_anpr_plate(
        self, event_id: str, new_plate: str, timestamp: Optional[datetime] = None
    ) -> List[str]:
        """Propagate a burst plate overwrite to state already derived from an
        ANPR event.

        "Bind now, overwrite within 3s" means the first read of a burst may have
        already produced a confirmed session before the corrected read arrives.
        This updates the pending event and any session created from it directly,
        intentionally bypassing ``link_plate_to_session``'s no-overwrite guard
        (that guard protects ReID-fusion callers, not ANPR self-correction).

        The displayed plate label and any linked-slot label follow ``session.plate``
        through ``get_display_id_for_track`` / ``get_plate_for_track`` on the next
        frame, so no direct state-machine poke is needed. In the common case the
        3s burst completes while the car is still at the gate, before parking.

        Returns the parking ``slot_id``s freed by evicting any stale same-plate
        session as the corrected plate becomes authoritative; the caller clears
        those DB rows off-lock (this method does no DB I/O).
        """
        now = timestamp or self._clock()
        released: List[str] = []
        with self._lock:
            event = self._pending_events.get(event_id)
            if event is not None:
                event.plate = new_plate

            # Prefer the event's recorded session_id; fall back to a scan in case
            # the session was created via a path that did not back-link it.
            target_sid = event.session_id if event else None
            sessions = (
                [self._sessions[target_sid]]
                if target_sid and target_sid in self._sessions
                else [s for s in self._sessions.values() if s.event_id == event_id]
            )
            for session in sessions:
                old = session.plate
                if old != new_plate:
                    # The corrected plate is authoritative: evict any OTHER stale
                    # same-plate session before this one adopts the plate, so one
                    # plate never holds two active slots.
                    released.extend(
                        self._claim_plate_globally(
                            new_plate,
                            keep_session_id=session.session_id,
                            timestamp=now,
                        )
                    )
                session.plate = new_plate
                self._gallery_index_upsert(session)
                logger.info(
                    "[ANPR] Rebound session %s plate %s -> %s (burst last-wins)",
                    session.session_id,
                    old,
                    new_plate,
                )
        return released

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
                self._track_first_seen.pop(key, None)
                self._track_view_quality.pop(key, None)
                session_id = self._track_session_map.pop(key, None)
                # Also clean up the session's observing_tracks entry
                if session_id:
                    session = self._sessions.get(session_id)
                    if session and key[0] in session.observing_tracks:
                        del session.observing_tracks[key[0]]
                        # Keep ownership maps in sync: a camera that no longer
                        # observes the car cannot keep its score or ownership.
                        session.observing_scores.pop(key[0], None)
                        if session.owner_camera == key[0]:
                            session.owner_camera = None

            # Orphan sweep: match_global_session also stamps _track_first_seen
            # for query tracks that never bind to a session (so _mark_track_seen
            # never fires). Those keys have no _track_last_seen twin, so the pass
            # above never reaches them. Age them out by their own timestamp —
            # otherwise the dict grows unboundedly, and worse, a stale anchor
            # survives to silently disable the temporal entry gate if the tracker
            # later reuses that integer track id for a different car.
            orphan_first_seen = [
                key
                for key, seen_at in self._track_first_seen.items()
                if key not in self._track_last_seen
                and (now - seen_at).total_seconds() > self.TRACK_MAPPING_EXPIRY_SECONDS
            ]
            for key in orphan_first_seen:
                self._track_first_seen.pop(key, None)

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

            # Drop burst pointers whose event has expired/closed so a new read
            # on that camera starts a fresh burst instead of resurrecting one.
            # "confirmed" is kept: the 3s window guard still gates coalescing,
            # and the pointer is overwritten on the next non-coalesced entry.
            for cam_id, eid in list(self._last_anpr_entry.items()):
                ev = self._pending_events.get(eid)
                if ev is None or ev.status not in ("pending", "provisional", "confirmed"):
                    self._last_anpr_entry.pop(cam_id, None)

            # Prune re-entry-grace stamps far past any useful window (memory
            # hygiene; a plate this old can never satisfy _has_fresh_reentry).
            _reentry_ttl = (
                max(self.REENTRY_DB_GRACE_SECONDS, self.PENDING_ANPR_EXPIRY_SECONDS)
                + 300
            )
            for _plate, _entry_at in list(self._last_anpr_entry_at.items()):
                _exit_at = self._last_anpr_exit_at.get(_plate, _entry_at)
                if (
                    (now - _entry_at).total_seconds() > _reentry_ttl
                    and (now - _exit_at).total_seconds() > _reentry_ttl
                ):
                    self._last_anpr_entry_at.pop(_plate, None)
                    self._last_anpr_exit_at.pop(_plate, None)

            candidates_to_delete = []
            for candidate_id, candidate in self._park_entry_candidates.items():
                if candidate.status in ("open", "provisional"):
                    # D3: Use entered_at (immutable, set once) instead of last_seen_at (refreshed every frame).
                    # A stationary car refreshes last_seen_at perpetually and never expires.
                    # Using entered_at ensures candidates expire after CANDIDATE_EXPIRY_SECONDS
                    # regardless of whether the car is moving or still.
                    age = (now - candidate.entered_at).total_seconds() if candidate.entered_at else 0
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

            seeded = getattr(self, "_park_entry_gallery_seeded", None)
            for candidate_id in candidates_to_delete:
                self._park_entry_candidates.pop(candidate_id, None)
                # Drop the per-visit gallery-seed marker so the set can't grow
                # without bound (see seed_gallery_from_park_entry).
                if seeded is not None:
                    seeded.discard(candidate_id)

        # Phase 2 / T2.2 — drop stale per-track vote buffers. Held outside
        # the registry lock so the voter's own RLock is the only contention
        # point. ``match_voter`` is always present (constructed in
        # ``VehicleRegistry.__init__``); the no-op guard is defensive against
        # subclasses that bypass the standard constructor.
        voter = getattr(self, "_match_voter", None)
        if voter is not None:
            try:
                voter.cleanup_stale()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[VOTER] cleanup_stale raised: %r", exc
                )

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

    def expire_park_entry_candidate(self, candidate_id: str) -> None:
        """D3: Mark a Park_Entry candidate as expired when it leaves the zone.

        Called by the exit path when a track leaves. Prevents a car that queued,
        left, and re-entered from re-using an old candidate that may have been
        sitting in zone for longer than PENDING_ANPR_BIND_TTL_SECONDS.
        """
        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate and candidate.status in ("open", "provisional"):
                candidate.status = "expired"

    def plate_for_park_entry_candidate(self, candidate_id: str) -> Optional[str]:
        """Resolve the plate a Park_Entry candidate is bound to, if any.

        After the one-shot ``bind_next_pending_anpr_to_candidate`` fires the
        candidate goes ``open`` → ``provisional`` and later frames get None from
        that method — but the binding persists on ``candidate.bound_event_id``.
        This lets a later frame (with a better crop) still resolve the plate and
        re-attempt the idempotent CAM-23 top-view seed. Returns None when the
        candidate is unknown or not yet bound to a pending event."""
        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None or not candidate.bound_event_id:
                return None
            event = self._pending_events.get(candidate.bound_event_id)
            return event.plate if event is not None else None

    def latest_park_entry_candidate_for_plate(self, plate: str) -> Optional[str]:
        """Most-recent Park_Entry candidate bound to ``plate`` that still holds a
        usable crop. Used by the CAM-03 confirmation fallback to seed the CAM-23
        top view when the in-zone Park_Entry seed never fired this visit."""
        with self._lock:
            best = None
            for cand in self._park_entry_candidates.values():
                if not cand.bound_event_id:
                    continue
                event = self._pending_events.get(cand.bound_event_id)
                if event is None or event.plate != plate:
                    continue
                if cand.snapshot_image is None or getattr(
                    cand.snapshot_image, "size", 0
                ) == 0:
                    continue
                if best is None or cand.last_seen_at > best.last_seen_at:
                    best = cand
            return best.candidate_id if best is not None else None

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

        # Multishot is on when either the legacy env-var shim
        # (REID_USE_MULTISHOT) or the documented config flag
        # (matching.use_multishot, e.g. from config.yaml) requests it. The
        # env-var path is kept so tests that patch the module global still
        # steer this branch; production now honours the YAML flag.
        multishot_on = REID_USE_MULTISHOT or bool(
            getattr(self._matching_config, "use_multishot", False)
        )
        if multishot_on:
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
                    logger.debug(
                        "[PARK_ENTRY] candidate=%s snapshot rejected: new_quality=%.1f <= current=%.1f",
                        candidate_id, quality_score, candidate.quality_score,
                    )
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

        # Keep up to multishot_ref_top_k sharpest references on EVERY camera
        # (was hardcoded to only give the entry camera "CAM-03" a 3-shot
        # gallery, leaving every other camera single-shot). Matching scores
        # max-cosine over these references, so more good references = more
        # robust identity without any model change.
        top_k = max(1, int(getattr(self._matching_config, "multishot_ref_top_k", 3)))
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
