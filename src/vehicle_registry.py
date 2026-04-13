"""
vehicle_registry.py — Vehicle plate-to-slot tracking.

Two assignment strategies (use one or both):

  SIMPLE QUEUE (try_assign_plate):
    1. ANPR sends a plate → stored in a pending queue
    2. Any detected car without a plate gets the OLDEST pending plate
    3. Pending plates older than 30 seconds are auto-expired

  IMAGE MATCHING (try_match_by_image):
    1. ANPR sends a plate + car image → both stored
    2. For each detected car, crop is compared vs ANPR images
    3. Uses multi-feature matching (dominant color, regional color, SSIM, edges)
    4. Best match above threshold gets the plate assigned

Thread-safe for concurrent access from the API and engine.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# [CHANGE 6] Replace all print() calls with a module-level logger.
# Callers can configure handlers, levels, and formatters externally.
logger = logging.getLogger(__name__)


@dataclass
class PendingANPREvent:
    """Plate received from ANPR, waiting for the real car body at Park_Entry."""
    event_id: str
    plate: str
    direction: str
    timestamp: datetime
    camera_id: Optional[str] = None

    status: str = "pending"   # pending, provisional, confirmed, expired, dropped
    candidate_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ParkEntryCandidate:
    """A real car seen in the Park_Entry zone."""
    candidate_id: str
    camera_id: str
    track_id: int
    entered_at: datetime
    last_seen_at: datetime

    snapshot_path: Optional[str] = None
    snapshot_image: Optional[np.ndarray] = None
    feature_vector: Optional[np.ndarray] = None
    quality_score: float = 0.0

    status: str = "open"   # open, provisional, confirmed, expired, dropped
    bound_event_id: Optional[str] = None


@dataclass
class VehicleSession:
    """Final confirmed identity: this plate belongs to this exact car."""
    session_id: str
    plate: str
    event_id: str
    candidate_id: str

    confirmed_at: datetime
    current_camera_id: Optional[str] = None
    current_track_id: Optional[int] = None

    linked_slot: Optional[str] = None
    linked_slot_name: Optional[str] = None
    linked_camera: Optional[str] = None
    linked_floor: Optional[str] = None
    linked_zone_id: Optional[str] = None
    linked_zone_name: Optional[str] = None
    linked_at: Optional[datetime] = None
    snapshot_path: Optional[str] = None


class VehicleRegistry:
    """
    Central registry for ANPR -> Park_Entry -> B1_Entrance -> parking slot flow.
    """

    PENDING_ANPR_EXPIRY_SECONDS = 30
    CANDIDATE_EXPIRY_SECONDS = 30

    # [CHANGE 3] How often the background GC thread wakes up (seconds).
    _GC_INTERVAL_SECONDS = 5

    def __init__(self, image_dir: str = "vehicle_images"):
        self._lock = threading.RLock()

        # [CHANGE 1] Dedicated lock so only one thread ever instantiates the matcher.
        self._matcher_lock = threading.Lock()

        # FIFO ANPR queue
        self._pending_event_order: List[str] = []
        self._pending_events: Dict[str, PendingANPREvent] = {}

        # Real cars seen at Park_Entry
        self._park_entry_candidates: Dict[str, ParkEntryCandidate] = {}

        # Final confirmed identities
        self._sessions: Dict[str, VehicleSession] = {}

        # Track -> session map
        # Key: (camera_id, track_id), Value: session_id
        self._track_session_map: Dict[Tuple[str, int], str] = {}

        # Slot -> session map for currently parked cars
        self._parked: Dict[str, VehicleSession] = {}

        # Completed visits
        self._history: List[VehicleSession] = []

        self._matcher = None
        self._image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

        # [CHANGE 3] Start background GC daemon thread.
        # daemon=True so it never blocks clean process shutdown.
        self._gc_thread = threading.Thread(
            target=self._gc_loop,
            name="VehicleRegistry-GC",
            daemon=True,
        )
        self._gc_thread.start()
        logger.debug("Background GC thread started (interval=%ds)", self._GC_INTERVAL_SECONDS)

    # ------------------------------------------------------------------
    # [CHANGE 3] Background garbage-collection loop
    # ------------------------------------------------------------------

    def _gc_loop(self) -> None:
        """Runs in a daemon thread; periodically purges stale state."""
        while True:
            time.sleep(self._GC_INTERVAL_SECONDS)
            try:
                self._cleanup_stale_data(datetime.now())
            except Exception:
                # Never let the GC thread die silently.
                logger.exception("Unhandled error in VehicleRegistry GC loop")

    # ------------------------------------------------------------------
    # [CHANGE 1] Thread-safe lazy matcher property
    # ------------------------------------------------------------------

    @property
    def matcher(self):
        """
        Lazy-load the image matcher exactly once, even under concurrent access.

        The double-checked locking pattern avoids acquiring the lock on every
        call once the matcher is initialised.
        """
        if self._matcher is None:                        # fast path (no lock)
            with self._matcher_lock:
                if self._matcher is None:                # slow path (one winner)
                    from src.image_matcher import VehicleImageMatcher
                    self._matcher = VehicleImageMatcher()
                    logger.debug("VehicleImageMatcher instantiated")
        return self._matcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        [CHANGE 3] _cleanup_stale_data is no longer called here;
        the background GC thread handles it so this hot path stays fast.
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
            # 1. Expire pending/provisional ANPR events and clear broken links.
            active_orders = []
            for eid in self._pending_event_order:
                e = self._pending_events.get(eid)
                if not e:
                    continue
                if e.status in ("pending", "provisional"):
                    age = (now - e.timestamp).total_seconds()
                    if age > self.PENDING_ANPR_EXPIRY_SECONDS:
                        e.status = "expired"
                        if e.candidate_id:
                            candidate = self._park_entry_candidates.get(e.candidate_id)
                            if candidate is not None:
                                candidate.status = "expired"
                                candidate.bound_event_id = None
                                # [CHANGE 4] Release bulky numpy arrays immediately.
                                candidate.snapshot_image = None
                                candidate.feature_vector = None
                            e.candidate_id = None
                    else:
                        active_orders.append(eid)
            self._pending_event_order = active_orders

            # 2. Expire or retire Park_Entry candidates and clear their event links.
            c_to_delete = []
            for cid, c in self._park_entry_candidates.items():
                if c.status in ("open", "provisional"):
                    age = (now - c.last_seen_at).total_seconds()
                    if age > self.CANDIDATE_EXPIRY_SECONDS:
                        c.status = "expired"
                        if c.bound_event_id:
                            event = self._pending_events.get(c.bound_event_id)
                            if event is not None and event.status in ("pending", "provisional"):
                                event.status = "expired"
                                event.candidate_id = None
                        c.bound_event_id = None
                        # [CHANGE 4] Release bulky numpy arrays immediately.
                        c.snapshot_image = None
                        c.feature_vector = None
                        c_to_delete.append(cid)
                elif c.status in ("confirmed", "dropped", "expired"):
                    c_to_delete.append(cid)

            for cid in c_to_delete:
                self._park_entry_candidates.pop(cid, None)

    def open_park_entry_candidate(
        self,
        camera_id: str,
        track_id: int,
        timestamp: Optional[datetime] = None,
    ) -> ParkEntryCandidate:
        """
        Create a candidate for a car entering the Park_Entry zone.
        """
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
            if not candidate:
                return

            candidate.last_seen_at = now

            if quality_score <= candidate.quality_score:
                return

            candidate.quality_score = quality_score
            candidate.snapshot_image = image.copy()
            candidate.feature_vector = feature_vector

            filename = f"{candidate.candidate_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self._image_dir, filename)

            cv2.imwrite(filepath, image)
            candidate.snapshot_path = filepath

            logger.debug(
                "[PARK_ENTRY] Updated snapshot for %s (score=%.3f)",
                candidate.candidate_id,
                quality_score,
            )
            return filepath

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
                    else:
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
        similarity_threshold: float = 0.35,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Confirm that a B1_Entrance car is the same car that was provisionally
        captured at Park_Entry.

        [CHANGE 2] After image matching (done outside the lock to avoid
        blocking), the lock is re-acquired and we re-verify that both the
        winning candidate AND its bound event are still in their expected
        states before committing the session. This closes the TOCTOU window
        that existed in the original implementation.

        On success:
          - create VehicleSession
          - map this track to that session
          - return the plate

        On failure:
          - keep provisional bindings intact and wait for a later match
        """
        now = timestamp or datetime.now()

        # Snapshot the provisional pairs while holding the lock.
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

        # Image comparison runs outside the lock — it's CPU-bound and must not
        # block ANPR ingestion or the GC thread.
        best_candidate = None
        best_score = 0.0
        for _, candidate in provisional_pairs:
            # candidate.snapshot_image could theoretically be None if the GC
            # thread cleared it between our snapshot and now; guard defensively.
            snap = candidate.snapshot_image
            if snap is None:
                continue
            score = self.matcher.compare(image, snap)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None or best_score < similarity_threshold:
            return None

        # [CHANGE 2] Re-acquire lock and re-validate before committing.
        with self._lock:
            # Re-fetch from the live registry — our local reference may be stale.
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

            session = VehicleSession(
                session_id=f"sess_{uuid.uuid4().hex[:12]}",
                plate=event.plate,
                event_id=event.event_id,
                candidate_id=live_candidate.candidate_id,
                confirmed_at=now,
                current_camera_id=camera_id,
                current_track_id=track_id,
                snapshot_path=live_candidate.snapshot_path
            )

            self._sessions[session.session_id] = session
            self._track_session_map[(camera_id, track_id)] = session.session_id

            event.status = "confirmed"
            event.session_id = session.session_id
            event.candidate_id = live_candidate.candidate_id

            live_candidate.status = "confirmed"
            live_candidate.last_seen_at = now

            # [CHANGE 4] Free numpy memory now that matching is complete.
            live_candidate.snapshot_image = None
            live_candidate.feature_vector = None

            logger.info(
                "[B1] Confirmed plate=%s for track (%s, %d) — session %s (score=%.3f)",
                session.plate,
                camera_id,
                track_id,
                session.session_id,
                best_score,
            )

            return session.plate

    def drop_provisional_binding(self, candidate_id: str) -> None:
        """
        Remove a failed provisional match.
        """
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

            # [CHANGE 4] Release numpy arrays on drop.
            candidate.snapshot_image = None
            candidate.feature_vector = None

    def attach_session_to_track(
        self,
        camera_id: str,
        track_id: int,
        session_id: str,
    ) -> None:
        """
        Attach a confirmed session to a new camera/track.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            session.current_camera_id = camera_id
            session.current_track_id = track_id
            self._track_session_map[(camera_id, track_id)] = session_id

    def get_plate_for_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> Optional[str]:
        """
        Resolve plate through confirmed session mapping.
        """
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            if session_id is None:
                return None

            session = self._sessions.get(session_id)
            return session.plate if session else None

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

            session.linked_slot = slot_id
            session.linked_slot_name = slot_name
            session.linked_camera = camera_id
            session.linked_floor = floor
            session.linked_zone_id = zone_id
            session.linked_zone_name = zone_name
            session.linked_at = timestamp
            self._parked[slot_id] = session

            return session.plate

    def _handle_exit(self, plate: str, timestamp: datetime) -> None:
        """
        Close a parked session when ANPR sends an exit event.
        """
        with self._lock:
            slot_to_remove = None
            for slot_id, session in self._parked.items():
                if session.plate == plate:
                    slot_to_remove = slot_id
                    break

            if slot_to_remove is not None:
                session = self._parked.pop(slot_to_remove)
                self._sessions.pop(session.session_id, None)
                self._history.append(session)

                if session.current_camera_id and session.current_track_id is not None:
                    self._track_session_map.pop(
                        (session.current_camera_id, session.current_track_id),
                        None,
                    )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_slot_plate(self, slot_id: str) -> Optional[str]:
        with self._lock:
            session = self._parked.get(slot_id)
            return session.plate if session else None

    def _get_snapshot_url(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return f"/images/{os.path.basename(path)}"

    def get_plate_location(self, plate: str) -> Optional[Dict]:
        with self._lock:
            for slot_id, session in self._parked.items():
                if session.plate == plate:
                    return {
                        "plate_number": session.plate,
                        "slot_id": slot_id,
                        "slot_name": session.linked_slot_name,
                        "zone_id": session.linked_zone_id,
                        "zone_name": session.linked_zone_name,
                        "camera_id": session.linked_camera,
                        "floor": session.linked_floor,
                        "track_id": session.current_track_id,
                        "parked_at": session.linked_at.isoformat() if session.linked_at else None,
                        "confirmed_at": session.confirmed_at.isoformat(),
                        "snapshot_url": self._get_snapshot_url(session.snapshot_path)
                    }
            return None

    def get_all_parked(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "plate_number": s.plate,
                    "slot_id": slot_id,
                    "slot_name": s.linked_slot_name,
                    "zone_id": s.linked_zone_id,
                    "zone_name": s.linked_zone_name,
                    "camera_id": s.linked_camera,
                    "floor": s.linked_floor,
                    "track_id": s.current_track_id,
                    "parked_at": s.linked_at.isoformat() if s.linked_at else None,
                    "confirmed_at": s.confirmed_at.isoformat(),
                    "snapshot_url": self._get_snapshot_url(s.snapshot_path)
                }
                for slot_id, s in self._parked.items()
            ]

    def get_pending_entries(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "event_id": e.event_id,
                    "plate_number": e.plate,
                    "timestamp": e.timestamp.isoformat(),
                    "status": e.status,
                    "candidate_id": e.candidate_id,
                    "session_id": e.session_id,
                }
                for e in self._pending_events.values()
                if e.direction == "entry" and e.status in ("pending", "provisional")
            ]

    def get_stats(self) -> Dict:
        """
        [CHANGE 5] active_sessions now reads len(self._sessions) directly.

        _handle_exit already removes exited sessions from self._sessions via
        .pop(), so the count is always accurate without building an O(n) set
        against self._history.
        """
        with self._lock:
            return {
                "parked_count": len(self._parked),
                "pending_entries": sum(
                    1 for e in self._pending_events.values()
                    if e.direction == "entry" and e.status in ("pending", "provisional")
                ),
                "total_visits": len(self._history),
                "active_sessions": len(self._sessions),   # O(1), always correct
                "tracked_ids": len(self._track_session_map),
            }
