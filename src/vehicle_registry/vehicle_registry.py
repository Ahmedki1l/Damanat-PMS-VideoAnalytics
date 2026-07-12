"""
Public VehicleRegistry entrypoint.

The heavy implementation is split into focused helper modules so this file can
stay easy to navigate while preserving the existing import path.
"""

import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.config import MatchingConfig, ReIDPreprocessingConfig
from src.matching import GalleryIndex, MatchDecision, MatchVoter
from src.vehicle_registry.vehicle_registry_core import VehicleRegistryCoreMixin
from src.vehicle_registry.vehicle_registry_identity import VehicleRegistryIdentityMixin
from src.vehicle_registry.vehicle_registry_models import (
    ParkEntryCandidate,
    PendingANPREvent,
    VehicleSession,
)
from src.vehicle_registry.vehicle_registry_queries import VehicleRegistryQueryMixin
from src.zoning.trace import enabled as _area_trace_enabled, trace as _area_trace

logger = logging.getLogger(__name__)


def _session_label(session) -> str:
    """Readable identity for traces: plate if known, else short session id."""
    if getattr(session, "plate", None):
        return session.plate
    sid = session.session_id or ""
    return f"#{sid[-6:]}" if sid else "#?"


class VehicleRegistry(
    VehicleRegistryCoreMixin,
    VehicleRegistryIdentityMixin,
    VehicleRegistryQueryMixin,
):
    """
    Central registry for ANPR -> Park_Entry -> B1_Entrance -> parking slot flow.
    """

    PENDING_ANPR_EXPIRY_SECONDS = 30
    # How long an UNCONSUMED pending ANPR entry stays eligible to be FIFO-bound to
    # a live-track Park_Entry candidate (which carries no plate of its own). Kept
    # SHORTER than PENDING_ANPR_EXPIRY_SECONDS on purpose: a legitimate car reaches
    # the B1 gate within a few seconds of its gate read, whereas a stale plate left
    # over from a lingering/mis-read PREVIOUS car (the night gate identity-swap)
    # must stop being bindable quickly so the NEXT car cannot inherit it. The event
    # itself still lives the full expiry for re-entry grace / specific-event binds.
    PENDING_ANPR_BIND_TTL_SECONDS = 10
    # D3 linger guard. The ANPR read happens at the gate, UPSTREAM of the CAM-23
    # Park_Entry polygon, so the car that triggered a read reaches the zone AFTER
    # it. A candidate that was ALREADY sitting in the zone when the plate was read
    # therefore cannot be the car that was read — it is a lingerer (it entered
    # without an ANPR read of its own, or its read was missed), and FIFO-binding it
    # hands it the ARRIVING car's plate.
    #
    # This is a check on the candidate's entry time RELATIVE TO THE READ, not on
    # its absolute age: a car may legitimately dwell in the zone for minutes at the
    # barrier, and an absolute age cap is exactly what force-expired dwelling cars
    # and had to be reverted (b3ef313). Both clocks are the registry's own
    # (`_clock()`), so there is no integrator skew to absorb — but `event.timestamp`
    # is when the event was RECEIVED, not when the plate was physically read, so
    # this grace must comfortably exceed the ANPR integrator's POST latency.
    PARK_ENTRY_LINGER_GRACE_SECONDS = 5
    # Re-entry DB-grace: a car can exit (parking_sessions -> closed) and RE-ENTER
    # before PMS-AI inserts the new open row. During that window the freshest DB
    # row is still the exit's closed row, so is_plate_inside would wrongly report
    # "already exited". If VA itself saw an ANPR exit followed by a NEWER ANPR
    # entry within this many seconds, treat the plate as inside despite the
    # lagging DB. Bounds the worst-case double-fault ghost to this many seconds.
    REENTRY_DB_GRACE_SECONDS = 60
    CANDIDATE_EXPIRY_SECONDS = 30
    # Effectively infinite; track mappings are now primarily cleared via the ANPR Exit event.
    TRACK_MAPPING_EXPIRY_SECONDS = 86400 * 30
    SESSION_HANDOFF_GUARD_SECONDS = 10
    _GC_INTERVAL_SECONDS = 5

    def __init__(
        self,
        image_dir: str = "vehicle_images",
        reid_preprocessing_config: ReIDPreprocessingConfig = None,
        public_base_url: str = "",
        snapshot_url_prefix: str = "/pms-video-analytics/snapshots",
        gateway_path_prefix: str = "",
        matching_config: Optional[MatchingConfig] = None,
        db_checker: Optional[Callable[[Optional[str]], bool]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        match_decision: Optional[MatchDecision] = None,
        area_registry=None,
        camera_floors: Optional[Dict[str, str]] = None,
        db_manager: Optional[Any] = None,
    ):
        # Optional DB handle (exposes SessionLocal) used for best-effort writes
        # such as clearing a stale slot's plate binding after an ANPR eviction
        # (_clear_slot_db_binding). None ⇒ VA-local in-memory only (fail-open,
        # mirrors is_plate_inside's optional db_checker DI).
        self.db_manager = db_manager
        self._lock = threading.RLock()
        self._matcher_lock = threading.Lock()

        # camera_id → floor label, used by the identity gate to skip ReID /
        # plate matching on ground-floor cameras (see is_reid_disabled_floor /
        # _is_reid_disabled_camera). Empty map ⇒ every camera is treated as
        # identity-enabled (legacy behaviour).
        self._camera_floors: Dict[str, str] = dict(camera_floors or {})

        # Cameras that host parking slots. Ownership prefers these over slotless
        # transit/aisle cameras (see _resolve_owner_camera) so a car's identity
        # is attributed to the camera that can actually bind its plate to a slot.
        # Populated by the engine after pipelines are built (set_cameras_with_slots);
        # empty ⇒ no slot preference (legacy behaviour).
        self._cameras_with_slots: Set[str] = set()

        # (camera_id, track_id) → latest view quality in [0, 1]: how fully the
        # camera sees this car (1.0 = bbox fully inside the frame, lower when
        # clipped by an edge). Fed per frame by the engine (record_track_view)
        # and used as the second ownership tie-breaker after slot preference —
        # a camera with the whole car outranks one that only sees part of it.
        # Empty ⇒ no view preference (legacy behaviour).
        self._track_view_quality: Dict[Tuple[str, int], float] = {}

        self._pending_event_order: List[str] = []
        self._pending_events: Dict[str, PendingANPREvent] = {}
        # ANPR burst coalescing: maps an ANPR camera_id → the event_id of its
        # currently-open burst, so rapid re-reads (last-wins within the window)
        # overwrite the plate instead of creating competing events.
        self._last_anpr_entry: Dict[str, str] = {}
        # Re-entry grace (skew-free, VA-internal). NOT the burst map above:
        # these are plate -> datetime of the latest ANPR entry / exit VA itself
        # observed, consulted by _has_fresh_reentry to override a stale 'closed'
        # DB row PMS-AI has not yet re-opened after a genuine re-entry. Bounded
        # by distinct plate count; pruned in _cleanup_stale_data.
        self._last_anpr_entry_at: Dict[str, datetime] = {}
        self._last_anpr_exit_at: Dict[str, datetime] = {}
        self._park_entry_candidates: Dict[str, ParkEntryCandidate] = {}
        self._sessions: Dict[str, VehicleSession] = {}
        self._track_session_map: Dict[Tuple[str, int], str] = {}
        self._track_last_seen: Dict[Tuple[str, int], datetime] = {}
        # First time each camera-local track was observed. Stamped once (never
        # overwritten) and used by match_global_session as the "entry anchor":
        # a car already parked BEFORE this track first appeared cannot be this
        # track, so parked-before candidates are excluded from its match pool.
        self._track_first_seen: Dict[Tuple[str, int], datetime] = {}
        self._parked: Dict[str, VehicleSession] = {}
        # Plate-lock: slots whose plate binding has been frozen. A locked slot is
        # refused by the relocate / stale-release / theft paths in
        # try_link_to_slot until it is explicitly unlocked (slot goes VACANT).
        self._locked_slots: set[str] = set()
        self._history: List[VehicleSession] = []

        # Zoning: area_id → {session_id} for the bounded (per-area) candidate
        # pool. Maintained by set_session_area / _drop_session_from_area_index.
        # ``area_registry`` (optional) resolves camera_id → area_id at runtime;
        # None = un-zoned (legacy all-sessions behaviour everywhere).
        self._area_registry = area_registry
        self._area_sessions: Dict[str, Set[str]] = defaultdict(set)
        # Cross-area handoff candidate builder — only when zoned. Lazy import
        # keeps the zoning package out of the un-zoned import path.
        self._handoff_matcher = None
        # Per-car area lifecycle (IN_AREA/IN_TRANSIT/…). Lives here, next to the
        # sessions it mutates; the engine feeds it observations + boundary
        # crossings via settle_track_area / apply_boundary_crossing. None when
        # un-zoned, so those driver methods are no-ops.
        self._area_state_machine = None
        if area_registry is not None:
            from src.zoning.handoff_matcher import CrossAreaHandoffMatcher
            from src.zoning.area_state_machine import AreaStateMachine

            self._handoff_matcher = CrossAreaHandoffMatcher(
                area_registry, clock=clock or datetime.now
            )
            self._area_state_machine = AreaStateMachine(
                self, area_registry, clock=clock or datetime.now
            )

        self._matcher = None
        self._reid_matcher = None
        self._image_dir = image_dir
        self._public_base_url = public_base_url
        self._snapshot_url_prefix = snapshot_url_prefix.strip("/")
        self._gateway_path_prefix = gateway_path_prefix.strip("/")
        self._reid_preprocessing_config = (
            reid_preprocessing_config or ReIDPreprocessingConfig()
        )

        # Matching configuration + decision chokepoint. When not injected,
        # fall back to defaults that preserve historical behaviour. The
        # NoopColorClassifier inside MatchDecision needs the legacy image
        # matcher for ``_compare_dominant_colors``; pass a zero-arg factory
        # so we don't force its construction at __init__ time (the matcher
        # property is intentionally lazy).
        self._matching_config: MatchingConfig = matching_config or MatchingConfig()
        if match_decision is not None:
            self._match_decision = match_decision
        else:
            self._match_decision = self._build_match_decision(
                self._matching_config,
                lambda: self.matcher,
            )

        # Phase 2 / T2.2 — temporal voting. Feature-flagged off by default;
        # ``try_link_to_slot`` routes its commit through this voter when
        # ``matching_config.voting_enabled`` is True.
        self._match_voter: MatchVoter = MatchVoter(self._matching_config)

        # Phase 3 / T3.2 — FAISS-CPU gallery index for O(log n) global
        # session lookup. Always instantiated so callers can probe its
        # state; ``match_global_session`` only routes through it when
        # ``matching_config.use_faiss_index`` is True. The index is rebuilt
        # from ``self._sessions`` at boot, so no on-disk persistence is
        # required — see src.matching.gallery_index for the design notes.
        self._gallery_index: GalleryIndex = GalleryIndex(
            dimension=self._matching_config.faiss_index_dimension,
            nlist=self._matching_config.faiss_index_nlist,
            metric="cosine",
        )

        # Per-car persistent gallery (folder per plate). Lazily built on first
        # use (needs the ReID backend name for the model tag); None when
        # ``gallery_persist_enabled`` is False. ``_gallery_last_add`` throttles
        # accumulation to one write per (plate, camera) per configured interval.
        self._gallery_store = None
        self._gallery_last_add: Dict[Tuple[str, str], float] = {}

        # DI seams (T0.5). ``db_checker`` lets tests replace the SQL probe in
        # is_plate_inside; ``clock`` lets tests advance time deterministically.
        # Both default to None — callsites still use the real
        # implementations exactly as before.
        self._db_checker: Optional[Callable[[Optional[str]], bool]] = db_checker
        self._clock: Callable[[], datetime] = clock or datetime.now

        os.makedirs(image_dir, exist_ok=True)

        self._gc_thread = threading.Thread(
            target=self._gc_loop,
            name="VehicleRegistry-GC",
            daemon=True,
        )
        self._gc_thread.start()
        logger.debug(
            "Background GC thread started (interval=%ds)",
            self._GC_INTERVAL_SECONDS,
        )

    @property
    def matching_config(self) -> MatchingConfig:
        """Active matching configuration (thresholds + feature flags)."""
        return self._matching_config

    @property
    def gallery_store(self):
        """Lazy per-car gallery store. None when persistence is disabled.

        The model tag (``backend:model-dir``) is captured at first use so a
        later ReID model swap invalidates the cached vectors (reload re-embeds
        from the retained crops)."""
        if not getattr(self._matching_config, "gallery_persist_enabled", False):
            return None
        if self._gallery_store is None:
            from src.vehicle_registry.gallery_store import VehicleGalleryStore

            model_dir = (self._matching_config.reid_openvino_model_dir or "").rstrip("/\\")
            tag = f"{getattr(self.reid_matcher, 'backend', '?')}:{os.path.basename(model_dir) or 'default'}"
            self._gallery_store = VehicleGalleryStore(
                self._image_dir, tag, self._matching_config.gallery_max_refs_per_car
            )
            logger.info("[gallery] persistent per-car gallery enabled (tag=%s)", tag)
        return self._gallery_store

    @property
    def match_decision(self) -> MatchDecision:
        """Centralised match decision chokepoint."""
        return self._match_decision

    @property
    def match_voter(self) -> MatchVoter:
        """Temporal vote aggregator used by ``try_link_to_slot``."""
        return self._match_voter

    @property
    def gallery_index(self) -> GalleryIndex:
        """Phase 3 / T3.2 FAISS-CPU gallery index.

        Returns the live :class:`GalleryIndex` instance regardless of
        whether ``matching_config.use_faiss_index`` is True — callers can
        inspect its state for diagnostics. ``match_global_session`` only
        consults the index when the feature flag is True.
        """
        return self._gallery_index

    # --- Zoning: per-area session index -------------------------------- #
    def set_session_area(self, session: VehicleSession, area_id: str) -> None:
        """Assign a session to an area and keep the ``_area_sessions`` index in
        sync. No-op when the area is unchanged. Empty ``area_id`` leaves the
        session un-zoned (removed from any area bucket)."""
        old = session.current_area
        if old == (area_id or ""):
            return
        with self._lock:
            if old:
                self._area_sessions.get(old, set()).discard(session.session_id)
            session.current_area = area_id or ""
            if area_id:
                self._area_sessions[area_id].add(session.session_id)
                if session.area_entered_at is None:
                    session.area_entered_at = self._clock()

    def _drop_session_from_area_index(self, session: VehicleSession) -> None:
        """Remove a session from its area bucket (called on exit/archival)."""
        if session.current_area:
            self._area_sessions.get(session.current_area, set()).discard(
                session.session_id
            )

    def sessions_in_area(self, area_id: str) -> List[VehicleSession]:
        """Live confirmed/parked sessions currently in an area (O(bucket))."""
        with self._lock:
            return [
                self._sessions[sid]
                for sid in self._area_sessions.get(area_id, set())
                if sid in self._sessions
            ]

    def area_snapshot(self) -> Dict[str, object]:
        """Read-only map of where cars currently are (for the SNAPSHOT trace):
        ``{"areas": {area_id: [labels]}, "transit": [labels]}``."""
        with self._lock:
            per_area: Dict[str, List[str]] = {}
            for area_id, sids in self._area_sessions.items():
                labels = [
                    _session_label(self._sessions[sid])
                    for sid in sids
                    if sid in self._sessions
                ]
                if labels:
                    per_area[area_id] = sorted(labels)
            transit = sorted(
                _session_label(s)
                for s in self._sessions.values()
                if s.area_state != "IN_AREA" and s.status in ("confirmed", "parked")
            )
        return {"areas": per_area, "transit": transit}

    def log_area_snapshot(self) -> None:
        """Emit a periodic SNAPSHOT trace line. No-op unless zoned AND
        ZONING_TRACE is on (so the snapshot is not even built in production)."""
        if self._area_state_machine is None or not _area_trace_enabled():
            return
        snap = self.area_snapshot()
        areas = (
            " | ".join(
                f"{a}:[{','.join(v)}]" for a, v in sorted(snap["areas"].items())
            )
            or "(none)"
        )
        transit = ",".join(snap["transit"]) or "-"
        _area_trace("SNAPSHOT", areas=areas, transit=transit)

    # --- Ownership priority inputs (fed by the engine) ------------------ #
    def set_cameras_with_slots(self, camera_ids) -> None:
        """Declare which cameras host parking slots. Ownership prefers these
        over slotless transit cameras. Called once after pipelines are built."""
        with self._lock:
            self._cameras_with_slots = set(camera_ids or ())

    def record_track_view(
        self, camera_id: str, track_id: int, view_quality: float
    ) -> None:
        """Record how fully ``camera_id`` currently sees track ``track_id``
        (1.0 = whole car in frame, lower when clipped). Used as the second
        ownership tie-breaker after slot preference. Cheap per-frame call."""
        if track_id is None or track_id < 0:
            return
        with self._lock:
            self._track_view_quality[(camera_id, track_id)] = float(view_quality)

    # --- Zoning: per-frame area-state drivers (no-ops when un-zoned) ---- #
    def settle_track_area(self, camera_id: str, track_id: int) -> None:
        """Settle this track's session into the observing camera's area, but
        only when that camera currently *owns* the identity.

        ``current_area`` is derived from the owner camera's area (the engine
        calls this every frame for live tracks). Gating on ownership means a
        single owner drives the area — neighbouring cameras that merely glimpse
        the car never thrash ``current_area``. No-op when un-zoned or the
        observing camera is un-zoned.
        """
        if self._area_state_machine is None:
            return
        area_id = self._area_registry.area_for_camera(camera_id)
        if not area_id:
            return
        with self._lock:  # RLock — set_session_area below re-acquires safely
            session_id = self._track_session_map.get((camera_id, track_id))
            session = self._sessions.get(session_id) if session_id else None
            if session is None:
                return
            if self._resolve_owner_camera(session, self._clock()) != camera_id:
                return
            old_area, old_state = session.current_area, session.area_state
            self._area_state_machine.on_area_observed(session, area_id)
            if session.current_area != old_area or session.area_state != old_state:
                _area_trace(
                    "SETTLE",
                    car=_session_label(session),
                    area=session.current_area,
                    was=old_area or "-",
                    prev=old_state,
                    cam=camera_id,
                    track=track_id,
                )

    def apply_boundary_crossing(
        self, camera_id: str, track_id: int, area_from: str, area_to: str
    ) -> None:
        """Apply a boundary crossing detected on a camera to the track's session
        (IN_AREA → IN_TRANSIT, clearing ``current_area``). Direction is resolved
        by the state machine from the car's current area, so one bidirectional
        band serves both ways. No-op when un-zoned or the track has no session.
        """
        if self._area_state_machine is None:
            return
        with self._lock:
            session_id = self._track_session_map.get((camera_id, track_id))
            session = self._sessions.get(session_id) if session_id else None
            if session is None:
                return
            old_area = session.current_area
            self._area_state_machine.on_boundary_cross(session, area_from, area_to)
            _area_trace(
                "CROSS",
                car=_session_label(session),
                cam=camera_id,
                band=f"{area_from}->{area_to}",
                departed=session.departed_from_area or "-",
                state=session.area_state,
                was=old_area or "-",
            )

    # ------------------------------------------------------------------ #
    # Gallery-index sync (Phase 3 / T3.2)
    # ------------------------------------------------------------------ #

    def _gallery_index_upsert(self, session) -> None:
        """Push the session's current feature vector into the gallery index.

        Idempotent and silently no-op when the session is anonymous, has no
        feature vector, or the index is disabled by config. The check is in
        one place so the identity-mixin callsites stay readable.
        """
        if not self._matching_config.use_faiss_index:
            return
        if session is None:
            return
        if session.status not in ("confirmed", "parked"):
            return
        vec = getattr(session, "feature_vector", None)
        if vec is None:
            return
        try:
            self._gallery_index.add(session.session_id, vec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[gallery_index] add(%s) raised %r — skipping.",
                session.session_id,
                exc,
            )

    def _gallery_index_remove(self, session_id: str) -> None:
        """Drop ``session_id`` from the gallery index.

        Skipped when the index is feature-flagged off; that keeps the
        index empty so any later toggle starts from a clean rebuild.
        """
        if not self._matching_config.use_faiss_index:
            return
        try:
            self._gallery_index.remove(session_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "[gallery_index] remove(%s) raised %r — skipping.",
                session_id,
                exc,
            )

    # ------------------------------------------------------------------ #
    # Plugin instantiation (Phase 2 T2.1)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_match_decision(
        config: MatchingConfig,
        image_matcher_factory,
    ) -> MatchDecision:
        """Construct a :class:`MatchDecision` with real plugins when their
        model files exist on disk; otherwise fall back to Noop plugins.

        Path resolution is explicit (no magic auto-discovery): each plugin's
        path is taken from ``MatchingConfig`` and the file is probed with
        ``os.path.exists`` before the heavy import happens. Missing models
        produce a log line and a Noop instance — never a hard import error
        at boot. This keeps test bootstrap fast and prevents a fresh checkout
        from crashing before any model is trained.
        """
        from src.matching import NoopColorClassifier, NoopPlateOCR, NoopTypeClassifier

        # --- Color classifier -------------------------------------------- #
        color_plugin = None
        color_path = (config.color_classifier_model or "").strip()
        if color_path and os.path.exists(color_path):
            try:
                from src.classifiers.color_classifier import OpenVINOColorClassifier

                color_plugin = OpenVINOColorClassifier(
                    model_path=color_path,
                    config=config,
                )
                logger.info(
                    "[MatchDecision] color classifier loaded from %s",
                    color_path,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[MatchDecision] failed to load color classifier at %s: %r — "
                    "falling back to NoopColorClassifier",
                    color_path,
                    exc,
                )
                color_plugin = None
        if color_plugin is None:
            color_plugin = NoopColorClassifier(
                image_matcher=image_matcher_factory,
                hsv_h_tol=config.hsv_h_tol,
                hsv_s_tol=config.hsv_s_tol,
                hsv_v_tol=config.hsv_v_tol,
            )

        # --- Type classifier --------------------------------------------- #
        type_plugin = None
        type_path = (config.type_classifier_model or "").strip()
        if type_path and os.path.exists(type_path):
            try:
                from src.classifiers.type_classifier import OpenVINOTypeClassifier

                type_plugin = OpenVINOTypeClassifier(model_path=type_path)
                logger.info(
                    "[MatchDecision] type classifier loaded from %s",
                    type_path,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[MatchDecision] failed to load type classifier at %s: %r — "
                    "falling back to NoopTypeClassifier",
                    type_path,
                    exc,
                )
                type_plugin = None
        if type_plugin is None:
            type_plugin = NoopTypeClassifier()

        # --- Plate OCR ---------------------------------------------------- #
        # The OCR plugin is heavy (PaddleOCR pulls ~50 MB of weights on first
        # use); only instantiate when ``plate_ocr_model`` is configured non-
        # empty so a fresh checkout boots without paddle installed.
        # Config semantics:
        #   ""                 -> NoopPlateOCR (disabled)
        #   <path-to-dir>      -> PaddlePlateOCR with that PaddleOCR home dir
        #   <anything-else>    -> PaddlePlateOCR with default cache
        #                         (~/.paddlex/official_models, auto-download)
        ocr_plugin = None
        ocr_path = (config.plate_ocr_model or "").strip()
        if ocr_path:
            try:
                from pathlib import Path as _P
                from src.ocr.plate_ocr import PaddlePlateOCR

                model_dir = ocr_path if _P(ocr_path).is_dir() else None
                ocr_plugin = PaddlePlateOCR(model_dir=model_dir)
                logger.info(
                    "[MatchDecision] plate OCR plugin loaded (model_dir=%s)",
                    model_dir or "<paddleocr default cache>",
                )
            except Exception as exc:
                logger.warning(
                    "[MatchDecision] failed to load plate OCR plugin (%r); "
                    "falling back to NoopPlateOCR",
                    exc,
                )
                ocr_plugin = None
        if ocr_plugin is None:
            ocr_plugin = NoopPlateOCR()

        return MatchDecision(
            config,
            color_classifier=color_plugin,
            type_classifier=type_plugin,
            plate_ocr=ocr_plugin,
        )
