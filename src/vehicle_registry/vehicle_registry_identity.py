import logging
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.matching.match_decision import Decision
from src.matching.plate_ocr_match import confirm_plate
from src.vehicle_registry.vehicle_registry_models import ParkEntryCandidate, VehicleSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedCandidate:
    """One car ReID thinks the query crop might be, with the evidence behind it.

    ``score`` is what the ranking sorts on. ``same_view_score`` is the part that comes
    from a reference taught by the SAME camera that is now looking at the car — a car
    that has parked here before. It is broken out because a warm match is much stronger
    evidence than a cold one (measured: warm rank-1 100%, cold 87.8%), and the decision
    layer and the ranker both need to tell them apart.
    """

    plate: str
    session_id: str
    score: float
    same_view_score: float
    cross_view_score: float
    warm: bool
    rank: int
    session: Any = None  # live VehicleSession; never serialised


@dataclass(frozen=True)
class SlotOcrPlan:
    """The main-thread half of a slot-identify OCR read, ready to hand off.

    ``plan_slot_ocr`` builds this using the ReID matcher and gallery (both
    main-thread-only). The heavy PaddleOCR read that follows takes only
    ``candidates``/``allow_retry`` and the crop, so it can run on a worker
    thread; ``confirm_slot_ocr`` then folds the read back in — on the main
    thread again, because the decision log touches the colour/type classifiers.
    """

    slot_id: str
    camera_id: Optional[str]
    candidates: List[str]
    kept: List["RankedCandidate"]
    rejected: List["RejectedCandidate"]
    allow_retry: bool
    decision_ctx: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TrackOcrPlan:
    """The same three-phase hand-off as :class:`SlotOcrPlan`, for the APPROACH read.

    The approach read (a still-driving car, ``try_ocr_identify_track``) differs from
    the slot read in one way that matters here: it ranks AFTER the read, not before
    (``reid_shortlist`` is only consulted once there is text to confirm). So the
    plan phase has no ReID work to do and carries only what the read itself needs —
    everything expensive stays in phase 2 (worker) and phase 3 (main thread).
    """

    camera_id: str
    track_id: int
    allow_retry: bool


@dataclass(frozen=True)
class RejectedCandidate:
    """A candidate a deterministic rule threw out, and why.

    Kept rather than silently dropped: the reason code is what makes a decision
    auditable, and the rejects are what let us MEASURE whether a rule earns its keep
    instead of assuming it does.
    """

    plate: str
    session_id: str
    raw_rank: int
    score: float
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

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
# Floors where identity work is ALWAYS off. Ground runs YOLO occupancy only —
# identity there comes from ANPR plates, not appearance.
_REID_DISABLED_FLOORS = {"ground", "ground floor"}

# Extra floors / cameras to disable at runtime, from VA_IDENTITY_DISABLED
# (comma-separated, case-insensitive). Exists so identity can be switched off
# for a subset WITHOUT a code change, to answer "does the identity work affect
# occupancy latency?" as a controlled experiment. Cached — read per frame.
_identity_disabled_cache: Optional[frozenset] = None


def identity_disabled_tokens() -> frozenset:
    """Floors/cameras disabled via ``VA_IDENTITY_DISABLED``, lowercased."""
    global _identity_disabled_cache
    if _identity_disabled_cache is None:
        raw = os.environ.get("VA_IDENTITY_DISABLED", "") or ""
        _identity_disabled_cache = frozenset(
            t.strip().lower() for t in raw.split(",") if t.strip()
        )
        if _identity_disabled_cache:
            logger.info(
                "[identity] DISABLED for: %s (VA_IDENTITY_DISABLED) — these run "
                "YOLO occupancy only: no ReID embedding, no approach-OCR, no "
                "slot-OCR, no plate binding",
                ", ".join(sorted(_identity_disabled_cache)),
            )
    return _identity_disabled_cache


def is_reid_disabled_floor(floor: str) -> bool:
    """True when identity work is off for this floor.

    Gates BOTH halves: the per-frame ReID embedding + approach-OCR in
    ``_process_special_zones``, and ``plate_matching_enabled`` (slot-OCR,
    plate binding, _resolve_locked_plate) in ``_update_slot_occupancy``.
    """
    f = (floor or "").strip().lower()
    return f in _REID_DISABLED_FLOORS or f in identity_disabled_tokens()


def is_identity_disabled(camera_id: str, floor: str) -> bool:
    """As :func:`is_reid_disabled_floor`, but also honours a CAMERA id.

    Camera granularity matters because ReID and the approach-OCR are computed
    per camera FRAME, not per slot — there is no way to disable them "for one
    slot". Disabling CAM-04 therefore covers every slot that camera watches.
    """
    if is_reid_disabled_floor(floor):
        return True
    return (camera_id or "").strip().lower() in identity_disabled_tokens()

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

# A gallery reference must be ONE WHOLE CAR. A box far wider than a car is not a
# car: it is two of them merged by the detector, or a car plus the one beside it.
# Mirrors the engine's _VQ_MAX_ASPECT.
#
# The engine applies that cap while computing view_quality, so accumulate_reference
# inherits it. But the two AUTHORITATIVE paths — _seed_gallery (the gate shot) and
# save_parked_reference (the OCR-verified parked pose) — call store.save_ref DIRECTLY
# with a hardcoded quality (998/999/1.0) and so were never geometry-checked at all.
# On 2026-07-12 that let a 1527x519 CAM-03 box (aspect 2.94) into EEB-80's gallery
# twice, and a 722x258 one (aspect 2.80) into DJS-7842's. "Authoritative identity"
# says the crop belongs to THIS PLATE; it says nothing about the crop being a car.
_REF_MAX_ASPECT = 2.2
_REF_MIN_SIDE_PX = 24

# ...and a crop is not a car if it is the WHOLE PICTURE. Aspect alone cannot say
# so: a 2688x1552 gate frame is aspect 1.73, well inside the 2.2 cap, so it
# sailed through this guard as "one whole car". A camera frame is a scene — road,
# sky, buildings, often several cars — and embeds as such.
#
# This is a BACKSTOP, not the fix. The fix is not handing this function a frame in
# the first place (see latest_park_entry_candidate_for_plate's camera filter); the
# bar here only has to catch a raw sensor frame without eating real detections.
# Calibrated against all 307 refs in the live gallery, 2026-07-15: the streams are
# 2688x1552, and the largest GENUINE car crop measured 2012x1309 (a car close to
# the lens). 2400 sits in that gap — it refuses every full frame by width with
# room to spare, and passes all 245 real crops. A box wider than this is ~90% of
# the sensor: either the frame itself, or a car so close it is a poor reference
# anyway. Re-derive if the stream resolution changes.
_REF_MAX_SIDE_PX = 2400


def is_plausible_car_crop(crop_bgr) -> bool:
    """Is this crop shaped like a single car? Geometry only — no appearance."""
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return False
    try:
        h, w = crop_bgr.shape[:2]
    except Exception:
        return False
    if h < _REF_MIN_SIDE_PX or w < _REF_MIN_SIDE_PX:
        return False
    if h >= _REF_MAX_SIDE_PX or w >= _REF_MAX_SIDE_PX:
        return False
    return (w / float(h)) <= _REF_MAX_ASPECT


class VehicleRegistryIdentityMixin:
    def is_plate_inside(self, plate: Optional[str]) -> bool:
        """Returns True iff `plate` has an open `parking_sessions` row in the
        shared DB. Single source of truth — avoids in-memory drift between VA
        and PMS-AI when the gate's exit-ANPR event was missed.

        Re-entry override: when the DB probe says NOT inside (latest row
        closed), this still returns True if VA itself just observed a genuine
        re-entry — an ANPR exit followed by a NEWER ANPR entry within
        ``REENTRY_DB_GRACE_SECONDS`` (see :meth:`has_recent_reentry`). This
        covers the window between a real re-entry and PMS-AI inserting the new
        open row, without resurrecting a plain missed-exit ghost (which never
        has a post-exit entry).

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
                # DB says not-inside — honour it UNLESS VA just saw a genuine,
                # recent re-entry the DB has not re-opened yet.
                return bool(db_checker(plate)) or self.has_recent_reentry(plate)
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
        # status == 'open' means inside; on 'closed' fall through to the
        # re-entry-grace override before declaring the car exited.
        return row[0] == "open" or self.has_recent_reentry(plate)

    def _has_fresh_reentry(self, plate: Optional[str], now: datetime) -> bool:
        """True iff VA observed an ANPR exit for ``plate`` and then a NEWER ANPR
        entry within ``REENTRY_DB_GRACE_SECONDS`` — a genuine re-entry the shared
        DB has not caught up on (PMS-AI has not yet inserted the new open row).

        Stays False for a plain missed-exit (an exit with no later entry, or an
        entry with no prior exit), preserving the anti-ghost guard: only a real,
        authoritative re-entry ANPR read can flip it True.

        Lock-free by design — each read is a single atomic ``dict.get`` and the
        maps are only ever written with whole-datetime assignments, so a torn
        read is impossible and a stale read is benign (grace boundary off by one
        event).
        """
        if not plate:
            return False
        entry_at = self._last_anpr_entry_at.get(plate)
        exit_at = self._last_anpr_exit_at.get(plate)
        if entry_at is None or exit_at is None:
            return False
        if entry_at <= exit_at:
            return False
        return (now - entry_at).total_seconds() <= self.REENTRY_DB_GRACE_SECONDS

    def has_recent_reentry(self, plate: Optional[str]) -> bool:
        """Public, DB-free re-entry-grace check. Consulted by is_plate_inside (to
        override a stale 'closed' DB read) and by the engine exit-janitor (to
        skip purging a plate that just genuinely re-entered)."""
        return self._has_fresh_reentry(plate, self._clock())

    def last_anpr_entry_at(self, plate: Optional[str]) -> Optional[datetime]:
        """When VA's OWN ANPR last watched this plate drive in through the gate.

        This is VA's authoritative answer to "is the car inside", independent of the
        shared parking_sessions table (which PMS-AI owns and can lag on, or — as on
        2026-07-11 — simply stop writing to). The exit-janitor compares it against the
        entry_time of the 'closed' row it is about to act on: if VA saw the car enter
        AFTER that row began, the row describes an older visit and must not be used to
        delete the car currently inside. Unlike the time-boxed re-entry grace, this
        holds for a lag of any length.
        """
        if not plate:
            return None
        return self._last_anpr_entry_at.get(plate)

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

    def _prepare_session_gallery(
        self,
        images: List[np.ndarray],
        timestamp: Optional[datetime] = None,
        precomputed_features: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Heavy half of gallery persistence: dedupe, embed, write snapshot
        files. MUST run OUTSIDE ``self._lock`` — it does model inference and
        disk I/O, and every camera thread's per-frame identity lookups block
        on that lock. Returns ``(ordered_images, feature_vectors,
        stored_paths)`` or ``None`` when no usable image remains.

        ``precomputed_features`` lets a caller that already embedded the same
        (deduped) image list — e.g. the B1 confirmation match — skip a second
        identical inference pass. It is only trusted when its length matches
        the deduped image list; otherwise features are re-extracted.
        """
        ordered_images = self._dedupe_valid_images(images)
        if not ordered_images:
            return None
        if (
            precomputed_features is not None
            and len(precomputed_features) == len(ordered_images)
        ):
            feature_vectors = list(precomputed_features)
        else:
            feature_vectors = self.reid_matcher.extract_features_batch(ordered_images)
        stored_paths = self._store_session_reference_snapshots(
            ordered_images,
            timestamp=timestamp,
        )
        return ordered_images, feature_vectors, stored_paths

    def _apply_session_gallery(
        self,
        session: VehicleSession,
        prepared,
        primary_snapshot_index: int = 0,
        source_camera: str = "",
    ) -> bool:
        """Cheap half of gallery persistence: MERGE the prepared gallery onto the
        session. Safe under ``self._lock`` — no inference, no disk I/O.

        Merges rather than replaces so a car's live gallery ACCUMULATES the
        genuinely different viewpoints it produces across every camera/zone it
        passes — a front view on an approaching camera, a rear view on the
        exit-view of a zone it drives out of — instead of being overwritten with
        only the latest camera's timeline (which left a cross-camera /
        cross-floor query, e.g. a B1->B2 handover, scoring near-zero against its
        own gallery). A new view that is a near-duplicate of one already held is
        skipped (nothing new to learn); when the merged set exceeds the per-car
        cap it is pruned to the most mutually-dissimilar subset — max viewpoint
        coverage — the same policy the on-disk store uses. On the first
        application the existing gallery is empty, so a merge is identical to the
        old replace.
        """
        ordered_images, feature_vectors, stored_paths = prepared
        primary_snapshot_index = max(
            0,
            min(primary_snapshot_index, len(ordered_images) - 1),
        )
        cfg = self._matching_config
        dedup = float(getattr(cfg, "gallery_dedup_cosine", 0.97))
        cap = max(1, int(getattr(cfg, "gallery_max_refs_per_car", 20)))

        # New (path, vector, source_camera) triples from this crossing (aligned;
        # skip empty vecs). The source camera drives match-time trust weighting.
        new_pairs = [
            (stored_paths[i] if i < len(stored_paths) else "", feature_vectors[i], source_camera)
            for i in range(len(feature_vectors))
            if feature_vectors[i] is not None
        ]
        if not new_pairs:
            return False

        # Existing gallery as aligned (path, vector, source_camera) triples.
        existing_vecs = list(session.reference_feature_vectors or [])
        existing_paths = list(session.reference_snapshot_paths or [])
        existing_cams = list(session.reference_source_cameras or [])
        merged = [
            (
                existing_paths[i] if i < len(existing_paths) else "",
                existing_vecs[i],
                existing_cams[i] if i < len(existing_cams) else "",
            )
            for i in range(len(existing_vecs))
            if existing_vecs[i] is not None
        ]

        # Identity gate (open-set novelty rejection): a new view may only join
        # this identity's gallery when it agrees, in appearance, with the
        # identity's ESTABLISHED refs. This stops a foreign car — wrongly bound
        # to a plate at the gate — from grafting its crops into that plate's
        # gallery (the reported contamination: "the other car's images ended up
        # in my gallery"), and stops the diversity cap from preferentially
        # keeping that foreign, dissimilar vector. Bootstrap (empty gallery)
        # admits unconditionally; disabled when the floor is 0.
        identity_floor = float(getattr(cfg, "gallery_min_identity_similarity", 0.0))
        established_vecs = [v for _, v, _ in merged]

        # Append each new view that clears the identity floor and is not a
        # near-duplicate of one already held.
        for path, vec, cam in new_pairs:
            if identity_floor > 0.0 and established_vecs:
                id_sim = max(
                    self.reid_matcher.compute_similarity(vec, ev)
                    for ev in established_vecs
                )
                if id_sim < identity_floor:
                    logger.warning(
                        "[gallery] Rejected foreign crop for session %s "
                        "(plate=%s): identity similarity %.3f < floor %.2f — "
                        "contamination guard.",
                        getattr(session, "session_id", "?"),
                        getattr(session, "plate", "") or "",
                        id_sim,
                        identity_floor,
                    )
                    continue
            if all(
                self.reid_matcher.compute_similarity(vec, kept_vec) <= dedup
                for _, kept_vec, _ in merged
            ):
                merged.append((path, vec, cam))

        # Diversity cap: keep the `cap` most mutually-dissimilar views.
        if len(merged) > cap:
            from src.vehicle_registry.gallery_store import select_diverse_indices

            keep = set(select_diverse_indices([v for _, v, _ in merged], cap))
            merged = [pair for i, pair in enumerate(merged) if i in keep]

        session.reference_snapshot_paths = [p for p, _, _ in merged]
        session.reference_feature_vectors = [v for _, v, _ in merged]
        session.reference_source_cameras = [c for _, _, c in merged]

        # Primary = this crossing's chosen canonical view (freshest best shot).
        if stored_paths:
            session.snapshot_path = stored_paths[
                min(primary_snapshot_index, len(stored_paths) - 1)
            ]
        if feature_vectors:
            primary_feature = feature_vectors[primary_snapshot_index]
            if primary_feature is not None:
                session.feature_vector = primary_feature
        return True

    def _persist_session_gallery(
        self,
        session: VehicleSession,
        images: List[np.ndarray],
        now: datetime,
        primary_snapshot_index: int = 0,
        seed_gallery: bool = False,
        gate_only: bool = False,
        source_camera: str = "",
    ) -> bool:
        """Prepare + apply in one call, for callers NOT holding ``self._lock``
        (e.g. ``confirm_anpr_session_directly``). Lock-holding callers must
        instead call ``_prepare_session_gallery`` outside the lock and
        ``_apply_session_gallery`` inside it, so inference and disk I/O never
        run under the registry lock."""
        prepared = self._prepare_session_gallery(images, timestamp=now)
        if prepared is None:
            return False
        self._apply_session_gallery(
            session, prepared, primary_snapshot_index, source_camera=source_camera
        )

        # Seed the durable per-plate gallery folder (vehicle_images/gallery/
        # <plate>/) so a folder exists on disk the moment a car is confirmed —
        # instead of only when live-tracking accumulation happens to fire (which
        # may never happen for a short-lived / low-FPS car). Reuses the
        # already-extracted feature. ONLY when the caller passes seed_gallery.
        # The wide gate-only ANPR shot from the direct session must pass
        # gate_only=True: the folder (and entry photo) are created, but the ref
        # is flagged so warm-start matching ignores it — persisting it as a
        # matchable ref would reintroduce the false-match-against-a-parked-car
        # risk (build_session_from_gallery loads matchable refs primary-first).
        if seed_gallery:
            ordered_images, feature_vectors, _ = prepared
            self._seed_plate_gallery_reference(
                session,
                ordered_images,
                feature_vectors,
                max(0, min(primary_snapshot_index, len(ordered_images) - 1)),
                gate_only=gate_only,
            )
        return True

    def _seed_plate_gallery_reference(
        self,
        session,
        ordered_images: List[np.ndarray],
        feature_vectors: List[Optional[np.ndarray]],
        primary_snapshot_index: int,
        gate_only: bool = False,
    ) -> None:
        """Persist the primary confirmation crop + embedding to the plate's
        on-disk gallery. No-op when persistence is off, the plate is unknown, or
        the primary image/feature is missing. Best-effort — disk errors are
        swallowed so confirmation never fails on a storage hiccup.

        ``gate_only`` seeds the wide gate ANPR shot: the folder is created the
        moment the car enters, but the ref is excluded from warm-start matching
        (see VehicleGalleryStore.save_ref)."""
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
        # Contamination guard on the durable folder (mirrors the in-memory
        # identity gate): don't persist a crop whose appearance disagrees with the
        # session's established gallery — otherwise a foreign car wrongly bound to
        # this plate poisons the on-disk refs and warm-start reloads the mix. A
        # legitimate primary is itself already in the (gated) session gallery, so
        # it self-matches; a foreign primary was rejected there and scores low
        # here. Skipped for gate_only seeds (bootstrap, non-matchable) and when
        # the gallery is still empty.
        identity_floor = float(
            getattr(self._matching_config, "gallery_min_identity_similarity", 0.0)
        )
        established = [
            v
            for v in (getattr(session, "reference_feature_vectors", None) or [])
            if v is not None
        ]
        if identity_floor > 0.0 and established and not gate_only:
            id_sim = max(
                self.reid_matcher.compute_similarity(feature, ev) for ev in established
            )
            if id_sim < identity_floor:
                logger.warning(
                    "[gallery] Skipped foreign on-disk seed for plate %s: identity "
                    "similarity %.3f < floor %.2f (contamination guard).",
                    plate,
                    id_sim,
                    identity_floor,
                )
                return
        if not is_plausible_car_crop(image):
            logger.warning(
                "[gallery] Skipped mis-shaped seed for plate %s: %s is not one car.",
                plate,
                "x".join(str(d) for d in image.shape[1::-1]),
            )
            return
        try:
            store.save_ref(
                plate,
                image,
                feature,
                # Authoritative references outrank live views at prune time;
                # the gate shot sits just below them so it never displaces one.
                quality=998.0 if gate_only else 999.0,
                camera_id=getattr(session, "last_seen_camera", "") or "ANPR",
                gate_only=gate_only,
            )
        except Exception as exc:  # pragma: no cover - disk best-effort
            logger.debug("[gallery] seed for %s failed: %r", plate, exc)

    def seed_gallery_from_park_entry(self, candidate_id: str, plate: str) -> bool:
        """Persist the CAM-23 Park_Entry (top-view) crop as a MATCHABLE
        ground-truth reference the moment a car is bound to a plate there.

        CAM-23 is one of the three ground-truth cameras (top view); its crop is a
        canonical viewpoint we want in every car's gallery. It is added even when
        the folder ALREADY exists (the ANPR gate `entry` already seeded the front
        view) — the top view is a distinct viewpoint, not a redundant re-seed.
        The crop is written to the durable per-plate folder (survives restart /
        warm-start) AND enriched onto the live session's in-memory gallery so it
        is immediately match-usable once the session becomes active at CAM-03.

        Fires once per visit (the caller only calls this on the one-shot
        open->provisional bind). NOTE: the plate binding here is still provisional
        (FIFO) — a mis-bind injects a wrong-plate CAM-23 crop; the shortened
        pending-bind TTL (PENDING_ANPR_BIND_TTL_SECONDS) is the mitigation.
        No-op when persistence is off, the plate is unknown, or the candidate has
        no usable crop. Best-effort — disk errors are swallowed."""
        store = self.gallery_store
        if store is None or not plate:
            logger.info(
                "[gallery] seed_gallery_from_park_entry no-op: gallery_store=%s plate=%r",
                store is not None, plate,
            )
            return False
        # Idempotent per visit: each Park_Entry visit is one candidate, and the
        # caller only fires on the one-shot open->provisional bind — but guard
        # against any double-call so a single visit never adds two identical
        # CAM-23 crops. (A returning car is a NEW candidate → a fresh top view.)
        seeded = getattr(self, "_park_entry_gallery_seeded", None)
        if seeded is None:
            seeded = self._park_entry_gallery_seeded = set()
        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None:
                logger.info("[gallery] seed_gallery_from_park_entry: candidate=%s not found", candidate_id)
                return False
            if candidate_id in seeded:
                logger.debug("[gallery] seed_gallery_from_park_entry: candidate=%s already seeded this visit", candidate_id)
                return False
            crop = candidate.snapshot_image
            feature = candidate.feature_vector
            quality = float(candidate.quality_score)
            camera_id = candidate.camera_id or "CAM-23"
        # Only mark the candidate seeded on SUCCESS: a first frame whose crop was
        # not yet usable must not permanently block the cross-frame retry (Fix 4)
        # — the seed re-fires on a later, better in-zone crop.
        if crop is None or getattr(crop, "size", 0) == 0:
            logger.info("[gallery] seed_gallery_from_park_entry candidate=%s: crop is empty", candidate_id)
            return False
        # Geometry gate — this path writes a MATCHABLE ref straight to disk and was
        # the only save_ref caller with no shape check at all, which is how the raw
        # gate frame got in (see latest_park_entry_candidate_for_plate). Cheap, and
        # it belongs here regardless of which camera the candidate came from.
        if not is_plausible_car_crop(crop):
            logger.warning(
                "[gallery] seed_gallery_from_park_entry candidate=%s plate=%s: "
                "crop %s is not one car — refusing to seed.",
                candidate_id, plate,
                "x".join(str(d) for d in crop.shape[1::-1]),
            )
            return False
        if feature is None:
            feature = self.reid_matcher.extract_feature(crop)
            if feature is None:
                logger.info("[gallery] seed_gallery_from_park_entry candidate=%s: ReID feature extraction failed", candidate_id)
                return False

        # D2: Apply identity-similarity floor to prevent mis-binds from poisoning the gallery.
        # When the gallery already has established references (i.e., not the first crop),
        # check that the new crop matches the plate's existing identity. If not, reject it.
        # This prevents a mis-bind at the gate from creating a permanent foreign ref.
        identity_floor = float(
            getattr(self._matching_config, "gallery_min_identity_similarity", 0.0)
        )
        # load_vectors returns (vectors, model_tag, source_cameras) — compare
        # against the VECTORS, not the tuple. Iterating the tuple fed the whole
        # vectors list (and the tag/cameras) into compute_similarity, raising and
        # breaking every seed once the floor is active (config.yaml sets 0.35).
        established, _, _ = store.load_vectors(plate)  # excludes gate_only refs
        if identity_floor > 0.0 and established:
            id_sim = max(
                self.reid_matcher.compute_similarity(feature, ev) for ev in established
            )
            if id_sim < identity_floor:
                logger.warning(
                    "[gallery] Rejected foreign Park_Entry seed for %s (id_sim=%.3f < %.2f)",
                    plate, id_sim, identity_floor
                )
                return False

        try:
            store.save_ref(
                plate, crop, feature, quality=quality, camera_id=camera_id,
                gate_only=False,
            )
        except Exception as exc:  # pragma: no cover - disk best-effort
            logger.warning("[gallery] seed_gallery_from_park_entry candidate=%s: store.save_ref failed: %r", candidate_id, exc)
            return False
        seeded.add(candidate_id)

        # Enrich the live session (created at the gate) so the top view is
        # immediately match-usable, tagged with its ground-truth source camera.
        # Deduped + capped, mirroring accumulate_reference. Best-effort: a car
        # without a live session yet still has the durable disk ref above.
        with self._lock:
            session = next(
                (
                    s
                    for s in self._sessions.values()
                    if s.plate == plate
                    and s.status in ("confirmed", "parked", "provisional")
                ),
                None,
            )
            if session is not None:
                # Anchor the canonical colour from the CAM-23 top view if no
                # earlier ground-truth crop did — a no-op once already set.
                self._maybe_set_ground_truth_hsv(session, crop)
                cfg = self._matching_config
                known = [session.feature_vector] + list(
                    session.reference_feature_vectors
                )
                is_dup = any(
                    ref is not None
                    and self.reid_matcher.compute_similarity(feature, ref)
                    > cfg.gallery_dedup_cosine
                    for ref in known
                )
                if not is_dup:
                    session.reference_feature_vectors.append(feature)
                    self._sync_reference_cameras(session)
                    session.reference_source_cameras[-1] = camera_id
                    cap = max(1, int(cfg.gallery_max_refs_per_car))
                    if len(session.reference_feature_vectors) > cap:
                        session.reference_feature_vectors = (
                            session.reference_feature_vectors[-cap:]
                        )
                        session.reference_source_cameras = (
                            session.reference_source_cameras[-cap:]
                        )
                    self._gallery_index_upsert(session)

        logger.info(
            "[gallery] Park_Entry (%s) added top-view ground-truth reference for "
            "plate=%s", camera_id, plate,
        )
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
            # Gate image only — keep this session out of global ReID matching
            # until CAM-03 attaches a real B1 reference (confirm_b1_entrance_by_plate).
            # Prevents the wide gate shot from false-matching a car already parked
            # inside the garage the instant the entry event arrives.
            gate_reference_only=True,
            # Hold this ANPR (car-cropped) embedding for promotion to a matchable
            # reference once CAM-03 confirms the plate — see the promotion block in
            # confirm_b1_entrance_by_plate. Excluded from matching until then.
            pending_anpr_vector=feature_vector,
        )
        # Anchor the canonical body colour from the ANPR front crop — the
        # earliest, most reliable frontal view. Drives the colour vetoes in
        # accumulate_reference / match_global_session.
        self._maybe_set_ground_truth_hsv(session, image)

        # Persist the ANPR image as the primary gallery reference (UI gallery /
        # reference_snapshot_paths) AND seed the durable per-plate folder
        # (vehicle_images/gallery/<plate>/) the moment the car passes the gate —
        # so EVERY entering car has a folder even if the CAM-03 B-entry reference
        # never arrives. The ANPR front crop (cropped to the car at the API
        # boundary) is a GROUND-TRUTH front viewpoint, so it is written
        # gate_only=False: VehicleGalleryStore.load_vectors/load_crops include it
        # and a returning car can warm-start-match on its own ANPR front view.
        # NOTE: live entry-time parked-car latching is still prevented by the
        # SEPARATE session-level ``gate_reference_only`` guard (this session is
        # excluded from match_global_session until CAM-03 attaches a floor
        # reference) — flipping the disk flag does not touch that protection.
        # CAM-03 B-entry (confirm_b1_entrance_by_plate) still adds the primary
        # matchable reference; seed_gallery_from_park_entry remains a fallback
        # seeder (it no-ops once a folder exists).
        self._persist_session_gallery(
            session,
            [image],
            now,
            primary_snapshot_index=0,
            seed_gallery=True,
            gate_only=False,
            source_camera=getattr(session, "last_seen_camera", "") or "ANPR",
        )

        evicted_slots: List[str] = []
        with self._lock:
            # Fast path: a same-arrival burst (an existing same-plate session
            # < 120s old) means the car DID pass through B1_Entrence in a race —
            # reuse it instead of creating a duplicate.
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

            # No recent burst match ⇒ a genuine (re-)entry. This ANPR gate read
            # is authoritative: evict any STALE (>=120s) same-plate session still
            # open from a missed exit so one plate never holds two active slots.
            evicted_slots = self._claim_plate_globally(
                plate, keep_session_id=session_id, timestamp=now
            )

            self._sessions[session_id] = session
            # Phase 3 / T3.2 — track the new confirmed session in the FAISS
            # gallery (no-op when use_faiss_index is False).
            self._gallery_index_upsert(session)

        # DB clear off-lock — never hold self._lock across DB I/O.
        for slot_id in evicted_slots:
            self._clear_slot_db_binding(slot_id)

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

        D3 linger guard: a candidate may only take a plate that was read at-or-after
        it entered the zone (within PARK_ENTRY_LINGER_GRACE_SECONDS). A car already
        sitting in the zone when the read arrived did not trigger that read, so
        binding it would steal the plate of the car that did. The comparison is
        `entered_at` vs the EVENT's timestamp — not the candidate's absolute age,
        which is what wrongly force-expired legitimately-dwelling cars in b3ef313.

        Older pending events are still offered to this candidate, so a lingerer whose
        OWN read merely arrived late can still bind its own plate.
        """
        now = timestamp or self._clock()

        with self._lock:
            candidate = self._park_entry_candidates.get(candidate_id)
            if candidate is None or candidate.status != "open":
                return None

            event = None
            for event_id in self._pending_event_order:
                pending = self._pending_events.get(event_id)
                if not (
                    pending
                    and pending.direction == "entry"
                    and pending.status == "pending"
                ):
                    continue
                age = (now - pending.timestamp).total_seconds()
                if age > self.PENDING_ANPR_EXPIRY_SECONDS:
                    pending.status = "expired"
                    continue
                # D3: was this candidate ALREADY in the zone when the plate was read?
                # If so it cannot be the car that was read — skip this event (but keep
                # looking, in case an OLDER pending event is genuinely this car's own).
                lingered = (
                    pending.timestamp - candidate.entered_at
                ).total_seconds()
                if lingered > self.PARK_ENTRY_LINGER_GRACE_SECONDS:
                    logger.info(
                        "[PARK_ENTRY] candidate %s was in-zone %.1fs before event %s "
                        "(plate=%s) was read (grace=%ss) — lingerer, not eligible for "
                        "this plate",
                        candidate_id, lingered, event_id, pending.plate,
                        self.PARK_ENTRY_LINGER_GRACE_SECONDS,
                    )
                    continue
                # Only FIFO-bind a FRESH pending plate to a plateless live-track
                # candidate. An older-but-not-yet-expired plate is stale residue
                # from a previous (lingering / mis-read) car — leaving it bindable
                # for the full 30s let the NEXT car's candidate grab it and adopt
                # its plate (the night gate identity-swap). It still lives out the
                # expiry for specific-event binds and re-entry grace.
                if age <= self.PENDING_ANPR_BIND_TTL_SECONDS:
                    event = pending
                    break
                else:
                    # Log when an event is skipped for being older than bind TTL
                    logger.info(
                        "[PARK_ENTRY] pending event %s (plate=%s) age=%.1fs > bind_ttl=%ss "
                        "(but <= expiry=%ss) — skipped for bind",
                        event_id, pending.plate, age, self.PENDING_ANPR_BIND_TTL_SECONDS,
                        self.PENDING_ANPR_EXPIRY_SECONDS,
                    )

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
            # Appearance floor on the single-candidate fallback: compute the lone
            # survivor's actual ReID score against the crossing track and pass it
            # to MatchDecision so the ``single_candidate_min_reid`` floor can
            # reject an appearance-blind bind. Previously this passed 0.0, so the
            # lone candidate was confirmed at ANY agreement — the mechanism by
            # which the NEXT car adopted a PREVIOUS car's stale pending plate.
            survivor_score = 0.0
            if fallback_pairs and current_reid_feat is not None:
                survivor = fallback_pairs[0][1]
                survivor_vectors = [
                    f
                    for f in list(getattr(survivor, "feature_vectors", []) or [])
                    if f is not None
                ]
                if not survivor_vectors and getattr(survivor, "feature_vector", None) is not None:
                    survivor_vectors = [survivor.feature_vector]
                if survivor_vectors:
                    survivor_score = max(
                        self.reid_matcher.compute_similarity(current_reid_feat, v)
                        for v in survivor_vectors
                    )
            fallback_verdict = decision.decide_b1(
                survivor_score,
                is_anpr_candidate=False,
                candidate_count=len(fallback_pairs),
            )
            if (
                fallback_verdict.verdict == "confirm"
                and fallback_verdict.reason == "single_candidate_fallback"
            ):
                best_candidate = fallback_pairs[0][1]
                logger.warning(
                    "[B1] Falling back to the only provisional candidate %s for "
                    "track (%s, %d); ReID=%.3f cleared the single-candidate floor "
                    "(strict visual threshold not met)",
                    best_candidate.candidate_id,
                    camera_id,
                    track_id,
                    survivor_score,
                )
            else:
                if fallback_pairs:
                    logger.info(
                        "[B1] Refused appearance-blind single-candidate bind for "
                        "track (%s, %d): lone candidate %s ReID=%.3f below floor "
                        "single_candidate_min_reid — not adopting its plate (guards "
                        "against the stale-pending-plate identity swap).",
                        camera_id,
                        track_id,
                        fallback_pairs[0][1].candidate_id,
                        survivor_score,
                    )
                return None

        session_feature_vector = (
            current_reid_feat
            if current_reid_feat is not None
            else best_candidate.feature_vector
        )
        reference_feature_vectors = [
            feature for feature in session_reference_features if feature is not None
        ]

        # Heavy half of gallery persistence (embedding + snapshot writes)
        # happens BEFORE the registry lock; only the cheap assignment runs
        # under it. Reuses the features already extracted for the match above
        # instead of re-embedding the same images a second time.
        prepared_gallery = self._prepare_session_gallery(
            session_reference_images,
            timestamp=now,
            precomputed_features=session_reference_features,
        )

        evicted_slots: List[str] = []
        result_plate: Optional[str] = None
        confirmed_session = None
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

            # Authoritative gate read: evict any OTHER stale same-plate session
            # still open from a missed exit, keeping the one we're about to
            # upgrade/create. DB clear happens off-lock after this block.
            evicted_slots = self._claim_plate_globally(
                event.plate,
                keep_session_id=session.session_id if session else None,
                timestamp=now,
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
                if prepared_gallery is not None:
                    self._apply_session_gallery(
                        session,
                        prepared_gallery,
                        primary_snapshot_index=primary_snapshot_index,
                        source_camera=camera_id,
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
                if prepared_gallery is not None:
                    self._apply_session_gallery(
                        session,
                        prepared_gallery,
                        primary_snapshot_index=primary_snapshot_index,
                        source_camera=camera_id,
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
            result_plate = session.plate
            confirmed_session = session

            # Cross-session identity reconciliation (config-gated, off by
            # default): if this car is a near-identical duplicate of another
            # freshly-confirmed identity — one physical car the ANPR misread as
            # two plates — collapse the duplicate so the car keeps ONE identity.
            reconciled_slots = self._reconcile_duplicate_identity(session, now)
            if reconciled_slots:
                evicted_slots.extend(reconciled_slots)

        # DB clear off-lock — never hold self._lock across DB I/O.
        for slot_id in evicted_slots:
            self._clear_slot_db_binding(slot_id)
        # Off-lock: seed the durable per-plate gallery folder with the live
        # CAM-03 confirmation crop. This is the in-engine counterpart of the
        # seed in confirm_b1_entrance_by_plate — without it, a car whose
        # B-entry API push never arrives gets no folder until live-tracking
        # accumulation happens to pass its quality gates (which it may never).
        if confirmed_session is not None and prepared_gallery is not None:
            ordered, feature_vectors, _ = prepared_gallery
            self._seed_plate_gallery_reference(
                confirmed_session,
                ordered,
                feature_vectors,
                max(0, min(primary_snapshot_index, len(ordered) - 1)),
            )
        return result_plate

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

        def _find_session_id() -> Optional[str]:
            best = None
            for s in self._sessions.values():
                if s.plate != plate:
                    continue
                if s.status not in ("confirmed", "parked", "provisional"):
                    continue
                if best is None or s.last_seen_at > best.last_seen_at:
                    best = s
            return best.session_id if best is not None else None

        with self._lock:
            session_id = _find_session_id()
        if session_id is None:
            return None

        # Heavy half (embedding + snapshot write) OFF the registry lock; only
        # the cheap assignment below runs under it.
        prepared = self._prepare_session_gallery([image], timestamp=now)
        if prepared is None:
            return None

        evicted_slots: List[str] = []
        result_sid: Optional[str] = None
        seeded_session = None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.plate != plate:
                # The session was closed/rebound while we were embedding —
                # re-resolve once; a genuine miss aborts.
                session_id = _find_session_id()
                session = self._sessions.get(session_id) if session_id else None
                if session is None:
                    return None

            session.last_seen_at = now
            session.last_seen_camera = "CAM-03"
            # Anchor the canonical colour from CAM-03 if the ANPR front never did
            # (e.g. the gate entry was missed) — a no-op once already set.
            self._maybe_set_ground_truth_hsv(session, image)
            self._apply_session_gallery(
                session, prepared, primary_snapshot_index=0, source_camera="CAM-03"
            )
            # The primary reference is now the real CAM-03 shot, not the wide gate
            # image — the session is a trustworthy match target again.
            session.gate_reference_only = False
            # Promote the ANPR frontal crop (stashed at gate entry, held out of
            # matching until now) to a matchable reference. The car is anchored to
            # this session by the CAM-03 confirmation, so the extra frontal view
            # enriches the appearance profile without the entry-time swap risk.
            # Deduped + FIFO-capped, mirroring accumulate_reference so one view
            # can't dominate or duplicate the gallery.
            pend = session.pending_anpr_vector
            if pend is not None:
                cfg = self._matching_config
                known = [session.feature_vector] + list(session.reference_feature_vectors)
                if not any(
                    ref is not None
                    and self.reid_matcher.compute_similarity(pend, ref)
                    > cfg.gallery_dedup_cosine
                    for ref in known
                ):
                    session.reference_feature_vectors.append(pend)
                    # The stashed vector is the ANPR frontal crop — tag it so
                    # match-time weighting treats it as a ground-truth reference.
                    session.reference_source_cameras.append("ANPR")
                    cap = max(1, int(cfg.gallery_max_refs_per_car))
                    if len(session.reference_feature_vectors) > cap:
                        session.reference_feature_vectors = (
                            session.reference_feature_vectors[-cap:]
                        )
                        session.reference_source_cameras = (
                            session.reference_source_cameras[-cap:]
                        )
                session.pending_anpr_vector = None
            self._gallery_index_upsert(session)
            # Convergence: this authoritative CAM-03 B-entry read is a good moment
            # to collapse any OTHER stale same-plate session left open by a missed
            # exit onto this one. Cannot itself create a duplicate.
            evicted_slots = self._claim_plate_globally(
                plate, keep_session_id=session.session_id, timestamp=now
            )
            logger.info(
                "[B1] CAM-03 B-entry snapshot attached for plate=%s -> session %s "
                "(primary identity reference at B1)",
                plate,
                session.session_id,
            )
            result_sid = session.session_id
            seeded_session = session

        # Off-lock: DB clears and the durable per-plate gallery seed (disk
        # write). The CAM-03 B-entry crop is the authoritative primary identity
        # reference — safe to persist (unlike the wide gate-only shot in
        # confirm_anpr_session_directly).
        for slot_id in evicted_slots:
            self._clear_slot_db_binding(slot_id)
        if seeded_session is not None:
            ordered_images, feature_vectors, _ = prepared
            self._seed_plate_gallery_reference(
                seeded_session, ordered_images, feature_vectors, 0
            )

        # Fix 4 — CAM-23 fallback: if the in-zone Park_Entry seed never fired this
        # visit (no CAM-23 top view on the session), pull the top view from the
        # most recent Park_Entry candidate for this plate now that the car is
        # confirmed. seed_gallery_from_park_entry is idempotent per candidate, so
        # this is a no-op when Park_Entry already seeded it.
        if seeded_session is not None and "CAM-23" not in (
            getattr(seeded_session, "reference_source_cameras", None) or []
        ):
            fallback_cid = self.latest_park_entry_candidate_for_plate(
                plate, camera_id="CAM-23"
            )
            if fallback_cid is not None:
                self.seed_gallery_from_park_entry(fallback_cid, plate)
        return result_sid

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

        # Extract feature vector for gallery persistence
        feature_vector = self.reid_matcher.extract_feature(image)

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
            path = self.add_session_snapshot(
                session.session_id, image, timestamp=now, source_camera=source_cam
            )
            if path is None:
                return None
            self._gallery_index_upsert(session)
            logger.info(
                "[B1] %s snapshot added to plate=%s -> session %s "
                "(secondary ReID reference)",
                source_cam, plate, session.session_id,
            )
            session_id = session.session_id

        # Persist to durable gallery folder OFF the registry lock
        if feature_vector is not None:
            try:
                self.gallery_store.save_ref(
                    plate,
                    image,
                    feature_vector,
                    quality=900.0,  # Secondary ref, below primary (999.0)
                    camera_id=source_cam,
                    timestamp=now,
                )
            except Exception as exc:
                logger.debug("[gallery] save_ref for ramp-entry %s failed: %r", plate, exc)

        return session_id

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
            if session_id is None or session_id not in self._sessions:
                return False

        # Heavy half (embedding + snapshot writes) OFF the registry lock.
        prepared = self._prepare_session_gallery(ordered_images, timestamp=now)
        if prepared is None:
            return False

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.last_seen_at = now
            session.last_seen_camera = camera_id
            session.last_seen_track_id = track_id
            applied = self._apply_session_gallery(
                session,
                prepared,
                primary_snapshot_index=primary_snapshot_index,
                source_camera=camera_id,
            )

        # Off-lock: persist the final CAM-03 timeline's primary crop to the
        # durable per-plate folder, same as the confirm paths — this is often
        # the best full-view reference the car ever produces.
        if applied:
            ordered, feature_vectors, _ = prepared
            self._seed_plate_gallery_reference(
                session,
                ordered,
                feature_vectors,
                max(0, min(primary_snapshot_index, len(ordered) - 1)),
            )
        return applied

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

    def get_session_plate(self, session_id: str) -> Optional[str]:
        """Plate bound to ``session_id``, regardless of camera ownership.

        Unlike :meth:`get_plate_for_track` this is NOT filtered by the
        single-camera ownership rule — it answers "is this session already
        identified?", not "may this camera display/attribute the plate?".
        Callers deciding whether to run identity matching at all should use
        this; without it, a non-owner camera sees ``None`` for a plated
        session and re-enters the anonymous-track ReID path.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            return session.plate if session is not None else None

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
        # Min-size gate: a distant, low-resolution crop makes a useless ReID
        # reference (upscaling to the model input is mostly interpolation) and
        # lets a far camera flood the gallery with tiny views. Reject on absolute
        # crop resolution — independent of the caller's view_quality, so it holds
        # for every accumulation caller. (Registry-side backstop to the engine's
        # size-aware _bbox_view_quality.)
        try:
            ch, cw = crop_bgr.shape[:2]
            if float(ch * cw) < float(getattr(cfg, "gallery_min_crop_area", 0.0) or 0.0):
                return False
        except Exception:
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

        # Fix 1 — colour veto: a floor-camera crop whose body colour is
        # incompatible with the car's ground-truth colour is a DIFFERENT car
        # (e.g. a white car passing next to a dark one). Reject it before it can
        # poison the gallery, regardless of whatever ReID score bound the track.
        # Fail-open when the session has no anchored ground-truth colour yet, and
        # `color_compatible` only rejects obvious mismatches, so legitimate
        # lighting/view variation still passes. Cheap: one centre-crop HSV.
        gt_hsv = getattr(session, "ground_truth_hsv", None)
        if gt_hsv is not None:
            try:
                from src.reid_matcher import body_colour_compatible, dominant_color_hsv

                crop_hsv = dominant_color_hsv(crop_bgr)
                if crop_hsv is not None and not body_colour_compatible(crop_hsv, gt_hsv):
                    return False
            except Exception:
                pass

        vec = self.reid_matcher.extract_feature(crop_bgr)
        if vec is None:
            return False

        # Fix 3 — feedback-loop guard: a new reference must resemble the
        # GROUND-TRUTH appearance (never the possibly-poisoned secondary refs),
        # scoring at least a lenient floor against the best ground-truth anchor.
        # This stops a wrong-car crop that slipped past matching from being added
        # and then reinforcing more wrong matches. Fail-open (None) when there is
        # no ground-truth appearance to compare against yet.
        gt_sim = self._best_ground_truth_similarity(vec, session)
        if gt_sim is not None and gt_sim < float(
            getattr(cfg, "gallery_accumulate_min_gt_similarity", 0.0) or 0.0
        ):
            return False

        with self._lock:
            live = self._sessions.get(session.session_id)
            if live is None or not live.plate:
                return False
            for ref in [live.feature_vector] + list(live.reference_feature_vectors):
                if ref is not None and self.reid_matcher.compute_similarity(vec, ref) > cfg.gallery_dedup_cosine:
                    return False  # near-duplicate — nothing new to learn
            live.reference_feature_vectors.append(vec)
            # Keep the source-camera list aligned (index-parallel) so match-time
            # trust weighting sees the right camera for this live reference.
            self._sync_reference_cameras(live)
            live.reference_source_cameras[-1] = camera_id or ""
            cap = max(1, int(cfg.gallery_max_refs_per_car))
            if len(live.reference_feature_vectors) > cap:
                # Diversity cap: keep the `cap` most mutually-dissimilar vectors
                # (max viewpoint coverage) rather than FIFO, which collapses the
                # gallery onto whichever viewpoint produced the most recent crops
                # and leaves a query from another camera/angle unmatchable. Seed
                # from the newest vector (just appended) so a genuinely novel view
                # is always retained.
                from src.vehicle_registry.gallery_store import select_diverse_indices

                vecs = live.reference_feature_vectors
                cams = live.reference_source_cameras
                keep_idx = set(
                    select_diverse_indices(vecs, cap, seed_index=len(vecs) - 1)
                )
                live.reference_feature_vectors = [
                    v for i, v in enumerate(vecs) if i in keep_idx
                ]
                live.reference_source_cameras = [
                    cams[i] for i in range(len(vecs)) if i in keep_idx
                ]
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

        vectors, tag, cameras = store.load_vectors(plate)
        if (not vectors) or tag != store._model_tag:
            crops, crop_cameras = store.load_crops(plate)
            if crops:
                feats = self.reid_matcher.extract_features_batch(crops)
                vectors, cameras = [], []
                for f, cam in zip(feats, crop_cameras):
                    if f is not None:
                        vectors.append(f)
                        cameras.append(cam)
        if not vectors:
            return None
        # Defensive: keep cameras index-aligned with vectors (secondary default).
        if len(cameras) < len(vectors):
            cameras = cameras + [""] * (len(vectors) - len(cameras))

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
                self._sync_reference_cameras(existing)
                if existing.feature_vector is None:
                    existing.feature_vector = vectors[0]
                    extra, extra_cams = vectors[1:], cameras[1:]
                else:
                    extra, extra_cams = vectors, cameras
                existing.reference_feature_vectors.extend(extra)
                existing.reference_source_cameras.extend(extra_cams)
                # Cap after enrich: a repeated warm-start (duplicate ANPR entry,
                # or a reload racing a live session) must not grow the ref list
                # without bound — more refs cost matching time AND widen the
                # false-match surface. Keep the most recent up to the cap.
                cap = max(1, int(self._matching_config.gallery_max_refs_per_car))
                if len(existing.reference_feature_vectors) > cap:
                    existing.reference_feature_vectors = (
                        existing.reference_feature_vectors[-cap:]
                    )
                    existing.reference_source_cameras = (
                        existing.reference_source_cameras[-cap:]
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
                reference_source_cameras=list(cameras[1:]),
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

    @staticmethod
    def _sync_reference_cameras(session) -> None:
        """Pad/truncate ``reference_source_cameras`` to match the length of
        ``reference_feature_vectors`` (index-aligned). Missing entries pad with
        "" (treated as a non-ground-truth/secondary source at match time — the
        conservative default). Called before writing a per-index camera so the
        two lists never drift."""
        vecs = session.reference_feature_vectors
        cams = session.reference_source_cameras
        if len(cams) < len(vecs):
            cams.extend([""] * (len(vecs) - len(cams)))
        elif len(cams) > len(vecs):
            del cams[len(vecs):]

    def _reference_weight(self, camera_id: str) -> float:
        """Trust weight for a gallery reference by its SOURCE camera.

        Ground-truth cameras (canonical viewpoints — ANPR front, CAM-23 top,
        CAM-03 front+back) score at full weight; any other camera's crop is
        down-weighted so an oblique side view can only win a match when it is
        substantially stronger than every ground-truth view. Unknown / empty
        source is treated as secondary (conservative)."""
        cfg = self._matching_config
        gt = getattr(cfg, "ground_truth_cameras", None)
        if gt and camera_id in gt:
            return 1.0
        return float(getattr(cfg, "secondary_camera_weight", 1.0))

    def _is_ground_truth_camera(self, camera_id: str) -> bool:
        """True when ``camera_id`` is one of the canonical ground-truth cameras
        (ANPR / CAM-23 / CAM-03). Mirrors :meth:`_reference_weight` semantics."""
        gt = getattr(self._matching_config, "ground_truth_cameras", None)
        return bool(gt and camera_id in gt)

    def _is_reattach_excluded_camera(self, camera_id: str) -> bool:
        """True when ``camera_id`` is a configured static slot camera whose
        anonymous tracks must not donate an identity via reattach (see
        ``MatchingConfig.reattach_excluded_cameras``). Such a camera permanently
        frames a parked car and would otherwise adopt another car's session on a
        borderline cross-camera score."""
        excluded = getattr(self._matching_config, "reattach_excluded_cameras", None)
        return bool(excluded and camera_id in excluded)

    def _maybe_set_ground_truth_hsv(self, session, image) -> None:
        """Anchor ``session.ground_truth_hsv`` from a ground-truth camera crop,
        once. Called from the ANPR / CAM-03 / CAM-23 seed paths with that
        camera's image. The first ground-truth view seen wins (ANPR front is the
        earliest and most reliable frontal colour); later ground-truth crops do
        not overwrite it. Best-effort — a crop that yields no colour is a no-op.
        This is the anchor for the colour vetoes in :meth:`accumulate_reference`
        and :meth:`match_global_session`."""
        if session is None or getattr(session, "ground_truth_hsv", None) is not None:
            return
        if image is None or getattr(image, "size", 0) == 0:
            return
        try:
            from src.reid_matcher import dominant_color_hsv

            hsv = dominant_color_hsv(image)
        except Exception:
            hsv = None
        if hsv is not None:
            session.ground_truth_hsv = hsv

    def _best_ground_truth_similarity(self, vec, session) -> Optional[float]:
        """Best cosine of ``vec`` against the session's GROUND-TRUTH appearance
        only — the primary ``feature_vector`` plus references whose source camera
        is a ground-truth camera. Poisoned secondary refs are excluded so they
        cannot lower the bar for admitting yet another wrong-car crop.

        Returns None when the session has no ground-truth appearance to compare
        against yet (caller fails open)."""
        anchors = []
        primary = getattr(session, "feature_vector", None)
        if primary is not None:
            anchors.append(primary)
        refs = getattr(session, "reference_feature_vectors", None) or []
        cams = getattr(session, "reference_source_cameras", None) or []
        for i, ref in enumerate(refs):
            if ref is None:
                continue
            cam = cams[i] if i < len(cams) else ""
            if self._is_ground_truth_camera(cam):
                anchors.append(ref)
        if not anchors:
            return None
        return max(
            self.reid_matcher.compute_similarity(vec, a) for a in anchors
        )

    def _best_weighted_score(self, query_vector, session) -> float:
        """Best source-camera-weighted ReID similarity of ``query_vector`` to a
        session's gallery. The PRIMARY (``feature_vector``) is always full weight
        (it is only ever set from a confirmation/seed path). Each entry in
        ``reference_feature_vectors`` is weighted by its index-aligned source
        camera in ``reference_source_cameras`` (missing -> secondary). Replaces
        the plain ``max(similarity)`` so the three ground-truth cameras dominate
        while other cameras still contribute recall. Returns 0.0 when nothing is
        scorable."""
        if query_vector is None:
            return 0.0
        best = 0.0
        primary = getattr(session, "feature_vector", None)
        if primary is not None:
            best = self.reid_matcher.compute_similarity(query_vector, primary)
        refs = getattr(session, "reference_feature_vectors", None) or []
        cams = getattr(session, "reference_source_cameras", None) or []
        for i, ref in enumerate(refs):
            if ref is None:
                continue
            cam = cams[i] if i < len(cams) else ""
            score = self.reid_matcher.compute_similarity(query_vector, ref) * self._reference_weight(cam)
            if score > best:
                best = score
        return best

    def _slot_pose_score(self, query_vector, session, slot_camera: str):
        """Score a PARKED-CAR query against a session's gallery, for the slot path.

        Identical to :meth:`_best_weighted_score` except that a reference captured by
        ``slot_camera`` itself is weighted by ``slot_camera_ref_weight`` (0.80) rather
        than the ordinary ``secondary_camera_weight`` (0.60). Such a reference is the car
        photographed in the exact pose we are now looking at, taught by an earlier
        OCR-confirmed park (``save_parked_reference``), and at 0.60 a car's own parked
        pose loses to a different car's full-weight gate photo — so it can never be
        recognised on a return visit.

        The uplift stops short of full weight on purpose: it applies to EVERY candidate,
        so it also lifts the regulars who park at this camera daily, and a same-view match
        between two *different* cars outscores a cross-view match on the *same* car. See
        the swept table on ``MatchingConfig.slot_camera_ref_weight`` — 1.0 buys nothing
        over 0.80 on warm and costs accuracy on cold.

        Returns ``(score, same_view, cross_view)``:
          * ``same_view``  — best RAW (unweighted) similarity over refs from
            ``slot_camera``; 0.0 if this car has never parked here. Non-zero = "warm".
          * ``cross_view`` — the ordinary weighted score over everything else.
          * ``score``      — what the ranking sorts on.

        The parts are returned separately because they are not interchangeable evidence:
        a warm match is far stronger than a cold one (rank-1 100% vs 88%), and both the
        decision gate and the ranker need to tell them apart.
        """
        if query_vector is None:
            return 0.0, 0.0, 0.0

        same_w = float(
            getattr(self._matching_config, "slot_camera_ref_weight", 0.80) or 0.80
        )

        same_view = 0.0     # raw, unweighted — the evidence, not the vote
        cross_view = 0.0    # weighted
        best = 0.0          # what we rank on

        primary = getattr(session, "feature_vector", None)
        if primary is not None:
            cross_view = self.reid_matcher.compute_similarity(query_vector, primary)
            best = cross_view

        refs = getattr(session, "reference_feature_vectors", None) or []
        cams = getattr(session, "reference_source_cameras", None) or []
        for i, ref in enumerate(refs):
            if ref is None:
                continue
            cam = cams[i] if i < len(cams) else ""
            sim = self.reid_matcher.compute_similarity(query_vector, ref)
            if slot_camera and cam == slot_camera:
                same_view = max(same_view, sim)
                best = max(best, sim * same_w)
            else:
                weighted = sim * self._reference_weight(cam)
                cross_view = max(cross_view, weighted)
                best = max(best, weighted)

        return best, same_view, cross_view

    @staticmethod
    def _temporally_eligible(session, anchor: Optional[datetime]) -> bool:
        """Temporal entry gate shared by both identity-match forks.

        A query track may only bind to a car that is still NOT parked, or one
        that parked (its slot went vacant->occupied) at or AFTER ``anchor`` —
        the moment this car's appearance began. A car already parked before
        that cannot be this moving track, and symmetrically an already-parked
        (old) car is never re-labelled onto a newer track.

        ``anchor`` derivation is the caller's responsibility and differs by
        fork by design: ``match_global_session`` (no session exists yet) uses
        the per-track first-seen time; ``reattach_track_to_confirmed_session``
        uses the anonymous ``current_session.first_seen_at`` (the car's
        appearance birth, which predates any single camera-local track). Both
        are lower-bound proxies for "when this car entered".

        Fail-open: a missing ``anchor`` (untracked / crop-only query) or a
        parked car with no ``linked_at`` timestamp keeps the candidate eligible.
        """
        if anchor is None:
            return True
        if session.status != "parked" and not session.linked_slot:
            return True
        linked_at = session.linked_at
        if linked_at is None:
            return True
        return linked_at >= anchor

    def match_global_session(
        self,
        query_vector: Optional[np.ndarray],
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        max_time_gap_seconds: float = 600.0,
        similarity_threshold: Optional[float] = None,
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
            # Entry anchor: the first time we ever saw THIS query track. A car
            # that was already parked BEFORE this moment cannot be the car we
            # are now tracking (it was sitting stationary in a slot while this
            # track was driving in), so parked-before candidates are dropped
            # from the pool. This is the "a new entrant may only match cars that
            # are still not parked, or cars that parked AFTER it entered" rule —
            # equivalently, an already-parked (old) car is never matched to a
            # newer track. Stamped once; an earlier stamp from a prior binding
            # is kept. Untracked/crop-only queries (no camera/track) leave the
            # anchor None and the rule stays inert (legacy all-sessions pool).
            entry_anchor = None
            if camera_id is not None and track_id is not None:
                entry_anchor = self._track_first_seen.setdefault(
                    (camera_id, track_id), now
                )

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
                        and self._temporally_eligible(session, entry_anchor)
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
                        and self._temporally_eligible(session, entry_anchor)
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

                # Slot transition override: same-camera recent parks can reattach
                # even if parked BEFORE this new track's entry_anchor (handles
                # adjacent slot movement, track dropout recovery). Typical case:
                # car moves B2→B3 on same camera, ByteTrack loses track briefly,
                # new track appears but entry_anchor blocks it. Solution: allow
                # reattach if last parked on THIS camera < 120 seconds ago.
                is_same_camera_recent_park = (
                    session.linked_camera == camera_id
                    and session.linked_at is not None
                    and (now - session.linked_at).total_seconds() < 120.0
                )

                if not has_live_track_elsewhere or is_same_camera_recent_park:
                    guarded_sessions.append(session)

        if not guarded_sessions:
            return None

        # Fix 2 — colour veto at match time. The live crop is in hand here, but
        # the modality cascade below is invoked with query_crop=None, so a
        # confident colour contradiction (a white car querying a dark session)
        # is otherwise treated as silence and the match rides on ReID alone.
        # Compute the query colour once and drop any candidate whose anchored
        # ground-truth colour is incompatible. Fail-open: no crop, or a
        # candidate without an anchored colour, is not vetoed.
        query_hsv = None
        if query_crop is not None:
            try:
                from src.reid_matcher import dominant_color_hsv

                query_hsv = dominant_color_hsv(query_crop)
            except Exception:
                query_hsv = None

        # Rank-5: score every guarded candidate once, then keep only the top-5
        # nearest by ReID similarity. The correct identity is far more reliably
        # within the 5 nearest than exactly rank-1 on these oblique views, so we
        # let the secondary signal decide among the 5 instead of trusting the
        # single argmax.
        scored = []
        for session in guarded_sessions:
            if query_hsv is not None:
                gt_hsv = getattr(session, "ground_truth_hsv", None)
                if gt_hsv is not None:
                    try:
                        from src.reid_matcher import body_colour_compatible

                        if not body_colour_compatible(query_hsv, gt_hsv):
                            continue  # different-coloured car — cannot be this one
                    except Exception:
                        pass
            # Source-camera-weighted best score: ground-truth cameras (ANPR,
            # CAM-23, CAM-03) dominate; other cameras' refs are down-weighted.
            score = self._best_weighted_score(query_vector, session)
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
        # Fallback mirrors the MatchingConfig.reid_solo_confirm default (0.70)
        # so a stub config without the attribute behaves like a real one.
        _solo = getattr(cfg, "reid_solo_confirm", 0.70) or 0.70
        _margin = getattr(cfg, "global_match_margin", 0.05) or 0.0
        _ocr_worth_it = (_best < _solo) or ((_best - _runner) < _margin)
        if query_crop is not None and top_candidates and _ocr_worth_it:
            ocr = getattr(self._match_decision, "plate_ocr", None)
            if ocr is not None and hasattr(ocr, "read"):
                try:
                    # Oblique slot/aisle view: plate is at the bottom of the box,
                    # not in the gate's bottom-centre ROI — read the full crop.
                    ocr_text, ocr_conf = ocr.read(query_crop, apply_plate_roi=False)
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
        # Track WHICH session is the runner-up, not just its score. A near-tie is
        # usually not two different cars — it is ONE car competing with itself: the
        # slot worker builds an anonymous (plate=None) session from its own
        # viewpoint, which then out-scores the car's real plated session (whose refs
        # are all gate views, so they only match cross-view at ~0.70). Without the
        # ids and plates in the log, that is indistinguishable from a genuine
        # two-car ambiguity, and the two call for opposite fixes.
        runner_up_sid = None
        session_plate = {}
        decision = self.match_decision
        for session, score in top_candidates:
            session_plate[session.session_id] = session.plate
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
                    if best_score > runner_up_score:
                        runner_up_score = best_score
                        runner_up_sid = best_sid
                best_score = score
                best_sid = session.session_id
            elif score > runner_up_score:
                runner_up_score = score
                runner_up_sid = session.session_id

        # ---- Slot acquisition by elimination --------------------------------
        # A car arriving at its slot cannot recognise ITSELF: its first view there is a
        # small oblique crop scoring ~0.58 against its own front-gate photos (below the
        # bar), so it either matches nothing (and mints an anonymous session) or matches
        # a nameless copy of itself. Either way the plate never reaches the slot, and no
        # threshold fixes it — from that viewpoint the ranking is INVERTED, so a looser
        # bar binds the WRONG plate.
        #
        # Geography and timing decide what appearance cannot. Inside the AREA-SCOPED
        # pool, if exactly ONE plated car is still in flight (entered recently, not yet
        # parked), it is the only car this can be. Appearance is then only asked not to
        # object. Requires area scoping to be active — without the spatial constraint
        # this would be a bare guess, so an un-zoned camera never takes this path.
        cfg_m = self._matching_config
        if bool(getattr(cfg_m, "slot_acquire_by_elimination", True)):
            best_plate_now = session_plate.get(best_sid) if best_sid else None
            # Only step in when appearance has FAILED to name the car: either nothing
            # confirmed, or the winner is an anonymous (nameless) session.
            if best_sid is None or best_plate_now is None:
                floor = float(
                    getattr(cfg_m, "slot_acquire_min_similarity", 0.50) or 0.50
                )
                inflight = self._inflight_plated_candidates(query_vector, now)
                if len(inflight) == 1 and inflight[0][1] >= floor:
                    claim_session, claim_score = inflight[0]
                    claim_sid = claim_session.session_id

                    # Is the car we are LOOKING AT plausibly the car that arrived?
                    #
                    # "Only one car in flight" says WHICH plate is unaccounted for. It
                    # says nothing about whether THIS car is that car. On 2026-07-11
                    # that gap bound DJS-7842 to a Nissan Sunny that had been parked in
                    # B25 for hours and merely never been identified: the B2 worker saw
                    # an anonymous car, could not name it, found exactly one car in
                    # flight, and claimed it. A wrong bind is worse than no bind.
                    #
                    # A car already being tracked BEFORE the plate was read at the gate
                    # cannot be the car that was read. Same reasoning as the D3 gate
                    # linger guard, one level up.
                    gate_read_at = self._last_anpr_entry_at.get(claim_session.plate)
                    incumbent = (
                        self._sessions.get(best_sid) if best_sid is not None else None
                    )
                    seen_before_gate = (
                        incumbent is not None
                        and gate_read_at is not None
                        and incumbent.first_seen_at is not None
                        and incumbent.first_seen_at < gate_read_at
                    )
                    if seen_before_gate:
                        logger.info(
                            "[GLOBAL] cam=%s elimination REFUSED for %s: this car was "
                            "already being tracked at %s, BEFORE the gate read at %s — "
                            "it was here first, so it cannot be the arriving car",
                            camera_id, claim_session.plate,
                            incumbent.first_seen_at, gate_read_at,
                        )
                        return None
                    # If a nameless copy of this car already won, fold it in so it
                    # cannot keep out-scoring the real identity on later frames.
                    if best_sid is not None and best_plate_now is None:
                        self._absorb_anonymous_session(best_sid, claim_sid)
                    logger.info(
                        "[GLOBAL] cam=%s ACQUIRED %s by elimination "
                        "(reid=%.3f >= floor %.2f; the ONLY car in flight) — "
                        "appearance could not name it, the gate and the clock did",
                        camera_id, claim_session.plate, claim_score, floor,
                    )
                    best_sid = claim_sid
                    best_score = max(best_score, claim_score)
                    runner_up_score = -1.0  # decided by elimination, not by margin
                elif len(inflight) > 1:
                    logger.info(
                        "[GLOBAL] cam=%s elimination declined: %d cars in flight "
                        "(%s) — ambiguous, refusing to guess",
                        camera_id, len(inflight),
                        ",".join(s.plate for s, _ in inflight),
                    )
                elif len(inflight) == 1:
                    logger.info(
                        "[GLOBAL] cam=%s elimination declined: only car in flight is "
                        "%s but reid=%.3f < floor %.2f — appearance CONTRADICTS",
                        camera_id, inflight[0][0].plate, inflight[0][1], floor,
                    )
                else:
                    # NOTHING in flight. Never leave this silent: it is the case that
                    # stranded DJS-7842 at CAM-04 on 2026-07-11 — CAM-06 had acquired it
                    # 25s earlier, yet by the time it reached its slot the in-flight pool
                    # was empty and the elimination was skipped without a word. Say
                    # exactly WHY each plated car was ruled out, or the next failure is
                    # another guessing game.
                    reasons = []
                    for sess in list(self._sessions.values()):
                        if not sess.plate:
                            continue
                        if getattr(sess, "linked_slot", None):
                            reasons.append(
                                "%s: already linked to slot %s"
                                % (sess.plate, sess.linked_slot)
                            )
                        elif sess.status not in ("confirmed", "parked"):
                            reasons.append("%s: status=%s" % (sess.plate, sess.status))
                        elif self._last_anpr_entry_at.get(sess.plate) is None:
                            reasons.append("%s: no ANPR gate read on record" % sess.plate)
                        else:
                            age = (
                                now - self._last_anpr_entry_at[sess.plate]
                            ).total_seconds()
                            reasons.append(
                                "%s: gate read %.0fs ago (window %.0fs)"
                                % (sess.plate, age, float(
                                    getattr(cfg_m, "slot_acquire_inflight_seconds", 300.0)
                                ))
                            )
                    logger.info(
                        "[GLOBAL] cam=%s elimination SKIPPED: no car in flight. "
                        "Plated sessions ruled out -> [%s]",
                        camera_id, "; ".join(reasons) or "no plated sessions at all",
                    )

        if best_sid is not None and runner_up_score >= 0.0:
            margin = float(
                getattr(self._matching_config, "global_match_margin", 0.05) or 0.0
            )
            if margin > 0.0 and (best_score - runner_up_score) < margin:
                best_plate = session_plate.get(best_sid)
                runner_plate = session_plate.get(runner_up_sid)
                # Exactly one side carries a plate ⇒ this is not two different cars,
                # it is ONE car competing with itself: its own anonymous session
                # (built from this camera's viewpoint, so it scores higher) against
                # its real plated session (gate views only, so it scores lower).
                # Abstaining here is what strands the car with plate=(none).
                self_competition = (best_plate is None) != (runner_plate is None)
                resolved = False
                if self_competition:
                    plated_sid = best_sid if best_plate else runner_up_sid
                    anon_sid = runner_up_sid if best_plate else best_sid
                    if self._absorb_anonymous_session(anon_sid, plated_sid):
                        logger.info(
                            "[GLOBAL] cam=%s self-competition resolved: %s(%.3f) and "
                            "%s(%.3f) were the SAME car -> bound to plate %s",
                            camera_id,
                            best_sid, best_score,
                            runner_up_sid, runner_up_score,
                            session_plate.get(plated_sid),
                        )
                        # Fall through to the normal success path below with the
                        # surviving (plated) session, rather than re-implementing it.
                        best_sid = plated_sid
                        best_score = max(best_score, runner_up_score)
                        resolved = True

                if not resolved:
                    logger.info(
                        "[GLOBAL] cam=%s abstain: ambiguous match "
                        "(best=%.3f %s plate=%s | runner_up=%.3f %s plate=%s | "
                        "margin=%.2f)",
                        camera_id,
                        best_score,
                        best_sid,
                        best_plate or "NONE(anonymous)",
                        runner_up_score,
                        runner_up_sid,
                        runner_plate or "NONE(anonymous)",
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

    def _inflight_plated_candidates(self, query_vector, now):
        """Every plated car that is still IN FLIGHT, scored against this query.

        In flight = has a plate, is NOT yet linked to a slot, and its ANPR GATE READ
        was recent.

        Deliberately searches ALL sessions, not the area-scoped candidate pool. A car
        in transit is BY DEFINITION leaving the area it was last seen in: DJS-7842
        entered at CAM-03 (area B1-A) and parked at CAM-04 (area B1-C), and the
        adjacency graph makes B1-C a neighbour of B1-B and RAMP-UP but NOT of B1-A —
        so the area pool excluded the car from its own destination and it was never
        even scored. Area scoping is right for narrowing ordinary appearance matches;
        it is wrong for a car whose whole state is "moving between areas".

        The safety here is not spatial, it is the conjunction below:

        Two exclusions carry the safety of this whole path:

        * **Already parked.** A car sitting in a slot is accounted for — it cannot also
          be the car arriving at a different one. This is what stops a long-parked car
          from stealing the acquisition, and it matters because from a slot camera's
          viewpoint a parked car can OUT-SCORE the real arrival (measured: DJS-7842
          0.634 vs RDJ-9640's own 0.583 on RDJ-9640's own crop).

        * **Anchored on the ANPR gate read**, never on the session's own timestamps.
          ``_restore_vehicle_galleries`` rebuilds a session with ``first_seen_at=now``,
          so after a restart EVERY car already inside would look freshly-entered and
          claim to be in flight. Anchoring on ``_last_anpr_entry_at`` (stamped only by
          a real gate read) means a restored car is NOT in flight until it actually
          drives through the gate again. No gate read on record ⇒ not in flight; the
          fail-safe direction is to decline.
        """
        cfg = self._matching_config
        window = float(
            getattr(cfg, "slot_acquire_inflight_seconds", 300.0) or 300.0
        )
        out = []
        for session in list(self._sessions.values()):
            if not session.plate:
                continue
            if session.status not in ("confirmed", "parked"):
                continue
            if getattr(session, "linked_slot", None):
                continue  # already parked somewhere — accounted for
            entered_at = self._last_anpr_entry_at.get(session.plate)
            if entered_at is None:
                continue  # no gate read on record — cannot claim to be arriving
            if (now - entered_at).total_seconds() > window:
                continue  # entered too long ago to still be driving to a slot
            score = (
                self._best_weighted_score(query_vector, session)
                if query_vector is not None
                else -1.0
            )
            out.append((session, score))
        return out

    def _absorb_anonymous_session(self, anon_sid: str, plated_sid: str) -> bool:
        """Fold a car's own ANONYMOUS session into its real PLATED one.

        A car entering the garage is plated at the gate (CAM-03), but the worker that
        owns the slot cameras also builds its own anonymous (plate=None) session for
        the same car from ITS viewpoint. Those two sessions then compete:

            anonymous  ~0.73  (slot viewpoint vs slot viewpoint — near-identical)
            plated     ~0.70  (gate views vs slot viewpoint — cross-view)

        The nameless copy WINS, and because the two land inside global_match_margin
        the matcher abstains — so the car parks with plate=(none) forever. It is not
        two cars being confused; it is one car competing with itself, and the side
        that carries the identity is the side that loses.

        The asymmetry is also the cure: the anonymous session's vectors ARE the
        parked-pose views the plated session lacks. Absorbing them makes the plated
        session match strongly from the slot cameras from here on, which is what
        closes the cross-view gap that no threshold could.

        Returns True when the merge happened. ``self._lock`` is an RLock, so this is
        safe whether or not the caller already holds it.
        """
        with self._lock:
            anon = self._sessions.get(anon_sid)
            plated = self._sessions.get(plated_sid)
            # Only ever collapse a nameless session INTO a named one — never the
            # reverse, and never merge two plated sessions: that would be a real
            # identity swap, the exact failure this whole guard exists to prevent.
            if anon is None or plated is None or anon.plate or not plated.plate:
                return False

            cap = int(
                getattr(self._matching_config, "gallery_max_refs_per_car", 20) or 20
            )
            carried = 0
            for vec in [anon.feature_vector, *(anon.reference_feature_vectors or [])]:
                if vec is None:
                    continue
                if len(plated.reference_feature_vectors) >= cap:
                    break
                plated.reference_feature_vectors.append(vec)
                carried += 1
            self._sync_reference_cameras(plated)

            # Carry the live observers across, so the track keeps its owner and the
            # slot camera does not immediately mint a FRESH anonymous session for the
            # same car on the very next frame (which would re-create the competition).
            for obs_cam, obs_tid in list(anon.observing_tracks.items()):
                plated.observing_tracks[obs_cam] = obs_tid
                plated.observing_scores[obs_cam] = anon.observing_scores.get(
                    obs_cam, 0.0
                )

            # Repoint every track that was bound to the nameless session.
            for key, sid in list(self._track_session_map.items()):
                if sid == anon_sid:
                    self._track_session_map[key] = plated_sid

            plated.last_seen_at = self._clock()
            anon.status = "merged"
            self._sessions.pop(anon_sid, None)
            self._gallery_index_remove(anon_sid)
            self._gallery_index_upsert(plated)

            logger.info(
                "[GLOBAL] merged anonymous session %s into plated %s (plate=%s): "
                "carried %d slot-viewpoint vector(s) the plated session was missing",
                anon_sid, plated_sid, plated.plate, carried,
            )
            return True

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

    def _session_floor(self, session) -> str:
        """Resolve a session's floor for the reattach same-floor guard.

        Prefers the area the car is currently settled in (``current_area``);
        falls back to its last-seen camera's area. Returns "" when zoning is off
        or the session isn't placed in any area yet — callers treat "" as
        "unknown floor" (excluded when the querying camera is zoned)."""
        if self._area_registry is None or not self._area_registry.enabled:
            return ""
        area = getattr(session, "current_area", "") or ""
        if not area:
            last_cam = getattr(session, "last_seen_camera", "") or ""
            if last_cam:
                area = self._area_registry.area_for_camera(last_cam)
        return self._area_registry.floor(area) if area else ""

    def reattach_track_to_confirmed_session(
        self,
        camera_id: str,
        track_id: int,
        query_vector: Optional[np.ndarray],
        similarity_threshold: Optional[float] = None,
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
        # Static slot cameras (permanently framing a parked car) must not donate
        # their anonymous tracks into a confirmed session — that is the CAM-24
        # parked-Lexus-adopts-the-entrant leak. They still track locally and can
        # be found by the forward global-match path; they just cannot reattach.
        if self._is_reattach_excluded_camera(camera_id):
            return None

        with self._lock:
            current_sid = self._track_session_map.get((camera_id, track_id))
            current_session = self._sessions.get(current_sid) if current_sid else None
            if current_session is None or current_session.plate:
                return current_sid

            # Same-floor guard: an anonymous track must not reattach to a plated
            # session on a DIFFERENT floor. aca072f bounded the global-match fork
            # by area but left this reattach fork unbounded — letting a B2 track
            # adopt a B1 car's plate on a cross-camera ReID score as low as
            # reattach_cross_camera (0.43). Only enforced when the querying camera
            # is zoned; un-zoned (Ground / unmapped) keeps the legacy all-sessions
            # pool. Floor (not area) is the right granularity: a legitimate
            # reattach is a car moving between aisles on ONE floor.
            query_floor = ""
            handoff_eligible_ids: set = set()
            if self._area_registry is not None and self._area_registry.enabled:
                query_area = self._area_registry.area_for_camera(camera_id)
                query_floor = self._area_registry.floor(query_area)
                # Cross-floor exception: a car that is genuinely transiting the
                # inter-floor ramp toward THIS camera's area (departed a
                # ramp-adjacent area, still inside the transit window) is a
                # legitimate arrival — the same eligibility the bounded global
                # matcher uses. Without this the same-floor filter below would
                # strand the plate on the origin floor when the car drives the
                # ramp (e.g. B1->RAMP-DN->B2). An unrelated cross-floor car has
                # no in-transit record, so it stays refused (the B1<->B2 leak).
                if self._handoff_matcher is not None and query_area:
                    handoff_eligible_ids = self._handoff_matcher.candidate_session_ids(
                        query_area, list(self._sessions.values())
                    )

            # Temporal gate (see _temporally_eligible): this anonymous track may
            # only reattach to a car that is still not parked, or one that parked
            # at or after this car's own appearance began. Anchored on the
            # anonymous session's first_seen_at (its appearance birth).
            reattach_anchor = current_session.first_seen_at

            candidates = [
                session
                for sid, session in self._sessions.items()
                if (
                    sid != current_sid
                    and session.status in ("confirmed", "parked")
                    and session.plate
                    and session.feature_vector is not None
                    and self._temporally_eligible(session, reattach_anchor)
                    and (
                        not query_floor
                        or self._session_floor(session) == query_floor
                        or sid in handoff_eligible_ids
                    )
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
            # Source-camera-weighted best score (ground-truth cameras dominate).
            score = self._best_weighted_score(query_vector, session)
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
                # Append as a secondary reference instead of replacing the
                # primary identity vector: reattach can confirm as low as the
                # cross-camera threshold, and one borderline false match must
                # not poison the authoritative B1/ANPR embedding. Dedup + FIFO
                # cap mirror accumulate_reference.
                cfg = self._matching_config
                known_vectors = [target_session.feature_vector] + list(
                    target_session.reference_feature_vectors
                )
                is_duplicate = any(
                    ref is not None
                    and self.reid_matcher.compute_similarity(query_vector, ref)
                    > cfg.gallery_dedup_cosine
                    for ref in known_vectors
                )
                # Learn-gate (Fix 1): associating a track can confirm as low as
                # reattach_cross_camera (0.43), but LEARNING its appearance as a
                # permanent gallery reference demands more — the vector must
                # resemble the session's GROUND-TRUTH appearance (never the
                # possibly-poisoned secondary refs) by the same floor the
                # disk-accumulate path uses. A borderline 0.43-0.60 reattach of a
                # wrong car (e.g. a static slot camera's parked neighbour) still
                # moves the track below; it just never poisons the gallery.
                # Fail-open when the session has no ground truth to compare yet.
                gt_sim = self._best_ground_truth_similarity(
                    query_vector, target_session
                )
                learn_floor = float(
                    getattr(cfg, "gallery_accumulate_min_gt_similarity", 0.0) or 0.0
                )
                if not is_duplicate and (gt_sim is None or gt_sim >= learn_floor):
                    target_session.reference_feature_vectors.append(query_vector)
                    self._sync_reference_cameras(target_session)
                    target_session.reference_source_cameras[-1] = camera_id or ""
                    cap = max(1, int(cfg.gallery_max_refs_per_car))
                    if len(target_session.reference_feature_vectors) > cap:
                        target_session.reference_feature_vectors = (
                            target_session.reference_feature_vectors[-cap:]
                        )
                        target_session.reference_source_cameras = (
                            target_session.reference_source_cameras[-cap:]
                        )
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
        allow_auto_lock: bool = True,
    ) -> Optional[str]:
        """
        Link a slot only from a confirmed session.
        No blind fallback to recent ANPR events.

        ``allow_auto_lock`` gates the deterministic "single unlocked occupied
        slot + single pending ANPR plate" auto-lock heuristic. It must be True
        ONLY on a real vacant->occupied transition (the ``vehicle_parked``
        event); the per-frame resolver for an already-occupied slot passes
        False. Without this gate a slot that was already occupied at startup (or
        parked earlier, before its plate was known) keeps hitting the per-frame
        path and would grab the next arrival's pending plate at confidence 1.0 —
        stamping a stranger's plate onto a long-parked car. The existing
        anti-swap guard does not catch this because the parked car has no plate
        yet (it only blocks overwriting a DIFFERENT plate).
        """
        if track_id is None:
            return None
        if is_reid_disabled_floor(floor):
            return None

        # ONLY A READ PLATE MAY REACH current_plate.
        #
        # This path binds a plate to a slot on APPEARANCE, and appearance at the slot is
        # not merely weak — it is INVERTED. Measured 2026-07-11: a car scored 0.583
        # against its OWN gate references while a DIFFERENT car scored 0.634. That
        # afternoon this path stamped RDJ-9640 onto B1_CRO, which held DJS-7842, and
        # dragged RDJ-9640's plate off its real slot (B10_CTO) in the same move.
        #
        # The OCR path (try_ocr_identify_track / try_ocr_identify_slot) READS the
        # characters off the car and binds only what exactly one car inside can explain.
        # Where OCR can see a plate, current_plate is right; where it cannot, the slot
        # stays NULL. A visible gap is fine. A stranger's plate in a customer's slot is
        # not — it propagates silently into billing and occupancy.
        #
        # So a session may only take a slot once its identity was READ. Set
        # slot_plate_requires_ocr=False to restore the old appearance-based behaviour.
        # NOTE the guard is UNCONDITIONAL, not "unless the car was OCR-confirmed".
        #
        # The first version checked `session.ocr_confirmed`, which is a property of the
        # CAR ("was this car ever read?"), not of THIS binding ("was it read at THIS
        # slot?"). On 2026-07-12 that let NDD-4141 — legitimately read at CAM-20 and so
        # flagged confirmed — be bound by APPEARANCE to a SECOND slot, B1_CRO, on CAM-21,
        # a camera that has never read a plate in its life. Same for HER-9235 across B2
        # and B6_Reserved. A car cannot be in two slots.
        #
        # Only the OCR path (bind_plate_to_slot, called after try_ocr_identify_slot
        # actually READ the characters off the car in THAT slot) may write current_plate.
        if bool(getattr(self._matching_config, "slot_plate_requires_ocr", True)):
            return None

        evicted_slots_to_clear = []
        result_plate = None

        with self._lock:
            # 1. Skip if slot already locked
            if slot_id in self._locked_slots:
                existing = self._parked.get(slot_id)
                if existing is not None:
                    return existing.plate
                return None

            # 2. Count other unlocked occupied slots
            other_unlocked = [
                sid for sid in self._parked
                if sid != slot_id and sid not in self._locked_slots
            ]

            # 3. Determine unique candidate plate values
            now = self._clock()
            active_pending_events = [
                e for e in self._pending_events.values()
                if e.direction == "entry" and e.status in ("pending", "provisional")
            ]
            valid_pending_events = [
                e for e in active_pending_events
                if (now - e.timestamp).total_seconds() <= self.PENDING_ANPR_EXPIRY_SECONDS
            ]
            unique_candidate_plates = list(set(e.plate for e in valid_pending_events))

            if allow_auto_lock and len(other_unlocked) == 0 and len(unique_candidate_plates) == 1:
                suggested_plate = unique_candidate_plates[0]
                plate_locked_elsewhere = any(
                    s.plate == suggested_plate and sid in self._locked_slots
                    for sid, s in self._parked.items()
                )

                # Anti-swap guard: never auto-lock over an already-identified car.
                # If the track's session already carries a DIFFERENT plate, this
                # "one slot + one pending plate" heuristic would overwrite a real
                # identity — car X sitting in the only unlocked slot relabelled
                # with a new arrival Y's plate. Skip auto-lock and let the normal
                # ReID/OCR path below keep X's existing binding.
                _existing_sid = self._track_session_map.get((camera_id, track_id))
                _existing_sess = (
                    self._sessions.get(_existing_sid) if _existing_sid else None
                )
                would_swap_identity = (
                    _existing_sess is not None
                    and bool(_existing_sess.plate)
                    and _existing_sess.plate != suggested_plate
                )
                if would_swap_identity:
                    logger.info(
                        "[AUTO-LOCK] Skipped: track (%s, %s) session %s already "
                        "bound to plate=%s (auto-lock would swap it for pending "
                        "plate=%s)",
                        camera_id, track_id, _existing_sid,
                        _existing_sess.plate, suggested_plate,
                    )

                if not plate_locked_elsewhere and not would_swap_identity:
                    session_id = self._track_session_map.get((camera_id, track_id))
                    evicted_slots = self._claim_plate_globally(
                        plate=suggested_plate,
                        keep_session_id=session_id,
                        reason="plate_reclaimed_elsewhere",
                        timestamp=now,
                    )
                    evicted_slots_to_clear.extend(evicted_slots)

                    is_new_session = False
                    if session_id is None:
                        session_id = f"sess_{uuid.uuid4().hex[:12]}"
                        session = VehicleSession(
                            session_id=session_id,
                            first_seen_at=now,
                            last_seen_at=now,
                            last_seen_camera=camera_id,
                            last_seen_track_id=track_id,
                            status="confirmed",
                        )
                        is_new_session = True
                    else:
                        session = self._sessions.get(session_id)
                        if session is None:
                            session = VehicleSession(
                                session_id=session_id,
                                first_seen_at=now,
                                last_seen_at=now,
                                last_seen_camera=camera_id,
                                last_seen_track_id=track_id,
                                status="confirmed",
                            )
                            is_new_session = True

                    session_backup = None
                    if not is_new_session:
                        session_backup = {
                            "plate": session.plate,
                            "status": session.status,
                            "new_pipeline_score": session.new_pipeline_score,
                            "old_pipeline_score": session.old_pipeline_score,
                            "linked_slot": session.linked_slot,
                            "linked_slot_name": session.linked_slot_name,
                            "linked_camera": session.linked_camera,
                            "linked_floor": session.linked_floor,
                            "linked_zone_id": session.linked_zone_id,
                            "linked_zone_name": session.linked_zone_name,
                            "linked_at": session.linked_at,
                        }

                    try:
                        session.plate = suggested_plate
                        session.new_pipeline_score = 1.0
                        session.old_pipeline_score = 1.0
                        session.status = "parked"
                        session.linked_slot = slot_id
                        session.linked_slot_name = slot_name
                        session.linked_camera = camera_id
                        session.linked_floor = floor
                        session.linked_zone_id = zone_id
                        session.linked_zone_name = zone_name
                        session.linked_at = timestamp

                        if is_new_session:
                            self._sessions[session_id] = session
                            self._track_session_map[(camera_id, track_id)] = session_id

                        self._gallery_index_upsert(session)

                        self._locked_slots.add(slot_id)
                        self._parked[slot_id] = session
                        self._drop_other_track_mappings_for_session(
                            session_id,
                            keep=(camera_id, track_id),
                        )

                        target_event = None
                        for event in valid_pending_events:
                            if event.plate == suggested_plate:
                                target_event = event
                                break
                        if target_event is not None:
                            target_event.status = "confirmed"
                            target_event.session_id = session_id

                        logger.info(
                            "[AUTO-LOCK] Auto-locked suggested plate=%s to slot=%s (track=%d, session=%s, reason=%s)",
                            suggested_plate,
                            slot_id,
                            track_id,
                            session_id,
                            "single_unlocked_slot_single_pending_plate",
                        )

                        result_plate = suggested_plate

                    except Exception as e:
                        if is_new_session:
                            self._sessions.pop(session_id, None)
                            self._track_session_map.pop((camera_id, track_id), None)
                        elif session_backup is not None:
                            for k, v in session_backup.items():
                                setattr(session, k, v)
                            try:
                                self._gallery_index_upsert(session)
                            except Exception:
                                pass
                        logger.exception("[AUTO-LOCK] Failed during lock transaction: %r", e)
                        raise e

            if result_plate is None:
                session_id = self._track_session_map.get((camera_id, track_id))
                if session_id is None:
                    return None

                session = self._sessions.get(session_id)
                if session is None:
                    return None

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

                if session.plate:
                    stale_slots = [
                        sid
                        for sid, s in self._parked.items()
                        if sid != slot_id
                        and s.session_id != session.session_id
                        and s.plate == session.plate
                        and sid not in self._locked_slots
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

                if session.linked_slot and session.linked_slot != slot_id:
                    old_slot_id = session.linked_slot
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

                result_plate = session.plate

        # DB clear off-lock - never hold self._lock across DB I/O.
        for sid in evicted_slots_to_clear:
            self._clear_slot_db_binding(sid)

        return result_plate

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

    @staticmethod
    def _has_appearance_evidence(session) -> bool:
        """Has VA ever actually SEEN this car — does it hold any image of it?

        A session hydrated from PMS-AI's ``parking_sessions`` carries a plate and nothing
        else. When that plate is an ANPR misread it names a car that does not exist, and
        VA has no photo of it because there is nothing to photograph.
        """
        if getattr(session, "feature_vector", None) is not None:
            return True
        return bool(getattr(session, "reference_feature_vectors", None))

    def plates_inside(self, require_appearance: Optional[bool] = None) -> List[str]:
        """Every plate VA believes is currently inside the facility.

        VA's OWN view (plated sessions in memory), not PMS-AI's parking_sessions table —
        on 2026-07-11 that table stopped receiving rows entirely while cars kept driving
        in.

        **Phantom plates are excluded.** VA hydrates a session for every open
        ``parking_sessions`` row, and the ANPR gate misreads: measured 2026-07-13, 18 of
        51 "cars inside" had no gallery folder because they are not cars — they are
        mis-OCR'd spellings of real ones (``BJA-7842``/``DJA-7842`` alongside the real
        ``DJS-7842``). They cannot be matched, because there is no photo of a car that
        does not exist; all they can do is collide. ``confirm_plate`` matches on the DIGIT
        RUN and abstains when a read fits more than one candidate, so those two phantoms
        turn a *perfect* read of ``DJS-7842`` into a three-way tie and the slot stays
        NULL. A candidate VA has never seen can only ever subtract.

        NOTE this is the FALLBACK candidate set, used only when ReID cannot shortlist (no
        query vector). Matching OCR against the whole facility is what poisoned the
        gallery on 2026-07-12 — see :meth:`reid_shortlist`. The ReID path is already
        immune (a phantom has no vector, so it is never scored); this closes the fallback.
        """
        if require_appearance is None:
            require_appearance = bool(
                getattr(
                    self._matching_config,
                    "candidates_require_appearance_evidence",
                    True,
                )
            )
        with self._lock:
            return sorted(
                {
                    s.plate
                    for s in self._sessions.values()
                    if s.plate
                    and (not require_appearance or self._has_appearance_evidence(s))
                }
            )

    def _is_locked_elsewhere(self, session, slot_id: Optional[str]):
        """Is this car already parked-and-locked in a DIFFERENT slot?

        A car cannot occupy two slots, so such a session is not the car we are looking
        at — no matter how much it resembles it. This is the cheapest and most reliable
        non-appearance signal available, and it kills a whole class of look-alike error:
        measured 2026-07-13, TRS-9117 in B5 (CAM-08) was consistently outranked by
        HSR-8327, which was already locked into B26 on another camera.

        Two sources, because a slot on another camera may be owned by another worker
        process (the supervisor runs several) and this registry cannot see its sessions:
          * LOCAL  — ``self._locked_slots`` / ``self._parked``, this worker's own binds.
          * DB     — ``_external_plate_locks``, refreshed from ``parking_slots`` on the
            existing session-sync tick.

        Returns ``(locked, reason, detail)``; ``locked`` False means keep the candidate.
        """
        plate = getattr(session, "plate", None)
        if not plate:
            return False, "", {}
        return self._locked_elsewhere_reason(
            plate, slot_id, linked_slot=getattr(session, "linked_slot", None)
        )

    def _locked_elsewhere_reason(
        self, plate: str, slot_id: Optional[str], linked_slot: Optional[str] = None
    ):
        """The rule itself, keyed by plate so both the ReID path and the
        ``plates_inside`` fallback can share it. Returns ``(locked, reason, detail)``."""
        if not plate:
            return False, "", {}

        if linked_slot and linked_slot != slot_id and linked_slot in self._locked_slots:
            return True, "LOCKED_ELSEWHERE_LOCAL", {"locked_slot": linked_slot}

        lock = (getattr(self, "_external_plate_locks", None) or {}).get(plate)
        if lock and lock.get("slot_id") and lock.get("slot_id") != slot_id:
            # ...unless the car has since left and driven back in. The lock is then a
            # ghost of the previous visit, and the car really is parking again, in a new
            # slot. Deliberately NOT a TTL: a car parked overnight holds a 12-hour-old
            # lock that is perfectly valid, and a TTL would silently un-exclude it.
            locked_at = lock.get("locked_at")
            if locked_at is not None:
                entered_at = self.last_anpr_entry_at(plate)
                if entered_at is not None and entered_at > locked_at:
                    return False, "", {}
            return True, "LOCKED_ELSEWHERE_DB", {
                "locked_slot": lock.get("slot_id"),
                "locked_camera": lock.get("camera_id"),
                "locked_at": locked_at,
            }

        return False, "", {}

    def take_released_slots(self) -> List[str]:
        """Slots a relocating car vacated, drained once. The engine nulls their DB rows —
        without this, parking_slots keeps the old row and the same car is reported in two
        places."""
        with self._lock:
            released = sorted(getattr(self, "_released_slots", set()) or set())
            self._released_slots = set()
        return released

    def _is_plate_locked_elsewhere(self, plate: str, slot_id: Optional[str]) -> bool:
        if not getattr(
            self._matching_config, "exclude_plates_locked_elsewhere", True
        ):
            return False
        locked, _reason, _detail = self._locked_elsewhere_reason(plate, slot_id)
        return locked

    def reid_rank(
        self,
        query_vector,
        *,
        slot_id: Optional[str] = None,
        slot_camera: Optional[str] = None,
        k: int = GLOBAL_MATCH_RANK,
        apply_mutual_exclusion: bool = True,
    ):
        """Rank every plausible car for a query vector, WITH its scores.

        This is :meth:`reid_shortlist` with the reasoning left in. The shortlist throws
        the scores away and returns bare plates, which is why nothing downstream could
        ever say *how sure* it was, and why no training data existed: the losing
        candidates — the hard negatives — vanished at the return statement.

        Two things happen here that the plain scan did not do:

        * **Scoring is slot-aware.** With ``slot_camera``, a reference taught by that same
          camera scores at full weight (see :meth:`_slot_pose_score`) instead of taking
          the secondary-camera discount meant for oblique guesses.

        * **Rejects come out before the top-k slice**, not after. That ordering matters:
          dropping a locked-elsewhere car PROMOTES rank-6 into the shortlist, so the gate
          can only ever ADD the right car to the candidate set. It cannot turn a correct
          bind into a wrong one.

        Returns ``(kept, rejected)``. ``kept`` is the top-k surviving
        :class:`RankedCandidate`, best first, re-ranked 1..k. ``rejected`` carries the
        reason each excluded candidate was dropped, for the decision log.
        """
        if query_vector is None:
            return [], []

        use_pose = bool(slot_camera)

        scored: List[RankedCandidate] = []
        with self._lock:
            for s in self._sessions.values():
                if not s.plate or s.feature_vector is None:
                    continue
                if use_pose:
                    score, same_view, cross_view = self._slot_pose_score(
                        query_vector, s, slot_camera
                    )
                else:
                    score = self._best_weighted_score(query_vector, s)
                    same_view, cross_view = 0.0, score
                scored.append(
                    RankedCandidate(
                        plate=s.plate,
                        session_id=s.session_id,
                        score=float(score),
                        same_view_score=float(same_view),
                        cross_view_score=float(cross_view),
                        warm=bool(same_view > 0.0),
                        rank=0,
                        session=s,
                    )
                )

            scored.sort(key=lambda c: c.score, reverse=True)

            kept: List[RankedCandidate] = []
            rejected: List[RejectedCandidate] = []
            for raw_rank, cand in enumerate(scored, start=1):
                if apply_mutual_exclusion and getattr(
                    self._matching_config, "exclude_plates_locked_elsewhere", True
                ):
                    locked, reason, detail = self._is_locked_elsewhere(
                        cand.session, slot_id
                    )
                    if locked:
                        rejected.append(
                            RejectedCandidate(
                                plate=cand.plate,
                                session_id=cand.session_id,
                                raw_rank=raw_rank,
                                score=cand.score,
                                reason=reason,
                                detail=detail,
                            )
                        )
                        continue
                kept.append(replace(cand, rank=len(kept) + 1))
                if len(kept) >= k:
                    break

        return kept, rejected

    def reid_shortlist(
        self,
        crop_bgr,
        k: int = GLOBAL_MATCH_RANK,
        *,
        slot_id: Optional[str] = None,
        slot_camera: Optional[str] = None,
    ) -> List[str]:
        """The top-k plates ReID considers plausible for this crop.

        REID NARROWS, OCR CONFIRMS. This is the candidate set OCR is allowed to pick
        from, and it is the fix for the 2026-07-12 gallery poisoning.

        OCR previously matched against EVERY car inside — 28 plates. 'EEB-80' carries
        only the digits '80', and '80' is a substring of nearly any read, so a Range
        Rover's plate "confirmed" EEB-80 and its photo was filed under a matte-grey
        Porsche's name. Six cars ended up in EEB-80's gallery.

        Appearance would have vetoed that instantly: a Range Rover is nowhere near a
        Porsche in ReID space, so EEB-80 would never appear in its top-5. Two independent
        witnesses must now agree — the car must LOOK like the candidate (ReID) AND its
        plate must READ as the candidate (OCR). Either alone is demonstrably unsafe, and
        OCR alone collides on short plates.

        Note ReID only has to place the right car in the TOP FIVE, not first — and it
        does: measured 2026-07-13 on the live gallery, recall@5 is 97.7% cold even
        though rank-1 is only 87.8%. That gap is exactly why the shortlist is the
        contract here, and why a bare top-1 is not trusted on its own.

        Thin wrapper over :meth:`reid_rank` — use that directly when you need the scores.
        """
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return []
        try:
            q = self.reid_matcher.extract_feature(crop_bgr)
        except Exception:
            return []
        kept, _rejected = self.reid_rank(
            q, slot_id=slot_id, slot_camera=slot_camera, k=k
        )
        return [c.plate for c in kept]

    def try_reid_identify_slot(
        self,
        slot_id: str,
        crop_bgr,
        camera_id: Optional[str] = None,
        *,
        is_reserved: bool = False,
        decision_ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Identify a parked car by APPEARANCE ALONE, when OCR cannot read its plate.

        The fallback for a slot that has no readable plate, ever. CAM-21 frames B1_CRO in
        pure side profile and has logged 455 attempts with zero reads; demanding an OCR
        witness there means the slot is NULL forever. So appearance stands alone — but
        only when it is genuinely certain, and it abstains loudly otherwise.

        The bar is the MARGIN, not the score. Measured on 311 real parked-pose queries: a
        wrong car's top-1 score reaches 0.762, above any score floor worth having, but a
        wrong car NEVER wins by more than 0.099. Requiring a 0.15 margin clears that with
        0.05 headroom in both the warm and cold regimes, for zero false accepts.

        It will refuse a car it has never seen, and that is the point: an unknown vehicle
        scores ~0.6 against everything with a near-zero margin (B22: top-1 0.598, margin
        0.025), and binding it would stamp a random known plate onto a stranger.
        ``current_plate`` is CORRECT or NULL.

        Returns the plate, or None. Does NOT teach the gallery — a solo bind is not
        evidence, and a wrong one would poison the very references it learned from.
        """
        cfg = self._matching_config
        if not getattr(cfg, "slot_reid_solo_enabled", False):
            return None
        if self.get_slot_plate(slot_id):
            return None
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return None

        min_score = float(
            cfg.slot_reid_solo_min_score_reserved if is_reserved
            else cfg.slot_reid_solo_min_score
        )
        min_margin = float(
            cfg.slot_reid_solo_min_margin_reserved if is_reserved
            else cfg.slot_reid_solo_min_margin
        )

        try:
            qvec = self.reid_matcher.extract_feature(crop_bgr)
        except Exception:
            return None
        if qvec is None:
            return None

        kept, rejected = self.reid_rank(
            qvec, slot_id=slot_id, slot_camera=camera_id, k=GLOBAL_MATCH_RANK
        )
        if not kept:
            return None

        top = kept[0]
        margin = top.score - (kept[1].score if len(kept) > 1 else 0.0)
        accept = top.score >= min_score and margin >= min_margin

        self._log_slot_decision(
            slot_id=slot_id, camera_id=camera_id, crop_bgr=crop_bgr,
            kept=kept, rejected=rejected, ocr_text="", ocr_conf=0.0,
            bound_plate=top.plate if accept else None,
            ctx={**(decision_ctx or {}), "path": "reid_solo"},
        )

        if not accept:
            logger.info(
                "[reid-solo] slot=%s ABSTAIN — best is %s at %.3f (margin %.3f); needs "
                "score>=%.2f margin>=%.2f%s. A low score AND a flat margin means no car "
                "in the gallery looks like this one — binding it would stamp a stranger "
                "with a known plate.",
                slot_id, top.plate, top.score, margin, min_score, min_margin,
                " [RESERVED — stricter]" if is_reserved else "",
            )
            return None

        logger.info(
            "[reid-solo] slot=%s ACCEPT %s — appearance alone: score %.3f, clear of the "
            "runner-up by %.3f (a wrong car never wins by more than 0.099). OCR cannot "
            "read this slot, so there is no second witness to wait for.",
            slot_id, top.plate, top.score, margin,
        )
        return top.plate

    def _candidate_attrs(self, plate: str):
        """(colour, type) for a candidate, as ``((label, conf), (label, conf))``.

        Cached per plate: the classifiers are OpenVINO inferences (~15ms each) and a
        car's colour does not change between frames. Computed once from a gallery crop,
        then free forever. Returns ``((None,0.0),(None,0.0))`` when unavailable.
        """
        cache = getattr(self, "_attr_cache", None)
        if cache is None:
            cache = self._attr_cache = {}
        if plate in cache:
            return cache[plate]

        blank = ((None, 0.0), (None, 0.0))
        store = getattr(self, "gallery_store", None)
        if store is None:
            cache[plate] = blank
            return blank
        try:
            crops = store.load_crops(plate) or []
            if not crops:
                cache[plate] = blank
                return blank
            col = self._match_decision._color_classifier.predict(crops[0])
            typ = self._match_decision._type_classifier.predict(crops[0])
            cache[plate] = (col, typ)
        except Exception:
            cache[plate] = blank
        return cache[plate]

    def _log_slot_decision(
        self, *, slot_id, camera_id, crop_bgr, kept, rejected,
        ocr_text, ocr_conf, bound_plate, ctx,
    ) -> None:
        """Record this decision — the candidates, their features, and the answer.

        ``bound_plate`` is the LABEL when OCR confirmed one; the other candidates are
        hard negatives (cars ReID genuinely believed at this instant). This is the only
        place the tie-breaking signals — locked_elsewhere, the OCR read, crop quality —
        are captured, and they cannot be reconstructed after the fact. Best-effort:
        logging must never break identification.
        """
        log = getattr(self, "decision_log", None)
        if log is None or not getattr(log, "enabled", False) or not kept:
            return
        try:
            from src.matching.decision_log import build_record
            from src.matching.slot_rank_features import (
                FEATURE_NAMES, CandidateSignals, QuerySignals, build_features,
            )
            from src.reid_matcher.reid_burst import sharpness_score

            ctx = ctx or {}
            h, w = crop_bgr.shape[:2]
            qcol, qtyp = (
                self._match_decision._color_classifier.predict(crop_bgr),
                self._match_decision._type_classifier.predict(crop_bgr),
            )
            query = QuerySignals(
                crop_sharpness=float(sharpness_score(crop_bgr)),
                crop_area=float(h * w),
                crop_aspect=float(w / max(h, 1)),
                active_candidates=len(kept),
            )
            sigs = []
            for c in kept:
                sess = c.session
                cams = list(getattr(sess, "reference_source_cameras", None) or [])
                ccol, ctyp_ = self._candidate_attrs(c.plate)
                locked, _r, _d = self._locked_elsewhere_reason(
                    c.plate, slot_id, getattr(sess, "linked_slot", None)
                )
                sigs.append(
                    CandidateSignals(
                        plate=c.plate,
                        reid_score=c.score,
                        same_view_score=c.same_view_score,
                        cross_view_score=c.cross_view_score,
                        warm=c.warm,
                        rank=c.rank,
                        n_refs=len(cams),
                        n_same_view_refs=sum(1 for x in cams if x == camera_id),
                        best_ref_is_gate=not c.warm,
                        color_match=(qcol[0] == ccol[0]) if (qcol[0] and ccol[0]) else None,
                        color_conf_query=float(qcol[1] or 0.0),
                        color_conf_cand=float(ccol[1] or 0.0),
                        type_match=(qtyp[0] == ctyp_[0]) if (qtyp[0] and ctyp_[0]) else None,
                        type_conf_query=float(qtyp[1] or 0.0),
                        locked_elsewhere=bool(locked),
                    )
                )
            feats = build_features(sigs, query)
            solo = (ctx.get("path") == "reid_solo")
            if bound_plate:
                decision = "BIND_REID_SOLO" if solo else "BIND"
            elif solo:
                decision = "ABSTAIN_REID_NOT_CONFIDENT"
            elif not ocr_text:
                decision = "ABSTAIN_NO_READ"
            else:
                decision = "ABSTAIN_READ_MATCHES_NOBODY"
            log.emit(
                build_record(
                    slot_id=slot_id,
                    camera_id=camera_id or "",
                    floor=ctx.get("floor"),
                    area=ctx.get("area"),
                    is_reserved=bool(ctx.get("is_reserved")),
                    reserved_for=ctx.get("reserved_for"),
                    attempt=int(ctx.get("attempt", 0)),
                    max_attempts=int(ctx.get("max_attempts", 0)),
                    query=query,
                    candidates=sigs,
                    feature_matrix=feats,
                    feature_names=FEATURE_NAMES,
                    rejected=rejected,
                    ocr_text=ocr_text,
                    ocr_conf=ocr_conf,
                    decision=decision,
                    bound_plate=bound_plate,
                    label_source="ocr" if bound_plate else None,
                )
            )
        except Exception as exc:
            logger.debug("[decision-log] emit failed for slot=%s: %r", slot_id, exc)

    def try_ocr_identify_slot(
        self,
        slot_id: str,
        crop_bgr,
        camera_id: Optional[str] = None,
        *,
        decision_ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """READ the plate off a parked car and bind it — no prior binding needed.

        REID NARROWS, OCR CONFIRMS: appearance proposes a shortlist, and the plate read
        off the car picks exactly one of them. Two independent witnesses must agree.
        Neither is trusted alone — OCR alone collides on short plates (a Range Rover's
        read "confirmed" EEB-80 and poisoned a Porsche's gallery on 2026-07-12), and a
        bare ReID top-1 is right only 87.8% of the time cold.

        Bind only when the read confirms exactly one shortlisted car; otherwise leave the
        slot NULL. ``current_plate`` must be CORRECT or NULL — a visible gap is fine, a
        stranger's plate is a liability.

        ``camera_id`` is the camera looking at the slot. Passing it lets the ranking
        recognise a reference this same camera taught on an earlier park (the car's own
        parked pose) and score it at full weight instead of discounting it as an oblique
        secondary view — worth +1.5pt of rank-1 on cars that have parked here before.

        Distinct from :meth:`try_ocr_confirm_slot`, which only CONFIRMS a plate already
        bound and cannot identify a car from nothing.

        This is the SYNCHRONOUS composition of the three phases below (plan → read →
        confirm), preserved verbatim for callers that want the read inline. The engine's
        async path calls the three directly so the PaddleOCR read runs off-thread; see
        matching.slot_ocr_async.
        """
        existing = self.get_slot_plate(slot_id)
        if existing:
            return existing  # already identified; nothing to do

        plan = self.plan_slot_ocr(slot_id, crop_bgr, camera_id, decision_ctx=decision_ctx)
        if plan is None:
            return None  # no OCR plugin / no crop
        ocr_text, ocr_conf = self.read_slot_plate(crop_bgr, plan.allow_retry)
        return self.confirm_slot_ocr(plan, crop_bgr, ocr_text, ocr_conf)

    def plan_slot_ocr(
        self,
        slot_id: str,
        crop_bgr,
        camera_id: Optional[str] = None,
        *,
        decision_ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional["SlotOcrPlan"]:
        """Phase 1 (MAIN THREAD): rank the candidates for a parked-slot read.

        Returns ``None`` when there is nothing to read (no OCR plugin, no crop) —
        the caller has already handled the already-identified case. Otherwise returns
        the shortlist + retry budget the read needs, carrying the kept/rejected
        candidates so the decision log can be written after the read.

        Runs on the main thread because it uses the ReID matcher, which shares its
        OpenVINO infer request with the tracking loop and is not safe to call from a
        second thread.
        """
        ocr = getattr(self._match_decision, "plate_ocr", None)
        if ocr is None or not hasattr(ocr, "read") or crop_bgr is None:
            return None

        # RANK FIRST, then read. The order matters: a blank OCR read used to return here
        # immediately, so the candidate list — the hard negatives a ranker needs — was
        # thrown away on exactly the 2,269 attempts we most want to understand. Scoring
        # costs one embedding (~15ms) against OCR's ~80-470ms, so it is nearly free.
        kept: List[RankedCandidate] = []
        rejected: List[RejectedCandidate] = []
        try:
            qvec = self.reid_matcher.extract_feature(crop_bgr)
        except Exception:
            qvec = None
        if qvec is not None:
            kept, rejected = self.reid_rank(
                qvec, slot_id=slot_id, slot_camera=camera_id, k=GLOBAL_MATCH_RANK
            )

        candidates = [c.plate for c in kept]
        if not candidates:
            # ReID unavailable — degrade, don't guess. Still drop cars parked elsewhere:
            # this fallback matched against EVERY plate inside and is the path that let
            # the 2026-07-12 poisoning through.
            candidates = [
                p
                for p in self.plates_inside()
                if not self._is_plate_locked_elsewhere(p, slot_id)
            ]

        # The enlarged-retry pass rescues a distant plate (B13: '' -> '9990BHD'), but on
        # a slot whose plate is simply not in frame it costs ~670ms and finds nothing. So
        # spend it on the FIRST few attempts only: if the plate were visible at all, an
        # early frame would have caught it. Beyond that we are paying to re-confirm a
        # camera-angle fact, 12 times a park, on the frame loop.
        attempt = int((decision_ctx or {}).get("attempt", 1))
        retry_budget = int(
            getattr(self._matching_config, "ocr_upscale_retry_max_attempts", 4)
        )
        return SlotOcrPlan(
            slot_id=slot_id,
            camera_id=camera_id,
            candidates=candidates,
            kept=kept,
            rejected=rejected,
            allow_retry=(attempt <= retry_budget),
            decision_ctx=decision_ctx,
        )

    def read_slot_plate(self, crop_bgr, allow_retry: bool) -> Tuple[str, float]:
        """Phase 2 (ANY THREAD): the heavy PaddleOCR read, and nothing else.

        This is the only phase safe to run off the main loop — it touches solely the
        plate-OCR plugin, whose ``read`` serialises itself with an internal lock. The
        slot camera looks DOWN, so the plate is at the bottom of the whole-car box, not
        in the gate's bottom-centre band: read the full crop (``apply_plate_roi=False``).
        """
        ocr = getattr(self._match_decision, "plate_ocr", None)
        if ocr is None or not hasattr(ocr, "read") or crop_bgr is None:
            return ("", 0.0)
        try:
            return ocr.read(crop_bgr, allow_retry=allow_retry, apply_plate_roi=False)
        except TypeError:
            # A PlateOCR implementation without the retry knob (e.g. NoopPlateOCR).
            return ocr.read(crop_bgr)
        except Exception as exc:
            logger.debug("[ocr-id] read failed: %r", exc)
            return ("", 0.0)

    def confirm_slot_ocr(
        self, plan: "SlotOcrPlan", crop_bgr, ocr_text: str, ocr_conf: float
    ) -> Optional[str]:
        """Phase 3 (MAIN THREAD): fold the read back in — confirm, log, decide.

        Returns the plate only when the read confirms exactly one shortlisted car;
        otherwise ``None`` (slot stays NULL). Runs on the main thread: the decision log
        scores the crop through the colour/type classifiers, which — like the ReID
        matcher — share their OpenVINO requests with the main loop.
        """
        slot_id, camera_id = plan.slot_id, plan.camera_id
        candidates = plan.candidates
        plate = confirm_plate(ocr_text, candidates) if ocr_text else None

        self._log_slot_decision(
            slot_id=slot_id,
            camera_id=camera_id,
            crop_bgr=crop_bgr,
            kept=plan.kept,
            rejected=plan.rejected,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
            bound_plate=plate,
            ctx=plan.decision_ctx,
        )

        if not ocr_text:
            # Never silent. A side-on slot (CAM-21 frames B1_CRO in pure profile) reads
            # nothing at all in the settled pose — the plate is simply not in the frame.
            # That is a CAMERA-ANGLE fact, not an OCR failure, and it must be visible in
            # the log or it looks like the pass never ran.
            logger.info(
                "[ocr-id] slot=%s read NOTHING — no plate visible in this view "
                "(side-on slot? the plate is only in frame while the car TURNS IN)",
                slot_id,
            )
            return None

        if not plate:
            logger.info(
                "[ocr-id] slot=%s read=%r confirms none of ReID's top-%d %s — "
                "leaving NULL",
                slot_id, ocr_text, len(candidates), candidates,
            )
            return None

        logger.info(
            "[ocr-id] slot=%s CONFIRMED %s: ReID shortlisted it and OCR read %r off the "
            "car — two independent witnesses agree",
            slot_id, plate, ocr_text,
        )
        return plate

    def ocr_transit_candidates(self, now=None):
        """Cars whose plate was READ moments ago and that have not parked yet.

        Every car entering this facility passes CAM-20, and CAM-20 reads plates reliably.
        So a car's identity is established by OCR *before* it reaches its slot, and a car
        that was read there but has not yet linked to a slot is definitively IN TRANSIT —
        seconds away, in the same worker process.

        That is what makes the one remaining hop safe. Contrast the earlier
        acquisition-by-elimination, which anchored on the ANPR gate read minutes and a
        whole garage away, and bound DJS-7842 onto a Nissan Sunny that had been parked
        for hours. Here the identity is READ (not inferred), the window is seconds, and
        the anchor is a camera metres from the slot.
        """
        now = now or self._clock()
        window = float(
            getattr(self._matching_config, "ocr_transit_window_seconds", 120.0) or 120.0
        )
        out = []
        with self._lock:
            for s in self._sessions.values():
                if not s.plate or not getattr(s, "ocr_confirmed", False):
                    continue
                if getattr(s, "linked_slot", None):
                    continue  # already parked — accounted for, cannot also be arriving
                seen_at = getattr(s, "ocr_identified_at", None)
                if seen_at is None:
                    continue
                if (now - seen_at).total_seconds() <= window:
                    out.append((s, seen_at))
        return out

    def adopt_transit_identity(
        self, camera_id: str, track_id: int, session_id: str
    ) -> Optional[str]:
        """Give an anonymous track the identity of the one car in OCR transit."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.plate:
                return None
            self._track_session_map[(camera_id, track_id)] = session_id
            session.observing_tracks[camera_id] = track_id
            session.observing_scores[camera_id] = 1.0
            session.last_seen_at = self._clock()
            session.last_seen_camera = camera_id
            return session.plate

    def save_parked_reference(self, plate: str, crop_bgr, camera_id: str) -> bool:
        """Teach the gallery this car's PARKED POSE, on an OCR-VERIFIED identity.

        This is the reference the system has never been able to get, and the reason it
        could never recognise a parked car. Every gallery reference until now came from
        the GATE (ANPR / CAM-23 / CAM-03) — front-on, close, at the barrier. A car in a
        slot is seen small, oblique, from above. Matching one against the other is the
        cross-view problem, and it does not merely score low, it scores WRONG: measured
        2026-07-11, a car scored 0.583 against its own gate references while a DIFFERENT
        car scored 0.634.

        We could not capture the parked pose before because doing so requires knowing
        WHO the car is, and appearance could not tell us — the chicken-and-egg that made
        the whole afternoon circular. OCR breaks it: the plate is READ off the car, so
        the identity is evidence, not inference, and it is safe to learn from.

        Once this reference is on disk, the same car parking again is a SAME-VIEW match
        (self-gallery rank-1 0.976 in the identity-disjoint eval) rather than a cross-view
        one (0.736) — which is what will let a side-on slot like B1_CRO, whose camera can
        never see a plate, be recognised on the car's second visit.

        Deliberately bypasses the accumulate path's occupied-slot guard: that guard exists
        to stop a MIS-BOUND car poisoning a gallery, and there is no mis-binding here.
        """
        if not plate or crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return False
        store = getattr(self, "gallery_store", None)
        if store is None:
            return False
        # OCR proves WHOSE car this is. It does not prove the box contains one car —
        # a detector box spanning this car and its neighbour still reads the plate.
        if not is_plausible_car_crop(crop_bgr):
            logger.warning(
                "[ocr-id] Refused mis-shaped parked-pose ref for %s: %s is not one car.",
                plate,
                "x".join(str(d) for d in crop_bgr.shape[1::-1]),
            )
            return False
        try:
            feature = self.reid_matcher.extract_feature(crop_bgr)
            if feature is None:
                return False
            store.save_ref(
                plate,
                crop_bgr,
                feature,
                quality=1.0,
                camera_id=camera_id,
                gate_only=False,
            )
        except Exception as exc:
            logger.warning("[ocr-id] parked-pose save failed for %s: %r", plate, exc)
            return False

        with self._lock:
            session = next(
                (s for s in self._sessions.values() if s.plate == plate), None
            )
            if session is not None and feature is not None:
                session.reference_feature_vectors.append(feature)
                self._sync_reference_cameras(session)
                # _sync_reference_cameras only PADS with "", so the ref would score as
                # an unknown (secondary) camera until a restart reloaded it from disk —
                # where save_ref did record camera_id. The slot-pose scorer keys off this
                # tag to recognise a same-view reference, so it must be set here.
                session.reference_source_cameras[-1] = camera_id or ""
                self._gallery_index_upsert(session)

        logger.info(
            "[ocr-id] gallery LEARNED the parked pose of %s from %s — next time this "
            "car parks it can be recognised by appearance, not just by reading it",
            plate, camera_id,
        )
        return True

    def try_ocr_identify_track(
        self, camera_id: str, track_id: int, crop_bgr
    ) -> Optional[str]:
        """READ the plate off a still-moving car and bind its session to this track.

        This is what makes side-on slots solvable. CAM-21 frames B1_CRO in pure profile,
        so a car parked there NEVER shows a plate — but it drove up the aisle to get
        there, front or rear toward the camera, and the plate was plainly visible then.
        Identify the track on the approach and the plate is already known by the time the
        car parks; the existing slot-linking then carries it into current_plate with no
        appearance guessing anywhere in the chain.

        Same contract as everywhere else: bind only what was READ, and only when exactly
        one car inside can explain the read. Otherwise leave it anonymous.

        SYNCHRONOUS PATH. Composed from the same three phases the async path uses
        (``plan_track_ocr`` -> ``read_slot_plate`` -> ``confirm_track_ocr``) so there is
        exactly one implementation of the logic and the two paths cannot drift.
        """
        plan = self.plan_track_ocr(camera_id, track_id, crop_bgr)
        if plan is None:
            return None
        ocr_text, _conf = self.read_slot_plate(crop_bgr, plan.allow_retry)
        return self.confirm_track_ocr(plan, crop_bgr, ocr_text)

    def plan_track_ocr(
        self, camera_id: str, track_id: int, crop_bgr, *, allow_retry: bool = True
    ) -> Optional["TrackOcrPlan"]:
        """Phase 1 (MAIN THREAD): decide whether an approach read is possible at all.

        Returns ``None`` when there is no OCR plugin or no crop — i.e. when phase 2
        would be pure waste. Unlike ``plan_slot_ocr`` this does NO ReID work: the
        approach path ranks after the read (see :class:`TrackOcrPlan`), so phase 1 is
        deliberately trivial and nothing expensive is left on the caller's thread.

        ``allow_retry`` defaults to True to preserve the historical behaviour of this
        path exactly (``ocr.read`` itself defaults to True and the old inline call passed
        no override). Worth revisiting separately: the enlarged pass costs ~670ms and the
        approach path sees the same car over many frames, so a second look may well be
        cheaper than an upscaled re-read of this one — but that is a behaviour change to
        measure on its own, not to smuggle into a threading change.
        """
        ocr = getattr(self._match_decision, "plate_ocr", None)
        if ocr is None or not hasattr(ocr, "read") or crop_bgr is None:
            return None
        return TrackOcrPlan(
            camera_id=camera_id, track_id=track_id, allow_retry=allow_retry
        )

    def confirm_track_ocr(
        self, plan: "TrackOcrPlan", crop_bgr, ocr_text: str
    ) -> Optional[str]:
        """Phase 3 (MAIN THREAD): fold the approach read back in — narrow, confirm, bind.

        Main-thread-only for two reasons: ``reid_shortlist`` uses the ReID matcher
        (shared OpenVINO infer request), and the bind mutates ``_track_session_map`` and
        live ``VehicleSession`` fields, which the registry does not synchronize per
        transaction. Returns the plate only when exactly one car inside explains the
        read; otherwise ``None`` and the track stays anonymous.
        """
        camera_id, track_id = plan.camera_id, plan.track_id
        if not ocr_text:
            return None

        # ReID narrows, OCR confirms — same two-witness rule as the slot path.
        candidates = self.reid_shortlist(crop_bgr) or self.plates_inside()
        plate = confirm_plate(ocr_text, candidates)
        if not plate:
            return None

        with self._lock:
            session = next(
                (s for s in self._sessions.values() if s.plate == plate), None
            )
            if session is None:
                return None
            self._track_session_map[(camera_id, track_id)] = session.session_id
            session.observing_tracks[camera_id] = track_id
            session.observing_scores[camera_id] = 1.0  # read, not inferred
            session.last_seen_at = self._clock()
            session.last_seen_camera = camera_id
            session.ocr_confirmed = True
            # Anchors the transit hop: this car's plate was READ at this moment, so
            # until it parks it is definitively in transit toward a slot.
            session.ocr_identified_at = self._clock()
        logger.info(
            "[ocr-id] track (%s, %s) read=%r -> %s (bound by evidence)",
            camera_id, track_id, ocr_text, plate,
        )
        return plate

    def bind_plate_to_slot(
        self,
        slot_id: str,
        plate: str,
        camera_id: str,
        floor: Optional[str] = None,
        *,
        source: str = "ocr",
    ) -> bool:
        """Park the session carrying ``plate`` into ``slot_id``.

        ``source="ocr"`` means the plate was READ off the car — evidence, so the session
        is marked ``ocr_confirmed`` and downstream may trust it (the lock gate fires, and
        the parked pose may be taught to the gallery).

        ``source="reid_solo"`` means it was INFERRED from appearance because no plate is
        readable at this slot. That must never masquerade as a read: ``ocr_confirmed``
        stays False, so the lock gate does not fire and ``save_parked_reference`` will not
        learn from it. A wrong solo bind would otherwise poison the very gallery it was
        inferred from.
        """
        if not slot_id or not plate:
            return False
        with self._lock:
            session = next(
                (s for s in self._sessions.values() if s.plate == plate), None
            )
            if session is None:
                return False

            # If this car was parked somewhere else, release that slot first —
            # a car cannot occupy two slots, and a stale link would keep the old
            # slot showing this plate forever.
            #
            # Releasing it in MEMORY is not enough: parking_slots keeps the old row, and
            # /api/slots then reports the same car in two places. Record the release so
            # the engine can null the DB row on its next sync tick.
            for other_slot, parked in list(self._parked.items()):
                if other_slot != slot_id and parked is session:
                    self._parked.pop(other_slot, None)
                    self._locked_slots.discard(other_slot)
                    if not hasattr(self, "_released_slots"):
                        self._released_slots = set()
                    self._released_slots.add(other_slot)

            session.status = "parked"
            session.linked_slot = slot_id
            session.linked_camera = camera_id
            if floor:
                session.linked_floor = floor
            session.linked_at = self._clock()
            read = source == "ocr"
            if read:
                session.ocr_confirmed = True  # it was READ, not guessed
            self._parked[slot_id] = session

            # A slot CLAIMS a car whatever named it — that is what mutual exclusion reads,
            # and one car still cannot be in two slots. Leaving solo binds out of
            # _locked_slots (on the reasoning that only a READ plate should "freeze")
            # conflated two different ideas and broke the contract: ERS-7949 ended up bound
            # to B17 AND B19 at once, because B17 could not see that B19 already held it.
            #
            # "Correctable by OCR" is a SEPARATE property, and it lives on the state
            # machine (bind_identity(lock=...)) and on parking_slots.plate_locked — not
            # here. A solo bind is still correctable; it is just no longer invisible.
            self._locked_slots.add(slot_id)
            self._gallery_index_upsert(session)
            return True

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
            # Slot camera: read the full car crop, not the gate's bottom-centre ROI.
            ocr_text, ocr_conf = ocr.read(crop_bgr, apply_plate_roi=False)
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

        Boot-restore dedup: the persisted DB may hold the same plate on two slots
        (a missed exit left a stale binding). One plate may occupy at most one
        slot, so we keep the most-recent binding and evict the older — even a
        locked one. This registry is a single shared instance across all cameras,
        so enforcing it here is sufficient.
        """
        released_slots: List[str] = []
        skip_restore = False
        with self._lock:
            if slot_id in self._parked:
                return

            if plate:
                rivals = [
                    (oslot, other)
                    for oslot, other in self._parked.items()
                    if oslot != slot_id and other.plate == plate
                ]
                if rivals:
                    # Incoming wins only when STRICTLY newer than every rival; on
                    # a tie the already-restored rival (first restore) wins, so
                    # restore order is deterministic.
                    incoming_newest = all(
                        (other.linked_at or other.first_seen_at or timestamp) < timestamp
                        for _, other in rivals
                    )
                    if incoming_newest:
                        for _, other in rivals:
                            released = self._close_session(other, reason="restore_dedup_stale")
                            if released is not None:
                                released_slots.append(released)
                    else:
                        # A rival is newer: keep the single newest rival, evict
                        # the rest, and skip restoring the stale incoming row.
                        skip_restore = True
                        winner = max(
                            rivals,
                            key=lambda pair: (
                                pair[1].linked_at or pair[1].first_seen_at or timestamp
                            ),
                        )[1]
                        for _, other in rivals:
                            if other is winner:
                                continue
                            released = self._close_session(other, reason="restore_dedup_stale")
                            if released is not None:
                                released_slots.append(released)

            if skip_restore:
                # The incoming row lost the tie-break — clear its own DB binding
                # so a later restart does not resurrect it.
                released_slots.append(slot_id)
            else:
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

        # DB clear off-lock — never hold self._lock across DB I/O.
        for sid in released_slots:
            self._clear_slot_db_binding(sid)

    def _close_session(self, session: VehicleSession, reason: str) -> Optional[str]:
        """Tear one session down completely and move it to history.

        Sets ``status``/``exit_reason``, drops all cross-camera track bindings,
        removes the session from ``_sessions``, the per-area index and the FAISS
        gallery, and — when the session owns a parking slot — pops that slot from
        ``_parked`` and releases any plate-lock on it. Returns the released
        ``slot_id`` (or ``None``) so the caller can clear that slot's DB row
        *outside* the registry lock: this method performs **no** DB I/O.

        Must be called under ``self._lock`` (RLock — safe to nest). Shared by the
        normal exit path (``_handle_exit``) and the ANPR eviction chokepoint
        (``_claim_plate_globally``); the distinct log line is left to callers.
        """
        session.status = "exited"
        session.exit_reason = reason

        # Clean up track bindings across all cameras
        for obs_cam, obs_tid in list(session.observing_tracks.items()):
            self._track_session_map.pop((obs_cam, obs_tid), None)
            self._track_last_seen.pop((obs_cam, obs_tid), None)
            self._track_first_seen.pop((obs_cam, obs_tid), None)
        session.observing_tracks.clear()
        session.observing_scores.clear()
        session.owner_camera = None

        # Legacy cleanup
        if session.last_seen_camera and session.last_seen_track_id is not None:
            self._track_session_map.pop(
                (session.last_seen_camera, session.last_seen_track_id),
                None,
            )

        # Slot teardown — a parked session releases its slot and any plate-lock.
        # Only release the slot this session actually owns (guards against a
        # stale linked_slot pointing at a slot now held by another session).
        released_slot: Optional[str] = None
        slot_id = session.linked_slot
        if slot_id is not None and self._parked.get(slot_id) is session:
            self._parked.pop(slot_id, None)
            self._locked_slots.discard(slot_id)
            released_slot = slot_id
        # A closed session must never dangle a slot pointer (it would otherwise
        # ghost-bind on a later frame or misreport a location). Mirrors the
        # link-clearing in try_link_to_slot's stale-release path.
        session.linked_slot = None
        session.linked_slot_name = None
        session.linked_camera = None
        session.linked_floor = None
        session.linked_zone_id = None
        session.linked_zone_name = None
        session.linked_at = None

        self._sessions.pop(session.session_id, None)
        # Zoning — drop the closed car from its area bucket so the bounded
        # (per-area) candidate pool never matches a car that has left.
        self._drop_session_from_area_index(session)
        # Phase 3 / T3.2 — drop from the FAISS gallery so a future query never
        # matches against a car that has already left the facility.
        self._gallery_index_remove(session.session_id)
        self._history.append(session)
        return released_slot

    def _clear_slot_db_binding(self, slot_id: str) -> None:
        """Best-effort: null the persisted plate binding on one ``parking_slots``
        row after an ANPR eviction, so a restart does not resurrect the stale
        plate on that slot (see ``restore_parked_binding`` / _load_camera_db_state).

        No-op when no ``db_manager`` is wired (VA-local/test env), mirroring
        ``is_plate_inside``'s fail-open DI. MUST be called **outside**
        ``self._lock`` — it opens a DB session, and DB I/O must never run under
        the registry lock.
        """
        db_manager = getattr(self, "db_manager", None)
        if db_manager is None or not slot_id:
            return
        try:
            from src.services.slot_status_service import clear_slot_plate_binding

            session = db_manager.SessionLocal()
            try:
                clear_slot_plate_binding(session, slot_id)
            finally:
                session.close()
        except Exception as exc:
            logger.warning(
                "[REGISTRY] Failed to clear DB plate binding for slot %s: %r",
                slot_id, exc,
            )

    def _reconcile_duplicate_identity(self, session, now) -> List[str]:
        """Collapse a near-identical DUPLICATE identity created for ONE physical
        car that the ANPR misread as two different plates ("one car entered with
        two plates"). Complements ``_claim_plate_globally`` (which enforces one
        session per *plate string*) by enforcing one session per *car appearance*.

        Called under ``self._lock`` from the B1 confirmation path. Conservative
        and OFF by default (``identity_reconcile_min_similarity == 0``): fires
        only when another CONFIRMED, not-yet-parked, DIFFERENT-plate session has a
        max ReID similarity >= the (high) floor AND entered within one gate-dwell
        window. The high floor sits well above the same-car mean, so only
        near-identical gate views trigger and two similar-but-different cars are
        never merged; parked/locked identities are never touched. Keeps THIS
        (newer) session, closes the older duplicate, and returns the slot_ids it
        released (for off-lock DB clearing by the caller)."""
        cfg = self._matching_config
        floor = float(getattr(cfg, "identity_reconcile_min_similarity", 0.0))
        if floor <= 0.0 or session is None or not getattr(session, "plate", None):
            return []
        window = float(getattr(cfg, "identity_reconcile_window_seconds", 60.0))
        my_vecs = [
            v
            for v in ([session.feature_vector] + list(session.reference_feature_vectors or []))
            if v is not None
        ]
        if not my_vecs:
            return []
        my_start = getattr(session, "first_seen_at", None)

        best = None
        best_sim = 0.0
        for other in list(self._sessions.values()):
            if other.session_id == session.session_id:
                continue
            # Only a still-in-gate confirmed duplicate — never a parked/locked
            # identity, whose slot binding is authoritative and costly to undo.
            if other.status != "confirmed" or getattr(other, "linked_slot", None):
                continue
            if not other.plate or other.plate == session.plate:
                continue  # same-plate collapse is _claim_plate_globally's job
            if my_start is not None:
                other_start = getattr(other, "first_seen_at", None)
                if (
                    other_start is not None
                    and abs((my_start - other_start).total_seconds()) > window
                ):
                    continue
            other_vecs = [
                v
                for v in ([other.feature_vector] + list(other.reference_feature_vectors or []))
                if v is not None
            ]
            if not other_vecs:
                continue
            sim = max(
                self.reid_matcher.compute_similarity(a, b)
                for a in my_vecs
                for b in other_vecs
            )
            if sim >= floor and sim > best_sim:
                best_sim = sim
                best = other
        if best is None:
            return []

        logger.warning(
            "[RECONCILE] Session %s (plate=%s) is the same physical car as %s "
            "(plate=%s) at ReID %.3f — collapsing the duplicate so the car keeps "
            "one identity (ANPR likely misread one car as two plates).",
            session.session_id,
            session.plate,
            best.session_id,
            best.plate,
            best_sim,
        )
        released = self._close_session(best, reason="duplicate_identity_reconciled")
        return [released] if released else []

    def _claim_plate_globally(
        self,
        plate: str,
        keep_session_id: Optional[str],
        reason: str = "plate_reclaimed_elsewhere",
        timestamp: Optional[datetime] = None,  # reserved; eviction is not time-stamped
    ) -> List[str]:
        """Enforce 'one plate = one active session': force-close every OTHER
        active (confirmed/parked) session for ``plate`` and return the parking
        ``slot_id``s they released.

        This is the single chokepoint that makes a genuine ANPR gate read the
        authority for a plate — a real re-entry evicts a stale session left open
        by a missed exit-camera detection, even a plate-locked one. Only ANPR
        callers invoke it; the ReID / OCR in-frame paths keep their locked-slot
        refusal untouched.

        Purely in-memory. It does NOT touch the DB — callers clear the returned
        slots via ``_clear_slot_db_binding`` AFTER releasing ``self._lock``, so no
        DB I/O ever runs under the lock. It does NOT ``stamp_exit`` (the car has
        not left the facility, so the gallery-TTL clock must not start) and does
        NOT purge pending events plate-wide (that would destroy the current
        re-entry's own pending event; the 30 s ``_cleanup_stale_data`` GC ages
        out the stale one). Must be called under ``self._lock`` (RLock).
        """
        if not plate:
            return []
        # Snapshot the victims first — _close_session mutates self._sessions.
        victims = [
            s
            for s in self._sessions.values()
            if s.plate == plate
            and s.status in ("confirmed", "parked")
            and s.session_id != keep_session_id
        ]
        released: List[str] = []
        for session in victims:
            slot_id = self._close_session(session, reason=reason)
            logger.warning(
                "[REGISTRY] Force-evicted stale session %s for plate %s "
                "(slot=%s, reason=%s) — likely a missed exit-camera detection",
                session.session_id, plate, slot_id, reason,
            )
            if slot_id is not None:
                released.append(slot_id)
        return released

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
        # Re-entry grace: stamp the exit time for BOTH callers of _handle_exit
        # (the ANPR exit branch AND the exit-janitor). A plain missed-exit keeps
        # entry_at <= exit_at (predicate False -> still evicted); only a NEW ANPR
        # entry after this stamp makes _has_fresh_reentry True.
        if plate:
            self._last_anpr_exit_at[plate] = timestamp
        # Drop this plate's accumulation-throttle entries so the map doesn't
        # grow unbounded over a long-running process.
        for key in [k for k in self._gallery_last_add if k[0] == plate]:
            self._gallery_last_add.pop(key, None)

        with self._lock:
            # 1. Find and close any active sessions for this plate (parked or driving).
            sessions_to_remove = [
                s for s in self._sessions.values() if s.plate == plate
            ]
            # Also include any parked session keyed under this plate whose object
            # has already left _sessions, so its slot still gets torn down.
            seen_ids = {s.session_id for s in sessions_to_remove}
            for sess in self._parked.values():
                if sess.plate == plate and sess.session_id not in seen_ids:
                    sessions_to_remove.append(sess)
                    seen_ids.add(sess.session_id)

            # _close_session owns the per-session + slot/plate-lock teardown.
            for session in sessions_to_remove:
                self._close_session(session, reason="exit_event")
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
