from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PendingANPREvent:
    """Plate received from ANPR, waiting for the real car body at Park_Entry."""

    event_id: str
    plate: str
    direction: str
    timestamp: datetime
    camera_id: Optional[str] = None

    status: str = "pending"  # pending, provisional, confirmed, expired, dropped
    candidate_id: Optional[str] = None
    session_id: Optional[str] = None

    # Optional metadata published by Phase-1 WS-B / WS-C plugins. Phase 0
    # writes never set these — they exist so the data flows without a
    # follow-up schema migration.
    detected_color: Optional[str] = None
    vehicle_type: Optional[str] = None


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
    snapshot_images: List[np.ndarray] = field(default_factory=list)
    snapshot_paths: List[str] = field(default_factory=list)
    feature_vectors: List[np.ndarray] = field(default_factory=list)

    color_hsv: Optional[Tuple[float, float, float]] = None
    color_hsv_values: List[Tuple[float, float, float]] = field(default_factory=list)

    status: str = "open"  # open, provisional, confirmed, expired, dropped
    bound_event_id: Optional[str] = None

    # "anpr_image"  → candidate was seeded from the image sent by pms_ai on /api/anpr/event
    # "zone_crop"   → candidate was created from a live Park_Entry / confirmation-zone frame
    source: str = "zone_crop"

    # Optional metadata published by Phase-1 plugins. Phase 0 leaves these
    # untouched; the fields exist so WS-B / WS-C / WS-D can persist their
    # predictions without an additional schema migration.
    color_class: Optional[str] = None
    color_class_conf: float = 0.0
    type_class: Optional[str] = None
    type_class_conf: float = 0.0
    ocr_plate: Optional[str] = None
    ocr_plate_conf: float = 0.0


@dataclass
class VehicleSession:
    """
    Final confirmed identity: linked to a plate or a unique ReID signature.
    """

    session_id: str
    plate: Optional[str] = None
    feature_vector: Optional[np.ndarray] = None

    first_seen_at: datetime = datetime.now()
    last_seen_at: datetime = datetime.now()
    last_seen_camera: str = ""
    last_seen_track_id: Optional[int] = None

    event_id: Optional[str] = None
    candidate_id: Optional[str] = None

    status: str = "confirmed"
    # Why the session was closed, recorded when it moves to _history. In-memory
    # only (no DB column). Values: "exit_event" (normal exit-camera detection),
    # "plate_reclaimed_elsewhere" (evicted by a fresh ANPR gate read of the same
    # plate — a likely missed exit), "restore_dedup_stale" (boot-restore dropped
    # an older duplicate binding for the same plate).
    exit_reason: Optional[str] = None

    linked_slot: Optional[str] = None
    linked_slot_name: Optional[str] = None
    linked_camera: Optional[str] = None
    linked_floor: Optional[str] = None
    linked_zone_id: Optional[str] = None
    linked_zone_name: Optional[str] = None
    linked_at: Optional[datetime] = None
    snapshot_path: Optional[str] = None
    reference_snapshot_paths: List[str] = field(default_factory=list)
    reference_feature_vectors: List[np.ndarray] = field(default_factory=list)
    
    gate_snapshot_paths: List[str] = field(default_factory=list)

    # True while the session's ONLY ReID reference is the wide ANPR gate image
    # (created by confirm_anpr_session_directly at the moment of the gate `entry`
    # event). The gate shot is a poor cross-camera reference — different
    # viewpoint/scale/lighting than the overhead parking cameras — so a session
    # in this state is EXCLUDED from match_global_session: it must not be able to
    # latch onto a car already parked inside before the real car reaches a floor.
    # Cleared once CAM-03 (B1_Entrence) attaches a genuine primary reference via
    # confirm_b1_entrance_by_plate, after which the session becomes matchable.
    gate_reference_only: bool = False

    # A car-cropped ANPR gate-image embedding held for delayed promotion. Stashed
    # at the gate `entry` event (confirm_anpr_session_directly) but kept OUT of
    # matching — like gate_reference_only — until CAM-03 confirms the plate. At
    # that point confirm_b1_entrance_by_plate appends it to
    # reference_feature_vectors: the car is anchored by then, so the extra frontal
    # ANPR view enriches the appearance profile without the entry-time swap risk.
    pending_anpr_vector: Optional[np.ndarray] = None

    # Maps camera_id → track_id for all cameras currently observing this car.
    # Enables multi-camera simultaneous identity display.
    observing_tracks: Dict[str, int] = field(default_factory=dict)

    # Per-camera latest ReID similarity score. Drives single-camera ownership:
    # the live observer with the highest score owns the identity (display + data).
    observing_scores: Dict[str, float] = field(default_factory=dict)
    # The camera currently designated as the sole owner of this car's identity.
    owner_camera: Optional[str] = None

    # --- Zoning / area state machine (foundation; inert until set by the
    # area state machine + bounded matcher). ``current_area`` is the area the
    # car is currently in (derived from its owner camera's area). ``area_state``
    # progresses IN_AREA → DEPARTING → IN_TRANSIT → ARRIVING → IN_AREA.
    # ``area_entered_at`` stamps the last area entry (transit-time gating).
    # ``departed_from_area`` records the area a DEPARTING/IN_TRANSIT car left,
    # read by the cross-area handoff matcher. Empty/default = un-zoned. ---
    current_area: str = ""
    area_state: str = "IN_AREA"
    area_entered_at: Optional[datetime] = None
    departed_from_area: str = ""

    new_pipeline_score: float = 0.0
    old_pipeline_score: float = 0.0

    # Optional Phase-1 metadata. Plugins (WS-B/C/D) may populate these on
    # confirmation; Phase 0 leaves them as defaults.
    color_class: Optional[str] = None
    type_class: Optional[str] = None
    ocr_plate_variants: List[str] = field(default_factory=list)

    # Plate-lock: set True once OCR has actually read this car's plate and it
    # agreed with ``self.plate`` (B1 confirm path, or the forced-OCR dead-zone
    # pass). The lock gate reads this directly — it is the OCR arm of the
    # "ReID >= lock_confidence OR ocr_confirmed" freeze condition, and does not
    # rely on the synthetic Decision built for the voter (which carries no OCR).
    ocr_confirmed: bool = False

    @property
    def display_id(self) -> str:
        """Returns plate if available, otherwise session ID."""
        return self.plate if self.plate else f"ID-{self.session_id[-6:]}"
