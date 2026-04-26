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

    # Maps camera_id → track_id for all cameras currently observing this car.
    # Enables multi-camera simultaneous identity display.
    observing_tracks: Dict[str, int] = field(default_factory=dict)

    new_pipeline_score: float = 0.0
    old_pipeline_score: float = 0.0

    @property
    def display_id(self) -> str:
        """Returns plate if available, otherwise session ID."""
        return self.plate if self.plate else f"ID-{self.session_id[-6:]}"
