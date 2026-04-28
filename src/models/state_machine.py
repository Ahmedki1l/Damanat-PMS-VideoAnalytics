"""
state_machine.py — Per-slot state machine with debounce logic.

State transitions:
  VACANT → ENTERING    : Vehicle detected in slot (1 frame).
  ENTERING → OCCUPIED  : Vehicle present for confirm_enter_frames consecutive frames.
  ENTERING → VACANT    : Vehicle disappears before confirmation.
  OCCUPIED → LEAVING   : Vehicle not detected in slot (1 frame).
  LEAVING → VACANT     : Vehicle absent for confirm_leave_frames consecutive frames.
  LEAVING → OCCUPIED   : Vehicle re-detected before confirmation.

Debounce prevents flicker from:
  - Momentary detection gaps (occlusion, tracker loss).
  - Vehicles passing by a slot without parking.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class SlotState(Enum):
    """Possible states for a parking slot."""
    VACANT = "VACANT"
    ENTERING = "ENTERING"
    OCCUPIED = "OCCUPIED"
    LEAVING = "LEAVING"


@dataclass
class SlotEvent:
    """
    Structured event emitted on state transitions.

    Attributes:
        event_type: One of vehicle_entering, vehicle_parked,
                    vehicle_leaving, slot_vacant.
        slot_id: Which slot this event relates to.
        track_id: The vehicle track ID (None for slot_vacant).
        timestamp: ISO-format timestamp of the event.
        camera_id: Which camera observed this event.
        floor: Which floor (B1, B2, etc.).
        plate: Recognized license plate (if available).
        is_alert: Whether this event is a security/violation alert.
        severity: Severity level (info, warning, critical).
    """
    event_type: str
    slot_id: str
    track_id: Optional[int]
    timestamp: str
    camera_id: str = ""
    floor: str = ""
    plate_number: str = ""
    is_alert: bool = False
    severity: str = "info"
    slot_name: str = ""
    zone_id: str = ""
    zone_name: str = ""
    snapshot_url: str = ""
    snapshot_path: str = ""
    alert_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary with specific field order."""
        return {
            "is_alert": self.is_alert,
            "severity": self.severity,
            "alert_type": self.event_type,
            "slot_id": self.slot_id,
            "slot_name": self.slot_name or self.slot_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "camera_id": self.camera_id,
            "floor": self.floor,
            "plate_number": self.plate_number,
            "snapshot_url": self.snapshot_url,
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
        }


class SlotStateMachine:
    """
    State machine for a single parking slot.

    Tracks the current state, the assigned vehicle track ID,
    and uses frame counters for debounced transitions.
    """

    def __init__(
        self,
        slot_id: str,
        confirm_enter_frames: int = 5,
        confirm_leave_frames: int = 8,
        is_violation_zone: bool = False,
        initial_state: SlotState = SlotState.VACANT,
    ):
        """
        Args:
            slot_id: Unique identifier for this slot.
            confirm_enter_frames: Consecutive frames with a vehicle
                                  before confirming OCCUPIED.
            confirm_leave_frames: Consecutive frames without a vehicle
                                  before confirming VACANT.
            is_violation_zone: Whether this slot is a restricted/intrusion zone
                               (loaded from the database, not inferred from name).
            initial_state: Starting state (useful for syncing with DB on startup).
        """
        self.slot_id = slot_id
        self.confirm_enter_frames = confirm_enter_frames
        self.confirm_leave_frames = confirm_leave_frames

        # Current state
        self.state: SlotState = initial_state
        self.assigned_track_id: Optional[int] = None
        self.latest_detection_bbox: Optional[tuple[float, float, float, float]] = None
        self.plate_number: str = ""
        self.snapshot_url: str = ""
        self.last_update_time: str = datetime.now().isoformat()

        # Store violation zone status from DB
        self._is_violation_zone: bool = is_violation_zone

        # Debounce counters
        self._enter_counter: int = 0
        self._leave_counter: int = 0

    @property
    def is_occupied(self) -> bool:
        """True if the slot is confirmed occupied."""
        return self.state == SlotState.OCCUPIED

    @property
    def is_violation_zone(self) -> bool:
        """True if the slot is a restricted/intrusion zone (set from database)."""
        return self._is_violation_zone

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the slot's current status."""
        return {
            "slot_id": self.slot_id,
            "state": self.state.value,
            "assigned_track_id": self.assigned_track_id,
            "latest_detection_bbox": self.latest_detection_bbox,
            "occupied": self.is_occupied,
            "last_update_time": self.last_update_time,
            "is_violation_zone": self.is_violation_zone,
            "plate_number": self.plate_number,
            "snapshot_url": self.snapshot_url,
        }

    def bind_identity(self, plate_number: Optional[str], snapshot_url: str = "") -> None:
        """Persist the confirmed identity for this slot until vacancy is confirmed."""
        if plate_number:
            self.plate_number = plate_number
        if snapshot_url:
            self.snapshot_url = snapshot_url

    def clear_identity(self) -> None:
        """Clear the persisted identity after the slot is confirmed vacant."""
        self.plate_number = ""
        self.snapshot_url = ""
        self.latest_detection_bbox = None

    def update(
        self,
        vehicle_present: bool,
        track_id: Optional[int] = None,
    ) -> List[SlotEvent]:
        """
        Process one frame of observation and return any triggered events.

        Args:
            vehicle_present: Whether a vehicle is currently detected in this slot.
            track_id: The track ID of the detected vehicle (None if absent).

        Returns:
            List of SlotEvent objects (empty if no transition occurred).
        """
        events: List[SlotEvent] = []
        now = datetime.now().isoformat()

        # ----- VACANT -----
        if self.state == SlotState.VACANT:
            if vehicle_present:
                # Start entering: switch to ENTERING, begin counting
                self.state = SlotState.ENTERING
                self.assigned_track_id = track_id
                self._enter_counter = 1
                self.last_update_time = now
                events.append(SlotEvent(
                    event_type="vehicle_entering",
                    slot_id=self.slot_id,
                    track_id=track_id,
                    timestamp=now,
                ))

        # ----- ENTERING -----
        elif self.state == SlotState.ENTERING:
            if vehicle_present:
                # Update track ID in case tracker reassigned
                self.assigned_track_id = track_id
                self._enter_counter += 1
                self.last_update_time = now

                if self._enter_counter >= self.confirm_enter_frames:
                    # Confirmed: vehicle is parked
                    self.state = SlotState.OCCUPIED
                    self._enter_counter = 0
                    events.append(SlotEvent(
                        event_type="vehicle_parked",
                        slot_id=self.slot_id,
                        track_id=track_id,
                        timestamp=now,
                    ))
            else:
                # Vehicle disappeared before confirmation — revert
                self.state = SlotState.VACANT
                self.assigned_track_id = None
                self._enter_counter = 0
                self.clear_identity()
                self.last_update_time = now

        # ----- OCCUPIED -----
        elif self.state == SlotState.OCCUPIED:
            if vehicle_present:
                # Still parked — update track ID (may change after re-track)
                self.assigned_track_id = track_id
                self.last_update_time = now
            else:
                # Vehicle not detected — start leaving
                self.state = SlotState.LEAVING
                self._leave_counter = 1
                self.last_update_time = now
                events.append(SlotEvent(
                    event_type="vehicle_leaving",
                    slot_id=self.slot_id,
                    track_id=self.assigned_track_id,
                    timestamp=now,
                ))

        # ----- LEAVING -----
        elif self.state == SlotState.LEAVING:
            if not vehicle_present:
                self._leave_counter += 1
                self.last_update_time = now

                if self._leave_counter >= self.confirm_leave_frames:
                    # Confirmed: slot is now vacant
                    old_track = self.assigned_track_id
                    self.state = SlotState.VACANT
                    self.assigned_track_id = None
                    self._leave_counter = 0
                    self.clear_identity()
                    events.append(SlotEvent(
                        event_type="slot_vacant",
                        slot_id=self.slot_id,
                        track_id=old_track,
                        timestamp=now,
                    ))
            else:
                # Vehicle re-detected — cancel leaving
                self.state = SlotState.OCCUPIED
                self.assigned_track_id = track_id
                self._leave_counter = 0
                self.last_update_time = now

        return events
