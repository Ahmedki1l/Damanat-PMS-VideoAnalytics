"""
state_machine.py — Per-slot state machine with debounce logic.

State transitions:
  VACANT → ENTERING    : Vehicle detected in slot.
  ENTERING → OCCUPIED  : Vehicle present for confirm_enter_frames consecutive frames.
  ENTERING → VACANT    : Absence persists through the cancellation gate.
  OCCUPIED → LEAVING   : Absence persists through the leave-start gate.
  LEAVING → VACANT     : Vehicle absent for confirm_leave_frames consecutive frames.
  LEAVING → OCCUPIED   : Vehicle re-detected before confirmation.

Debounce prevents flicker from:
  - Momentary detection gaps (occlusion, tracker loss).
  - Vehicles passing by a slot without parking.
"""

import logging
import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Occupancy-latency instrumentation. Every line is tagged [SLOTLAT] so the whole
# distribution can be pulled out of a running system with a single grep:
#
#   grep -h '\[SLOTLAT\]' logs/va_*.out.log
#
# CONFIRM lines carry the frames and the WALL-CLOCK seconds a flip actually took;
# RESET lines are the flicker events that make a flip take longer than its nominal
# debounce (the counters demand CONSECUTIVE frames, so one contrary frame throws the
# accumulated count away). Comparing CONFIRM elapsed_s against the nominal
# frames/fps tells you whether slow flips are the debounce itself or flicker.
_SLOTLAT = "[SLOTLAT]"


class SlotState(Enum):
    """Possible states for a parking slot."""
    VACANT = "VACANT"
    ENTERING = "ENTERING"
    OCCUPIED = "OCCUPIED"
    LEAVING = "LEAVING"


class SlotObservationKind(Enum):
    """The occupancy evidence produced by one inference opportunity."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SlotObservation:
    """One fresh, ordered observation for a slot.

    ``UNKNOWN`` is deliberately distinct from ``ABSENT``. A skipped frame,
    inference failure, stale completion, or reconnect gap carries no occupancy
    evidence and therefore cannot move a slot toward vacancy.
    """

    kind: SlotObservationKind
    observed_at: float
    timestamp: str
    track_id: Optional[int] = None


@dataclass(frozen=True)
class SlotObservationPolicy:
    """Configuration for the optional elapsed-time occupancy policy."""

    mode: str = "legacy"
    enter_seconds: float = 3.0
    leave_seconds: float = 20.0
    enter_min_observations: int = 2
    leave_min_observations: int = 3
    enter_cancel_seconds: float = 1.0
    enter_cancel_min_observations: int = 2
    leave_start_seconds: float = 1.0
    leave_start_min_observations: int = 2
    max_known_gap_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "shadow", "time"}:
            raise ValueError(
                "slot observation mode must be legacy, shadow, or time"
            )
        if (
            not math.isfinite(self.enter_seconds)
            or not math.isfinite(self.leave_seconds)
            or not math.isfinite(self.enter_cancel_seconds)
            or not math.isfinite(self.leave_start_seconds)
            or self.enter_seconds < 0
            or self.leave_seconds < 0
            or self.enter_cancel_seconds < 0
            or self.leave_start_seconds < 0
        ):
            raise ValueError("slot hysteresis durations cannot be negative")
        if any(
            count < 1
            for count in (
                self.enter_min_observations,
                self.leave_min_observations,
                self.enter_cancel_min_observations,
                self.leave_start_min_observations,
            )
        ):
            raise ValueError("slot hysteresis observation counts must be positive")
        if (
            self.leave_start_seconds > self.leave_seconds
            or self.leave_start_min_observations > self.leave_min_observations
        ):
            raise ValueError("leave-start gate cannot exceed vacancy confirmation")
        if (
            not math.isfinite(self.max_known_gap_seconds)
            or self.max_known_gap_seconds <= 0
        ):
            raise ValueError("max_known_gap_seconds must be positive")


class _TimedTransition(Enum):
    ENTERING = "vehicle_entering"
    PARKED = "vehicle_parked"
    LEAVING = "vehicle_leaving"
    VACANT = "slot_vacant"


class _TimedHysteresis:
    """Elapsed-time occupancy reducer used by shadow and authoritative modes."""

    def __init__(
        self,
        policy: SlotObservationPolicy,
        initial_state: SlotState,
    ) -> None:
        self.policy = policy
        self.state = initial_state
        self.assigned_track_id: Optional[int] = None
        self._run_started_at: Optional[float] = None
        self._last_known_at: Optional[float] = None
        self._evidence_count = 0
        self._opposite_run_started_at: Optional[float] = None
        self._opposite_evidence_count = 0

    def observe(self, observation: SlotObservation) -> List[_TimedTransition]:
        if observation.kind == SlotObservationKind.UNKNOWN:
            return []

        if self._known_gap_exceeded(observation.observed_at):
            # A long UNKNOWN interval breaks every pending proof, regardless of
            # whether the first fresh signal agrees with the old run. Keeping a
            # stale positive run across an intervening ABSENT would otherwise
            # let one later PRESENT immediately park the slot.
            self._clear_all_runs()
        transitions = self._advance_known(observation)
        self._last_known_at = observation.observed_at
        return transitions

    def _advance_known(
        self, observation: SlotObservation
    ) -> List[_TimedTransition]:
        present = observation.kind == SlotObservationKind.PRESENT
        if self.state == SlotState.VACANT:
            return self._from_vacant(observation, present)
        if self.state == SlotState.ENTERING:
            return self._from_entering(observation, present)
        if self.state == SlotState.OCCUPIED:
            return self._from_occupied(observation, present)
        return self._from_leaving(observation, present)

    def _from_vacant(
        self, observation: SlotObservation, present: bool
    ) -> List[_TimedTransition]:
        if not present:
            return []
        self.state = SlotState.ENTERING
        self.assigned_track_id = observation.track_id
        self._start_run(observation.observed_at)
        transitions = [_TimedTransition.ENTERING]
        if self._enter_ready(observation.observed_at):
            self.state = SlotState.OCCUPIED
            self._clear_run()
            transitions.append(_TimedTransition.PARKED)
        return transitions

    def _from_entering(
        self, observation: SlotObservation, present: bool
    ) -> List[_TimedTransition]:
        if not present:
            self._continue_opposite_run(observation.observed_at)
            if not self._run_values_ready(
                self._opposite_run_started_at,
                self._opposite_evidence_count,
                observation.observed_at,
                self.policy.enter_cancel_seconds,
                self.policy.enter_cancel_min_observations,
            ):
                return []
            self.state = SlotState.VACANT
            self.assigned_track_id = None
            self._clear_all_runs()
            return []

        self._clear_opposite_run()
        self.assigned_track_id = observation.track_id
        self._continue_run(observation.observed_at)
        if not self._enter_ready(observation.observed_at):
            return []
        self.state = SlotState.OCCUPIED
        self._clear_run()
        return [_TimedTransition.PARKED]

    def _from_occupied(
        self, observation: SlotObservation, present: bool
    ) -> List[_TimedTransition]:
        if present:
            self.assigned_track_id = observation.track_id
            self._clear_run()
            return []
        if self._run_started_at is None:
            self._start_run(observation.observed_at)
        else:
            self._continue_run(observation.observed_at)
        if not self._run_values_ready(
            self._run_started_at,
            self._evidence_count,
            observation.observed_at,
            self.policy.leave_start_seconds,
            self.policy.leave_start_min_observations,
        ):
            return []
        self.state = SlotState.LEAVING
        transitions = [_TimedTransition.LEAVING]
        if self._leave_ready(observation.observed_at):
            self.state = SlotState.VACANT
            self.assigned_track_id = None
            self._clear_run()
            transitions.append(_TimedTransition.VACANT)
        return transitions

    def _from_leaving(
        self, observation: SlotObservation, present: bool
    ) -> List[_TimedTransition]:
        if present:
            self.state = SlotState.OCCUPIED
            self.assigned_track_id = observation.track_id
            self._clear_all_runs()
            return []

        self._continue_run(observation.observed_at)
        if not self._leave_ready(observation.observed_at):
            return []
        self.state = SlotState.VACANT
        self.assigned_track_id = None
        self._clear_run()
        return [_TimedTransition.VACANT]

    def _start_run(self, observed_at: float) -> None:
        self._run_started_at = observed_at
        self._evidence_count = 1

    def _continue_run(self, observed_at: float) -> None:
        if self._run_started_at is None or self._known_gap_exceeded(observed_at):
            self._start_run(observed_at)
            return
        self._evidence_count += 1

    def _continue_opposite_run(self, observed_at: float) -> None:
        if (
            self._opposite_run_started_at is None
            or self._known_gap_exceeded(observed_at)
        ):
            self._opposite_run_started_at = observed_at
            self._opposite_evidence_count = 1
            return
        self._opposite_evidence_count += 1

    def _known_gap_exceeded(self, observed_at: float) -> bool:
        if self._last_known_at is None:
            return False
        return (
            observed_at - self._last_known_at
            > self.policy.max_known_gap_seconds
        )

    def _enter_ready(self, observed_at: float) -> bool:
        return self._run_ready(
            observed_at,
            self.policy.enter_seconds,
            self.policy.enter_min_observations,
        )

    def _leave_ready(self, observed_at: float) -> bool:
        return self._run_ready(
            observed_at,
            self.policy.leave_seconds,
            self.policy.leave_min_observations,
        )

    def _run_ready(
        self, observed_at: float, required_seconds: float, required_count: int
    ) -> bool:
        return self._run_values_ready(
            self._run_started_at,
            self._evidence_count,
            observed_at,
            required_seconds,
            required_count,
        )

    @staticmethod
    def _run_values_ready(
        run_started_at: Optional[float],
        evidence_count: int,
        observed_at: float,
        required_seconds: float,
        required_count: int,
    ) -> bool:
        if run_started_at is None:
            return False
        elapsed = observed_at - run_started_at
        return evidence_count >= required_count and elapsed >= required_seconds

    def _clear_run(self) -> None:
        self._run_started_at = None
        self._evidence_count = 0

    def _clear_opposite_run(self) -> None:
        self._opposite_run_started_at = None
        self._opposite_evidence_count = 0

    def _clear_all_runs(self) -> None:
        self._clear_run()
        self._clear_opposite_run()


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
        observation_policy: Optional[SlotObservationPolicy] = None,
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
        self.observation_policy = observation_policy or SlotObservationPolicy()

        # Current state
        self.state: SlotState = initial_state
        self.assigned_track_id: Optional[int] = None
        self.latest_detection_bbox: Optional[tuple[float, float, float, float]] = None
        self.plate_number: str = ""
        self.snapshot_url: str = ""
        self.last_update_time: str = datetime.now().isoformat()

        # Plate-lock: once a parked car's plate is confirmed above the bar it is
        # frozen (plate_locked=True) and cannot change until the slot goes VACANT.
        # plate_confidence is the score of the currently-bound plate; while
        # unlocked the binding only ever upgrades to a strictly higher score.
        self.plate_locked: bool = False
        self.plate_confidence: float = 0.0

        # Store violation zone status from DB
        self._is_violation_zone: bool = is_violation_zone

        # Debounce counters
        self._enter_counter: int = 0
        self._leave_counter: int = 0

        # [SLOTLAT] instrumentation only — never read by the state logic.
        # _*_started_at: monotonic stamp of the frame that opened the current
        # ENTERING/LEAVING run, so a CONFIRM can report the true wall-clock cost.
        # _*_resets: how many times flicker threw the run away since the last
        # confirmed flip — the number that separates "the debounce is long" from
        # "detection is unstable".
        self._enter_started_at: Optional[float] = None
        self._leave_started_at: Optional[float] = None
        self._enter_resets: int = 0
        self._leave_resets: int = 0

        self._timed_hysteresis: Optional[_TimedHysteresis] = None
        if self.observation_policy.mode in {"shadow", "time"}:
            self._timed_hysteresis = _TimedHysteresis(
                self.observation_policy,
                initial_state,
            )

    @property
    def is_occupied(self) -> bool:
        """True until departure is confirmed, including the LEAVING grace."""
        return self.state in {SlotState.OCCUPIED, SlotState.LEAVING}

    @property
    def is_violation_zone(self) -> bool:
        """True if the slot is a restricted/intrusion zone (set from database)."""
        return self._is_violation_zone

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the slot's current status."""
        status = {
            "slot_id": self.slot_id,
            "state": self.state.value,
            "assigned_track_id": self.assigned_track_id,
            "latest_detection_bbox": self.latest_detection_bbox,
            "occupied": self.is_occupied,
            "last_update_time": self.last_update_time,
            "is_violation_zone": self.is_violation_zone,
            "plate_number": self.plate_number,
            "snapshot_url": self.snapshot_url,
            "plate_locked": self.plate_locked,
            "plate_confidence": self.plate_confidence,
        }
        if self._timed_hysteresis is not None:
            shadow_state = self._timed_hysteresis.state
            status["time_policy_state"] = shadow_state.value
            status["time_policy_disagrees"] = shadow_state != self.state
        return status

    def is_plate_locked(self) -> bool:
        """True once the bound plate has been frozen for this slot."""
        return self.plate_locked

    def bind_identity(
        self,
        plate_number: Optional[str],
        snapshot_url: str = "",
        confidence: Optional[float] = None,
        lock: bool = False,
    ) -> None:
        """Persist the confirmed identity for this slot until vacancy is confirmed.

        Passing ``plate_number=None`` (or ``""``) clears the field — callers in
        engine_runtime.py rely on this to drop ghost plates when the registry
        no longer has a binding for the slot (e.g. CAM-01/CAM-02 where plate
        matching is disabled but slot detection still runs).

        Args:
            confidence: Score of this plate reading. When provided it becomes the
                slot's ``plate_confidence`` (the value the unlocked upgrade path
                compares against). ``None`` leaves the stored confidence untouched.
            lock: When True, hard-freeze the binding — subsequent lower/other
                readings are ignored until ``clear_identity`` (slot goes VACANT).
        """
        self.plate_number = plate_number or ""
        if snapshot_url:
            self.snapshot_url = snapshot_url
        if confidence is not None:
            self.plate_confidence = confidence
        if lock:
            self.plate_locked = True

    def clear_identity(self) -> None:
        """Clear the persisted identity after the slot is confirmed vacant."""
        self.plate_number = ""
        self.snapshot_url = ""
        self.latest_detection_bbox = None
        self.plate_locked = False
        self.plate_confidence = 0.0

    def update(
        self,
        vehicle_present: bool,
        track_id: Optional[int] = None,
    ) -> List[SlotEvent]:
        """Compatibility wrapper for callers with a fresh inference result."""
        observation = SlotObservation(
            kind=(
                SlotObservationKind.PRESENT
                if vehicle_present
                else SlotObservationKind.ABSENT
            ),
            observed_at=time.monotonic(),
            timestamp=datetime.now().isoformat(),
            track_id=track_id,
        )
        return self.observe(observation)

    def observe(self, observation: SlotObservation) -> List[SlotEvent]:
        """Apply known evidence or hold state for an UNKNOWN observation."""
        mode = self.observation_policy.mode
        if mode == "time":
            return self._update_timed(observation)

        if self._timed_hysteresis is not None:
            self._timed_hysteresis.observe(observation)
        if observation.kind == SlotObservationKind.UNKNOWN:
            return []
        return self._update_legacy(observation)

    def _update_timed(self, observation: SlotObservation) -> List[SlotEvent]:
        timed = self._timed_hysteresis
        if timed is None:
            raise RuntimeError("time mode requires a timed hysteresis reducer")

        previous_state = self.state
        old_track = self.assigned_track_id
        transitions = timed.observe(observation)
        self.state = timed.state
        self.assigned_track_id = timed.assigned_track_id
        if observation.kind != SlotObservationKind.UNKNOWN:
            self.last_update_time = observation.timestamp

        events = [
            self._timed_event(transition, observation, old_track)
            for transition in transitions
        ]
        if (
            _TimedTransition.VACANT in transitions
            or previous_state == SlotState.ENTERING
            and self.state == SlotState.VACANT
        ):
            self.clear_identity()
        return events

    def _timed_event(
        self,
        transition: _TimedTransition,
        observation: SlotObservation,
        old_track: Optional[int],
    ) -> SlotEvent:
        track_id = self.assigned_track_id
        if transition == _TimedTransition.LEAVING:
            track_id = old_track
        elif transition == _TimedTransition.VACANT:
            track_id = old_track
        return SlotEvent(
            event_type=transition.value,
            slot_id=self.slot_id,
            track_id=track_id,
            timestamp=observation.timestamp,
        )

    def _update_legacy(
        self,
        observation: SlotObservation,
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
        vehicle_present = observation.kind == SlotObservationKind.PRESENT
        track_id = observation.track_id
        now = observation.timestamp

        # ----- VACANT -----
        if self.state == SlotState.VACANT:
            if vehicle_present:
                # Start entering: switch to ENTERING, begin counting
                self.state = SlotState.ENTERING
                self.assigned_track_id = track_id
                self._enter_counter = 1
                self._enter_started_at = observation.observed_at
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
                    self._log_confirm(
                        "OCCUPIED",
                        need=self.confirm_enter_frames,
                        started_at=self._enter_started_at,
                        resets=self._enter_resets,
                    )
                    self._enter_counter = 0
                    self._enter_started_at = None
                    self._enter_resets = 0
                    events.append(SlotEvent(
                        event_type="vehicle_parked",
                        slot_id=self.slot_id,
                        track_id=track_id,
                        timestamp=now,
                    ))
            else:
                # Vehicle disappeared before confirmation — revert
                self._enter_resets += 1
                self._log_reset(
                    "ENTERING->VACANT",
                    had=self._enter_counter,
                    need=self.confirm_enter_frames,
                    started_at=self._enter_started_at,
                    resets=self._enter_resets,
                )
                self.state = SlotState.VACANT
                self.assigned_track_id = None
                self._enter_counter = 0
                self._enter_started_at = None
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
                self._leave_started_at = observation.observed_at
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
                    self._log_confirm(
                        "VACANT",
                        need=self.confirm_leave_frames,
                        started_at=self._leave_started_at,
                        resets=self._leave_resets,
                    )
                    self.assigned_track_id = None
                    self._leave_counter = 0
                    self._leave_started_at = None
                    self._leave_resets = 0
                    self.clear_identity()
                    events.append(SlotEvent(
                        event_type="slot_vacant",
                        slot_id=self.slot_id,
                        track_id=old_track,
                        timestamp=now,
                    ))
            else:
                # Vehicle re-detected — cancel leaving
                self._leave_resets += 1
                self._log_reset(
                    "LEAVING->OCCUPIED",
                    had=self._leave_counter,
                    need=self.confirm_leave_frames,
                    started_at=self._leave_started_at,
                    resets=self._leave_resets,
                )
                self.state = SlotState.OCCUPIED
                self.assigned_track_id = track_id
                self._leave_counter = 0
                self._leave_started_at = None
                self.last_update_time = now

        return events

    # ---- [SLOTLAT] instrumentation -------------------------------------------------
    # Measurement only: these never mutate state and never raise into the frame loop.

    def _log_confirm(
        self,
        to_state: str,
        *,
        need: int,
        started_at: Optional[float],
        resets: int,
    ) -> None:
        """A flip completed. Report what it actually cost in frames, seconds and resets.

        ``elapsed_s`` well above ``need / effective_fps`` means the debounce was NOT the
        bottleneck — the run kept getting thrown away. ``resets`` says how often.
        """
        elapsed = (
            time.monotonic() - started_at if started_at is not None else float("nan")
        )
        logger.info(
            "%s CONFIRM slot=%s -> %s after %d frames in %.2fs (resets=%d)",
            _SLOTLAT, self.slot_id, to_state, need, elapsed, resets,
        )

    def _log_reset(
        self,
        transition: str,
        *,
        had: int,
        need: int,
        started_at: Optional[float],
        resets: int,
    ) -> None:
        """Flicker threw the debounce run away. This is the unbounded-tail event."""
        elapsed = (
            time.monotonic() - started_at if started_at is not None else float("nan")
        )
        logger.info(
            "%s RESET   slot=%s %s at %d/%d frames after %.2fs "
            "(reset #%d for this flip; counters need CONSECUTIVE frames)",
            _SLOTLAT, self.slot_id, transition, had, need, elapsed, resets,
        )
