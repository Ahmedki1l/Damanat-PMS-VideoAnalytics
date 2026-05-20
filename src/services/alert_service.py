import os

from sqlalchemy.orm import Session
from src.model.alert import Alert
from src.repositories import AlertRepository, ParkingSlotRepository
from src.utils.datetime_helper import facility_now_naive

_RESTRICTED_GATED_TYPES = ("special_needs_violation", "vehicle_intrusion")

SLOT_SCOPED_VIOLATION_ALERT_TYPES = (
    "vehicle_violation",
    "named_slot_violation",
    "special_needs_violation",
    "vehicle_intrusion",
    # Legacy short-form values written by older versions of get_alert_type_for_slot.
    # Included so the auto-resolver clears pre-existing rows on slot-empty transitions.
    "violation",
    "intrusion",
)


def _restricted_zone_alerts_enabled() -> bool:
    return os.environ.get("ENABLE_RESTRICTED_ZONE_ALERTS", "false").lower() == "true"

def check_slot_restricted(db: Session, slot_id: str) -> bool:
    """True when a slot should trigger alerts (violation zone, employee, or special-needs)."""
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if not slot:
        return False
    return slot.is_violation_zone or slot.reservation_type in ("EMPLOYEE", "SPECIAL")

def get_alert_type_for_slot(db: Session, slot_id: str) -> str:
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot and slot.reservation_type == "SPECIAL":
        return "special_needs_violation"
    if slot and slot.reservation_type == "EMPLOYEE":
        return "vehicle_intrusion"
    if slot and "violation" in (slot.slot_name or "").lower():
        return "vehicle_violation"
    if "violation" in slot_id.lower():
        return "vehicle_violation"
    return "vehicle_intrusion"

def report_alert(
    db: Session,
    slot_id: str,
    plate_number: str = None,
    camera_id: str = None,
    severity: str = "critical",
    snapshot_path: str = None,
):
    """
    YOLO detects car in a slot -> check if restricted (is_violation_zone) -> create alert.
    Returns None if slot is not restricted or alert already active.
    """
    if not check_slot_restricted(db, slot_id):
        return None

    if not _restricted_zone_alerts_enabled():
        gated_alert_type = get_alert_type_for_slot(db, slot_id)
        if gated_alert_type in _RESTRICTED_GATED_TYPES:
            return None

    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    # Dedicated per-alert snapshots are the preferred source. Fall back to the
    # slot's rolling latest image only if the dedicated save path is unavailable.
    resolved_snapshot_path = snapshot_path or (slot.last_snapshot_path if slot else None)

    # Don't duplicate - check if there's already an active alert on this slot
    existing = AlertRepository.get_active_by_slot(db, slot_id)
    if existing:
        if resolved_snapshot_path and not existing.snapshot_path:
            existing.snapshot_path = resolved_snapshot_path
            db.commit()
            db.refresh(existing)
        return existing

    alert_type = get_alert_type_for_slot(db, slot_id)
    zone_id = slot.zone_id if slot and slot.zone_id else None
    zone_name = slot.zone_name if slot and slot.zone_name else zone_id or slot_id
    slot_name = slot.slot_name if slot and slot.slot_name else slot_id

    new_alert = Alert(
        alert_type=alert_type,
        camera_id=camera_id or "UNKNOWN",
        zone_id=zone_id,
        zone_name=zone_name,
        slot_id=slot_id,
        event_type="vehicle_detected",
        slot_number=slot_name,
        description=(
            f"Vehicle detected in special-needs reserved slot {slot_name}"
            if alert_type == "special_needs_violation"
            else f"Unauthorized vehicle detected in reserved slot {slot_name}"
            if alert_type == "vehicle_intrusion" and slot and slot.reservation_type == "EMPLOYEE"
            else f"Unauthorized vehicle detected in {slot_name}"
        ),
        snapshot_path=resolved_snapshot_path,
        is_resolved=False,
        triggered_at=facility_now_naive(),
        plate_number=plate_number,
        severity=severity
    )
    return AlertRepository.create(db, new_alert)

def resolve_alert(db: Session, slot_id: str):
    """Car leaves restricted slot -> resolve the active alert."""
    active = AlertRepository.get_active_by_slot(db, slot_id)
    if not active:
        return None
    return AlertRepository.resolve(db, active.id)


def auto_resolve_slot_violation_alerts(db: Session, slot_id: str) -> list[int]:
    """Bulk-resolve all unresolved slot-scoped violation alerts for a slot.

    Called by VA on a confirmed occupied -> empty transition. Resolves alerts
    whose alert_type is in SLOT_SCOPED_VIOLATION_ALERT_TYPES, regardless of
    which service originally wrote them. Skips is_test=True alerts (demo
    alerts have their own lifecycle).

    Returns the list of resolved alert ids (may be empty).
    """
    now = facility_now_naive()
    alerts = (
        db.query(Alert)
        .filter(
            Alert.slot_id == slot_id,
            Alert.is_resolved == False,  # noqa: E712 — SQLAlchemy needs ==, not `is`
            Alert.is_test == False,  # noqa: E712
            Alert.alert_type.in_(SLOT_SCOPED_VIOLATION_ALERT_TYPES),
        )
        .all()
    )
    resolved_ids: list[int] = []
    for alert in alerts:
        alert.is_resolved = True
        alert.resolved_at = now
        resolved_ids.append(alert.id)
    if resolved_ids:
        db.commit()
    return resolved_ids

def get_active_alerts(db: Session, alert_type: str = None):
    """Get all currently active (unresolved) alerts."""
    return AlertRepository.get_all(db, is_resolved=False, alert_type=alert_type)

def get_alert_history(db: Session, slot_id: str):
    """Get full alert history for a specific slot."""
    return AlertRepository.get_history_by_slot(db, slot_id)
