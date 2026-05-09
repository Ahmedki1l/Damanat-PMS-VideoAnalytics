from sqlalchemy.orm import Session
from src.model.alert import Alert
from src.repositories import AlertRepository, ParkingSlotRepository
from src.utils.datetime_helper import facility_now_naive

def check_slot_restricted(db: Session, slot_id: str) -> bool:
    """True when a slot should trigger alerts (violation zone, employee, or special-needs)."""
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if not slot:
        return False
    return slot.is_violation_zone or slot.parking_category in ("employee", "special")

def get_alert_type_for_slot(db: Session, slot_id: str) -> str:
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot and slot.parking_category == "special":
        return "special_needs_violation"
    if slot and slot.parking_category == "employee":
        return "vehicle_intrusion"
    if slot and "violation" in (slot.slot_name or "").lower():
        return "violation"
    if "violation" in slot_id.lower():
        return "violation"
    return "intrusion"

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
            if alert_type == "vehicle_intrusion" and slot and slot.parking_category == "employee"
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

def get_active_alerts(db: Session, alert_type: str = None):
    """Get all currently active (unresolved) alerts."""
    return AlertRepository.get_all(db, is_resolved=False, alert_type=alert_type)

def get_alert_history(db: Session, slot_id: str):
    """Get full alert history for a specific slot."""
    return AlertRepository.get_history_by_slot(db, slot_id)
