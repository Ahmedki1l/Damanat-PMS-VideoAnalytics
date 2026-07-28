import os

from sqlalchemy.orm import Session
from src.model.alert import Alert
from src.repositories import AlertRepository, ParkingSlotRepository
from src.utils.datetime_helper import facility_now_naive

_RESTRICTED_GATED_TYPES = (
    "special_needs_violation",
    "vehicle_intrusion",
    "reserved_slot_unidentified",
)

SLOT_SCOPED_VIOLATION_ALERT_TYPES = (
    "vehicle_violation",
    "named_slot_violation",
    "special_needs_violation",
    "vehicle_intrusion",
    # Raised when a car has occupied a named slot past the identity deadline and
    # neither OCR nor ReID could name it, so ownership could never be tested. Not a
    # proven intrusion — it says "nobody can vouch for this car". See
    # ParkingEngineRuntimeMixin._sweep_pending_ownership.
    "reserved_slot_unidentified",
    # Legacy short-form values written by older versions of get_alert_type_for_slot.
    # Included so the auto-resolver clears pre-existing rows on slot-empty transitions.
    "violation",
    "intrusion",
)

# Set from AppConfig via configure_alerts() at config-load time. None = never
# configured, fall back to the env var.
_ENABLE_RESTRICTED_ZONE_ALERTS: bool | None = None

# Alert types kept OUT of the real-time SSE stream (/api/alerts/stream).
# Notification-only: report_alert still writes the row and the REST endpoints
# still serve it — the dashboard just stops popping a live alert for it.
# Mirrors AlertsConfig.suppressed_notification_types; overwritten by
# configure_alerts() once YAML is loaded.
_SUPPRESSED_NOTIFICATION_TYPES: frozenset[str] = frozenset({"reserved_slot_unidentified"})

# Alert types turned OFF entirely: report_alert drops them before any row is
# written (no DB, no notification). Distinct from _SUPPRESSED_NOTIFICATION_TYPES,
# which silences only the live stream but still records the row. Set via
# configure_alerts() from AlertsConfig.disabled_alert_types.
_DISABLED_ALERT_TYPES: frozenset[str] = frozenset()


def configure_alerts(
    *,
    enable_restricted_zone_alerts: bool,
    suppressed_notification_types: tuple[str, ...] | None = None,
    disabled_alert_types: tuple[str, ...] | None = None,
) -> None:
    """Push the resolved AppConfig alert toggles into this module.

    Called by load_config(). This module is reached from request handlers and DB
    paths that have no access to the engine's AppConfig, so without this it read
    os.environ directly and ignored YAML entirely — meaning YAML could open the
    engine-side gate while this one stayed shut, yielding alert logs and snapshots
    with no corresponding rows in the alerts table.
    """
    global _ENABLE_RESTRICTED_ZONE_ALERTS, _SUPPRESSED_NOTIFICATION_TYPES
    global _DISABLED_ALERT_TYPES
    _ENABLE_RESTRICTED_ZONE_ALERTS = bool(enable_restricted_zone_alerts)
    if suppressed_notification_types is not None:
        _SUPPRESSED_NOTIFICATION_TYPES = frozenset(
            t.strip() for t in suppressed_notification_types if t and t.strip()
        )
    if disabled_alert_types is not None:
        _DISABLED_ALERT_TYPES = frozenset(
            t.strip() for t in disabled_alert_types if t and t.strip()
        )


def notification_suppressed(alert_type: str | None) -> bool:
    """True when this alert type must not be pushed to connected dashboards.

    Deliberately gates the STREAM, not report_alert: the alert is still a real
    record worth keeping and querying, it just isn't worth interrupting an
    operator over.
    """
    return bool(alert_type) and alert_type in _SUPPRESSED_NOTIFICATION_TYPES


def alert_type_disabled(alert_type: str | None) -> bool:
    """True when this alert type is turned off entirely (no row, no stream)."""
    return bool(alert_type) and alert_type in _DISABLED_ALERT_TYPES


def _restricted_zone_alerts_enabled() -> bool:
    if _ENABLE_RESTRICTED_ZONE_ALERTS is not None:
        return _ENABLE_RESTRICTED_ZONE_ALERTS
    # Unconfigured fallback. Mirrors AlertsConfig's default (True) rather than the
    # old hardcoded "false", which silently disabled the feature everywhere.
    return os.environ.get("ENABLE_RESTRICTED_ZONE_ALERTS", "true").strip().lower() == "true"

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

def _describe_alert(alert_type: str, slot_name: str, slot) -> str:
    if alert_type == "special_needs_violation":
        return f"Vehicle detected in special-needs reserved slot {slot_name}"
    if alert_type == "reserved_slot_unidentified":
        return (
            f"Unidentified vehicle occupying reserved slot {slot_name} — neither "
            f"plate OCR nor appearance matching could name it, so ownership could "
            f"not be verified"
        )
    if alert_type == "vehicle_intrusion" and slot and slot.reservation_type == "EMPLOYEE":
        return f"Unauthorized vehicle detected in reserved slot {slot_name}"
    return f"Unauthorized vehicle detected in {slot_name}"


def report_alert(
    db: Session,
    slot_id: str,
    plate_number: str = None,
    camera_id: str = None,
    severity: str = "critical",
    snapshot_path: str = None,
    alert_type: str = None,
):
    """
    YOLO detects car in a slot -> check if restricted (is_violation_zone) -> create alert.
    Returns None if slot is not restricted or alert already active.

    ``alert_type`` overrides the slot-derived type. The engine's deferred
    named-slot path uses it to distinguish a PROVEN non-owner (vehicle_intrusion)
    from a car nobody could identify in time (reserved_slot_unidentified) — a
    distinction this function cannot make from the slot row alone.
    """
    if not check_slot_restricted(db, slot_id):
        return None

    # Fully-disabled alert types are dropped before any row is written — no DB,
    # no stream. Distinct from notification suppression, which still records the
    # row. Resolve the effective type first so a slot-derived type is covered too.
    effective_type = alert_type or get_alert_type_for_slot(db, slot_id)
    if alert_type_disabled(effective_type):
        return None

    if not _restricted_zone_alerts_enabled():
        if effective_type in _RESTRICTED_GATED_TYPES:
            return None

    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    # Dedicated per-alert snapshots are the preferred source. Fall back to the
    # slot's rolling latest image only if the dedicated save path is unavailable.
    resolved_snapshot_path = snapshot_path or (slot.last_snapshot_path if slot else None)

    alert_type = alert_type or get_alert_type_for_slot(db, slot_id)
    slot_name = slot.slot_name if slot and slot.slot_name else slot_id

    # Don't duplicate - check if there's already an active alert on this slot
    existing = AlertRepository.get_active_by_slot(db, slot_id)
    if existing:
        dirty = False
        if resolved_snapshot_path and not existing.snapshot_path:
            existing.snapshot_path = resolved_snapshot_path
            dirty = True
        # Upgrade path: the slot alerted as "nobody could identify this car", and
        # identity has since landed and proved a non-owner. Same car, same
        # occupancy — promote the row in place instead of leaving a stale
        # unidentified alert next to a new intrusion one for the same slot.
        if (
            existing.alert_type == "reserved_slot_unidentified"
            and alert_type == "vehicle_intrusion"
        ):
            existing.alert_type = alert_type
            existing.description = _describe_alert(alert_type, slot_name, slot)
            existing.severity = severity
            if plate_number:
                existing.plate_number = plate_number
            dirty = True
        if dirty:
            db.commit()
            db.refresh(existing)
        return existing

    zone_id = slot.zone_id if slot and slot.zone_id else None
    zone_name = slot.zone_name if slot and slot.zone_name else zone_id or slot_id

    new_alert = Alert(
        alert_type=alert_type,
        camera_id=camera_id or "UNKNOWN",
        zone_id=zone_id,
        zone_name=zone_name,
        slot_id=slot_id,
        event_type="vehicle_detected",
        slot_number=slot_name,
        description=_describe_alert(alert_type, slot_name, slot),
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
