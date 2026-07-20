import logging

from sqlalchemy.orm import Session
from src.model import SlotStatus, CameraFeed
from src.repositories import SlotStatusRepository, ParkingSlotRepository
from src.utils.datetime_helper import facility_now_naive
from . import alert_service
from . import pms_api_client

logger = logging.getLogger(__name__)

def log_camera_feed_event(db: Session, event_type: str, camera_id: str, plate: str = None, slot_id: str = None, snapshot_path: str = None):
    """
    Log a security/violation event to the camera_feeds table.
    """
    slot = None
    if slot_id:
        slot = ParkingSlotRepository.get_by_id(db, slot_id)
        
    # Derive Location Label (e.g. "(B1-NORTH)")
    location_label = f"({camera_id})"
    if slot:
        floor = slot.floor or "B1"
        zone = slot.zone_name or slot.zone_id or "ZONE"
        location_label = f"({floor}-{zone})"

    # Map Event Type to Description
    description_map = {
        "vehicle_violation": "Unauthorized Parking Violation",
        "vehicle_intrusion": "Reserved Slot Ownership Violation",
        "special_needs_violation": "Special-Needs Slot Violation",
    }
    event_description = description_map.get(event_type, event_type.replace("_", " ").title())

    new_feed = CameraFeed(
        camera_id=camera_id,
        location_label=location_label,
        event_description=event_description,
        detection_source="Vision Detection",
        plate_number=plate,
        snapshot_path=snapshot_path,
        timestamp=facility_now_naive()
    )
    
    db.add(new_feed)
    db.commit()
    return new_feed

def log_vehicle_event(
    db: Session,
    slot_id: str,
    plate: str,
    is_parked: bool,
    camera_id: str = None,
    severity: str = None,
    snapshot_path: str = None,
):
    plate_value = (plate or None) if is_parked else None

    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot:
        slot.is_available = not is_parked
        # Identity mirror: parking_slots.current_plate and
        # slot_status.plate_number are authoritative twins carrying the same
        # value on the same session, so downstream consumers (the Gateway
        # prefers current_plate and falls back to slot_status.plate_number)
        # can read either. report_alert may flush this row marginally before
        # the occupancy row is created — the value is identical either way.
        #
        # OCCUPANCY MAY NOT ERASE IDENTITY. This is an occupancy event, and by
        # design it carries no plate: identity (OCR/ReID) runs AFTER occupancy is
        # published, so `plate` is empty on essentially every vehicle_parked
        # (engine_runtime._get_slot_alert_type says so in as many words). Mirroring
        # that empty value onto current_plate makes every occupancy write a plate
        # ERASER — an already-identified slot re-emitting an occupied event (a
        # re-park, an alert-typed event, a restart replay) is silently reset to
        # NULL, and a LOCKED slot never rewrites it because _resolve_locked_plate
        # freezes on is_plate_locked(). The slot then reads NULL for the rest of
        # the occupancy. So: only VACATING clears identity — that is the one
        # transition that actually proves the old plate is gone.
        if not is_parked:
            slot.current_plate = None
            slot.plate_confidence = 0.0
            slot.plate_locked = False
            slot.plate_locked_at = None
        elif plate_value:
            slot.current_plate = plate_value

    is_occupied_to_empty_transition = False
    if not is_parked:
        prev_status = SlotStatusRepository.get_latest_by_slot(db, slot_id)
        if prev_status is not None and prev_status.status == "occupied":
            is_occupied_to_empty_transition = True

    # Keep the twins in step: the new occupancy row carries whatever identity the
    # slot row now holds (the event's plate if it brought one, otherwise the plate
    # already bound to the slot) so the two columns never disagree. A vacate row
    # carries None, matching the clear above.
    new_log = SlotStatus(
        slot_id=slot_id,
        plate_number=(plate_value or getattr(slot, "current_plate", None)) if is_parked else None,
        status="occupied" if is_parked else "available"
    )

    alert_id = None
    if is_parked:
        # A NAMED SLOT WITH NO PLATE IS NOT OURS TO JUDGE.
        #
        # A slot reserved for a person can only be "intruded" by someone who is
        # NOT that person, and identity is not known at occupancy time — it
        # resolves seconds to minutes later. The engine owns that verdict
        # (_register_pending_ownership -> _evaluate_named_slot_ownership, with
        # _sweep_pending_ownership as the deadline) and raises the alert itself,
        # after an ownership check and carrying the plate.
        #
        # Raising here regardless is what produced 80 plateless vehicle_intrusion
        # alerts on 2026-07-20 — every executive flagged as an intruder in his own
        # slot — while the deferred path fired 0 times. report_alert with no
        # alert_type derives "vehicle_intrusion" from reservation_type == EMPLOYEE
        # alone (alert_service.py:66) and never looks at WHO parked, so the
        # ownership machinery was bypassed entirely.
        #
        # When a plate IS present the engine has already ruled (engine_runtime.py
        # :2570 returns vehicle_intrusion only for a proven non-owner), so that
        # case still reports here and keeps its plate.
        named_slot = getattr(slot, "reservation_type", None) == "EMPLOYEE"
        if named_slot and not plate_value:
            logger.debug(
                "[alert] slot=%s is reserved for %r and no plate is known yet — "
                "deferring the ownership verdict to the engine",
                slot_id, getattr(slot, "reserved_for", None),
            )
        else:
            alert = alert_service.report_alert(
                db,
                slot_id,
                plate,
                camera_id=camera_id,
                severity=severity,
                snapshot_path=snapshot_path,
            )
            if alert:
                alert_id = alert.id
    else:
        alert = alert_service.resolve_alert(db, slot_id)
        if alert:
            alert_id = alert.id

    created = SlotStatusRepository.create(db, new_log)

    if is_occupied_to_empty_transition:
        try:
            resolved_ids = alert_service.auto_resolve_slot_violation_alerts(db, slot_id)
            if resolved_ids and alert_id is None:
                alert_id = resolved_ids[0]
        except Exception as exc:
            print(
                f"[WARN] auto-resolve slot violation alerts failed for slot {slot_id}: {exc}"
            )

    if slot and plate:
        if is_parked:
            pms_api_client.bind_slot_session(
                plate_number=plate,
                slot_id=slot.slot_id,
                slot_number=slot.slot_name or slot.slot_id,
                zone_id=slot.zone_id,
                zone_name=slot.zone_name,
                floor=slot.floor,
                camera_id=camera_id,
                parked_at=created.time.isoformat() if created.time else None,
            )
        else:
            pms_api_client.unbind_slot_session(
                plate_number=plate,
                slot_id=slot.slot_id,
                slot_number=slot.slot_name or slot.slot_id,
                camera_id=camera_id,
                left_at=created.time.isoformat() if created.time else None,
            )

    return created, alert_id


def update_current_slot_plate(
    db: Session,
    slot_id: str,
    plate: str,
    camera_id: str = None,
):
    """
    Update the latest occupied slot_status row with the recognized plate.

    This is used when the slot becomes occupied first and ANPR/ReID resolves
    the identity slightly later. In that case we should enrich the current
    occupied record instead of creating another duplicate occupied row.
    """
    plate_value = plate or None

    latest_status = SlotStatusRepository.get_latest_by_slot(db, slot_id)
    if latest_status and latest_status.status == "occupied":
        previous_plate = latest_status.plate_number
        latest_status.plate_number = plate_value

        # Mirror the identity onto parking_slots.current_plate in the SAME
        # commit. The slot row is fetched before the commit (not after) so a
        # crash between the two writes cannot leave the columns disagreeing.
        slot = ParkingSlotRepository.get_by_id(db, slot_id)
        if slot:
            slot.current_plate = plate_value

        db.commit()
        db.refresh(latest_status)

        if slot and plate_value != previous_plate:
            if plate_value:
                pms_api_client.bind_slot_session(
                    plate_number=plate_value,
                    slot_id=slot.slot_id,
                    slot_number=slot.slot_name or slot.slot_id,
                    zone_id=slot.zone_id,
                    zone_name=slot.zone_name,
                    floor=slot.floor,
                    camera_id=camera_id,
                    parked_at=latest_status.time.isoformat() if latest_status.time else None,
                )
            elif previous_plate:
                # Clear the session in PMS API as well
                pms_api_client.unbind_slot_session(
                    plate_number=previous_plate,
                    slot_id=slot.slot_id,
                    slot_number=slot.slot_name or slot.slot_id,
                    camera_id=camera_id,
                    left_at=facility_now_naive().isoformat(),
                )
        return latest_status

    if not plate:
        return None

    created, _ = log_vehicle_event(
        db=db,
        slot_id=slot_id,
        plate=plate,
        is_parked=True,
        camera_id=camera_id,
    )
    return created


def get_vehicle_current_location(db: Session, plate: str):
    last_event = SlotStatusRepository.get_latest_by_plate(db, plate)
    if last_event and last_event.status == "occupied":
        return last_event
    return None


def reset_all_slot_plates(db: Session) -> int:
    """Wipe every plate identity from the parking (VA-local, occupancy intact).

    Nulls the persisted identity columns on every ``parking_slots`` row
    (``current_plate`` / ``plate_confidence`` / ``plate_locked`` /
    ``plate_locked_at`` — the fields ``_load_camera_db_state`` restores on boot)
    and clears the ``plate_number`` on each slot's latest ``slot_status`` row so
    the live API/UI shows no plate. ``is_available`` / occupancy is left
    untouched — only identities are cleared.

    Deliberately VA-local: it does NOT call ``pms_api_client.unbind_slot_session``
    (unlike ``update_current_slot_plate``), so no cross-system side effects.

    Returns the number of slots whose plate binding was cleared.
    """
    cleared = 0
    slots = ParkingSlotRepository.get_all(db)
    for slot in slots:
        had_plate = bool(getattr(slot, "current_plate", None))
        slot.current_plate = None
        slot.plate_confidence = 0.0
        slot.plate_locked = False
        slot.plate_locked_at = None

        latest_status = SlotStatusRepository.get_latest_by_slot(db, slot.slot_id)
        if latest_status is not None and latest_status.plate_number:
            latest_status.plate_number = None
            had_plate = True

        if had_plate:
            cleared += 1

    db.commit()
    return cleared


def clear_slot_plate_binding(db: Session, slot_id: str) -> bool:
    """Wipe the plate identity from a single ``parking_slots`` row (VA-local).

    One-slot analogue of :func:`reset_all_slot_plates`: nulls ``current_plate`` /
    ``plate_confidence`` / ``plate_locked`` / ``plate_locked_at`` (the fields
    ``_load_camera_db_state`` restores on boot) and clears ``plate_number`` on the
    slot's latest ``slot_status`` row so the live API/UI shows no plate.
    Occupancy (``is_available``) is left untouched — only the identity is cleared.

    Deliberately VA-local: does NOT call ``pms_api_client`` (the external session
    table is out of scope for this eviction). Returns True if the row existed.
    """
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot is None:
        return False

    slot.current_plate = None
    slot.plate_confidence = 0.0
    slot.plate_locked = False
    slot.plate_locked_at = None

    latest_status = SlotStatusRepository.get_latest_by_slot(db, slot_id)
    if latest_status is not None and latest_status.plate_number:
        latest_status.plate_number = None

    db.commit()
    return True
