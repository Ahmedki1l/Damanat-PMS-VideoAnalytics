from sqlalchemy.orm import Session
from src.model import SlotStatus, CameraFeed
from src.repositories import SlotStatusRepository, ParkingSlotRepository
from src.utils.datetime_helper import facility_now_naive
from . import alert_service
from . import pms_api_client

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
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot:
        slot.is_available = not is_parked

    is_occupied_to_empty_transition = False
    if not is_parked:
        prev_status = SlotStatusRepository.get_latest_by_slot(db, slot_id)
        if prev_status is not None and prev_status.status == "occupied":
            is_occupied_to_empty_transition = True

    new_log = SlotStatus(
        slot_id=slot_id,
        plate_number=plate if is_parked else None,
        status="occupied" if is_parked else "available"
    )

    alert_id = None
    if is_parked:
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
    latest_status = SlotStatusRepository.get_latest_by_slot(db, slot_id)
    if latest_status and latest_status.status == "occupied":
        previous_plate = latest_status.plate_number
        latest_status.plate_number = plate
        db.commit()
        db.refresh(latest_status)

        slot = ParkingSlotRepository.get_by_id(db, slot_id)
        if slot and plate != previous_plate:
            if plate:
                pms_api_client.bind_slot_session(
                    plate_number=plate,
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
