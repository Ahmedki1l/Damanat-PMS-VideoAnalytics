from sqlalchemy.orm import Session
from src.model import SlotStatus, CameraFeed
from src.repositories import SlotStatusRepository, ParkingSlotRepository
from datetime import datetime
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
        "vehicle_intrusion": "Security Intrusion Detected",
        "named_slot_violation": "Reserved Slot Ownership Violation",
    }
    event_description = description_map.get(event_type, event_type.replace("_", " ").title())

    new_feed = CameraFeed(
        camera_id=camera_id,
        location_label=location_label,
        event_description=event_description,
        detection_source="Vision Detection",
        plate_number=plate,
        snapshot_path=snapshot_path,
        timestamp=datetime.now()
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
    
    new_log = SlotStatus(
        slot_id=slot_id,
        plate_number=plate,
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
                    left_at=datetime.now().isoformat(),
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
