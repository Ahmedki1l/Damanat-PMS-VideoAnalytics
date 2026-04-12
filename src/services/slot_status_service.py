from sqlalchemy.orm import Session
from src.model import SlotStatus
from src.repositories import SlotStatusRepository, ParkingSlotRepository
from . import alert_service
from . import pms_api_client

def log_vehicle_event(db: Session, slot_id: str, plate: str, is_parked: bool, camera_id: str = None):
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot:
        slot.is_available = not is_parked
    
    new_log = SlotStatus(
        slot_id=slot_id,
        plate_number=plate,
        status="occupied" if is_parked else "available"
    )
    
    if is_parked:
        alert_service.report_alert(db, slot_id, plate, camera_id=camera_id)
    else:
        alert_service.resolve_alert(db, slot_id)

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

    return created

def get_vehicle_current_location(db: Session, plate: str):
    last_event = SlotStatusRepository.get_latest_by_plate(db, plate)
    if last_event and last_event.status == "occupied":
        return last_event
    return None
