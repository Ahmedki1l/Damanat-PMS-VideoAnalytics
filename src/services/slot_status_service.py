from sqlalchemy.orm import Session
from src.model import SlotStatus
from src.repositories import SlotStatusRepository, ParkingSlotRepository
from . import intrusion_service

def log_vehicle_event(db: Session, slot_id: str, plate: str, is_parked: bool):
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if slot:
        slot.is_available = not is_parked
    
    new_log = SlotStatus(
        slot_id=slot_id,
        plate_number=plate,
        status="occupied" if is_parked else "available"
    )
    
    if is_parked:
        intrusion_service.report_intrusion(db, slot_id, plate)
    else:
        intrusion_service.resolve_intrusion(db, slot_id)
    
    return SlotStatusRepository.create(db, new_log)

def get_vehicle_current_location(db: Session, plate: str):
    last_event = SlotStatusRepository.get_latest_by_plate(db, plate)
    if last_event and last_event.status == "occupied":
        return last_event
    return None