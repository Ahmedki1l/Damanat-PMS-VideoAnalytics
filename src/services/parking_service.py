from sqlalchemy.orm import Session
from fastapi import HTTPException
from src.model import ParkingSlot
from src.repositories import ParkingSlotRepository
from src.schemas import ParkingSlotUpdate

def get_slot_or_404(db: Session, slot_id: str):
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    if not slot:
        raise HTTPException(status_code=404, detail=f"Slot {slot_id} not found")
    return slot

def sync_slots_from_config(db: Session, slots_from_json: list, floor: int):
    for slot in slots_from_json:
        polygon_json = [[x, y] for x, y in list(slot.polygon.exterior.coords)]
        
        existing = ParkingSlotRepository.get_by_id(db, slot.id)
        if existing:
            # Update geometry and name ONLY — do NOT overwrite is_violation_zone
            # so that sync_restricted_slots.py settings are preserved
            existing.slot_name = slot.label or slot.id
            existing.floor = floor
            existing.polygon = polygon_json
        else:
            # New slot: infer violation zone from name as a sensible default
            new_slot = ParkingSlot(
                slot_id=slot.id,
                slot_name=slot.label or slot.id,
                floor=floor,
                polygon=polygon_json,
                is_available=True,
                is_violation_zone="violation" in slot.id.lower()
            )
            ParkingSlotRepository.create(db, new_slot)
    db.commit()

def update_slot_partial(db: Session, slot_id: str, data: ParkingSlotUpdate):
    existing = get_slot_or_404(db, slot_id)
    for key, value in data.dict(exclude_unset=True).items():
        setattr(existing, key, value)
    ParkingSlotRepository.update(db)
    return existing