from fastapi import APIRouter, Depends, HTTPException
from src.services import parking_service
from sqlalchemy.orm import Session
from src.database import get_db
from typing import List
from src.schemas import ParkingSlotResponse, ParkingSlotUpdate
from src.repositories import ParkingSlotRepository

router = APIRouter(prefix="/api/parking/slots", tags=["Parking Slots"])

@router.get("/", response_model=List[ParkingSlotResponse])
def get_all_slots(db: Session = Depends(get_db)):
    return parking_service.get_slot_or_404(db, slot_id="all")

@router.get("/available", response_model=List[ParkingSlotResponse])
def get_available_slots(db: Session = Depends(get_db)):
    return ParkingSlotRepository.filter_available_slots(db)

@router.get("/floor/{floor}", response_model=List[ParkingSlotResponse])
def get_slots_by_floor(floor: str, db: Session = Depends(get_db)):
    return ParkingSlotRepository.filter_floor_slots(db, floor)

@router.get("/{slot_id}", response_model=ParkingSlotResponse)
def get_slot_by_id(slot_id: str, db: Session = Depends(get_db)):
    return parking_service.get_slot_or_404(db, slot_id)

@router.put("/{slot_id}", response_model=ParkingSlotResponse)
def update_slot(slot_id: str, data: ParkingSlotUpdate, db: Session = Depends(get_db)):
    return parking_service.update_slot_partial(db, slot_id, data)

@router.delete("/{slot_id}")
def delete_slot(slot_id: str, db: Session = Depends(get_db)):
    slot = parking_service.get_slot_or_404(db, slot_id)
    ParkingSlotRepository.delete(db, slot)
    return {"detail": f"Slot {slot_id} deleted"}