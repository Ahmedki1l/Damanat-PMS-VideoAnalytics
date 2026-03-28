from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database import get_db
from src.repositories import SlotStatusRepository
from src.schemas import SlotStatusResponse
from src.services import slot_status_service
from typing import List
from pydantic import BaseModel

class VehicleEventPayload(BaseModel):
    slot_id: str
    plate: str = None
    is_parked: bool

router = APIRouter(prefix="/api/status", tags=["Parking Status"])

@router.get("/history/{slot_id}", response_model=List[SlotStatusResponse])
def get_slot_history(slot_id: str, db: Session = Depends(get_db)):
    return SlotStatusRepository.get_history_by_slot(db, slot_id)

@router.get("/latest/{slot_id}", response_model=SlotStatusResponse)
def get_latest_slot_status(slot_id: str, db: Session = Depends(get_db)):
    return SlotStatusRepository.get_latest_by_slot(db, slot_id)

@router.get("/vehicle/{plate}", response_model=SlotStatusResponse)
def get_vehicle_status(plate: str, db: Session = Depends(get_db)):
    return SlotStatusRepository.get_latest_by_plate(db, plate)

@router.post("/event", response_model=SlotStatusResponse)
def log_vehicle_event(payload: VehicleEventPayload, db: Session = Depends(get_db)):
    return slot_status_service.log_vehicle_event(db, payload.slot_id, payload.plate, payload.is_parked)

