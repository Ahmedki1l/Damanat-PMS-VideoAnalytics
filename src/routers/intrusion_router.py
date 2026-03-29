from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database import get_db
from src.services import intrusion_service
from src.schemas import IntrusionCreate, IntrusionResponse
from typing import List

router = APIRouter(prefix="/api/intrusions", tags=["Intrusions"])


@router.post("/report", response_model=IntrusionResponse)
def report_intrusion(payload: IntrusionCreate, db: Session = Depends(get_db)):
    """YOLO camera detects car in restricted slot → report intrusion."""
    result = intrusion_service.report_intrusion(
        db, payload.slot_id, payload.plate_number, payload.camera_id
    )
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=200, detail=f"Slot {payload.slot_id} is not restricted")
    return result


@router.post("/resolve/{slot_id}", response_model=IntrusionResponse)
def resolve_intrusion(slot_id: str, db: Session = Depends(get_db)):
    """Car leaves restricted slot → resolve the active intrusion."""
    result = intrusion_service.resolve_intrusion(db, slot_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"No active intrusion on slot {slot_id}")
    return result


@router.get("/active", response_model=List[IntrusionResponse])
def get_active_intrusions(db: Session = Depends(get_db)):
    """Get all currently active (unresolved) intrusions."""
    return intrusion_service.get_active_intrusions(db)


@router.get("/history/{slot_id}", response_model=List[IntrusionResponse])
def get_intrusion_history(slot_id: str, db: Session = Depends(get_db)):
    """Get full intrusion history for a specific slot."""
    return intrusion_service.get_intrusion_history(db, slot_id)


@router.get("/check/{slot_id}")
def check_slot_restricted(slot_id: str, db: Session = Depends(get_db)):
    """Check if a slot is in the restricted list."""
    return {
        "slot_id": slot_id,
        "restricted": intrusion_service.check_slot_restricted(db, slot_id)
    }
