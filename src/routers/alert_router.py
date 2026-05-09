from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database import get_db
from src.schemas import AlertResponse
from src.services import alert_service
from typing import List

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertResponse])
def get_alerts(alert_type: str = None, db: Session = Depends(get_db)):
    """Get all active alerts, optionally filtered by type ('vehicle_violation', 'vehicle_intrusion', 'special_needs_violation', 'intrusion')."""
    return alert_service.get_active_alerts(db, alert_type)

@router.get("/history/{slot_id}", response_model=List[AlertResponse])
def get_alert_history(slot_id: str, db: Session = Depends(get_db)):
    """Get full alert history for a specific slot/zone."""
    return alert_service.get_alert_history(db, slot_id)

@router.post("/{slot_id}/resolve", response_model=AlertResponse)
def resolve_alert(slot_id: str, db: Session = Depends(get_db)):
    """Manually resolve an active alert for a slot."""
    alert = alert_service.resolve_alert(db, slot_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Active alert not found for this slot")
    return alert
