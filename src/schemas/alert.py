from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AlertBase(BaseModel):
    alert_type: str
    camera_id: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    slot_id: Optional[str] = None
    region_id: Optional[int] = None
    slot_number: Optional[str] = None
    event_type: Optional[str] = None
    description: Optional[str] = None
    snapshot_path: Optional[str] = None
    is_test: bool = False
    is_resolved: bool = False
    plate_number: Optional[str] = None

class AlertCreate(AlertBase):
    pass

class AlertUpdate(BaseModel):
    is_resolved: bool
    resolved_at: Optional[datetime] = None

class AlertResponse(AlertBase):
    id: int
    triggered_at: datetime
    resolved_at: Optional[datetime] = None

    class Config:
        from_attributes = True
