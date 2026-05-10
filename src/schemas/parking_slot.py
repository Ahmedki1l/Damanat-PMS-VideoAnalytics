from pydantic import BaseModel
from typing import Optional

class ParkingSlotCreate(BaseModel):
    slot_id: str
    slot_name: Optional[str] = None
    camera_id: Optional[str] = None
    floor: Optional[str] = None
    slot_type: Optional[str] = None
    polygon: Optional[list] = None
    is_available: Optional[bool] = True
    is_violation_zone: Optional[bool] = False
    parking_category: Optional[str] = "GENERAL"
    reserved_for: Optional[str] = None

class ParkingSlotUpdate(BaseModel):
    slot_name: Optional[str] = None
    camera_id: Optional[str] = None
    floor: Optional[str] = None
    slot_type: Optional[str] = None
    polygon: Optional[list] = None
    is_available: Optional[bool] = None
    is_violation_zone: Optional[bool] = None
    parking_category: Optional[str] = None
    reserved_for: Optional[str] = None

class ParkingSlotResponse(BaseModel):
    slot_id: str
    slot_name: Optional[str]
    camera_id: Optional[str]
    floor: Optional[str]
    slot_type: Optional[str]
    polygon: Optional[list]
    is_available: bool
    is_violation_zone: bool
    parking_category: str
    reserved_for: Optional[str]

    class Config:
        from_attributes = True
