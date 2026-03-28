from pydantic import BaseModel
from typing import Optional

class ParkingSlotCreate(BaseModel):
    slot_id: str
    slot_name: Optional[str] = None
    floor: Optional[str] = None
    polygon: Optional[list] = None
    is_available: Optional[bool] = True
    is_violation_zone: Optional[bool] = False

class ParkingSlotUpdate(BaseModel):
    slot_name: Optional[str] = None
    floor: Optional[str] = None
    polygon: Optional[list] = None
    is_available: Optional[bool] = None
    is_violation_zone: Optional[bool] = None

class ParkingSlotResponse(BaseModel):
    slot_id: str
    slot_name: Optional[str]
    floor: Optional[str]
    polygon: Optional[list]
    is_available: bool
    is_violation_zone: bool

    class Config:
        from_attributes = True
