from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class IntrusionCreate(BaseModel):
    slot_id: str = Field(..., example="G1")
    plate_number: Optional[str] = Field(None, example="ABC123")
    camera_id: Optional[str] = Field(None, example="CAM_01")

class IntrusionResponse(BaseModel):
    id: int
    slot_id: str
    plate_number: Optional[str]
    status: str
    detected_at: datetime
    resolved_at: Optional[datetime]
    camera_id: Optional[str]

    class Config:
        from_attributes = True
