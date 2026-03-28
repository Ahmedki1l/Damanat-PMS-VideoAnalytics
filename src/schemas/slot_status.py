from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SlotStatusCreate(BaseModel):
    slot_id: str = Field(..., example="A1")
    plate_number: Optional[str] = Field(None, example="ABC123")
    status: str = Field(..., example="occupied")

class SlotStatusUpdate(BaseModel):
    plate_number: Optional[str]
    status: Optional[str]

class SlotStatusResponse(BaseModel):
    id: int
    slot_id: str
    plate_number: Optional[str]
    status: str
    time: datetime  

    class Config:
        from_attributes = True
