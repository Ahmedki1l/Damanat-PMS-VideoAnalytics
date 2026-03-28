from sqlalchemy import Column, Integer, String, Boolean, JSON
from sqlalchemy.orm import relationship
from src.database import Base

class ParkingSlot(Base):
    __tablename__ = "parking_slots"
    slot_id = Column(String, primary_key=True, index=True,nullable=False)
    slot_name = Column(String, index=True)
    floor = Column(String, index=True)
    polygon = Column(JSON)
    is_available = Column(Boolean, default=True)
    is_violation_zone = Column(Boolean, default=False)

    statuses = relationship(
        "SlotStatus",          
        back_populates="slot",
        cascade="all, delete"
    )