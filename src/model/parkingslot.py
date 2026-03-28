from sqlalchemy import Column, Integer, String, Boolean, JSON
from sqlalchemy.orm import relationship
from src.database import Base

class ParkingSlot(Base):
    __tablename__ = "parking_slots"
    slot_id = Column(String(50), primary_key=True, index=True,nullable=False)
    slot_name = Column(String(100), index=True)
    floor = Column(String(50), index=True)
    polygon = Column(JSON)
    is_available = Column(Boolean, default=True)
    is_violation_zone = Column(Boolean, default=False)

    statuses = relationship(
        "SlotStatus",          
        back_populates="slot",
        cascade="all, delete"
    )

    intrusions = relationship(
        "Intrusion",
        back_populates="slot",
        cascade="all, delete"
    )