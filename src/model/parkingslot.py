from sqlalchemy import Column, Integer, String, Boolean, JSON
from sqlalchemy.orm import relationship
from src.database import Base

class ParkingSlot(Base):
    __tablename__ = "parking_slots"
    slot_id = Column(String(50), primary_key=True, index=True, nullable=False)
    slot_name = Column(String(100), index=True)
    camera_id = Column(String(50), index=True, nullable=True)
    floor = Column(String(50), index=True)
    zone_id = Column(String(100), nullable=True)
    zone_name = Column(String(100), nullable=True)
    slot_type = Column(String(30), nullable=False, default="parking")
    polygon = Column(JSON)
    last_snapshot_path = Column(String(255), nullable=True)
    is_available = Column(Boolean, default=True)
    is_violation_zone = Column(Boolean, default=False)
    reservation_type = Column(String(20), nullable=False, default="GENERAL")
    reserved_for = Column(String(255), nullable=True)

    statuses = relationship(
        "SlotStatus",          
        back_populates="slot",
        cascade="all, delete"
    )

    alerts = relationship(
        "Alert",
        back_populates="slot",
        cascade="all, delete"
    )
