from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, JSON
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

    # Plate-lock runtime binding: the plate frozen onto this slot while a car is
    # parked. Persisted so the binding survives a restart (see restart recovery
    # in engine_runtime._load_camera_db_state). Cleared when the slot goes VACANT.
    current_plate = Column(String(50), nullable=True)
    plate_confidence = Column(Float, nullable=False, default=0.0)
    plate_locked = Column(Boolean, nullable=False, default=False)
    plate_locked_at = Column(DateTime, nullable=True)

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
