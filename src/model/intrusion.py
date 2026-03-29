from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from src.database import Base
from datetime import datetime


class Intrusion(Base):
    __tablename__ = "intrusions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    slot_id = Column(String(50), ForeignKey("parking_slots.slot_id"), nullable=False)
    plate_number = Column(String(20), index=True, nullable=True)
    status = Column(String(20), default="active")  # "active" | "resolved"
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    camera_id = Column(String(50), nullable=True)

    slot = relationship(
        "ParkingSlot",
        back_populates="intrusions"
    )
