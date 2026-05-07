from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from src.database import Base
from src.utils.datetime_helper import facility_now_naive

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False, index=True)
    camera_id = Column(String(50), nullable=False)
    zone_id = Column(String(100), nullable=True)
    zone_name = Column(String(100))
    slot_id = Column(String(50), ForeignKey("parking_slots.slot_id"), nullable=True, index=True)
    region_id = Column(Integer)
    slot_number = Column(String(100), nullable=True)
    event_type = Column(String(100))
    description = Column(Text)
    snapshot_path = Column(Text, nullable=True)
    is_test = Column(Boolean, default=False, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    triggered_at = Column(DateTime, default=facility_now_naive, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Custom addition to track plates natively
    plate_number = Column(String(50), nullable=True, index=True)
    severity = Column(String(20), nullable=False, default="info")

    slot = relationship("ParkingSlot", back_populates="alerts", foreign_keys=[slot_id])

    def __repr__(self):
        return f"<Alert {self.id} type={self.alert_type} zone={self.zone_id} resolved={self.is_resolved}>"
