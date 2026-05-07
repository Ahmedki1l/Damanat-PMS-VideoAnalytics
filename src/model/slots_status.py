from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from src.database import Base
from src.utils.datetime_helper import facility_now_naive


class SlotStatus(Base):
    __tablename__ = "slot_status"
    id = Column(Integer, primary_key=True, index=True , autoincrement=True)
    slot_id = Column(String(50), ForeignKey("parking_slots.slot_id"),nullable=False)
    plate_number = Column(String(20), index=True)
    status = Column(String(20))
    time = Column(DateTime, default=facility_now_naive)

    slot = relationship(
        "ParkingSlot",
        back_populates="statuses"
    )
