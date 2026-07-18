from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Index
from sqlalchemy.orm import relationship
from src.database import Base, _SLOT_STATUS_LOOKUP_INDEX
from src.utils.datetime_helper import facility_now_naive


class SlotStatus(Base):
    """Append-only history of slot occupancy transitions.

    NOTE: this table has no retention policy — it only ever grows. The composite
    index below is what keeps ``get_latest_by_slot`` (``WHERE slot_id = ? ORDER BY
    time DESC``) off a full scan; without it that query sits on the engine's
    consumer thread, between a slot going vacant and the ``is_available`` commit
    the frontend reads, and gets slower every day the system runs.
    """

    __tablename__ = "slot_status"
    id = Column(Integer, primary_key=True, index=True , autoincrement=True)
    slot_id = Column(String(50), ForeignKey("parking_slots.slot_id"),nullable=False)
    plate_number = Column(String(20), index=True)
    status = Column(String(20))
    time = Column(DateTime, default=facility_now_naive)

    # Matches the back-fill in database._ensure_schema_updates, which is what
    # adds this to an ALREADY-EXISTING table (create_all will not).
    __table_args__ = (
        Index(_SLOT_STATUS_LOOKUP_INDEX, "slot_id", time.desc()),
    )

    slot = relationship(
        "ParkingSlot",
        back_populates="statuses"
    )
