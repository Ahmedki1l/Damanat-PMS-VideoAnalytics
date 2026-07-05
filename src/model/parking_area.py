from sqlalchemy import Boolean, Column, Integer, String, JSON
from src.database import Base


class ParkingArea(Base):
    """
    A parking *area* (zoning) — a camera group within a floor (aisle or ramp).

    This repo owns the table; it is created by ``Base.metadata.create_all``
    (``DatabaseManager.create_tables``) and seeded from config.yaml when empty.
    At runtime ``sync_areas_from_db`` makes these rows the source of truth for
    ``AppConfig.areas``.
    """
    __tablename__ = "parking_areas"

    area_id = Column(String(50), primary_key=True, index=True)  # "B1-E", "B2-RAMP"
    name = Column(String(100), nullable=True)
    floor = Column(String(50), index=True, nullable=True)       # "B1", "B2"
    capacity = Column(Integer, nullable=False, default=0)        # soft cap on bounded gallery
    # {neighbor_area_id: transit_seconds} — gates the cross-area handoff matcher.
    adjacency_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    def __repr__(self):
        return f"<ParkingArea {self.area_id} floor={self.floor} cap={self.capacity}>"
