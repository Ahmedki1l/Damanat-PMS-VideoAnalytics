from sqlalchemy import Column, String, JSON
from src.database import Base


class Boundary(Base):
    """
    An area-to-area crossing polygon (zoning), authored per camera.

    A boundary is a *pairwise* gate: it connects exactly two areas
    (``area_from`` -> ``area_to``). A junction where 3+ areas meet is modelled
    as several boundary rows, one per crossable pair. Kept in its own table so
    ``parking_slots`` stays purely about parking slots.
    """
    __tablename__ = "boundaries"

    boundary_id = Column(String(50), primary_key=True, index=True)  # e.g. "ramp_C_to_RAMP"
    camera_id = Column(String(50), index=True, nullable=True)        # camera that sees it
    floor = Column(String(50), index=True, nullable=True)
    polygon = Column(JSON)                                           # pixel coords (camera view)
    area_from = Column(String(50), nullable=True)                    # area a car is leaving
    area_to = Column(String(50), nullable=True)                      # area a car is entering

    def __repr__(self):
        return (
            f"<Boundary {self.boundary_id} cam={self.camera_id} "
            f"{self.area_from}->{self.area_to}>"
        )
