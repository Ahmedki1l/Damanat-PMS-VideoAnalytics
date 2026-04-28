from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class VehicleRecord:
    plate_number: str
    title: str | None


class VehicleRepository:
    @staticmethod
    def get_by_plate(db: Session, plate_number: str) -> VehicleRecord | None:
        if not plate_number:
            return None

        row = db.execute(
            text(
                """
                SELECT plate_number, title
                FROM vehicles
                WHERE plate_number = :plate_number
                """
            ),
            {"plate_number": plate_number},
        ).mappings().first()

        if not row:
            return None

        return VehicleRecord(
            plate_number=row.get("plate_number", ""),
            title=row.get("title"),
        )
