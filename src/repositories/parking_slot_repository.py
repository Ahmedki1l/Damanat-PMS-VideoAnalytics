from sqlalchemy.orm import Session
from src.model import ParkingSlot

class ParkingSlotRepository:
    @staticmethod
    def get_all(db: Session):
        return db.query(ParkingSlot).all()

    @staticmethod
    def get_by_id(db: Session, slot_id: str):
        return db.query(ParkingSlot).filter(ParkingSlot.slot_id == slot_id).first()

    @staticmethod
    def get_plate_locks(db: Session):
        """Every slot that currently claims a plate, across ALL cameras and workers.

        The cross-process truth for "this car is already parked somewhere else". Each
        worker owns only its own cameras' slots, so this is the only way one worker can
        know that a candidate it is scoring is in fact sitting in a slot on a camera it
        never sees. A handful of rows (13 today) — cheap enough for the 10s sync tick.
        """
        return (
            db.query(
                ParkingSlot.slot_id,
                ParkingSlot.camera_id,
                ParkingSlot.current_plate,
                ParkingSlot.plate_locked,
                ParkingSlot.plate_locked_at,
            )
            .filter(ParkingSlot.current_plate.isnot(None))
            .filter(ParkingSlot.current_plate != "")
            .all()
        )

    @staticmethod
    def get_by_camera_and_slot_id(db: Session, camera_id: str, slot_id: str):
        return (
            db.query(ParkingSlot)
            .filter(
                ParkingSlot.camera_id == camera_id,
                ParkingSlot.slot_id == slot_id,
            )
            .first()
        )

    @staticmethod
    def create(db: Session, slot_model: ParkingSlot):
        db.add(slot_model)
        db.commit()
        db.refresh(slot_model)
        return slot_model

    @staticmethod
    def update(db: Session):
        db.commit()

    @staticmethod
    def delete(db: Session, slot: ParkingSlot):
        db.delete(slot)
        db.commit()

    @staticmethod
    def filter_floor_slots(db: Session, floor: str):
        return db.query(ParkingSlot).filter(ParkingSlot.floor == floor).all()

    @staticmethod
    def filter_camera_slots(db: Session, camera_id: str):
        return (
            db.query(ParkingSlot)
            .filter(ParkingSlot.camera_id == camera_id)
            .all()
        )

    @staticmethod
    def filter_camera_slots_by_types(db: Session, camera_id: str, slot_types):
        return (
            db.query(ParkingSlot)
            .filter(
                ParkingSlot.camera_id == camera_id,
                ParkingSlot.slot_type.in_(list(slot_types)),
            )
            .all()
        )

    @staticmethod
    def filter_available_slots(db: Session):
        return db.query(ParkingSlot).filter(ParkingSlot.is_available == True).all()
