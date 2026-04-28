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
