from sqlalchemy.orm import Session
from src.model import SlotStatus

class SlotStatusRepository:
    @staticmethod
    def create(db: Session, status_model: SlotStatus):
        db.add(status_model)
        db.commit()
        db.refresh(status_model)
        return status_model

    @staticmethod
    def get_history_by_slot(db: Session, slot_id: str):
        return db.query(SlotStatus).filter(SlotStatus.slot_id == slot_id).all()

    @staticmethod
    def get_latest_by_slot(db: Session, slot_id: str):
        return db.query(SlotStatus).filter(SlotStatus.slot_id == slot_id).order_by(SlotStatus.time.desc()).first()

    @staticmethod
    def get_latest_by_plate(db: Session, plate: str):
        return db.query(SlotStatus).filter(SlotStatus.plate_number == plate).order_by(SlotStatus.time.desc()).first()
    
    