from sqlalchemy.orm import Session
from src.model import Intrusion
from datetime import datetime

class IntrusionRepository:
    @staticmethod
    def create(db: Session, intrusion_model: Intrusion):
        db.add(intrusion_model)
        db.commit()
        db.refresh(intrusion_model)
        return intrusion_model

    @staticmethod
    def get_active_by_slot(db: Session, slot_id: str):
        return db.query(Intrusion).filter(
            Intrusion.slot_id == slot_id,
            Intrusion.status == "active"
        ).first()

    @staticmethod
    def get_all_active(db: Session):
        return db.query(Intrusion).filter(Intrusion.status == "active").all()

    @staticmethod
    def get_history_by_slot(db: Session, slot_id: str):
        return db.query(Intrusion).filter(
            Intrusion.slot_id == slot_id
        ).order_by(Intrusion.detected_at.desc()).all()

    @staticmethod
    def resolve(db: Session, intrusion_id: int):
        intrusion = db.query(Intrusion).filter(Intrusion.id == intrusion_id).first()
        if intrusion:
            intrusion.status = "resolved"
            intrusion.resolved_at = datetime.utcnow()
            db.commit()
            db.refresh(intrusion)
        return intrusion
