from sqlalchemy import or_
from sqlalchemy.orm import Session
from src.model import Alert
from src.utils.datetime_helper import facility_now_naive
from typing import Optional

class AlertRepository:
    @staticmethod
    def create(db: Session, alert_model: Alert):
        db.add(alert_model)
        db.commit()
        db.refresh(alert_model)
        return alert_model

    @staticmethod
    def get_active_by_slot(db: Session, slot_id: str, alert_type: Optional[str] = None):
        query = db.query(Alert).filter(
            or_(Alert.slot_id == slot_id, Alert.zone_id == slot_id),
            Alert.is_resolved == False
        )
        if alert_type:
            query = query.filter(Alert.alert_type == alert_type)
        return query.first()

    @staticmethod
    def get_all(db: Session, is_resolved: Optional[bool] = None, alert_type: Optional[str] = None):
        query = db.query(Alert)
        if is_resolved is not None:
            query = query.filter(Alert.is_resolved == is_resolved)
        if alert_type:
            query = query.filter(Alert.alert_type == alert_type)
        return query.order_by(Alert.triggered_at.desc()).all()

    @staticmethod
    def get_history_by_slot(db: Session, slot_id: str):
        return db.query(Alert).filter(
            or_(Alert.slot_id == slot_id, Alert.zone_id == slot_id)
        ).order_by(Alert.triggered_at.desc()).all()

    @staticmethod
    def resolve(db: Session, alert_id: int):
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert and not alert.is_resolved:
            alert.is_resolved = True
            alert.resolved_at = facility_now_naive()
            db.commit()
            db.refresh(alert)
        return alert
