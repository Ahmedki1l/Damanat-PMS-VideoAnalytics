from sqlalchemy.orm import Session
from src.model import Intrusion
from src.repositories import IntrusionRepository, ParkingSlotRepository


def check_slot_restricted(db: Session, slot_id: str) -> bool:
    """Check if a slot has the is_violation_zone flag enabled in the DB."""
    slot = ParkingSlotRepository.get_by_id(db, slot_id)
    return slot.is_violation_zone if slot else False


def report_intrusion(db: Session, slot_id: str, plate_number: str = None, camera_id: str = None):
    """
    YOLO detects car in a slot → check if restricted (is_violation_zone) → create intrusion.
    Returns None if slot is not restricted or intrusion already active.
    """
    if not check_slot_restricted(db, slot_id):
        return None

    # Don't duplicate — check if there's already an active intrusion on this slot
    existing = IntrusionRepository.get_active_by_slot(db, slot_id)
    if existing:
        return existing

    new_intrusion = Intrusion(
        slot_id=slot_id,
        plate_number=plate_number,
        status="active",
        camera_id=camera_id
    )
    return IntrusionRepository.create(db, new_intrusion)


def resolve_intrusion(db: Session, slot_id: str):
    """Car leaves restricted slot → resolve the active intrusion."""
    active = IntrusionRepository.get_active_by_slot(db, slot_id)
    if not active:
        return None
    return IntrusionRepository.resolve(db, active.id)


def get_active_intrusions(db: Session):
    """Get all currently active (unresolved) intrusions."""
    return IntrusionRepository.get_all_active(db)


def get_intrusion_history(db: Session, slot_id: str):
    """Get full intrusion history for a specific slot."""
    return IntrusionRepository.get_history_by_slot(db, slot_id)
