"""Special-needs bays retain normal occupancy and identity facts without alerts."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from src.database import Base
from src.model import Alert, ParkingSlot
from src.services.named_slot_service import is_restricted_slot
from src.services import alert_service, slot_status_service


@pytest.fixture(autouse=True)
def _stub_pms_session_boundary(monkeypatch):
    monkeypatch.setattr(slot_status_service.pms_api_client, "bind_slot_session", lambda **_: None)
    monkeypatch.setattr(slot_status_service.pms_api_client, "unbind_slot_session", lambda **_: None)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _special_slot(*, is_violation_zone=False):
    return ParkingSlot(
        slot_id="G1_SPECIAL",
        slot_name="G1 Special Needs",
        camera_id="CAM-19",
        floor="G",
        zone_id="G-NORTH",
        zone_name="G North",
        reservation_type="SPECIAL",
        is_violation_zone=is_violation_zone,
        is_available=True,
    )


def test_special_occupancy_and_later_identity_remain_durable_without_alerts(monkeypatch):
    monkeypatch.setattr(alert_service, "_ENABLE_RESTRICTED_ZONE_ALERTS", True)
    db = _session()
    try:
        slot = _special_slot()
        db.add(slot)
        db.commit()

        occupied, alert_id = slot_status_service.log_vehicle_event(
            db, slot.slot_id, plate=None, is_parked=True, camera_id="CAM-19"
        )
        assert (occupied.status, occupied.plate_number, alert_id) == ("occupied", None, None)
        assert slot.is_available is False

        enriched = slot_status_service.update_current_slot_plate(
            db, slot.slot_id, "ABC-123", camera_id="CAM-19"
        )
        assert enriched.plate_number == "ABC-123"
        assert slot.current_plate == "ABC-123"
        assert db.query(Alert).filter(Alert.slot_id == slot.slot_id).count() == 0
    finally:
        db.close()


def test_special_slot_blocks_explicit_legacy_alert_types_even_if_marked_violation(monkeypatch):
    monkeypatch.setattr(alert_service, "_ENABLE_RESTRICTED_ZONE_ALERTS", True)
    monkeypatch.setattr(alert_service, "_DISABLED_ALERT_TYPES", frozenset())
    db = _session()
    try:
        slot = _special_slot(is_violation_zone=True)
        db.add(slot)
        db.commit()
        assert is_restricted_slot(slot) is False

        for alert_type in ("special_needs_review", "special_needs_violation", "vehicle_violation"):
            assert alert_service.report_alert(
                db, slot.slot_id, camera_id="CAM-19", alert_type=alert_type
            ) is None
        assert db.query(Alert).filter(Alert.slot_id == slot.slot_id).count() == 0

        # A physical violation bay remains actionable when it is not SPECIAL.
        slot.reservation_type = "GENERAL"
        db.commit()
        assert is_restricted_slot(slot) is True
        alert = alert_service.report_alert(
            db, slot.slot_id, camera_id="CAM-19", alert_type="vehicle_violation"
        )
        assert alert is not None

        # Explicit retired types are rejected regardless of the slot class.
        assert alert_service.report_alert(
            db, slot.slot_id, camera_id="CAM-19", alert_type="special_needs_violation"
        ) is None
    finally:
        db.close()
