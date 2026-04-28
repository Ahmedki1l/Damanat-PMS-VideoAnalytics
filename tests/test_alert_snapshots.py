import importlib.util
import os
from pathlib import Path
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append(".")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics_test"))

from src.models.state_machine import SlotEvent
from src.services import alert_service, slot_status_service

sys.modules.setdefault("src.camera_manager", types.SimpleNamespace(CameraConfig=object))
sys.modules.setdefault(
    "src.core.engine.camera_pipeline",
    types.SimpleNamespace(CameraPipeline=object),
)
sys.modules.setdefault(
    "src.detection.tracking_manager",
    types.SimpleNamespace(TrackingManager=object),
)
sys.modules.setdefault(
    "src.model.parkingslot",
    types.SimpleNamespace(ParkingSlot=object),
)
sys.modules.setdefault(
    "src.models.slot",
    types.SimpleNamespace(load_slots=lambda *args, **kwargs: ([], None)),
)
sys.modules.setdefault(
    "src.services.parking_service",
    types.SimpleNamespace(sync_slots_from_config=lambda *args, **kwargs: None),
)

engine_runtime_spec = importlib.util.spec_from_file_location(
    "test_engine_runtime_module",
    os.path.join("src", "core", "engine", "engine_runtime.py"),
)
engine_runtime_module = importlib.util.module_from_spec(engine_runtime_spec)
assert engine_runtime_spec.loader is not None
engine_runtime_spec.loader.exec_module(engine_runtime_module)
ParkingEngineRuntimeMixin = engine_runtime_module.ParkingEngineRuntimeMixin


class DummyEngine(ParkingEngineRuntimeMixin):
    def __init__(self):
        self.pipelines = {}
        self._recent_violators = []
        self._violation_history_limit = 60
        self._violation_match_threshold = 0.95
        self.vehicle_registry = None
        self.db_manager = None


def _make_repo_temp_dir() -> Path:
    root = Path(".codex_test_tmp").resolve()
    root.mkdir(exist_ok=True)
    path = root / f"alert_snapshot_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_named_slot_violation_creates_permanent_alert_snapshot(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine()
    engine.pipelines["CAM_04"] = SimpleNamespace(
        state_machines={"B3_CEO": SimpleNamespace(is_violation_zone=False)}
    )

    detection = SimpleNamespace(bbox=[10, 10, 80, 90])
    assignment = SimpleNamespace(slot_vehicle_map={"B3_CEO": (101, detection)})
    frame = np.full((120, 120, 3), 255, dtype=np.uint8)
    event = SlotEvent(
        event_type="vehicle_parked",
        slot_id="B3_CEO",
        track_id=101,
        timestamp="2026-04-28T10:00:00",
    )

    result = engine._filter_violation_events(frame, assignment, "CAM_04", [event])

    assert len(result) == 1
    assert result[0].event_type == "named_slot_violation"
    assert result[0].snapshot_path.startswith(os.path.join("vehicle_images", "alerts"))
    assert os.path.exists(temp_dir / result[0].snapshot_path)


def test_violation_slot_creates_permanent_alert_snapshot(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine()
    engine.pipelines["CAM_02"] = SimpleNamespace(
        state_machines={"Violation_1": SimpleNamespace(is_violation_zone=True)}
    )

    detection = SimpleNamespace(bbox=[15, 15, 75, 85])
    assignment = SimpleNamespace(slot_vehicle_map={"Violation_1": (202, detection)})
    frame = np.full((120, 120, 3), 127, dtype=np.uint8)
    event = SlotEvent(
        event_type="vehicle_parked",
        slot_id="Violation_1",
        track_id=202,
        timestamp="2026-04-28T10:05:00",
    )

    result = engine._filter_violation_events(frame, assignment, "CAM_02", [event])

    assert len(result) == 1
    assert result[0].event_type == "vehicle_violation"
    assert result[0].snapshot_path.startswith(os.path.join("vehicle_images", "alerts"))
    assert os.path.exists(temp_dir / result[0].snapshot_path)


def test_log_vehicle_event_passes_snapshot_path_to_report_alert():
    fake_db = MagicMock()
    fake_slot = SimpleNamespace(
        is_available=True,
        slot_id="Violation_1",
        slot_name="Violation 1",
        zone_id="B1",
        zone_name="B1",
        floor="B1",
    )
    created_status = SimpleNamespace(time=None)

    with patch.object(
        slot_status_service.ParkingSlotRepository,
        "get_by_id",
        return_value=fake_slot,
    ), patch.object(
        slot_status_service.SlotStatusRepository,
        "create",
        return_value=created_status,
    ), patch.object(
        alert_service,
        "report_alert",
        return_value=SimpleNamespace(id=42),
    ) as report_alert_mock:
        _, alert_id = slot_status_service.log_vehicle_event(
            db=fake_db,
            slot_id="Violation_1",
            plate=None,
            is_parked=True,
            camera_id="CAM_02",
            severity="critical",
            snapshot_path="vehicle_images/alerts/test_alert.jpg",
        )

    assert alert_id == 42
    report_alert_mock.assert_called_once_with(
        fake_db,
        "Violation_1",
        None,
        camera_id="CAM_02",
        severity="critical",
        snapshot_path="vehicle_images/alerts/test_alert.jpg",
    )


def test_update_current_slot_plate_updates_latest_occupied_row():
    fake_db = MagicMock()
    fake_slot = SimpleNamespace(
        slot_id="SLOT_01",
        slot_name="Slot 01",
        zone_id="B1",
        zone_name="North",
        floor="B1",
    )
    latest_status = SimpleNamespace(
        slot_id="SLOT_01",
        plate_number="",
        status="occupied",
        time=None,
    )

    with patch.object(
        slot_status_service.SlotStatusRepository,
        "get_latest_by_slot",
        return_value=latest_status,
    ), patch.object(
        slot_status_service.ParkingSlotRepository,
        "get_by_id",
        return_value=fake_slot,
    ), patch.object(
        slot_status_service.pms_api_client,
        "bind_slot_session",
    ) as bind_slot_session_mock, patch.object(
        slot_status_service,
        "log_vehicle_event",
    ) as log_vehicle_event_mock:
        updated = slot_status_service.update_current_slot_plate(
            db=fake_db,
            slot_id="SLOT_01",
            plate="ABC123",
            camera_id="CAM_04",
        )

    assert updated is latest_status
    assert latest_status.plate_number == "ABC123"
    fake_db.commit.assert_called_once()
    fake_db.refresh.assert_called_once_with(latest_status)
    bind_slot_session_mock.assert_called_once()
    log_vehicle_event_mock.assert_not_called()


def test_report_alert_backfills_existing_active_alert_snapshot():
    fake_db = MagicMock()
    existing_alert = SimpleNamespace(id=9, snapshot_path=None)
    fake_slot = SimpleNamespace(
        zone_id="B1",
        zone_name="B1",
        slot_name="Violation 1",
        last_snapshot_path="slot_Violation_1_latest.jpg",
    )

    with patch.object(alert_service, "check_slot_restricted", return_value=True), patch.object(
        alert_service.ParkingSlotRepository,
        "get_by_id",
        return_value=fake_slot,
    ), patch.object(
        alert_service.AlertRepository,
        "get_active_by_slot",
        return_value=existing_alert,
    ):
        result = alert_service.report_alert(
            fake_db,
            "Violation_1",
            snapshot_path="vehicle_images/alerts/permanent.jpg",
        )

    assert result is existing_alert
    assert existing_alert.snapshot_path == "vehicle_images/alerts/permanent.jpg"
    fake_db.commit.assert_called_once()
    fake_db.refresh.assert_called_once_with(existing_alert)


def test_report_alert_falls_back_to_slot_snapshot_when_dedicated_snapshot_missing():
    fake_db = MagicMock()
    fake_slot = SimpleNamespace(
        zone_id="B1",
        zone_name="B1",
        slot_name="Violation 1",
        last_snapshot_path="slot_Violation_1_latest.jpg",
    )

    with patch.object(alert_service, "check_slot_restricted", return_value=True), patch.object(
        alert_service.ParkingSlotRepository,
        "get_by_id",
        return_value=fake_slot,
    ), patch.object(
        alert_service.AlertRepository,
        "get_active_by_slot",
        return_value=None,
    ), patch.object(
        alert_service.AlertRepository,
        "create",
        side_effect=lambda db, alert_model: alert_model,
    ):
        alert = alert_service.report_alert(fake_db, "Violation_1", snapshot_path=None)

    assert alert.snapshot_path == "slot_Violation_1_latest.jpg"
