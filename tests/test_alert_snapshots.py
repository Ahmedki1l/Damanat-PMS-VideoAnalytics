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

# Stub the heavy/circular imports ONLY while loading engine_runtime standalone,
# then remove every stub we inserted. Leaving them in sys.modules poisoned every
# test module imported after this one (alphabetically: almost all of them) —
# e.g. `from src.core.engine.camera_pipeline import CameraPipeline` silently
# returned `object`, making test_slot_probe_geometry's ROI tests fail in full
# runs while passing in isolation.
_stubs = {
    "src.camera_manager": types.SimpleNamespace(CameraConfig=object),
    "src.core.engine.camera_pipeline": types.SimpleNamespace(CameraPipeline=object),
    "src.detection.tracking_manager": types.SimpleNamespace(TrackingManager=object),
    # detector pulls in ultralytics (and a YOLO weights load) purely for the
    # is_untracked helper — stub it so these alert tests don't need the CV stack.
    "src.detection.detector": types.SimpleNamespace(
        is_untracked=lambda *args, **kwargs: False
    ),
    "src.model.parkingslot": types.SimpleNamespace(ParkingSlot=object),
    "src.models.slot": types.SimpleNamespace(
        load_slots=lambda *args, **kwargs: ([], None)
    ),
    "src.services.parking_service": types.SimpleNamespace(
        sync_slots_from_config=lambda *args, **kwargs: None,
        bootstrap_camera_slots_from_json=lambda *args, **kwargs: None,
        load_camera_slots=lambda *args, **kwargs: ([], None),
    ),
}
_inserted = [name for name, stub in _stubs.items()
             if sys.modules.setdefault(name, stub) is stub]
try:
    engine_runtime_spec = importlib.util.spec_from_file_location(
        "test_engine_runtime_module",
        os.path.join("src", "core", "engine", "engine_runtime.py"),
    )
    engine_runtime_module = importlib.util.module_from_spec(engine_runtime_spec)
    assert engine_runtime_spec.loader is not None
    engine_runtime_spec.loader.exec_module(engine_runtime_module)
    ParkingEngineRuntimeMixin = engine_runtime_module.ParkingEngineRuntimeMixin
finally:
    for _name in _inserted:
        if sys.modules.get(_name) is _stubs[_name]:
            del sys.modules[_name]


class DummyEngine(ParkingEngineRuntimeMixin):
    def __init__(self, reserved_for=None, special=None, enable_restricted=True,
                 identity_timeout_s=300.0, violation_min_dwell_s=180.0):
        self.pipelines = {}
        self._recent_violators = []
        self._violation_history_limit = 60
        self._violation_match_threshold = 0.95
        self.vehicle_registry = None
        self.db_manager = None
        self._reserved_for_map = dict(reserved_for or {})
        self._special_slots = set(special or ())
        self.event_bus = SimpleNamespace(emit_batch=lambda events: None)
        self.config = SimpleNamespace(
            alerts=SimpleNamespace(
                enable_restricted_zone_alerts=enable_restricted,
                reserved_slot_identity_timeout_s=identity_timeout_s,
                violation_min_dwell_s=violation_min_dwell_s,
            ),
            output=SimpleNamespace(
                snapshot_base_dir="vehicle_images",
                public_base_url="",
                snapshot_url_prefix="",
            ),
        )


def _make_repo_temp_dir() -> Path:
    root = Path(".codex_test_tmp").resolve()
    root.mkdir(exist_ok=True)
    path = root / f"alert_snapshot_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _snapshot_disk_path(temp_dir: Path, url: str) -> Path:
    """'/snapshots/alerts/x.jpg' -> <temp_dir>/vehicle_images/alerts/x.jpg."""
    return temp_dir / "vehicle_images" / "alerts" / url.rsplit("/", 1)[-1]


def _parked_event(slot_id: str, track_id: int) -> SlotEvent:
    return SlotEvent(
        event_type="vehicle_parked",
        slot_id=slot_id,
        track_id=track_id,
        timestamp="2026-04-28T10:00:00",
    )


# ---------------------------------------------------------------------------- #
# Real-facility scenario: B13_COO on CAM-08.
#
# Every constant below is the production value, not an invention:
#   slot + label   slots/b1_cam08.json          -> "B13_COO" / "Slot B13 COO"
#   owner title    migrate_parking_categories.py -> ("B13_COO", "COO")
#   owner's car    DJS-7842, observed live 2026-07-11 driving gate -> B13 COO
#                  (tests/test_self_competition_merge.py)
#   other car      BHD-9990, read at B13/CAM-24 (src/config.py)
#
# This slot is why the feature was muted: the datasets carry 40 vehicle_intrusion
# alerts for B13_COO on 2026-05-07 ALONE (452 across all slots), because the old
# code decided ownership at vehicle_parked when the plate is never known yet. The
# kill switch landed 2026-05-20, a week later.
# ---------------------------------------------------------------------------- #
B13 = "B13_COO"
B13_CAM = "CAM-08"
B13_OWNER_TITLE = "COO"
COO_CAR = "DJS-7842"
OTHER_CAR = "BHD-9990"


def _b13_engine(vehicles, **kwargs):
    """Engine wired for B13_COO, with a fake `vehicles` table backing the real
    ownership lookup so _is_named_slot_vehicle_allowed runs for real.

    `vehicles` maps plate -> title, exactly as the vehicles table stores it.
    """
    engine = DummyEngine(reserved_for={B13: B13_OWNER_TITLE}, **kwargs)
    engine.pipelines[B13_CAM] = SimpleNamespace(
        state_machines={B13: SimpleNamespace(is_violation_zone=False)},
        slots=[
            SimpleNamespace(
                id=B13, label="Slot B13 COO", zone_id="B1", zone_name="B1-PARKING"
            )
        ],
        floor="B1",
    )
    engine.db_manager = SimpleNamespace(SessionLocal=lambda: MagicMock())
    return engine


def _patch_vehicles(vehicles):
    """Patch the vehicles table the ownership check reads."""
    import src.repositories as repos

    return patch.object(
        repos.VehicleRepository,
        "get_by_plate",
        staticmethod(
            lambda db, plate: (
                SimpleNamespace(plate_number=plate, title=vehicles[plate])
                if plate in vehicles
                else None
            )
        ),
    )


def test_b13_coo_owner_parks_no_alert(monkeypatch):
    """The COO parks in the COO's own slot. This must be SILENT.

    Under the old code this exact sequence produced a vehicle_intrusion alert,
    every time, because ownership was decided before the plate existed.
    """
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = _b13_engine(None)
    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type, k.get("plate"))
    )

    # --- t=0: the car settles into B13. Identity has NOT run yet. -----------
    detection = SimpleNamespace(bbox=[420, 0, 570, 60])
    assignment = SimpleNamespace(slot_vehicle_map={B13: (4471, detection)})
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    result = engine._filter_violation_events(
        frame, assignment, B13_CAM, [_parked_event(B13, 4471)]
    )

    assert result[0].event_type == "vehicle_parked", "occupancy must be untouched"
    assert result[0].is_alert is False
    assert raised == [], "THE REGRESSION: alerting here fired on the owner's own car"
    assert B13 in engine._pending_ownership()

    # --- t+~8s: OCR reads the plate and it is the COO's car. ---------------
    with _patch_vehicles({COO_CAR: B13_OWNER_TITLE}):
        engine._evaluate_named_slot_ownership(B13, B13_CAM, COO_CAR)

    assert raised == [], "the slot's own owner must never raise an alert"
    assert B13 not in engine._pending_ownership()


def test_b13_coo_stranger_parks_raises_intrusion(monkeypatch):
    """A different car takes the COO's slot -> a real, earned intrusion alert."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = _b13_engine(None)
    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type, k.get("plate"), k.get("severity"))
    )

    detection = SimpleNamespace(bbox=[420, 0, 570, 60])
    assignment = SimpleNamespace(slot_vehicle_map={B13: (4472, detection)})
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    engine._filter_violation_events(
        frame, assignment, B13_CAM, [_parked_event(B13, 4472)]
    )
    assert raised == []  # still silent until identity lands

    with _patch_vehicles({COO_CAR: B13_OWNER_TITLE, OTHER_CAR: ""}):
        engine._evaluate_named_slot_ownership(B13, B13_CAM, OTHER_CAR)

    assert raised == [(B13, "vehicle_intrusion", OTHER_CAR, "critical")]
    assert B13 not in engine._pending_ownership()


def test_b13_coo_identity_never_lands_falls_back_to_unidentified(monkeypatch):
    """CAM-08 films B13 at a shallow angle and the plate often never reads.

    docs/SLOT_IDENTITY_FINDINGS lists B13_COO under "plate not in frame". If the
    verdict simply waited forever, this slot would have no coverage at all — so
    the deadline converts silence into a lower-severity, honestly-worded alert.
    """
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = _b13_engine(None, identity_timeout_s=300.0)
    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type, k.get("plate"), k.get("severity"))
    )

    detection = SimpleNamespace(bbox=[420, 0, 570, 60])
    assignment = SimpleNamespace(slot_vehicle_map={B13: (4473, detection)})
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    engine._filter_violation_events(
        frame, assignment, B13_CAM, [_parked_event(B13, 4473)]
    )
    parked_at = engine._pending_ownership()[B13]["since"]

    # OCR keeps failing. Frames keep arriving; the sweep runs on each one.
    for elapsed in (30, 120, 299):
        engine._sweep_pending_ownership(parked_at + elapsed)
    assert raised == [], "must not alert while identity may still land"

    engine._sweep_pending_ownership(parked_at + 301)
    assert raised == [(B13, "reserved_slot_unidentified", None, "warning")]

    # The sweep runs every frame for as long as the car stays — one alert only.
    for elapsed in (400, 900, 3600):
        engine._sweep_pending_ownership(parked_at + elapsed)
    assert len(raised) == 1


def test_b13_coo_blank_vehicle_title_flags_the_owner(monkeypatch):
    """Guard on the DATA prerequisite, not the code.

    Ownership is `vehicles.title == parking_slots.reserved_for`. The findings doc
    records vehicles.title as blank for every car — and with it blank the COO's
    OWN car fails the check and is reported as an intruder. Deferring the verdict
    does not save us here; the vehicles table has to be populated. This test
    exists so that regression is loud instead of silent.
    """
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = _b13_engine(None)
    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type, k.get("plate"))
    )
    engine._register_pending_ownership(B13, B13_CAM, None, 1000.0)

    with _patch_vehicles({COO_CAR: ""}):  # title blank, as in production today
        engine._evaluate_named_slot_ownership(B13, B13_CAM, COO_CAR)

    assert raised == [(B13, "vehicle_intrusion", COO_CAR)], (
        "with vehicles.title blank the owner is flagged — populate title for the "
        "8 named-slot vehicles before enabling alerts in production"
    )


def test_named_slot_defers_alert_until_identity_lands(monkeypatch):
    """A named slot with no plate yet must NOT alert.

    This is the regression that made the feature unusable: identity runs AFTER
    occupancy, so plate_number is empty here essentially always, and the old code
    read that as "not the owner" and raised an intrusion alert on every single
    park — including the slot owner's own car.
    """
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B3_CEO": "CEO"})
    engine.pipelines["CAM_04"] = SimpleNamespace(
        state_machines={"B3_CEO": SimpleNamespace(is_violation_zone=False)},
        slots=[SimpleNamespace(id="B3_CEO", label="B3", zone_id="B", zone_name="B")],
        floor="B1",
    )

    detection = SimpleNamespace(bbox=[10, 10, 80, 90])
    assignment = SimpleNamespace(slot_vehicle_map={"B3_CEO": (101, detection)})
    frame = np.full((120, 120, 3), 255, dtype=np.uint8)
    event = _parked_event("B3_CEO", 101)

    result = engine._filter_violation_events(frame, assignment, "CAM_04", [event])

    # Occupancy still flows through, untouched and un-alerted.
    assert len(result) == 1
    assert result[0].event_type == "vehicle_parked"
    assert result[0].is_alert is False
    # ...and the verdict is now pending rather than lost.
    assert "B3_CEO" in engine._pending_ownership()
    assert engine._pending_ownership()["B3_CEO"]["owner_title"] == "CEO"


def test_named_slot_owner_never_alerts(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B3_CEO": "CEO"})
    engine.pipelines["CAM_04"] = SimpleNamespace(
        state_machines={"B3_CEO": SimpleNamespace(is_violation_zone=False)},
        slots=[SimpleNamespace(id="B3_CEO", label="B3", zone_id="B", zone_name="B")],
        floor="B1",
    )
    engine._register_pending_ownership("B3_CEO", "CAM_04", None, 1000.0)

    raised = []
    engine._raise_named_slot_alert = lambda *a, **k: raised.append(a)
    # The CEO's own car.
    engine._is_named_slot_vehicle_allowed = lambda plate, title: True

    engine._evaluate_named_slot_ownership("B3_CEO", "CAM_04", "ABC-123")

    assert raised == []
    assert "B3_CEO" not in engine._pending_ownership()


def test_named_slot_non_owner_raises_intrusion(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B3_CEO": "CEO"})
    engine.pipelines["CAM_04"] = SimpleNamespace(
        state_machines={"B3_CEO": SimpleNamespace(is_violation_zone=False)},
        slots=[SimpleNamespace(id="B3_CEO", label="B3", zone_id="B", zone_name="B")],
        floor="B1",
    )
    engine._register_pending_ownership("B3_CEO", "CAM_04", None, 1000.0)

    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type, k.get("plate"))
    )
    engine._is_named_slot_vehicle_allowed = lambda plate, title: False

    engine._evaluate_named_slot_ownership("B3_CEO", "CAM_04", "XYZ-999")

    assert raised == [("B3_CEO", "vehicle_intrusion", "XYZ-999")]
    assert "B3_CEO" not in engine._pending_ownership()


def test_unidentified_named_slot_alerts_after_timeout(monkeypatch):
    """B1_CRO has OCR disabled and appearance routinely abstains, so identity may
    never land. Without the sweep that slot would have zero intrusion coverage."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B1_CRO": "CRO"}, identity_timeout_s=300.0)
    engine.pipelines["CAM_21"] = SimpleNamespace(
        state_machines={"B1_CRO": SimpleNamespace(is_violation_zone=False)},
        slots=[SimpleNamespace(id="B1_CRO", label="B1", zone_id="B", zone_name="B")],
        floor="B1",
    )
    engine._register_pending_ownership("B1_CRO", "CAM_21", None, 1000.0)

    raised = []
    engine._raise_named_slot_alert = lambda slot_id, entry, alert_type, **k: raised.append(
        (slot_id, alert_type)
    )

    engine._sweep_pending_ownership(1000.0 + 299.0)
    assert raised == []  # still inside the deadline

    engine._sweep_pending_ownership(1000.0 + 301.0)
    assert raised == [("B1_CRO", "reserved_slot_unidentified")]

    # Latched: the sweep runs every frame and must not re-alert.
    engine._sweep_pending_ownership(1000.0 + 900.0)
    assert len(raised) == 1


def test_timeout_fallback_can_be_disabled(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B1_CRO": "CRO"}, identity_timeout_s=0.0)
    engine.pipelines["CAM_21"] = SimpleNamespace(slots=[], floor="B1")
    engine._register_pending_ownership("B1_CRO", "CAM_21", None, 1000.0)

    raised = []
    engine._raise_named_slot_alert = lambda *a, **k: raised.append(a)

    engine._sweep_pending_ownership(1000.0 + 100000.0)
    assert raised == []


def test_vacating_clears_pending_verdict(monkeypatch):
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(reserved_for={"B3_CEO": "CEO"})
    engine.pipelines["CAM_04"] = SimpleNamespace(slots=[], floor="B1")
    engine._register_pending_ownership("B3_CEO", "CAM_04", None, 1000.0)
    assert "B3_CEO" in engine._pending_ownership()

    engine._clear_pending_ownership("B3_CEO")

    raised = []
    engine._raise_named_slot_alert = lambda *a, **k: raised.append(a)
    engine._sweep_pending_ownership(1000.0 + 99999.0)
    assert raised == []


def test_special_needs_slot_still_alerts_immediately(monkeypatch):
    """Special-needs and violation zones do NOT depend on identity, so they must
    keep deciding at park time rather than being dragged into the deferral."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(special={"G1"})
    engine.pipelines["CAM_01"] = SimpleNamespace(
        state_machines={"G1": SimpleNamespace(is_violation_zone=False)},
        slots=[SimpleNamespace(id="G1", label="G1", zone_id="G", zone_name="G")],
        floor="G",
    )
    detection = SimpleNamespace(bbox=[10, 10, 80, 90])
    assignment = SimpleNamespace(slot_vehicle_map={"G1": (7, detection)})
    frame = np.full((120, 120, 3), 255, dtype=np.uint8)

    result = engine._filter_violation_events(
        frame, assignment, "CAM_01", [_parked_event("G1", 7)]
    )

    assert result[0].event_type == "special_needs_violation"
    assert result[0].is_alert is True
    assert "G1" not in engine._pending_ownership()


def _violation_setup(engine):
    engine.pipelines["CAM_02"] = SimpleNamespace(
        state_machines={"Violation_1": SimpleNamespace(is_violation_zone=True)},
        slots=[],
        floor="B1",
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
    return assignment, frame, event


def test_violation_slot_creates_permanent_alert_snapshot(monkeypatch):
    """The snapshot still lands — now on the deferred alert, once the car stays."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(violation_min_dwell_s=180.0)
    assignment, frame, event = _violation_setup(engine)

    emitted = []
    engine.event_bus = SimpleNamespace(emit_batch=emitted.extend)

    # Contact: occupancy passes through, no alert yet.
    result = engine._filter_violation_events(frame, assignment, "CAM_02", [event])
    assert len(result) == 1
    assert result[0].event_type == "vehicle_parked"
    assert emitted == []

    # Still there past the dwell — now it is a violation.
    entry = engine._pending_violations()["Violation_1"]
    engine._sweep_pending_violations(entry["since"] + 181.0)

    assert len(emitted) == 1
    assert emitted[0].event_type == "vehicle_violation"
    assert emitted[0].severity == "critical"
    # _save_alert_snapshot returns the externally-reachable URL, not a bare path.
    assert "/alerts/" in emitted[0].snapshot_path
    assert _snapshot_disk_path(temp_dir, emitted[0].snapshot_path).exists()


def test_a_car_passing_through_a_violation_zone_raises_nothing(monkeypatch):
    """The 2026-09-02 Violation-MAIN noise: 31 alerts, median dwell 64s, all traffic."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(violation_min_dwell_s=180.0)
    assignment, frame, event = _violation_setup(engine)

    emitted = []
    engine.event_bus = SimpleNamespace(emit_batch=emitted.extend)

    engine._filter_violation_events(frame, assignment, "CAM_02", [event])
    entry = engine._pending_violations()["Violation_1"]

    # 64 seconds later the slot goes VACANT — the car drove on.
    engine._sweep_pending_violations(entry["since"] + 64.0)
    engine._clear_pending_violation("Violation_1")

    assert emitted == []
    assert engine._pending_violations() == {}

    # And the sweep cannot resurrect it afterwards.
    engine._sweep_pending_violations(entry["since"] + 600.0)
    assert emitted == []


def test_violation_dwell_of_zero_restores_alert_on_contact(monkeypatch):
    """The escape hatch: dwell <= 0 is the pre-2026-09 behaviour, unchanged."""
    temp_dir = _make_repo_temp_dir()
    monkeypatch.chdir(temp_dir)
    engine = DummyEngine(violation_min_dwell_s=0.0)
    assignment, frame, event = _violation_setup(engine)

    result = engine._filter_violation_events(frame, assignment, "CAM_02", [event])

    assert len(result) == 1
    assert result[0].event_type == "vehicle_violation"
    assert "/alerts/" in result[0].snapshot_path
    assert _snapshot_disk_path(temp_dir, result[0].snapshot_path).exists()
    assert engine._pending_violations() == {}


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
    existing_alert = SimpleNamespace(
        id=9, snapshot_path=None, alert_type="vehicle_violation"
    )
    fake_slot = SimpleNamespace(
        zone_id="B1",
        zone_name="B1",
        slot_name="Violation 1",
        # A violation ZONE, not a reserved slot — get_alert_type_for_slot reads
        # reservation_type before falling through to the slot-name check.
        reservation_type="GENERAL",
        is_violation_zone=True,
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
        # A violation ZONE, not a reserved slot — get_alert_type_for_slot reads
        # reservation_type before falling through to the slot-name check.
        reservation_type="GENERAL",
        is_violation_zone=True,
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
