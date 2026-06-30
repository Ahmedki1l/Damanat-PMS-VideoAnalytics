"""Regression: _update_slot_state must skip plate-related registry calls on
ground-floor cameras (gated by ``is_reid_disabled_floor``), but must still call
them on cameras that DO participate in identity matching (e.g. CAM-03 on B1)."""
import importlib.util
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.append(".")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics_test"))

from src.models.state_machine import SlotState, SlotStateMachine  # noqa: E402

# Stub modules that engine_runtime.py imports at module scope but that pull
# in heavy ML deps we don't need for this gate test. Mirrors the boilerplate
# in tests/test_alert_snapshots.py.
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
    types.SimpleNamespace(
        sync_slots_from_config=lambda *args, **kwargs: None,
        bootstrap_camera_slots_from_json=lambda *args, **kwargs: None,
        load_camera_slots=lambda *args, **kwargs: ([], [], None, []),
    ),
)

engine_runtime_spec = importlib.util.spec_from_file_location(
    "test_engine_runtime_plate_gate_module",
    os.path.join("src", "core", "engine", "engine_runtime.py"),
)
engine_runtime_module = importlib.util.module_from_spec(engine_runtime_spec)
assert engine_runtime_spec.loader is not None
engine_runtime_spec.loader.exec_module(engine_runtime_module)
ParkingEngineRuntimeMixin = engine_runtime_module.ParkingEngineRuntimeMixin
is_reid_disabled_floor = engine_runtime_module.is_reid_disabled_floor


class _DummyEngine(ParkingEngineRuntimeMixin):
    def __init__(self):
        self.pipelines = {}
        self.vehicle_registry = MagicMock()
        # try_link_to_slot defaults to returning None (matching the registry
        # blocklist behaviour) so any assertion failure that lets the call
        # through still yields a sensible mock return.
        self.vehicle_registry.try_link_to_slot.return_value = None
        self.vehicle_registry.get_plate_for_track.return_value = None
        self.db_manager = None

    # Stub out the snapshot helpers so the test doesn't touch disk.
    def _save_slot_snapshot(self, *args, **kwargs):
        return None

    def _persist_slot_snapshot_path(self, *args, **kwargs):
        return None

    def _save_car_crop(self, *args, **kwargs):
        return None

    def _persist_late_slot_plate(self, *args, **kwargs):
        return None


def _build_pipeline(slot_id: str, floor: str):
    slot = SimpleNamespace(id=slot_id, label=slot_id, zone_id="Z", zone_name="Zone")
    # Pre-seed the state machine into ENTERING so a single update() call with
    # vehicle_present=True transitions ENTERING -> OCCUPIED and emits the
    # vehicle_parked event we want to inspect.
    state_machine = SlotStateMachine(slot_id=slot_id, confirm_enter_frames=1)
    state_machine.state = SlotState.ENTERING
    state_machine._enter_counter = 0
    return SimpleNamespace(
        slots=[slot],
        state_machines={slot_id: state_machine},
        floor=floor,
    )


def _vehicle_parked_event(engine, cam_id: str, slot_id: str, floor: str):
    detection = SimpleNamespace(bbox=(10.0, 10.0, 80.0, 90.0))
    track_id = 42
    frame = np.full((120, 120, 3), 255, dtype=np.uint8)
    assignment = SimpleNamespace(slot_vehicle_map={slot_id: (track_id, detection)})
    events = engine._update_slot_state(cam_id, frame, engine.pipelines[cam_id], assignment)
    return events, track_id


def test_disabled_camera_skips_plate_registry_calls():
    cam_id = "CAM-01"
    assert is_reid_disabled_floor("Ground Floor"), (
        "Sanity check: the ground floor must be identity-disabled"
    )
    engine = _DummyEngine()
    engine.pipelines[cam_id] = _build_pipeline("G1", "Ground Floor")

    events, _ = _vehicle_parked_event(engine, cam_id, "G1", "Ground Floor")

    # vehicle_parked event still fires (slot occupancy detection is unaffected).
    parked_events = [e for e in events if e.event_type == "vehicle_parked"]
    assert len(parked_events) == 1, (
        f"Expected exactly one vehicle_parked event on disabled cam; got {events!r}"
    )
    assert parked_events[0].plate_number == "", (
        "Plate must remain empty on a disabled camera"
    )

    # None of the plate-related registry calls should have happened.
    engine.vehicle_registry.get_plate_for_track.assert_not_called()
    engine.vehicle_registry.try_link_to_slot.assert_not_called()


def test_enabled_camera_still_calls_plate_registry():
    cam_id = "CAM-03"
    assert not is_reid_disabled_floor("B1"), (
        "Sanity check: B1 must NOT be identity-disabled"
    )
    engine = _DummyEngine()
    engine.pipelines[cam_id] = _build_pipeline("B1_01", "B1")

    events, track_id = _vehicle_parked_event(engine, cam_id, "B1_01", "B1")

    parked_events = [e for e in events if e.event_type == "vehicle_parked"]
    assert len(parked_events) == 1

    # Both plate-related registry calls must happen on an enabled camera.
    # try_link_to_slot is invoked twice per frame on OCCUPIED slots: once in
    # the vehicle_parked branch and once in the OCCUPIED/LEAVING refresh.
    engine.vehicle_registry.get_plate_for_track.assert_called_once_with(cam_id, track_id)
    assert engine.vehicle_registry.try_link_to_slot.call_count >= 1
    for call in engine.vehicle_registry.try_link_to_slot.call_args_list:
        assert call.kwargs["camera_id"] == cam_id
        assert call.kwargs["slot_id"] == "B1_01"
        assert call.kwargs["track_id"] == track_id


def test_startup_cleanup_frees_plate_on_disabled_camera_slot():
    """If a CAM-01/CAM-02 slot has a plate bound in slot_status from before
    plate matching was disabled, the startup cleanup must clear it (set
    plate_number=None) and call pms_api_client.unbind_slot_session. CAM-03
    slots must be left untouched."""
    from unittest.mock import patch

    from src.services import slot_status_service

    engine = _DummyEngine()
    engine.db_manager = MagicMock()
    engine.db_manager.SessionLocal.return_value = MagicMock()

    # Two disabled-camera pipelines (CAM-01 with G1; CAM-02 with G4) and
    # one enabled pipeline (CAM-03 with B1_01) — only G1 has a stale plate.
    engine.pipelines["CAM-01"] = _build_pipeline("G1", "Ground Floor")
    engine.pipelines["CAM-02"] = _build_pipeline("G4", "Ground Floor")
    engine.pipelines["CAM-03"] = _build_pipeline("B1_01", "B1")

    stale_row = SimpleNamespace(
        slot_id="G1",
        plate_number="ABC-123",
        status="occupied",
        time=None,
    )
    empty_row = SimpleNamespace(
        slot_id="G4",
        plate_number=None,
        status="occupied",
        time=None,
    )
    b1_row = SimpleNamespace(
        slot_id="B1_01",
        plate_number="XYZ-999",
        status="occupied",
        time=None,
    )
    rows_by_slot = {"G1": stale_row, "G4": empty_row, "B1_01": b1_row}

    def fake_get_latest_by_slot(session, slot_id):
        return rows_by_slot.get(slot_id)

    with patch.object(
        slot_status_service.SlotStatusRepository,
        "get_latest_by_slot",
        side_effect=fake_get_latest_by_slot,
    ), patch.object(
        slot_status_service,
        "update_current_slot_plate",
        wraps=slot_status_service.update_current_slot_plate,
    ) as update_mock, patch.object(
        engine_runtime_module,
        "update_current_slot_plate",
    ) as engine_update_mock:
        # Re-bind the SlotStatusRepository import in engine_runtime's
        # local namespace too, since _free_plates_on_disabled_cameras does
        # `from src.repositories import SlotStatusRepository` at call time.
        engine._free_plates_on_disabled_cameras()

    # update_current_slot_plate was called exactly once — for the only
    # disabled-camera slot that had a non-empty plate (G1 on CAM-01).
    assert engine_update_mock.call_count == 1, (
        f"Expected exactly one cleanup call; got "
        f"{engine_update_mock.call_args_list!r}"
    )
    call_kwargs = engine_update_mock.call_args.kwargs
    assert call_kwargs["slot_id"] == "G1"
    assert call_kwargs["plate"] is None
    assert call_kwargs["camera_id"] == "CAM-01"

    # G4 (disabled cam but no plate) and B1_01 (enabled cam) were NOT touched.
    cleaned_slot_ids = {c.kwargs["slot_id"] for c in engine_update_mock.call_args_list}
    assert "G4" not in cleaned_slot_ids
    assert "B1_01" not in cleaned_slot_ids


def test_startup_cleanup_no_op_when_db_manager_absent():
    engine = _DummyEngine()
    engine.db_manager = None
    # Must not raise even with no DB and no pipelines.
    engine._free_plates_on_disabled_cameras()


def test_disabled_camera_clears_stale_plate_safety_net():
    """If a stale plate is somehow left on a disabled camera's state machine,
    the safety-net branch clears it on the next frame with a car in slot."""
    cam_id = "CAM-02"
    engine = _DummyEngine()
    engine.pipelines[cam_id] = _build_pipeline("G4", "Ground Floor")
    state_machine = engine.pipelines[cam_id].state_machines["G4"]
    # Simulate a legacy stale binding from before the proactive gate landed.
    state_machine.state = SlotState.OCCUPIED
    state_machine.plate_number = "STALE-PLATE-123"

    _vehicle_parked_event(engine, cam_id, "G4", "Ground Floor")

    assert state_machine.plate_number == "", (
        "Stale plate must be cleared by the safety-net branch"
    )
    # Safety net also evicts the registry's _parked entry.
    engine.vehicle_registry.unlink_slot.assert_called_with("G4")
    # And it still never calls try_link_to_slot or get_plate_for_track.
    engine.vehicle_registry.get_plate_for_track.assert_not_called()
    engine.vehicle_registry.try_link_to_slot.assert_not_called()
