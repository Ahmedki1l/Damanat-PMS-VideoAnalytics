"""Per-camera state-machine debounce overrides (state_machine.camera_overrides).

The override exists so one camera with a known detector-recall gap (CAM-00, the
near-nadir fisheye roof view) can demand a longer absent-run before a slot flips
VACANT, without slowing vacancy detection on the other 25 cameras.
"""
from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from src.config import AppConfig, load_config
from src.core.engine.camera_pipeline import CameraPipeline
from src.models.slot import ParkingSlot


def _slot(slot_id: str = "G1") -> ParkingSlot:
    return ParkingSlot(id=slot_id, polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))


def _write(tmp_path, body: str) -> str:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(body, encoding="utf-8")
    return str(config_file)


def test_overrides_parse_and_normalize_camera_id(tmp_path) -> None:
    config = load_config(
        _write(
            tmp_path,
            """
state_machine:
  confirm_enter_frames: 5
  confirm_leave_frames: 8
  camera_overrides:
    Cam_00:
      confirm_leave_frames: "16"
""",
        )
    )

    # Globals untouched; the override is keyed by the normalized id.
    assert config.state_machine.confirm_enter_frames == 5
    assert config.state_machine.confirm_leave_frames == 8
    assert config.state_machine.camera_overrides == {"CAM00": {"confirm_leave_frames": 16}}


def test_unknown_override_key_fails_at_config_load(tmp_path) -> None:
    """A typo must raise, not silently leave the camera on the global debounce —
    the symptom (a slot flipping too early) is what the override was added to fix."""
    with pytest.raises(ValueError, match="unknown key"):
        load_config(
            _write(
                tmp_path,
                """
state_machine:
  camera_overrides:
    CAM-00:
      confirm_leave_franes: 16
""",
            )
        )


@pytest.mark.parametrize("bad", ["0", "-3", "banana"])
def test_non_positive_or_non_int_override_fails_at_config_load(tmp_path, bad) -> None:
    with pytest.raises(ValueError, match="must be an int|must be >= 1"):
        load_config(
            _write(
                tmp_path,
                f"""
state_machine:
  camera_overrides:
    CAM-00:
      confirm_leave_frames: {bad}
""",
            )
        )


def test_pipeline_applies_override_only_to_the_named_camera() -> None:
    config = AppConfig()
    config.state_machine.confirm_enter_frames = 5
    config.state_machine.confirm_leave_frames = 8
    config.state_machine.camera_overrides = {"CAM00": {"confirm_leave_frames": 16}}

    overridden = CameraPipeline("CAM-00", "Ground Floor", [_slot()], config)
    machine = overridden.state_machines["G1"]
    assert machine.confirm_leave_frames == 16
    # Unset keys inherit the global (DB-owned) value.
    assert machine.confirm_enter_frames == 5

    untouched = CameraPipeline("CAM-01", "Ground Floor", [_slot()], config)
    assert untouched.state_machines["G1"].confirm_leave_frames == 8


def test_resolve_for_camera_leaves_global_config_unmutated() -> None:
    config = AppConfig()
    config.state_machine.confirm_leave_frames = 8
    config.state_machine.camera_overrides = {"CAM00": {"confirm_leave_frames": 16}}

    resolved = config.state_machine.resolve_for_camera("CAM-00")

    assert resolved.confirm_leave_frames == 16
    assert config.state_machine.confirm_leave_frames == 8
    # The resolved copy must not carry the override map onward.
    assert resolved.camera_overrides == {}
    # A camera with no entry gets the shared instance back — no allocation.
    assert config.state_machine.resolve_for_camera("CAM-07") is config.state_machine
