from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.config import MotionSchedulingConfig, load_config
from src.core.fair_latest_output import (
    FairLatestOutputQueue,
    InferenceCompletionGate,
    InferenceOutput,
)
from src.core.motion_scheduler import (
    MotionFrame,
    MotionScheduler,
    MotionSignal,
    motion_iteration_plan,
    motion_bypass_cameras,
    rotated_camera_ids,
    validate_motion_runtime,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _config(**overrides) -> MotionSchedulingConfig:
    values = {
        "mode": "enforce",
        "analysis_fps": 2.0,
        "analysis_width": 32,
        "pixel_delta": 10,
        "changed_ratio": 0.05,
        "active_hold_seconds": 0.0,
        "sentinel_interval_seconds": 5.0,
        "stale_frame_seconds": 3.0,
    }
    values.update(overrides)
    return MotionSchedulingConfig(**values)


def _frame(value: int = 0) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def _sample(
    camera_id: str,
    sequence: int,
    *,
    value: int = 0,
    epoch: int = 1,
    age: float = 0.0,
) -> MotionFrame:
    return MotionFrame(camera_id, _frame(value), sequence, epoch, age)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_motion_modes_require_the_single_process_scheduler_runtime(mode) -> None:
    with pytest.raises(ValueError, match="VA_SINGLE_PROCESS=1"):
        validate_motion_runtime(mode, scheduler_available=False)

    validate_motion_runtime(mode, scheduler_available=True)


def test_legacy_motion_mode_is_valid_without_scheduler_runtime() -> None:
    validate_motion_runtime("legacy", scheduler_available=False)


def test_shadow_preserves_each_legacy_target_tick_between_motion_samples() -> None:
    analysis_ticks = [True, False, False, True, False]
    plans = [
        motion_iteration_plan(
            mode="shadow",
            bypass=False,
            target_due=True,
            analysis_due=analysis_due,
        )
        for analysis_due in analysis_ticks
    ]

    assert [should_read for should_read, _ in plans] == [True] * 5
    assert [should_analyze for _, should_analyze in plans] == analysis_ticks


def test_shadow_can_sample_motion_between_legacy_target_ticks_without_yolo() -> None:
    should_read, should_analyze = motion_iteration_plan(
        mode="shadow",
        bypass=False,
        target_due=False,
        analysis_due=True,
    )

    assert should_read is True
    assert should_analyze is True


def test_each_camera_has_an_independent_baseline_and_sequence() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(), monotonic_clock=clock)

    first_a = scheduler.examine(_sample("CAM-A", 1))
    assert first_a.signal is MotionSignal.UNKNOWN
    assert first_a.should_infer
    scheduler.mark_submitted("CAM-A", first_a.reason)

    clock.advance(0.5)
    quiet_transition = scheduler.examine(_sample("CAM-A", 2))
    assert quiet_transition.signal is MotionSignal.QUIET
    assert quiet_transition.reason == "transition"
    scheduler.mark_submitted("CAM-A", quiet_transition.reason)

    clock.advance(0.5)
    quiet_a = scheduler.examine(_sample("CAM-A", 3))
    assert not quiet_a.should_infer

    first_b = scheduler.examine(_sample("CAM-B", 1))
    assert first_b.signal is MotionSignal.UNKNOWN
    assert first_b.reason == "baseline_reset"
    assert first_b.should_infer

    metrics = scheduler.metrics()
    assert metrics["CAM-A"]["last_examined_sequence"] == 3
    assert metrics["CAM-B"]["last_examined_sequence"] == 1


def test_duplicate_sequence_consumes_one_analysis_interval() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(analysis_fps=2.0), monotonic_clock=clock)
    scheduler.examine(_sample("CAM-A", 1))

    clock.advance(0.5)
    assert scheduler.analysis_due("CAM-A")
    duplicate = scheduler.examine(_sample("CAM-A", 1))

    assert duplicate.reason == "duplicate_sequence"
    assert not scheduler.analysis_due("CAM-A")
    clock.advance(0.49)
    assert not scheduler.analysis_due("CAM-A")
    clock.advance(0.01)
    assert scheduler.analysis_due("CAM-A")


def test_shadow_duplicate_sequence_preserves_legacy_yolo_tick() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(
        _config(mode="shadow", analysis_fps=2.0),
        monotonic_clock=clock,
    )
    scheduler.examine(_sample("CAM-A", 1))

    clock.advance(0.5)
    duplicate = scheduler.examine(_sample("CAM-A", 1))

    assert duplicate.should_infer is True
    assert duplicate.would_gate is True
    assert duplicate.reason == "shadow_duplicate_sequence"


def test_shadow_passthrough_syncs_reconnect_epoch_before_completion() -> None:
    scheduler = MotionScheduler(
        _config(mode="shadow"),
        monotonic_clock=_Clock(),
    )
    scheduler.examine(_sample("CAM-A", 1, epoch=4))

    passthrough = scheduler.shadow_passthrough(
        _sample("CAM-A", 2, epoch=5, age=0.1)
    )

    assert passthrough.should_infer is True
    assert passthrough.frame_is_valid is True
    assert scheduler.completion_is_valid("CAM-A", 5, 0.1)
    assert not scheduler.completion_is_valid("CAM-A", 4, 0.1)


def test_shadow_passthrough_rejects_stale_frame() -> None:
    scheduler = MotionScheduler(
        _config(mode="shadow"),
        monotonic_clock=_Clock(),
    )

    decision = scheduler.shadow_passthrough(
        _sample("CAM-A", 1, epoch=3, age=4.0)
    )

    assert decision.frame_is_valid is False
    assert decision.reason == "stale_frame"


def test_transition_latch_survives_until_an_inference_is_submitted() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(), monotonic_clock=clock)

    first = scheduler.examine(_sample("CAM-A", 1))
    assert first.should_infer
    # Do not mark the first opportunity submitted: the latch must remain armed.
    clock.advance(0.5)
    second = scheduler.examine(_sample("CAM-A", 2))
    clock.advance(0.5)
    third = scheduler.examine(_sample("CAM-A", 3))
    assert second.reason == "transition"
    assert third.reason == "transition"

    scheduler.mark_submitted("CAM-A", third.reason)
    clock.advance(0.5)
    assert not scheduler.examine(_sample("CAM-A", 4)).should_infer


def test_quiet_camera_gets_a_mandatory_sentinel() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(), monotonic_clock=clock)

    first = scheduler.examine(_sample("CAM-A", 1))
    scheduler.mark_submitted("CAM-A", first.reason)
    clock.advance(0.5)
    transition = scheduler.examine(_sample("CAM-A", 2))
    scheduler.mark_submitted("CAM-A", transition.reason)
    clock.advance(0.5)
    assert scheduler.examine(_sample("CAM-A", 3)).reason == "quiet"

    clock.advance(4.5)
    sentinel = scheduler.examine(_sample("CAM-A", 4))
    assert sentinel.should_infer
    assert sentinel.reason == "sentinel"


def test_motion_and_motion_error_both_schedule_yolo() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(), monotonic_clock=clock)
    first = scheduler.examine(_sample("CAM-A", 1))
    scheduler.mark_submitted("CAM-A", first.reason)

    clock.advance(0.5)
    active = scheduler.examine(_sample("CAM-A", 2, value=255))
    assert active.signal is MotionSignal.ACTIVE
    assert active.should_infer

    def raise_cv_error(*_args, **_kwargs):
        raise cv2.error("motion failed")

    scheduler._low_resolution_gray = raise_cv_error
    clock.advance(0.5)
    failed = scheduler.examine(_sample("CAM-B", 1))
    assert failed.signal is MotionSignal.UNKNOWN
    assert failed.should_infer
    assert failed.reason == "motion_error"


def test_stale_frame_is_unknown_and_cannot_be_occupancy_evidence() -> None:
    scheduler = MotionScheduler(_config(), monotonic_clock=_Clock())

    decision = scheduler.examine(_sample("CAM-A", 1, age=4.0))

    assert decision.signal is MotionSignal.UNKNOWN
    assert not decision.frame_is_valid
    assert decision.reason == "stale_frame"


def test_legacy_mode_still_rejects_stale_provenance_without_motion_work() -> None:
    scheduler = MotionScheduler(
        _config(mode="legacy"),
        monotonic_clock=_Clock(),
    )

    decision = scheduler.examine(_sample("CAM-A", 1, age=4.0))

    assert not decision.frame_is_valid
    assert not scheduler.completion_is_valid("CAM-A", 1, 4.0)


def test_reconnect_resets_baseline_and_rejects_old_epoch_completion() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(_config(), monotonic_clock=clock)
    first = scheduler.examine(_sample("CAM-A", 1, epoch=4))
    scheduler.mark_submitted("CAM-A", first.reason)

    clock.advance(0.5)
    reconnected = scheduler.examine(_sample("CAM-A", 2, epoch=5))

    assert reconnected.signal is MotionSignal.UNKNOWN
    assert reconnected.reason == "baseline_reset"
    assert not scheduler.completion_is_valid("CAM-A", 4, 0.1)
    assert scheduler.completion_is_valid("CAM-A", 5, 0.1)


def test_shadow_mode_measures_gate_but_always_infers() -> None:
    clock = _Clock()
    scheduler = MotionScheduler(
        _config(mode="shadow"),
        monotonic_clock=clock,
    )
    first = scheduler.examine(_sample("CAM-A", 1))
    scheduler.mark_submitted("CAM-A", first.reason)
    clock.advance(0.5)
    transition = scheduler.examine(_sample("CAM-A", 2))
    scheduler.mark_submitted("CAM-A", transition.reason)
    clock.advance(0.5)

    quiet = scheduler.examine(_sample("CAM-A", 3))

    assert quiet.should_infer
    assert quiet.would_gate
    assert quiet.reason == "shadow_quiet"


def test_entry_cameras_and_entry_zones_bypass_motion_gate() -> None:
    bypass = motion_bypass_cameras(
        {
            "CAM-08": {"north_Park-Entry": object()},
            "CAM-09": {"B1_Entrence": object()},
            "CAM-11": {"B2_Entrance": object()},
            "CAM-12": {"B1-entry": object()},
            "CAM-10": {"ordinary-slot": object()},
        },
        entry_cameras={"CAM-24"},
    )
    assert {
        "CAM-23",
        "CAM-03",
        "CAM-08",
        "CAM-09",
            "CAM-11",
            "CAM-12",
        "CAM-24",
    } <= bypass
    assert "CAM-10" not in bypass

    scheduler = MotionScheduler(_config(), bypass_cameras=bypass)
    for sequence in range(1, 4):
        decision = scheduler.examine(_sample("CAM-23", sequence))
        assert decision.should_infer
        assert decision.reason == "bypass"


@pytest.mark.parametrize(
    ("mode", "analysis_fps", "bypassed", "valid"),
    [
        ("enforce", 0.999, False, False),
        ("enforce", 1.0, False, True),
        ("enforce", 0.5, True, True),
        ("shadow", 0.5, False, True),
        ("legacy", 0.5, False, True),
    ],
)
def test_motion_analysis_rate_covers_target_ticks_in_enforce_mode(
    mode, analysis_fps, bypassed, valid
) -> None:
    scheduler = MotionScheduler(
        _config(mode=mode, analysis_fps=analysis_fps),
        bypass_cameras={"CAM-A"} if bypassed else (),
    )

    if valid:
        scheduler.validate_target_rate(1.0, ["CAM-A"])
    else:
        with pytest.raises(ValueError, match="analysis_fps must be >="):
            scheduler.validate_target_rate(1.0, ["CAM-A"])


def test_motion_rate_validation_honors_per_camera_always_infer_override() -> None:
    scheduler = MotionScheduler(
        _config(
            analysis_fps=2.0,
            camera_overrides={
                "CAM07": {"analysis_fps": 0.5},
                "CAM08": {"analysis_fps": 0.5, "always_infer": True},
            },
        ),
    )

    with pytest.raises(ValueError, match="CAM-07:0.5"):
        scheduler.validate_target_rate(1.0, ["CAM-07", "CAM-08"])

    scheduler.validate_target_rate(1.0, ["CAM-08"])


def test_camera_rotation_changes_who_gets_first_access() -> None:
    cameras = ["A", "B", "C"]
    assert rotated_camera_ids(cameras, 0) == ["A", "B", "C"]
    assert rotated_camera_ids(cameras, 1) == ["B", "C", "A"]
    assert rotated_camera_ids(cameras, 2) == ["C", "A", "B"]


def _output(camera_id: str, sequence: int) -> InferenceOutput:
    return InferenceOutput(
        camera_id=camera_id,
        frame=None,
        detections=[],
        sequence=sequence,
        capture_ts=0.0,
        stream_epoch=1,
        submitted_at=0.0,
        inferred_at=0.0,
    )


def test_noisy_camera_can_replace_only_itself_and_cannot_starve_others() -> None:
    outputs = FairLatestOutputQueue()
    outputs.put_latest(_output("NOISY", 1))
    outputs.put_latest(_output("QUIET", 1))
    outputs.put_latest(_output("NOISY", 2))
    outputs.put_latest(_output("NOISY", 3))

    first = outputs.get(timeout=0.0)
    outputs.put_latest(_output("NOISY", 4))
    second = outputs.get(timeout=0.0)
    third = outputs.get(timeout=0.0)

    assert (first.camera_id, first.sequence) == ("NOISY", 3)
    assert (second.camera_id, second.sequence) == ("QUIET", 1)
    assert (third.camera_id, third.sequence) == ("NOISY", 4)
    assert outputs.replacements == 2


def test_reconnect_between_checks_does_not_accept_old_epoch_completion() -> None:
    gate = InferenceCompletionGate()
    completion = _output("CAM-A", 7)

    assert gate.rejection_reason(completion, current_stream_epoch=1) == ""
    assert gate.rejection_reason(completion, current_stream_epoch=2) == "stale_epoch"

    # A failed final epoch check must not consume the provenance.
    assert gate.rejection_reason(completion, current_stream_epoch=1) == ""
    gate.record_acceptance(completion)
    assert gate.rejection_reason(completion, current_stream_epoch=1) == "stale"


def test_motion_config_parses_typed_normalized_camera_overrides(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
motion_scheduler:
  mode: enforce
  always_infer: false
  camera_overrides:
    Cam_07:
      analysis_fps: "3.5"
      analysis_width: "80"
      always_infer: "yes"
""",
        encoding="utf-8",
    )

    config = load_config(str(config_file)).motion_scheduler

    assert config.mode == "enforce"
    assert config.always_infer is False
    assert config.camera_overrides["CAM07"] == {
        "analysis_fps": 3.5,
        "analysis_width": 80,
        "always_infer": True,
    }


def test_deployment_mode_environment_overrides_take_precedence(
    tmp_path,
    monkeypatch,
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
motion_scheduler:
  mode: legacy
state_machine:
  mode: legacy
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VA_MOTION_SCHEDULER_MODE", "shadow")
    monkeypatch.setenv("VA_SLOT_STATE_MODE", "time")

    config = load_config(str(config_file))

    assert config.motion_scheduler.mode == "shadow"
    assert config.state_machine.observation_policy.mode == "time"


def test_invalid_boolean_override_fails_at_config_load(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
motion_scheduler:
  camera_overrides:
    CAM-07:
      always_infer: maybe
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        load_config(str(config_file))


@pytest.mark.parametrize(
    "override_yaml",
    [
        "sentinel_interval_seconds: 9.0",
        "sentinel_interval_seconds: 5.0\n  camera_overrides:\n    CAM-07:\n      sentinel_interval_seconds: 9.0",
    ],
)
def test_enforced_motion_sentinel_cannot_exceed_timed_known_gap(
    tmp_path, override_yaml
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
motion_scheduler:
  mode: enforce
  {override_yaml}
state_machine:
  mode: time
  max_known_gap_seconds: 8.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sentinel_interval_seconds must be <="):
        load_config(str(config_file))


def test_enforced_motion_sentinel_equal_to_timed_known_gap_is_valid(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
motion_scheduler:
  mode: enforce
  sentinel_interval_seconds: 8.0
state_machine:
  mode: time
  max_known_gap_seconds: 8.0
""",
        encoding="utf-8",
    )

    config = load_config(str(config_file))
    assert config.motion_scheduler.sentinel_interval_seconds == 8.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_fps", float("nan")),
        ("changed_ratio", float("inf")),
        ("active_hold_seconds", float("nan")),
        ("sentinel_interval_seconds", float("inf")),
        ("stale_frame_seconds", float("nan")),
    ],
)
def test_non_finite_motion_configuration_fails_closed(field, value) -> None:
    with pytest.raises(ValueError):
        MotionScheduler(_config(**{field: value}))
