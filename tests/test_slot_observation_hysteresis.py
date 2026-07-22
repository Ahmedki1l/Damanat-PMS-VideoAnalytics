from __future__ import annotations

import pytest

from src.models.state_machine import (
    SlotObservation,
    SlotObservationKind,
    SlotObservationPolicy,
    SlotState,
    SlotStateMachine,
)


def _observation(
    kind: SlotObservationKind,
    at: float,
    track_id: int | None = None,
) -> SlotObservation:
    return SlotObservation(
        kind=kind,
        observed_at=at,
        timestamp=f"2026-07-21T00:00:{at:06.3f}",
        track_id=track_id,
    )


def test_unknown_does_not_advance_legacy_enter_or_leave_counters() -> None:
    machine = SlotStateMachine(
        "S1",
        confirm_enter_frames=3,
        confirm_leave_frames=3,
    )

    machine.observe(_observation(SlotObservationKind.PRESENT, 0.0, 11))
    for at in (1.0, 2.0, 30.0):
        assert machine.observe(_observation(SlotObservationKind.UNKNOWN, at)) == []
    machine.observe(_observation(SlotObservationKind.PRESENT, 31.0, 11))
    assert machine.state is SlotState.ENTERING
    parked = machine.observe(_observation(SlotObservationKind.PRESENT, 32.0, 11))
    assert [event.event_type for event in parked] == ["vehicle_parked"]

    machine.observe(_observation(SlotObservationKind.ABSENT, 33.0))
    assert machine.state is SlotState.LEAVING
    assert machine.is_occupied
    for at in (40.0, 80.0):
        machine.observe(_observation(SlotObservationKind.UNKNOWN, at))
    machine.observe(_observation(SlotObservationKind.ABSENT, 81.0))
    assert machine.state is SlotState.LEAVING
    vacant = machine.observe(_observation(SlotObservationKind.ABSENT, 82.0))
    assert [event.event_type for event in vacant] == ["slot_vacant"]


def test_time_policy_requires_duration_and_known_observation_count() -> None:
    machine = SlotStateMachine(
        "S1",
        observation_policy=SlotObservationPolicy(
            mode="time",
            enter_seconds=3.0,
            leave_seconds=20.0,
            enter_min_observations=2,
            leave_min_observations=3,
            max_known_gap_seconds=8.0,
        ),
    )

    entering = machine.observe(
        _observation(SlotObservationKind.PRESENT, 0.0, 41)
    )
    assert [event.event_type for event in entering] == ["vehicle_entering"]
    assert machine.state is SlotState.ENTERING

    machine.observe(_observation(SlotObservationKind.PRESENT, 2.0, 41))
    assert machine.state is SlotState.ENTERING
    machine.observe(_observation(SlotObservationKind.PRESENT, 3.0, 41))
    assert machine.state is SlotState.OCCUPIED

    first_miss = machine.observe(_observation(SlotObservationKind.ABSENT, 4.0))
    assert first_miss == []
    assert machine.state is SlotState.OCCUPIED
    leaving = machine.observe(_observation(SlotObservationKind.ABSENT, 5.0))
    assert [event.event_type for event in leaving] == ["vehicle_leaving"]
    assert machine.state is SlotState.LEAVING
    assert machine.is_occupied

    # UNKNOWN for minutes is not proof of absence. The next known observation
    # restarts the contiguous evidence window instead of declaring vacancy.
    machine.observe(_observation(SlotObservationKind.UNKNOWN, 100.0))
    machine.observe(_observation(SlotObservationKind.ABSENT, 100.0))
    machine.observe(_observation(SlotObservationKind.ABSENT, 107.0))
    assert machine.state is SlotState.LEAVING
    machine.observe(_observation(SlotObservationKind.ABSENT, 114.0))
    assert machine.state is SlotState.LEAVING
    vacant = machine.observe(_observation(SlotObservationKind.ABSENT, 120.0))
    assert [event.event_type for event in vacant] == ["slot_vacant"]
    assert not machine.is_occupied


def test_single_absent_does_not_emit_leaving_or_change_public_state() -> None:
    machine = SlotStateMachine(
        "S1",
        initial_state=SlotState.OCCUPIED,
        observation_policy=SlotObservationPolicy(
            mode="time",
            leave_seconds=20.0,
            leave_min_observations=3,
            leave_start_seconds=1.0,
            leave_start_min_observations=2,
        ),
    )

    missed = machine.observe(_observation(SlotObservationKind.ABSENT, 0.0))
    recovered = machine.observe(
        _observation(SlotObservationKind.PRESENT, 0.5, track_id=7)
    )

    assert missed == []
    assert recovered == []
    assert machine.state is SlotState.OCCUPIED


def test_single_absent_does_not_cancel_entering_state() -> None:
    machine = SlotStateMachine(
        "S1",
        observation_policy=SlotObservationPolicy(
            mode="time",
            enter_seconds=3.0,
            enter_min_observations=2,
            enter_cancel_seconds=1.0,
            enter_cancel_min_observations=2,
        ),
    )
    machine.observe(_observation(SlotObservationKind.PRESENT, 0.0, 7))

    missed = machine.observe(_observation(SlotObservationKind.ABSENT, 1.0))
    recovered = machine.observe(_observation(SlotObservationKind.PRESENT, 2.0, 7))

    assert missed == []
    assert recovered == []
    assert machine.state is SlotState.ENTERING


def test_long_unknown_gap_breaks_enter_run_even_if_absent_arrives_first() -> None:
    machine = SlotStateMachine(
        "S1",
        observation_policy=SlotObservationPolicy(
            mode="time",
            enter_seconds=3.0,
            enter_min_observations=2,
            enter_cancel_seconds=10.0,
            enter_cancel_min_observations=3,
            max_known_gap_seconds=5.0,
        ),
    )
    machine.observe(_observation(SlotObservationKind.PRESENT, 0.0, 7))
    machine.observe(_observation(SlotObservationKind.UNKNOWN, 19.0))
    machine.observe(_observation(SlotObservationKind.ABSENT, 20.0))

    parked = machine.observe(_observation(SlotObservationKind.PRESENT, 23.0, 7))

    assert parked == []
    assert machine.state is SlotState.ENTERING
    parked = machine.observe(_observation(SlotObservationKind.PRESENT, 26.0, 7))
    assert [event.event_type for event in parked] == ["vehicle_parked"]


def test_time_policy_unknown_gap_restarts_enter_evidence() -> None:
    machine = SlotStateMachine(
        "S1",
        observation_policy=SlotObservationPolicy(
            mode="time",
            enter_seconds=3.0,
            enter_min_observations=2,
            max_known_gap_seconds=5.0,
        ),
    )
    machine.observe(_observation(SlotObservationKind.PRESENT, 0.0, 7))
    machine.observe(_observation(SlotObservationKind.UNKNOWN, 20.0))
    machine.observe(_observation(SlotObservationKind.PRESENT, 20.0, 7))
    assert machine.state is SlotState.ENTERING

    parked = machine.observe(_observation(SlotObservationKind.PRESENT, 23.0, 7))
    assert [event.event_type for event in parked] == ["vehicle_parked"]


def test_shadow_policy_reports_disagreement_without_changing_legacy_result() -> None:
    machine = SlotStateMachine(
        "S1",
        confirm_enter_frames=2,
        observation_policy=SlotObservationPolicy(
            mode="shadow",
            enter_seconds=10.0,
            enter_min_observations=2,
        ),
    )
    machine.observe(_observation(SlotObservationKind.PRESENT, 0.0, 9))
    machine.observe(_observation(SlotObservationKind.PRESENT, 1.0, 9))

    status = machine.get_status()
    assert machine.state is SlotState.OCCUPIED
    assert status["time_policy_state"] == "ENTERING"
    assert status["time_policy_disagrees"] is True


def test_default_policy_remains_legacy() -> None:
    machine = SlotStateMachine("S1", confirm_enter_frames=2)

    assert machine.observation_policy.mode == "legacy"
    machine.update(True, track_id=1)
    machine.update(True, track_id=1)
    assert machine.state is SlotState.OCCUPIED


def test_leaving_is_publicly_occupied_until_vacancy_is_confirmed() -> None:
    machine = SlotStateMachine(
        "S1",
        initial_state=SlotState.OCCUPIED,
        confirm_leave_frames=2,
    )

    machine.update(False)

    assert machine.state is SlotState.LEAVING
    assert machine.is_occupied
    assert machine.get_status()["occupied"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enter_seconds", float("nan")),
        ("leave_seconds", float("inf")),
        ("max_known_gap_seconds", float("nan")),
    ],
)
def test_non_finite_hysteresis_configuration_fails_closed(field, value) -> None:
    with pytest.raises(ValueError):
        SlotObservationPolicy(**{field: value})
