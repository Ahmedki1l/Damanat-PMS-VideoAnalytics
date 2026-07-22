"""Regression coverage for source-ordered Entry V2 exit lifecycle races."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import pytest

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    AttemptInput,
    CrossingInput,
    CrossingRole,
    EntryMode,
    EntryUnavailable,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.identity import RegistryIdentityPublisher
from src.entry.settings import EntrySettings
from src.vehicle_registry.vehicle_registry import VehicleRegistry


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
PLATE = "AAA-1111"


def _settings(**overrides) -> EntrySettings:
    base = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=16,
        max_pending_crossings=16,
        max_pending_callbacks=8,
        receipt_capacity=64,
        journey_capacity=32,
        max_images_per_event=3,
        max_image_bytes=1024,
        reid_min_score=0.75,
        reid_row_margin=0.08,
        reid_column_margin=0.08,
        merge_min_score=0.90,
        merge_margin=0.08,
        ocr_min_confidence=0.75,
        correction_min_evidence=2,
        correction_min_cameras=2,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"RAMP-IN", "PARK-ENTRY-VA"}),
        primary_directions=frozenset({"ramp-entry"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B-IN"}),
        fallback_directions=frozenset({"b-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-key",
    )
    return replace(base, **overrides)


def _frame(
    event_id: str,
    camera_id: str,
    embedding: Sequence[float],
    *,
    source_role: str,
    plate: str = "",
) -> FrameEvidence:
    return FrameEvidence(
        evidence_id=f"{event_id}:0",
        embedding=tuple(embedding),
        plate=PlateEvidence(
            evidence_id=f"{event_id}:0",
            camera_id=camera_id,
            source_role=source_role,
            state=PlateReadState.READABLE if plate else PlateReadState.NO_PLATE,
            text=plate,
            confidence=0.99 if plate else 0.0,
        ),
    )


class _Processor:
    def __init__(self, evidence: Mapping[str, Sequence[FrameEvidence]]):
        self._evidence = evidence

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        del camera_id, source_role, images, metadata
        return tuple(self._evidence[event_id])


class _CommittedSink:
    def __init__(self):
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(
            delivered=True,
            attempts=1,
            publish_identity=True,
            session_committed=True,
            ack_result="created",
        )


class _RetryThenCommitSink(_CommittedSink):
    def __init__(self):
        super().__init__()
        self.should_deliver = False

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        if not self.should_deliver:
            return DeliveryResult(False, 1, "temporary_failure", retryable=True)
        return DeliveryResult(
            delivered=True,
            attempts=1,
            publish_identity=True,
            session_committed=True,
            ack_result="created",
        )


class _BlockingFirstCommitSink(_CommittedSink):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("test callback release timed out")
        return DeliveryResult(
            delivered=True,
            attempts=1,
            publish_identity=True,
            session_committed=True,
            ack_result="created",
        )


class _CrossingBarrierProcessor(_Processor):
    def __init__(self, evidence: Mapping[str, Sequence[FrameEvidence]]):
        super().__init__(evidence)
        self._crossing_barrier = threading.Barrier(2)

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        if event_id.startswith("crossing-callback-race-"):
            self._crossing_barrier.wait(timeout=5)
        return super().analyze(
            event_id=event_id,
            camera_id=camera_id,
            source_role=source_role,
            images=images,
            metadata=metadata,
        )


class _ExitBeforePublishSink(_CommittedSink):
    """Model the API's registry mutation winning before coordinator.record_exit."""

    def __init__(self, registry: VehicleRegistry, exit_at: datetime):
        super().__init__()
        self._registry = registry
        self._exit_at = exit_at

    def deliver(self, payload):
        result = super().deliver(payload)
        self._registry.register_anpr_event(
            PLATE,
            "exit",
            timestamp=self._exit_at,
        )
        return result


@dataclass(frozen=True)
class _Visit:
    attempt: AttemptInput
    crossing: CrossingInput
    attempt_image: bytes
    crossing_image: bytes


def _visit(
    identifier: str,
    *,
    attempt_at: datetime,
    crossing_at: datetime,
    crossing_source: str = "hikvision",
    line_id: str = "RAMP-IN",
) -> _Visit:
    return _Visit(
        attempt=AttemptInput(
            attempt_id=f"attempt-{identifier}",
            source_event_id=f"source-attempt-{identifier}",
            camera_id="ANPR-ENTRY",
            captured_at=attempt_at,
            reported_plate=PLATE,
            reported_confidence=0.99,
            metadata={"lane": "entry"},
        ),
        crossing=CrossingInput(
            crossing_id=f"crossing-{identifier}",
            source_event_id=f"source-crossing-{identifier}",
            camera_id="CAM-23",
            captured_at=crossing_at,
            line_id=line_id,
            direction="ramp-entry",
            role=CrossingRole.PRIMARY,
            metadata={"track": identifier, "crossing_source": crossing_source},
        ),
        attempt_image=f"attempt-image-{identifier}".encode(),
        crossing_image=f"crossing-image-{identifier}".encode(),
    )


def _visit_evidence(
    visit: _Visit,
    embedding: Sequence[float],
    *,
    crossing_plate: str = "AAA1111",
) -> dict[str, list[FrameEvidence]]:
    return {
        visit.attempt.attempt_id: [
            _frame(
                visit.attempt.attempt_id,
                "ANPR-ENTRY",
                embedding,
                source_role="anpr",
            )
        ],
        visit.crossing.crossing_id: [
            _frame(
                visit.crossing.crossing_id,
                "CAM-23",
                embedding,
                source_role="primary",
                plate=crossing_plate,
            )
        ],
    }


def _coordinator(
    evidence: Mapping[str, Sequence[FrameEvidence]],
    *,
    sink=None,
    publisher=None,
) -> tuple[EntryCoordinator, object]:
    active_sink = sink or _CommittedSink()
    return (
        EntryCoordinator(
            _settings(),
            _Processor(evidence),
            active_sink,
            identity_publisher=publisher,
        ),
        active_sink,
    )


def _confirm_visit(coordinator: EntryCoordinator, visit: _Visit):
    coordinator.ingest_attempt(visit.attempt, [visit.attempt_image])
    return coordinator.ingest_crossing(visit.crossing, [visit.crossing_image])


def test_registry_stamped_exit_before_coordinator_boundary_closes_tombstone(
    tmp_path,
):
    entry_at = NOW + timedelta(minutes=1)
    exit_at = NOW + timedelta(minutes=5)
    visit = _visit("race", attempt_at=NOW, crossing_at=entry_at)
    evidence = _visit_evidence(visit, (1.0, 0.0))
    registry = VehicleRegistry(image_dir=str(tmp_path), clock=lambda: exit_at)
    sink = _ExitBeforePublishSink(registry, exit_at)
    coordinator, _ = _coordinator(
        evidence,
        sink=sink,
        publisher=RegistryIdentityPublisher(registry),
    )

    result = _confirm_visit(coordinator, visit)

    summary = coordinator.state_summary()
    assert result.decision_status == "confirmed"
    assert result.callback_delivered is True
    assert registry._sessions == {}
    assert summary["pending_callback_count"] == 0
    assert summary["finalized_journey_count"] == 1
    assert summary["open_journey_count"] == 0
    assert summary["pending_exit_count"] == 0

    # This is the delayed coordinator half of the same API exit delivery.
    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 0

    attempt_retry = coordinator.ingest_attempt(
        visit.attempt,
        [visit.attempt_image],
    )
    crossing_retry = coordinator.ingest_crossing(
        visit.crossing,
        [visit.crossing_image],
    )
    assert attempt_retry.duplicate is True
    assert crossing_retry.duplicate is True
    assert attempt_retry.decision_id == result.decision_id
    assert crossing_retry.decision_id == result.decision_id
    assert crossing_retry.callback_delivered is True
    assert len(sink.payloads) == 1


def test_pending_exit_exact_replay_is_consumed_by_closed_finalization():
    exit_at = NOW + timedelta(minutes=5)
    visit = _visit(
        "pending-exit",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    coordinator, sink = _coordinator(_visit_evidence(visit, (1.0, 0.0)))

    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 1

    result = _confirm_visit(coordinator, visit)

    summary = coordinator.state_summary()
    assert result.decision_status == "confirmed"
    assert summary["finalized_journey_count"] == 1
    assert summary["open_journey_count"] == 0
    assert summary["pending_exit_count"] == 0
    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 0
    assert len(sink.payloads) == 1


def test_applied_old_exit_replay_does_not_poison_newer_open_visit():
    first = _visit(
        "first",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    second = _visit(
        "second",
        attempt_at=NOW + timedelta(minutes=10),
        crossing_at=NOW + timedelta(minutes=12),
    )
    evidence = {
        **_visit_evidence(first, (1.0, 0.0)),
        **_visit_evidence(second, (1.0, 0.0)),
    }
    coordinator, sink = _coordinator(evidence)
    first_exit = NOW + timedelta(minutes=5)
    second_exit = NOW + timedelta(minutes=15)

    _confirm_visit(coordinator, first)
    assert coordinator.record_exit(PLATE, first_exit) == 1
    _confirm_visit(coordinator, second)
    assert coordinator.state_summary()["open_journey_count"] == 1

    assert coordinator.record_exit(PLATE, first_exit) == 0
    summary = coordinator.state_summary()
    assert summary["open_journey_count"] == 1
    assert summary["pending_exit_count"] == 0

    assert coordinator.record_exit(PLATE, second_exit) == 1
    assert coordinator.state_summary()["open_journey_count"] == 0
    assert len(sink.payloads) == 2


def test_one_exit_closes_unique_latest_of_two_open_visits():
    older = _visit(
        "older",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    newer = _visit(
        "newer",
        attempt_at=NOW + timedelta(minutes=8),
        crossing_at=NOW + timedelta(minutes=10),
    )
    evidence = {
        **_visit_evidence(older, (1.0, 0.0)),
        **_visit_evidence(newer, (0.0, 1.0)),
    }
    coordinator, _ = _coordinator(evidence)
    _confirm_visit(coordinator, older)
    _confirm_visit(coordinator, newer)
    assert coordinator.state_summary()["open_journey_count"] == 2

    assert coordinator.record_exit(PLATE, NOW + timedelta(minutes=15)) == 1
    summary = coordinator.state_summary()
    assert summary["open_journey_count"] == 1
    assert summary["ambiguous_exit_count"] == 1

    # This earlier source boundary can close only the older remaining visit,
    # proving the first exit selected the uniquely latest one.
    assert coordinator.record_exit(PLATE, NOW + timedelta(minutes=5)) == 1
    assert coordinator.state_summary()["open_journey_count"] == 0


def test_pending_exit_waits_for_all_reserved_same_plate_callbacks():
    older = _visit(
        "callback-race-older",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    newer = _visit(
        "callback-race-newer",
        attempt_at=NOW + timedelta(minutes=8),
        crossing_at=NOW + timedelta(minutes=10),
    )
    evidence = {
        **_visit_evidence(older, (1.0, 0.0)),
        **_visit_evidence(newer, (0.0, 1.0)),
    }
    sink = _BlockingFirstCommitSink()
    coordinator = EntryCoordinator(
        _settings(),
        _CrossingBarrierProcessor(evidence),
        sink,
    )
    coordinator.ingest_attempt(older.attempt, [older.attempt_image])
    coordinator.ingest_attempt(newer.attempt, [newer.attempt_image])

    errors = []

    def ingest_crossing(visit):
        try:
            coordinator.ingest_crossing(visit.crossing, [visit.crossing_image])
        except Exception as exc:  # pragma: no cover - reported by assertion
            errors.append(exc)

    workers = [
        threading.Thread(target=ingest_crossing, args=(visit,))
        for visit in (older, newer)
    ]
    for worker in workers:
        worker.start()
    try:
        assert sink.started.wait(timeout=5)
        assert (
            sink.payloads[0]["entry_captured_at"]
            == older.crossing.captured_at.isoformat()
        )
        assert coordinator.record_exit(PLATE, NOW + timedelta(minutes=15)) == 0
        assert coordinator.state_summary()["pending_exit_count"] == 1
    finally:
        sink.release.set()
        for worker in workers:
            worker.join(timeout=5)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    summary = coordinator.state_summary()
    assert summary["pending_exit_count"] == 0
    assert summary["open_journey_count"] == 1

    # The 12:15 exit must close the newer 12:10 journey. Therefore the older
    # 12:02 journey remains eligible for this earlier boundary.
    assert coordinator.record_exit(PLATE, NOW + timedelta(minutes=5)) == 1
    assert coordinator.state_summary()["open_journey_count"] == 0


def test_materialized_inflight_attempt_is_counted_once_at_capacity():
    first = _visit(
        "capacity-first",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    second = _visit(
        "capacity-second",
        attempt_at=NOW + timedelta(minutes=5),
        crossing_at=NOW + timedelta(minutes=7),
    )
    evidence = {
        **_visit_evidence(first, (1.0, 0.0)),
        **_visit_evidence(second, (0.0, 1.0)),
    }
    sink = _BlockingFirstCommitSink()
    coordinator = EntryCoordinator(
        _settings(journey_capacity=2, max_pending_attempts=2),
        _Processor(evidence),
        sink,
    )
    coordinator.ingest_crossing(first.crossing, [first.crossing_image])

    errors = []

    def confirm_first():
        try:
            coordinator.ingest_attempt(first.attempt, [first.attempt_image])
        except Exception as exc:  # pragma: no cover - reported by assertion
            errors.append(exc)

    worker = threading.Thread(target=confirm_first)
    worker.start()
    try:
        assert sink.started.wait(timeout=5)
        blocked_summary = coordinator.state_summary()
        assert blocked_summary["group_count"] == 1
        assert blocked_summary["journey_capacity_load"] == 1

        accepted = coordinator.ingest_attempt(
            second.attempt,
            [second.attempt_image],
        )
        assert accepted.accepted is True
    finally:
        sink.release.set()
        worker.join(timeout=5)

    assert not errors
    assert not worker.is_alive()
    assert coordinator.state_summary()["journey_capacity_load"] == 2


def test_equal_latest_entry_times_close_none_and_retain_one_boundary():
    first = _visit(
        "equal-first",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=10),
    )
    second = _visit(
        "equal-second",
        attempt_at=NOW + timedelta(minutes=1),
        crossing_at=NOW + timedelta(minutes=10),
    )
    evidence = {
        **_visit_evidence(first, (1.0, 0.0)),
        **_visit_evidence(second, (0.0, 1.0)),
    }
    coordinator, _ = _coordinator(evidence)
    _confirm_visit(coordinator, first)
    _confirm_visit(coordinator, second)

    exit_at = NOW + timedelta(minutes=15)
    assert coordinator.record_exit(PLATE, exit_at) == 0
    summary = coordinator.state_summary()
    assert summary["open_journey_count"] == 2
    assert summary["ambiguous_exit_count"] == 1
    assert summary["pending_exit_count"] == 1

    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 1


def test_callback_retry_applies_exit_marker_before_finalization():
    visit = _visit(
        "retry",
        attempt_at=NOW,
        crossing_at=NOW + timedelta(minutes=2),
    )
    sink = _RetryThenCommitSink()
    coordinator, _ = _coordinator(
        _visit_evidence(visit, (1.0, 0.0)),
        sink=sink,
    )
    coordinator.ingest_attempt(visit.attempt, [visit.attempt_image])

    with pytest.raises(EntryUnavailable, match="entry_confirmation_delivery_failed"):
        coordinator.ingest_crossing(visit.crossing, [visit.crossing_image])

    exit_at = NOW + timedelta(minutes=5)
    assert coordinator.record_exit(PLATE, exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 1

    sink.should_deliver = True
    assert list(coordinator.retry_pending_callbacks().values()) == [True]
    summary = coordinator.state_summary()
    assert summary["pending_callback_count"] == 0
    assert summary["pending_exit_count"] == 0
    assert summary["finalized_journey_count"] == 1
    assert summary["open_journey_count"] == 0

    attempt_retry = coordinator.ingest_attempt(
        visit.attempt,
        [visit.attempt_image],
    )
    crossing_retry = coordinator.ingest_crossing(
        visit.crossing,
        [visit.crossing_image],
    )
    assert attempt_retry.decision_id == crossing_retry.decision_id
    assert crossing_retry.callback_delivered is True
    assert len(sink.payloads) == 2


def test_resolved_producer_family_receipts_share_delivered_decision():
    attempt = AttemptInput(
        attempt_id="attempt-family",
        source_event_id="source-attempt-family",
        camera_id="ANPR-ENTRY",
        captured_at=NOW,
        reported_plate=PLATE,
        reported_confidence=0.99,
        metadata={"lane": "entry"},
    )
    hikvision = CrossingInput(
        crossing_id="crossing-family-hikvision",
        source_event_id="source-crossing-family-hikvision",
        camera_id="CAM-23",
        captured_at=NOW + timedelta(minutes=2),
        line_id="RAMP-IN",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
        metadata={"crossing_source": "hikvision"},
    )
    local_zone = CrossingInput(
        crossing_id="crossing-family-local",
        source_event_id="source-crossing-family-local",
        camera_id="CAM-23",
        captured_at=NOW + timedelta(minutes=2, seconds=1),
        line_id="PARK-ENTRY-VA",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
        metadata={"crossing_source": "va_local_zone"},
    )
    evidence = {
        attempt.attempt_id: [
            _frame(
                attempt.attempt_id,
                "ANPR-ENTRY",
                (1.0, 0.0),
                source_role="anpr",
            )
        ],
        hikvision.crossing_id: [
            _frame(
                hikvision.crossing_id,
                "CAM-23",
                (1.0, 0.0),
                source_role="primary",
                plate="AAA1111",
            )
        ],
        local_zone.crossing_id: [
            _frame(
                local_zone.crossing_id,
                "CAM-23",
                (0.99995, 0.01),
                source_role="primary",
            )
        ],
    }
    coordinator, sink = _coordinator(evidence)
    hikvision_image = b"hikvision-family-image"
    local_image = b"local-family-image"
    attempt_image = b"attempt-family-image"

    coordinator.ingest_crossing(hikvision, [hikvision_image])
    coordinator.ingest_crossing(local_zone, [local_image])
    confirmed = coordinator.ingest_attempt(attempt, [attempt_image])

    assert confirmed.decision_status == "confirmed"
    assert confirmed.callback_delivered is True
    assert len(sink.payloads) == 1

    hikvision_retry = coordinator.ingest_crossing(hikvision, [hikvision_image])
    local_retry = coordinator.ingest_crossing(local_zone, [local_image])
    attempt_retry = coordinator.ingest_attempt(attempt, [attempt_image])
    for retry in (hikvision_retry, local_retry, attempt_retry):
        assert retry.duplicate is True
        assert retry.decision_id == confirmed.decision_id
        assert retry.decision_status == "confirmed"
        assert retry.callback_delivered is True
    assert len(sink.payloads) == 1
