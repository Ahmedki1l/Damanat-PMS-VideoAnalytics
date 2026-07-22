from dataclasses import replace
from datetime import datetime, timedelta, timezone
import threading

import pytest

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    AttemptInput,
    CrossingInput,
    CrossingRole,
    EntryMode,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)


def _settings() -> EntrySettings:
    return replace(
        EntrySettings(),
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=8,
        max_pending_crossings=8,
        max_pending_callbacks=8,
        receipt_capacity=32,
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
        primary_lines=frozenset({"PARK-ENTRY-VA", "RAMP-IN"}),
        primary_directions=frozenset({"ramp-entry"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B-IN"}),
        fallback_directions=frozenset({"b-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-key",
    )


def _frame(
    event_id: str,
    camera_id: str,
    embedding: tuple[float, ...],
    *,
    plate: str = "",
    role: str,
) -> FrameEvidence:
    state = PlateReadState.READABLE if plate else PlateReadState.NO_PLATE
    return FrameEvidence(
        evidence_id=f"{event_id}:0",
        embedding=embedding,
        plate=PlateEvidence(
            evidence_id=f"{event_id}:0",
            camera_id=camera_id,
            source_role=role,
            state=state,
            text=plate,
            confidence=0.99 if plate else 0.0,
        ),
    )


def _attempt(
    attempt_id: str,
    plate: str,
    *,
    captured_at: datetime,
) -> AttemptInput:
    return AttemptInput(
        attempt_id=attempt_id,
        source_event_id=f"source-{attempt_id}",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=plate,
        reported_confidence=0.99,
        metadata={"lane": "entry"},
    )


def _crossing(
    crossing_id: str,
    *,
    role: CrossingRole,
    captured_at: datetime,
    line_id: str | None = None,
    crossing_source: str = "hikvision",
) -> CrossingInput:
    is_primary = role == CrossingRole.PRIMARY
    return CrossingInput(
        crossing_id=crossing_id,
        source_event_id=f"source-{crossing_id}",
        camera_id="CAM-23" if is_primary else "CAM-03",
        captured_at=captured_at,
        line_id=line_id or ("RAMP-IN" if is_primary else "B-IN"),
        direction="ramp-entry" if is_primary else "b-entry",
        role=role,
        metadata={"track": 7, "crossing_source": crossing_source},
    )


class _Processor:
    def __init__(self, evidence_by_event):
        self.evidence_by_event = evidence_by_event
        self.calls: list[str] = []

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        del camera_id, source_role, images, metadata
        self.calls.append(event_id)
        return tuple(self.evidence_by_event[event_id])


class _ConcurrentProcessor(_Processor):
    def __init__(self, evidence_by_event, concurrent_event_ids):
        super().__init__(evidence_by_event)
        self._concurrent_event_ids = set(concurrent_event_ids)
        self._barrier = threading.Barrier(len(self._concurrent_event_ids))

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        if event_id in self._concurrent_event_ids:
            self._barrier.wait(timeout=2)
        return super().analyze(
            event_id=event_id,
            camera_id=camera_id,
            source_role=source_role,
            images=images,
            metadata=metadata,
        )


class _Sink:
    def __init__(self):
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(
            True,
            1,
            "",
            publish_identity=False,
            session_committed=True,
        )


class _BlockingSink(_Sink):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        self.started.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("test callback was not released")
        return DeliveryResult(
            True,
            1,
            "",
            publish_identity=False,
            session_committed=True,
        )


def _coordinator(evidence_by_event, *, processor=None):
    processor = processor or _Processor(evidence_by_event)
    sink = _Sink()
    coordinator = EntryCoordinator(_settings(), processor, sink)
    return coordinator, processor, sink


def _coordinator_with_sink(evidence_by_event, sink):
    processor = _Processor(evidence_by_event)
    coordinator = EntryCoordinator(_settings(), processor, sink)
    return coordinator, processor


def _confirm_first_journey(coordinator: EntryCoordinator):
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    return coordinator.ingest_crossing(
        _crossing(
            "fallback-1",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
        ),
        [b"fallback-1"],
    )


def test_late_primary_after_fallback_confirmation_is_quarantined():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-1": [
            _frame(
                "fallback-1",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
        "primary-late": [
            _frame(
                "primary-late",
                "CAM-23",
                (1.0, 0.0),
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    confirmed = _confirm_first_journey(coordinator)

    late = coordinator.ingest_crossing(
        _crossing(
            "primary-late",
            role=CrossingRole.PRIMARY,
            # Equal source time is the inclusive upper boundary for duplicate
            # notifications of the already-confirmed physical entry.
            captured_at=NOW + timedelta(minutes=5),
        ),
        [b"primary-late"],
    )

    assert confirmed.decision_status == "confirmed"
    assert late.accepted is False
    assert late.duplicate is True
    assert late.decision_id == confirmed.decision_id
    assert late.decision_status == "confirmed"
    assert late.callback_delivered is True
    assert [payload["crossing_id"] for payload in sink.payloads] == ["fallback-1"]
    assert coordinator.state_summary()["crossing_count"] == 0


def test_late_primary_retry_is_idempotent_without_reanalysis():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-1": [
            _frame(
                "fallback-1",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
        "primary-late": [
            _frame(
                "primary-late",
                "CAM-23",
                (1.0, 0.0),
                role="primary",
            )
        ],
    }
    coordinator, processor, sink = _coordinator(evidence)
    _confirm_first_journey(coordinator)
    request = _crossing(
        "primary-late",
        role=CrossingRole.PRIMARY,
        captured_at=NOW + timedelta(minutes=2),
    )

    first = coordinator.ingest_crossing(request, [b"primary-late"])
    retried = coordinator.ingest_crossing(request, [b"primary-late"])

    assert retried == first
    assert processor.calls.count("primary-late") == 1
    assert len(sink.payloads) == 1
    assert coordinator.state_summary()["crossing_count"] == 0


def test_late_primary_inserted_during_callback_is_reconciled_after_delivery():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-1": [
            _frame(
                "fallback-1",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
        "primary-late": [
            _frame(
                "primary-late",
                "CAM-23",
                (1.0, 0.0),
                role="primary",
            )
        ],
    }
    sink = _BlockingSink()
    coordinator, processor = _coordinator_with_sink(evidence, sink)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    fallback_results = []
    errors = []

    def confirm_with_fallback():
        try:
            fallback_results.append(
                coordinator.ingest_crossing(
                    _crossing(
                        "fallback-1",
                        role=CrossingRole.FALLBACK,
                        captured_at=NOW + timedelta(minutes=5),
                    ),
                    [b"fallback-1"],
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    callback_thread = threading.Thread(target=confirm_with_fallback)
    callback_thread.start()
    assert sink.started.wait(timeout=2)

    request = _crossing(
        "primary-late",
        role=CrossingRole.PRIMARY,
        captured_at=NOW + timedelta(minutes=2),
    )
    in_flight_delivery = coordinator.ingest_crossing(
        request,
        [b"primary-late"],
    )
    assert in_flight_delivery.decision_id is None
    assert coordinator.state_summary()["crossing_count"] == 2

    sink.release.set()
    callback_thread.join(timeout=3)

    assert not callback_thread.is_alive()
    assert errors == []
    assert fallback_results[0].decision_status == "confirmed"
    assert coordinator.state_summary()["crossing_count"] == 0
    reconciled_retry = coordinator.ingest_crossing(request, [b"primary-late"])
    assert reconciled_retry.accepted is False
    assert reconciled_retry.duplicate is True
    assert reconciled_retry.decision_id == fallback_results[0].decision_id
    assert processor.calls.count("primary-late") == 1
    assert len(sink.payloads) == 1


def test_quarantined_primary_cannot_confirm_a_later_unrelated_vehicle():
    confusing_new_car_embedding = (0.8, 0.6)
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-1": [
            _frame(
                "fallback-1",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
        # This stale event would exceed the ReID threshold for attempt-2 if it
        # remained pending, despite describing the already-entered first car.
        "primary-late": [
            _frame(
                "primary-late",
                "CAM-23",
                (1.0, 0.0),
                role="primary",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                confusing_new_car_embedding,
                plate="BBB2222",
                role="anpr",
            )
        ],
        "primary-2": [
            _frame(
                "primary-2",
                "CAM-23",
                confusing_new_car_embedding,
                plate="BBB2222",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    _confirm_first_journey(coordinator)
    coordinator.ingest_crossing(
        _crossing(
            "primary-late",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
        ),
        [b"primary-late"],
    )

    waiting = coordinator.ingest_attempt(
        _attempt(
            "attempt-2",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"anpr-2"],
    )

    assert waiting.decision_id is None
    assert [payload["attempt_id"] for payload in sink.payloads] == ["attempt-1"]
    confirmed = coordinator.ingest_crossing(
        _crossing(
            "primary-2",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=12),
        ),
        [b"primary-2"],
    )
    assert confirmed.decision_status == "confirmed"
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
    ]
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0


def test_concurrent_hikvision_and_local_primary_events_form_one_crossing_family():
    evidence = {
        "primary-hikvision": [
            _frame(
                "primary-hikvision",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-va-local": [
            _frame(
                "primary-va-local",
                "CAM-23",
                (0.999, 0.01),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
    }
    processor = _ConcurrentProcessor(
        evidence,
        {"primary-hikvision", "primary-va-local"},
    )
    coordinator, _, sink = _coordinator(evidence, processor=processor)
    errors = []

    def ingest(request, image):
        try:
            coordinator.ingest_crossing(request, [image])
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    hikvision = threading.Thread(
        target=ingest,
        args=(
            _crossing(
                "primary-hikvision",
                role=CrossingRole.PRIMARY,
                captured_at=NOW + timedelta(minutes=2),
                crossing_source="hikvision",
            ),
            b"hikvision",
        ),
    )
    local = threading.Thread(
        target=ingest,
        args=(
            _crossing(
                "primary-va-local",
                role=CrossingRole.PRIMARY,
                captured_at=NOW + timedelta(minutes=2, seconds=1),
                line_id="PARK-ENTRY-VA",
                crossing_source="va_local_zone",
            ),
            b"va-local",
        ),
    )
    hikvision.start()
    local.start()
    hikvision.join(timeout=3)
    local.join(timeout=3)

    assert not hikvision.is_alive()
    assert not local.is_alive()
    assert errors == []
    assert coordinator.state_summary()["crossing_count"] == 2

    confirmed = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert confirmed.decision_status == "confirmed"
    assert len(sink.payloads) == 1
    assert sink.payloads[0]["crossing_id"] in {
        "primary-hikvision",
        "primary-va-local",
    }
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0


def test_two_genuinely_ambiguous_primary_vehicles_are_not_collapsed():
    left_vehicle = (0.95, 0.31224989991991997)
    right_vehicle = (0.95, -0.31224989991991997)
    evidence = {
        "primary-left": [
            _frame(
                "primary-left",
                "CAM-23",
                left_vehicle,
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-right": [
            _frame(
                "primary-right",
                "CAM-23",
                right_vehicle,
                plate="BBB2222",
                role="primary",
            )
        ],
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "primary-left",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"left"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-right",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"right"],
    )

    waiting = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert waiting.decision_id is None
    assert sink.payloads == []
    assert coordinator.state_summary()["attempt_count"] == 1
    assert coordinator.state_summary()["crossing_count"] == 2


def test_equivalent_primary_family_keeps_conflicting_ocr_fail_closed():
    evidence = {
        "primary-hikvision": [
            _frame(
                "primary-hikvision",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-va-local": [
            _frame(
                "primary-va-local",
                "CAM-23",
                (1.0, 0.0),
                plate="BBB2222",
                role="primary",
            )
        ],
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "primary-hikvision",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"hikvision"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-va-local",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2, seconds=1),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"va-local"],
    )

    result = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert result.decision_status == "abstained"
    assert sink.payloads
    assert all(payload["status"] == "abstained" for payload in sink.payloads)
    assert all(payload["reason"] == "primary_ocr_conflict" for payload in sink.payloads)
    assert coordinator.state_summary()["attempt_count"] == 1
    assert coordinator.state_summary()["crossing_count"] == 2


@pytest.mark.parametrize(
    ("first_source", "first_line", "late_source", "late_line"),
    [
        ("hikvision", "RAMP-IN", "va_local_zone", "PARK-ENTRY-VA"),
        ("va_local_zone", "PARK-ENTRY-VA", "hikvision", "RAMP-IN"),
    ],
)
def test_primary_counterpart_just_after_confirmation_is_quarantined(
    first_source,
    first_line,
    late_source,
    late_line,
):
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-first": [
            _frame(
                "primary-first",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-late": [
            _frame(
                "primary-late",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    confirmed = coordinator.ingest_crossing(
        _crossing(
            "primary-first",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            line_id=first_line,
            crossing_source=first_source,
        ),
        [b"primary-first"],
    )

    duplicate = coordinator.ingest_crossing(
        _crossing(
            "primary-late",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2, seconds=1),
            line_id=late_line,
            crossing_source=late_source,
        ),
        [b"primary-late"],
    )

    assert confirmed.decision_status == "confirmed"
    assert duplicate.duplicate is True
    assert duplicate.accepted is False
    assert duplicate.decision_id == confirmed.decision_id
    assert len(sink.payloads) == 1
    assert coordinator.state_summary()["crossing_count"] == 0


def test_cam03_fallback_after_primary_confirmation_is_quarantined():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "fallback-later": [
            _frame(
                "fallback-later",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    confirmed = coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-1"],
    )

    downstream = coordinator.ingest_crossing(
        _crossing(
            "fallback-later",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="va_local_zone",
        ),
        [b"fallback-later"],
    )

    assert confirmed.decision_status == "confirmed"
    assert downstream.duplicate is True
    assert downstream.accepted is False
    assert downstream.decision_id == confirmed.decision_id
    assert len(sink.payloads) == 1
    assert coordinator.state_summary()["crossing_count"] == 0
    # Retain the exact match until the exit source timestamp proves whether it
    # is downstream evidence for this journey or a same-car re-entry delivered
    # before its exit webhook.
    assert coordinator.state_summary()["provisional_crossing_count"] == 1


def test_late_pre_exit_capture_is_compacted_after_journey_closes():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "fallback-late": [
            _frame(
                "fallback-late",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"attempt-1"],
    )
    confirmed = coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
        ),
        [b"primary-1"],
    )
    assert coordinator.record_exit(
        "AAA-1111",
        NOW + timedelta(minutes=5),
    ) == 1

    late = coordinator.ingest_crossing(
        _crossing(
            "fallback-late",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=4),
            crossing_source="va_local_zone",
        ),
        [b"fallback-late"],
    )

    assert late.duplicate is True
    assert late.decision_id == confirmed.decision_id
    assert len(sink.payloads) == 1
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_late_lookalike_with_conflicting_ocr_remains_eligible():
    lookalike = (0.92, 0.39191835884530846)
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "fallback-2": [
            _frame(
                "fallback-2",
                "CAM-03",
                lookalike,
                plate="BBB2222",
                role="fallback",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                lookalike,
                plate="BBB2222",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"attempt-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
        ),
        [b"primary-1"],
    )

    waiting = coordinator.ingest_crossing(
        _crossing(
            "fallback-2",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="va_local_zone",
        ),
        [b"fallback-2"],
    )

    assert waiting.accepted is True
    assert waiting.duplicate is False
    assert coordinator.state_summary()["crossing_count"] == 1
    assert coordinator.state_summary()["late_ocr_conflict_count"] == 1

    confirmed = coordinator.ingest_attempt(
        _attempt(
            "attempt-2",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=4),
        ),
        [b"attempt-2"],
    )

    assert confirmed.decision_status == "confirmed"
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
    ]


def test_post_entry_fallback_survives_unrelated_group_compaction_for_delayed_attempt():
    car_b_embedding = (0.92, 0.39191835884530846)
    evidence = {
        "attempt-a": [
            _frame(
                "attempt-a",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-b": [
            _frame(
                "fallback-b",
                "CAM-03",
                car_b_embedding,
                plate="BBB2222",
                role="fallback",
            )
        ],
        "primary-a": [
            _frame(
                "primary-a",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-b": [
            _frame(
                "attempt-b",
                "ANPR-ENTRY",
                car_b_embedding,
                plate="BBB2222",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-a", "AAA-1111", captured_at=NOW),
        [b"attempt-a"],
    )

    waiting = coordinator.ingest_crossing(
        _crossing(
            "fallback-b",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=10),
            crossing_source="va_local_zone",
        ),
        [b"fallback-b"],
    )
    confirmed_a = coordinator.ingest_crossing(
        _crossing(
            "primary-a",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-a"],
    )

    assert waiting.decision_status == "abstained"
    assert confirmed_a.decision_status == "confirmed"
    # Reliable conflicting OCR must never be swallowed by the finalized
    # look-alike journey. Keep it eligible for its source-earlier ANPR attempt.
    assert coordinator.state_summary()["crossing_count"] == 1
    assert coordinator.state_summary()["provisional_crossing_count"] == 0
    assert coordinator.state_summary()["late_ocr_conflict_count"] == 1

    confirmed_b = coordinator.ingest_attempt(
        _attempt(
            "attempt-b",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=9),
        ),
        [b"attempt-b"],
    )

    assert confirmed_b.decision_status == "confirmed"
    assert [
        payload["attempt_id"]
        for payload in sink.payloads
        if payload["status"] == "confirmed"
    ] == ["attempt-a", "attempt-b"]
    assert sink.payloads[-1]["crossing_id"] == "fallback-b"
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_strict_producer_twin_is_consumed_before_same_car_reentry_matching():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-1-twin": [
            _frame(
                "primary-1-twin",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-2": [
            _frame(
                "primary-2",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"attempt-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1-twin",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2, seconds=1),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"primary-1-twin"],
    )
    # A complementary Hikvision/local report inside the strict producer-pair
    # window is the completed journey's twin, not an unbounded provisional.
    assert coordinator.state_summary()["provisional_crossing_count"] == 0

    assert coordinator.record_exit(
        "AAA-1111",
        NOW + timedelta(minutes=5),
    ) == 1

    coordinator.ingest_attempt(
        _attempt(
            "attempt-2",
            "AAA-1111",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"attempt-2"],
    )
    assert coordinator.state_summary()["provisional_crossing_count"] == 0

    reentry = coordinator.ingest_crossing(
        _crossing(
            "primary-2",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=12),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"primary-2"],
    )

    assert reentry.decision_status == "confirmed"
    assert [
        payload["attempt_id"]
        for payload in sink.payloads
        if payload["status"] == "confirmed"
    ] == ["attempt-1", "attempt-2"]
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_unrelated_later_attempt_does_not_retire_delayed_attempt_evidence():
    evidence = {
        "attempt-a": [
            _frame(
                "attempt-a",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-a": [
            _frame(
                "primary-a",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "fallback-b": [
            _frame(
                "fallback-b",
                "CAM-03",
                (1.0, 0.0),
                plate="BBB2222",
                role="fallback",
            )
        ],
        "attempt-c": [
            _frame(
                "attempt-c",
                "ANPR-ENTRY",
                (0.0, 1.0),
                plate="CCC3333",
                role="anpr",
            )
        ],
        "attempt-b": [
            _frame(
                "attempt-b",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="BBB2222",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-a", "AAA-1111", captured_at=NOW),
        [b"attempt-a"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-a",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-a"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "fallback-b",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="va_local_zone",
        ),
        [b"fallback-b"],
    )
    coordinator.ingest_attempt(
        _attempt(
            "attempt-c",
            "CCC-3333",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"attempt-c"],
    )

    assert coordinator.state_summary()["crossing_count"] == 1
    assert coordinator.state_summary()["provisional_crossing_count"] == 0
    assert coordinator.state_summary()["late_ocr_conflict_count"] == 1
    delayed = coordinator.ingest_attempt(
        _attempt(
            "attempt-b",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=4),
        ),
        [b"attempt-b"],
    )

    assert delayed.decision_status == "confirmed"
    assert sink.payloads[-1]["attempt_id"] == "attempt-b"
    assert sink.payloads[-1]["crossing_id"] == "fallback-b"
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_crossing_captured_before_attempt_cannot_confirm_it():
    evidence = {
        "primary-old": [
            _frame(
                "primary-old",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-new": [
            _frame(
                "attempt-new",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "primary-old",
            role=CrossingRole.PRIMARY,
            captured_at=NOW,
            crossing_source="hikvision",
        ),
        [b"primary-old"],
    )
    result = coordinator.ingest_attempt(
        _attempt(
            "attempt-new",
            "AAA-1111",
            captured_at=NOW + timedelta(hours=6),
        ),
        [b"attempt-new"],
    )

    assert result.decision_id is None
    assert sink.payloads == []
    assert coordinator.state_summary()["attempt_count"] == 1
    assert coordinator.state_summary()["crossing_count"] == 1


@pytest.mark.parametrize(
    ("producer_skew", "expected_confirmed"),
    [(timedelta(seconds=5), True), (timedelta(seconds=5, milliseconds=1), False)],
)
def test_producer_family_respects_source_skew_boundary(
    producer_skew,
    expected_confirmed,
):
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-hikvision": [
            _frame(
                "primary-hikvision",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-local": [
            _frame(
                "primary-local",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    first_crossing_at = NOW + timedelta(minutes=2)
    coordinator.ingest_crossing(
        _crossing(
            "primary-hikvision",
            role=CrossingRole.PRIMARY,
            captured_at=first_crossing_at,
            crossing_source="hikvision",
        ),
        [b"primary-hikvision"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-local",
            role=CrossingRole.PRIMARY,
            captured_at=first_crossing_at + producer_skew,
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"primary-local"],
    )
    result = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"attempt-1"],
    )

    assert (result.decision_status == "confirmed") is expected_confirmed
    assert bool(sink.payloads) is expected_confirmed


def test_hours_apart_producers_do_not_form_one_crossing_family():
    evidence = {
        "primary-old-local": [
            _frame(
                "primary-old-local",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-new-hikvision": [
            _frame(
                "primary-new-hikvision",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-new": [
            _frame(
                "attempt-new",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "primary-old-local",
            role=CrossingRole.PRIMARY,
            captured_at=NOW,
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"primary-old-local"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-new-hikvision",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(hours=6),
            crossing_source="hikvision",
        ),
        [b"primary-new-hikvision"],
    )
    result = coordinator.ingest_attempt(
        _attempt(
            "attempt-new",
            "AAA-1111",
            captured_at=NOW + timedelta(hours=5),
        ),
        [b"attempt-new"],
    )

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["crossing_id"] == "primary-new-hikvision"
    assert (
        sink.payloads[0]["entry_captured_at"] == (NOW + timedelta(hours=6)).isoformat()
    )


def test_confirmation_partitions_source_future_attempt_into_new_journey():
    evidence = {
        "attempt-old": [
            _frame(
                "attempt-old",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "attempt-future": [
            _frame(
                "attempt-future",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="BBB2222",
                role="anpr",
            )
        ],
        "primary-old": [
            _frame(
                "primary-old",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "primary-future": [
            _frame(
                "primary-future",
                "CAM-23",
                (1.0, 0.0),
                plate="BBB2222",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-old", "AAA-1111", captured_at=NOW),
        [b"attempt-old"],
    )
    coordinator.ingest_attempt(
        _attempt(
            "attempt-future",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"attempt-future"],
    )
    old_result = coordinator.ingest_crossing(
        _crossing(
            "primary-old",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="hikvision",
        ),
        [b"primary-old"],
    )

    assert old_result.decision_status == "confirmed"
    assert sink.payloads[-1]["attempt_id"] == "attempt-old"
    assert coordinator.state_summary()["attempt_count"] == 1
    assert coordinator.state_summary()["group_count"] == 1

    future_result = coordinator.ingest_crossing(
        _crossing(
            "primary-future",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=12),
            crossing_source="hikvision",
        ),
        [b"primary-future"],
    )

    assert future_result.decision_status == "confirmed"
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-old",
        "attempt-future",
    ]
    assert [payload["canonical_plate"] for payload in sink.payloads] == [
        "AAA-1111",
        "BBB-2222",
    ]
    assert coordinator.state_summary()["attempt_count"] == 0


def test_source_future_attempt_ocr_cannot_confirm_an_older_crossing():
    evidence = {
        "attempt-old": [
            _frame(
                "attempt-old",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "attempt-future": [
            _frame(
                "attempt-future",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="BBB2222",
                role="anpr",
            )
        ],
        "primary-old": [
            _frame(
                "primary-old",
                "CAM-23",
                (1.0, 0.0),
                plate="BBB2222",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-old", "AAA-1111", captured_at=NOW),
        [b"attempt-old"],
    )
    coordinator.ingest_attempt(
        _attempt(
            "attempt-future",
            "BBB-2222",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"attempt-future"],
    )
    result = coordinator.ingest_crossing(
        _crossing(
            "primary-old",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="hikvision",
        ),
        [b"primary-old"],
    )

    assert result.decision_status == "abstained"
    assert not any(payload["status"] == "confirmed" for payload in sink.payloads)
    assert sink.payloads[-1]["attempt_id"] == "attempt-old"
    assert coordinator.state_summary()["attempt_count"] == 2


def test_reentry_evidence_arriving_before_exit_webhook_is_released_by_source_time():
    new_attempt_embedding = (0.8660254037844386, 0.5)
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                new_attempt_embedding,
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-2": [
            _frame(
                "fallback-2",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-1"],
    )
    provisional = coordinator.ingest_crossing(
        _crossing(
            "fallback-2",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=15),
            crossing_source="va_local_zone",
        ),
        [b"fallback-2"],
    )

    assert provisional.duplicate is True
    assert coordinator.state_summary()["provisional_crossing_count"] == 1
    reentry = coordinator.ingest_attempt(
        _attempt(
            "attempt-2",
            "AAA-1111",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"anpr-2"],
    )

    assert reentry.decision_status is None
    assert coordinator.record_exit(
        "AAA-1111",
        NOW + timedelta(minutes=5),
    ) == 1
    assert sink.payloads[-1]["attempt_id"] == "attempt-2"
    assert sink.payloads[-1]["reid_score"] == pytest.approx(new_attempt_embedding[0])
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_concurrent_reentry_attempt_and_crossing_release_without_arrival_order():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-2": [
            _frame(
                "fallback-2",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
    }
    processor = _ConcurrentProcessor(evidence, {"attempt-2", "fallback-2"})
    coordinator, _, sink = _coordinator(evidence, processor=processor)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-1"],
    )
    assert coordinator.record_exit(
        "AAA-1111",
        NOW + timedelta(minutes=5),
    ) == 1
    results = []
    errors = []

    def submit_attempt():
        try:
            results.append(
                coordinator.ingest_attempt(
                    _attempt(
                        "attempt-2",
                        "AAA-1111",
                        captured_at=NOW + timedelta(minutes=10),
                    ),
                    [b"anpr-2"],
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def submit_crossing():
        try:
            results.append(
                coordinator.ingest_crossing(
                    _crossing(
                        "fallback-2",
                        role=CrossingRole.FALLBACK,
                        captured_at=NOW + timedelta(minutes=15),
                        crossing_source="va_local_zone",
                    ),
                    [b"fallback-2"],
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [
        threading.Thread(target=submit_attempt),
        threading.Thread(target=submit_crossing),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert any(result.decision_status == "confirmed" for result in results)
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
    ]
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 0


def test_same_source_local_rearm_is_provisionally_quarantined():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "local-primary-1": [
            _frame(
                "local-primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "local-primary-repeat": [
            _frame(
                "local-primary-repeat",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "local-primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"local-primary-1"],
    )

    repeated = coordinator.ingest_crossing(
        _crossing(
            "local-primary-repeat",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=3),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
        [b"local-primary-repeat"],
    )

    assert repeated.duplicate is True
    assert len(sink.payloads) == 1
    assert coordinator.state_summary()["crossing_count"] == 0
    assert coordinator.state_summary()["provisional_crossing_count"] == 1


def test_newer_pending_attempt_allows_cam03_fallback_for_reentry():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "primary-1": [
            _frame(
                "primary-1",
                "CAM-23",
                (1.0, 0.0),
                plate="AAA1111",
                role="primary",
            )
        ],
        "attempt-2": [
            _frame(
                "attempt-2",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-2": [
            _frame(
                "fallback-2",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-1",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        [b"primary-1"],
    )
    assert coordinator.record_exit(
        "AAA-1111",
        NOW + timedelta(minutes=5),
    ) == 1
    coordinator.ingest_attempt(
        _attempt(
            "attempt-2",
            "AAA-1111",
            captured_at=NOW + timedelta(minutes=10),
        ),
        [b"anpr-2"],
    )

    reentry = coordinator.ingest_crossing(
        _crossing(
            "fallback-2",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=15),
            crossing_source="va_local_zone",
        ),
        [b"fallback-2"],
    )

    assert reentry.duplicate is False
    assert reentry.decision_status == "confirmed"
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
    ]
    assert coordinator.state_summary()["attempt_count"] == 0
    assert coordinator.state_summary()["crossing_count"] == 0


@pytest.mark.parametrize(
    ("strong_source", "ingest_order"),
    [
        ("hikvision", ("primary-hikvision", "primary-va-local")),
        ("va_local_zone", ("primary-va-local", "primary-hikvision")),
    ],
)
def test_primary_family_uses_best_reid_view_and_any_strong_exact_ocr(
    strong_source,
    ingest_order,
):
    hikvision_vector = (0.9396926207859084, 0.3420201433256687)
    local_vector = (0.9961946980917455, 0.08715574274765817)
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                role="anpr",
            )
        ],
        "primary-hikvision": [
            _frame(
                "primary-hikvision",
                "CAM-23",
                hikvision_vector,
                plate="AAA1111" if strong_source == "hikvision" else "",
                role="primary",
            )
        ],
        "primary-va-local": [
            _frame(
                "primary-va-local",
                "CAM-23",
                local_vector,
                plate="AAA1111" if strong_source == "va_local_zone" else "",
                role="primary",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    requests = {
        "primary-hikvision": _crossing(
            "primary-hikvision",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2),
            crossing_source="hikvision",
        ),
        "primary-va-local": _crossing(
            "primary-va-local",
            role=CrossingRole.PRIMARY,
            captured_at=NOW + timedelta(minutes=2, seconds=1),
            line_id="PARK-ENTRY-VA",
            crossing_source="va_local_zone",
        ),
    }
    for crossing_id in ingest_order:
        coordinator.ingest_crossing(requests[crossing_id], [crossing_id.encode()])

    result = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["crossing_id"] == "primary-va-local"
    assert sink.payloads[0]["reid_score"] == pytest.approx(local_vector[0])
    assert sink.payloads[0]["ocr_text"] == "AAA-1111"
    assert coordinator.state_summary()["crossing_count"] == 0


@pytest.mark.parametrize("strong_source", ["hikvision", "va_local_zone"])
def test_fallback_family_aggregates_nonconflicting_ocr(strong_source):
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                role="anpr",
            )
        ],
        "fallback-hikvision": [
            _frame(
                "fallback-hikvision",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111" if strong_source == "hikvision" else "",
                role="fallback",
            )
        ],
        "fallback-va-local": [
            _frame(
                "fallback-va-local",
                "CAM-03",
                (0.999, 0.01),
                plate="AAA1111" if strong_source == "va_local_zone" else "",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "fallback-hikvision",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="hikvision",
        ),
        [b"fallback-hikvision"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "fallback-va-local",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5, seconds=1),
            crossing_source="va_local_zone",
        ),
        [b"fallback-va-local"],
    )

    result = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert result.decision_status == "confirmed"
    assert len(sink.payloads) == 1
    assert sink.payloads[0]["ocr_source"] == "fallback"
    assert sink.payloads[0]["ocr_text"] == "AAA-1111"
    assert coordinator.state_summary()["crossing_count"] == 0


def test_conflicting_fallback_producers_remain_ambiguous_and_fail_closed():
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                plate="AAA1111",
                role="anpr",
            )
        ],
        "fallback-hikvision": [
            _frame(
                "fallback-hikvision",
                "CAM-03",
                (1.0, 0.0),
                plate="AAA1111",
                role="fallback",
            )
        ],
        "fallback-va-local": [
            _frame(
                "fallback-va-local",
                "CAM-03",
                # The views exceed the dedicated producer-duplicate floor;
                # their reliable OCR disagreement must remain fail closed.
                (0.96, 0.28),
                plate="BBB2222",
                role="fallback",
            )
        ],
    }
    coordinator, _, sink = _coordinator(evidence)
    coordinator.ingest_crossing(
        _crossing(
            "fallback-hikvision",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5),
            crossing_source="hikvision",
        ),
        [b"fallback-hikvision"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "fallback-va-local",
            role=CrossingRole.FALLBACK,
            captured_at=NOW + timedelta(minutes=5, seconds=1),
            crossing_source="va_local_zone",
        ),
        [b"fallback-va-local"],
    )

    waiting = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA-1111", captured_at=NOW),
        [b"anpr-1"],
    )

    assert waiting.decision_id is None
    assert sink.payloads == []
    assert coordinator.state_summary()["attempt_count"] == 1
    assert coordinator.state_summary()["crossing_count"] == 2
