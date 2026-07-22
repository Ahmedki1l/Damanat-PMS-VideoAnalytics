"""Regression specifications for adversarial Entry V2 ordering failures."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

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


def _settings(*, max_pending_crossings: int = 8) -> EntrySettings:
    return replace(
        EntrySettings(),
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=8,
        max_pending_crossings=max_pending_crossings,
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
    plate: str,
    source_role: str,
) -> FrameEvidence:
    return FrameEvidence(
        evidence_id=f"{event_id}:0",
        embedding=embedding,
        plate=PlateEvidence(
            evidence_id=f"{event_id}:0",
            camera_id=camera_id,
            source_role=source_role,
            state=PlateReadState.READABLE,
            text=plate,
            confidence=0.99,
        ),
    )


def _attempt(
    attempt_id: str,
    plate: str,
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
    role: CrossingRole,
    captured_at: datetime,
    crossing_source: str,
) -> CrossingInput:
    primary = role == CrossingRole.PRIMARY
    return CrossingInput(
        crossing_id=crossing_id,
        source_event_id=f"source-{crossing_id}",
        camera_id="CAM-23" if primary else "CAM-03",
        captured_at=captured_at,
        line_id="RAMP-IN" if primary else "B-IN",
        direction="ramp-entry" if primary else "b-entry",
        role=role,
        metadata={"track": 7, "crossing_source": crossing_source},
    )


class _Processor:
    def __init__(self, evidence_by_event):
        self._evidence_by_event = evidence_by_event

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        del camera_id, source_role, images, metadata
        return tuple(self._evidence_by_event[event_id])


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


def _coordinator(evidence_by_event, *, max_pending_crossings: int = 8):
    sink = _Sink()
    coordinator = EntryCoordinator(
        _settings(max_pending_crossings=max_pending_crossings),
        _Processor(evidence_by_event),
        sink,
    )
    return coordinator, sink


def test_late_producer_twins_cannot_exhaust_crossing_capacity():
    """Completed unrelated journeys must not make later entries unavailable."""
    embeddings = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    plates = ("AAA1111", "BBB2222", "CCC3333")
    evidence = {}
    for index, (embedding, plate) in enumerate(zip(embeddings, plates), start=1):
        evidence[f"attempt-{index}"] = [
            _frame(
                f"attempt-{index}",
                "ANPR-ENTRY",
                embedding,
                plate,
                "anpr",
            )
        ]
        evidence[f"hik-{index}"] = [
            _frame(f"hik-{index}", "CAM-23", embedding, plate, "primary")
        ]
        evidence[f"local-{index}"] = [
            _frame(f"local-{index}", "CAM-23", embedding, plate, "primary")
        ]

    coordinator, sink = _coordinator(evidence, max_pending_crossings=2)
    for index, plate in enumerate(plates[:2], start=1):
        attempt_at = NOW + timedelta(minutes=10 * index)
        coordinator.ingest_attempt(
            _attempt(f"attempt-{index}", plate, attempt_at),
            [b"anpr"],
        )
        coordinator.ingest_crossing(
            _crossing(
                f"hik-{index}",
                CrossingRole.PRIMARY,
                attempt_at + timedelta(minutes=1),
                "hikvision",
            ),
            [b"hik"],
        )
        coordinator.ingest_crossing(
            _crossing(
                f"local-{index}",
                CrossingRole.PRIMARY,
                attempt_at + timedelta(minutes=1, seconds=1),
                "va_local_zone",
            ),
            [b"local"],
        )

    third_attempt_at = NOW + timedelta(minutes=30)
    coordinator.ingest_attempt(
        _attempt("attempt-3", plates[2], third_attempt_at),
        [b"anpr"],
    )
    third = coordinator.ingest_crossing(
        _crossing(
            "hik-3",
            CrossingRole.PRIMARY,
            third_attempt_at + timedelta(minutes=1),
            "hikvision",
        ),
        [b"hik"],
    )

    assert third.decision_status == "confirmed"
    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
        "attempt-3",
    ]


def test_much_later_primary_cannot_steal_an_earlier_valid_fallback():
    """Camera-role priority must not override physical stage chronology."""
    evidence = {
        "attempt-1": [
            _frame(
                "attempt-1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                "AAA1111",
                "anpr",
            )
        ],
        "fallback-1": [
            _frame(
                "fallback-1",
                "CAM-03",
                (1.0, 0.0),
                "AAA1111",
                "fallback",
            )
        ],
        "primary-future": [
            _frame(
                "primary-future",
                "CAM-23",
                (1.0, 0.0),
                "AAA1111",
                "primary",
            )
        ],
    }
    coordinator, sink = _coordinator(evidence)
    fallback_at = NOW + timedelta(minutes=5)
    coordinator.ingest_crossing(
        _crossing(
            "fallback-1",
            CrossingRole.FALLBACK,
            fallback_at,
            "va_local_zone",
        ),
        [b"fallback"],
    )
    coordinator.ingest_crossing(
        _crossing(
            "primary-future",
            CrossingRole.PRIMARY,
            NOW + timedelta(hours=6),
            "hikvision",
        ),
        [b"primary"],
    )

    result = coordinator.ingest_attempt(
        _attempt("attempt-1", "AAA1111", NOW),
        [b"anpr"],
    )

    assert result.decision_status == "confirmed"
    assert len(sink.payloads) == 1
    assert sink.payloads[0]["crossing_id"] == "fallback-1"
    assert sink.payloads[0]["entry_captured_at"] == fallback_at.isoformat()


def test_normal_cam23_then_cam03_journeys_do_not_exhaust_crossing_capacity():
    """Exit source time compacts downstream evidence without a business TTL."""
    embeddings = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    plates = ("AAA1111", "BBB2222", "CCC3333")
    evidence = {}
    for index, (embedding, plate) in enumerate(zip(embeddings, plates), start=1):
        evidence[f"attempt-{index}"] = [
            _frame(
                f"attempt-{index}",
                "ANPR-ENTRY",
                embedding,
                plate,
                "anpr",
            )
        ]
        evidence[f"primary-{index}"] = [
            _frame(
                f"primary-{index}",
                "CAM-23",
                embedding,
                plate,
                "primary",
            )
        ]
        evidence[f"fallback-{index}"] = [
            _frame(
                f"fallback-{index}",
                "CAM-03",
                embedding,
                plate,
                "fallback",
            )
        ]

    coordinator, sink = _coordinator(evidence, max_pending_crossings=2)
    for index, plate in enumerate(plates, start=1):
        attempt_at = NOW + timedelta(minutes=10 * index)
        coordinator.ingest_attempt(
            _attempt(f"attempt-{index}", plate, attempt_at),
            [b"anpr"],
        )
        confirmation = coordinator.ingest_crossing(
            _crossing(
                f"primary-{index}",
                CrossingRole.PRIMARY,
                attempt_at + timedelta(minutes=1),
                "hikvision",
            ),
            [b"primary"],
        )
        assert confirmation.decision_status == "confirmed"
        coordinator.ingest_crossing(
            _crossing(
                f"fallback-{index}",
                CrossingRole.FALLBACK,
                attempt_at + timedelta(minutes=2),
                "va_local_zone",
            ),
            [b"fallback"],
        )
        assert coordinator.record_exit(
            plate,
            attempt_at + timedelta(minutes=3),
        ) == 1

    assert [payload["attempt_id"] for payload in sink.payloads] == [
        "attempt-1",
        "attempt-2",
        "attempt-3",
    ]
