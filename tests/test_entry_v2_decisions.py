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
    EntryCapacityExceeded,
    EntryConflict,
    EntryMode,
    EntryUnavailable,
    EvidenceUnavailable,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def settings(**overrides):
    base = EntrySettings(
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
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"ramp-entry"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B-IN"}),
        fallback_directions=frozenset({"b-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-key",
    )
    return replace(base, **overrides)


def frame(event_id, camera, vector, state, text="", confidence=0.0, role="anpr"):
    return FrameEvidence(
        evidence_id=f"{event_id}:0",
        embedding=tuple(vector),
        plate=PlateEvidence(
            evidence_id=f"{event_id}:0",
            camera_id=camera,
            source_role=role,
            state=state,
            text=text,
            confidence=confidence,
        ),
    )


class FakeProcessor:
    def __init__(self, evidence_by_event):
        self.evidence_by_event = evidence_by_event
        self.calls = []

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        self.calls.append((event_id, camera_id, source_role, len(images), dict(metadata)))
        return tuple(self.evidence_by_event[event_id])


class BarrierProcessor(FakeProcessor):
    def __init__(self, evidence_by_event, synchronized_ids):
        super().__init__(evidence_by_event)
        self.synchronized_ids = set(synchronized_ids)
        self.barrier = threading.Barrier(len(self.synchronized_ids))

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        if event_id in self.synchronized_ids:
            self.barrier.wait(timeout=2)
        return super().analyze(
            event_id=event_id,
            camera_id=camera_id,
            source_role=source_role,
            images=images,
            metadata=metadata,
        )


class RecordingSink:
    def __init__(
        self,
        delivered=True,
        publish_identity=True,
        session_committed=None,
        ack_result="",
    ):
        self.should_deliver = delivered
        self.publish_identity = publish_identity
        self.session_committed = session_committed
        self.ack_result = ack_result
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(
            self.should_deliver,
            1,
            "" if self.should_deliver else "down",
            publish_identity=self.publish_identity,
            session_committed=self.session_committed,
            ack_result=self.ack_result,
        )


class PermanentFailureSink:
    def deliver(self, payload):
        del payload
        return DeliveryResult(False, 1, "http_401", retryable=False)


class RecordingPublisher:
    def __init__(self, fail=False):
        self.fail = fail
        self.identities = []

    def publish(self, identity):
        self.identities.append(identity)
        if self.fail:
            raise RuntimeError("registry unavailable")


class BlockingSink:
    def __init__(self, delivered):
        self.delivered = delivered
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def deliver(self, payload):
        del payload
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("test callback was not released")
        return DeliveryResult(self.delivered, 1, "" if self.delivered else "down")


def attempt(identifier, plate, camera="ANPR-ENTRY", confidence=0.91):
    return AttemptInput(
        attempt_id=identifier,
        source_event_id=f"source-{identifier}",
        camera_id=camera,
        captured_at=NOW,
        reported_plate=plate,
        reported_confidence=confidence,
        metadata={"lane": "entry"},
    )


def crossing(
    identifier,
    *,
    camera="CAM-23",
    line="RAMP-IN",
    direction="ramp-entry",
    role=CrossingRole.PRIMARY,
    captured_at=NOW,
):
    crossing_source = (
        "va_local_zone" if role == CrossingRole.FALLBACK else "hikvision"
    )
    return CrossingInput(
        crossing_id=identifier,
        source_event_id=f"source-{identifier}",
        camera_id=camera,
        captured_at=captured_at,
        line_id=line,
        direction=direction,
        role=role,
        metadata={"track": 7, "crossing_source": crossing_source},
    )


def coordinator(evidence, *, cfg=None, sink=None, publisher=None):
    sink = sink or RecordingSink()
    return (
        EntryCoordinator(
            cfg or settings(),
            FakeProcessor(evidence),
            sink,
            identity_publisher=publisher,
        ),
        sink,
    )


def test_strict_reid_and_primary_ocr_exact_confirms_metadata_only():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "ABC1234", 0.96, "primary")],
    }
    coord, sink = coordinator(evidence)

    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt-image"])
    result = coord.ingest_crossing(
        crossing("c1", captured_at=NOW + timedelta(seconds=30)),
        [b"crossing-image"],
    )

    assert result.decision_status == "confirmed"
    assert result.callback_delivered is True
    assert sink.payloads[0]["status"] == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"
    assert sink.payloads[0]["entry_camera_id"] == "ANPR-ENTRY"
    assert sink.payloads[0]["entry_captured_at"] == (
        NOW + timedelta(seconds=30)
    ).isoformat()
    assert sink.payloads[0]["crossing_id"] == "c1"
    assert sink.payloads[0]["plate_confidence"] == pytest.approx(91.0)
    assert sink.payloads[0]["reid_score"] == pytest.approx(1.0)
    assert not any("image" in key or "path" in key for key in sink.payloads[0])
    assert coord.state_summary()["attempt_count"] == 0


def test_mixed_vehicle_anpr_burst_is_rejected_before_cached_ocr():
    evidence = {
        "a-mixed": [
            frame(
                "a-mixed-plate",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
            ),
            frame(
                "a-mixed-car",
                "ANPR-ENTRY",
                (0.0, 1.0),
                PlateReadState.NO_PLATE,
            ),
        ],
    }
    coord, sink = coordinator(evidence)

    with pytest.raises(EvidenceUnavailable, match="mixed_vehicle_evidence"):
        coord.ingest_attempt(attempt("a-mixed", "ABC-1234"), [b"car-a", b"car-b"])

    assert sink.payloads == []
    assert coord.state_summary()["attempt_count"] == 0


def test_mixed_primary_burst_cannot_combine_reid_ocr_or_registry_anchor():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR-ENTRY",
                (0.0, 1.0),
                PlateReadState.READABLE,
                "NEW2222",
                0.99,
            )
        ],
        "c-mixed": [
            # OCR belongs to car A, while car B is the ReID match for a1. The
            # old aggregate policy could combine them into one correction.
            frame(
                "c-car-a",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "NEW2222",
                0.99,
                "primary",
            ),
            frame(
                "c-car-b",
                "CAM-23",
                (0.0, 1.0),
                PlateReadState.NO_PLATE,
                role="primary",
            ),
        ],
    }
    publisher = RecordingPublisher()
    coord, sink = coordinator(evidence, publisher=publisher)
    coord.ingest_attempt(attempt("a1", "OLD-1111"), [b"anpr"])

    with pytest.raises(EvidenceUnavailable, match="mixed_vehicle_evidence"):
        coord.ingest_crossing(crossing("c-mixed"), [b"car-a", b"car-b"])

    assert sink.payloads == []
    assert publisher.identities == []
    assert coord.state_summary()["attempt_count"] == 1
    assert coord.state_summary()["crossing_count"] == 0


def test_same_vehicle_burst_remains_eligible_for_cached_anpr_ocr():
    evidence = {
        "a1": [
            frame("a1-0", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE),
            frame(
                "a1-1",
                "ANPR-ENTRY",
                (0.999, 0.01),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
            ),
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
    }
    publisher = RecordingPublisher()
    coord, sink = coordinator(evidence, publisher=publisher)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"burst-1", b"burst-2"])

    result = coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[-1]["ocr_source"] == "anpr_cached"
    assert len(publisher.identities) == 1
    assert len(publisher.identities[0].attempt_embeddings) == 2


def test_readable_primary_conflict_abstains_and_blocks_fallback():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.95, "primary")],
        "f1": [frame("f1", "CAM-03", (1.0, 0.0), PlateReadState.READABLE, "ABC1234", 0.98, "fallback")],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    first = coord.ingest_crossing(crossing("c1"), [b"c"])
    fallback = coord.ingest_crossing(
        crossing("f1", camera="CAM-03", line="B-IN", direction="B-entry", role=CrossingRole.FALLBACK),
        [b"f"],
    )

    assert first.decision_status == "abstained"
    assert sink.payloads[0]["reason"] == "correction_consensus_insufficient"
    assert fallback.decision_status == "abstained"
    assert sink.payloads[1]["reason"] == "readable_primary_blocks_fallback"
    assert coord.state_summary()["attempt_count"] == 1


def test_distinct_reliable_primary_reads_cannot_use_arrival_order_to_confirm():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "p-wrong": [
            frame(
                "p-wrong",
                "CAM-23",
                (0.8, 0.6),
                PlateReadState.READABLE,
                "XYZ9999",
                0.99,
                "primary",
            )
        ],
        "p-correct": [
            frame(
                "p-correct",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    first = coord.ingest_crossing(crossing("p-wrong"), [b"wrong"])
    second = coord.ingest_crossing(crossing("p-correct"), [b"correct"])

    assert first.decision_status == "abstained"
    assert second.decision_status == "abstained"
    assert sink.payloads[-1]["reason"] == "primary_ocr_conflict"
    assert sorted(sink.payloads[-1]["ocr_evidence_ids"]) == [
        "p-correct:0",
        "p-wrong:0",
    ]
    assert coord.state_summary()["attempt_count"] == 1


def test_low_confidence_primary_noise_cannot_veto_reliable_primary_ocr():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1-good",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "primary",
            ),
            frame(
                "c1-noise",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "XYZ9999",
                0.20,
                "primary",
            ),
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    result = coord.ingest_crossing(crossing("c1"), [b"good", b"noise"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[-1]["canonical_plate"] == "ABC-1234"
    assert sink.payloads[-1]["ocr_evidence_ids"] == ["c1-good:0"]


def test_pending_conflicting_primary_reads_are_seen_before_best_one_can_confirm():
    evidence = {
        "p-wrong": [
            frame(
                "p-wrong",
                "CAM-23",
                (0.8, 0.6),
                PlateReadState.READABLE,
                "XYZ9999",
                0.99,
                "primary",
            )
        ],
        "p-correct": [
            frame(
                "p-correct",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "primary",
            )
        ],
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_crossing(crossing("p-wrong"), [b"wrong"])
    coord.ingest_crossing(crossing("p-correct"), [b"correct"])

    result = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    assert result.decision_status == "abstained"
    assert sink.payloads
    assert all(payload["status"] == "abstained" for payload in sink.payloads)
    assert all(
        payload["reason"] == "primary_ocr_conflict" for payload in sink.payloads
    )
    assert coord.state_summary()["attempt_count"] == 1


def test_anpr_cached_ocr_is_used_only_when_primary_has_no_plate():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.93,
            )
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["ocr_source"] == "anpr_cached"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"


def test_digit_first_cached_ocr_does_not_match_letter_first_reported_plate():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "1234ABC",
                0.93,
            )
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert all(payload["status"] == "abstained" for payload in sink.payloads)
    assert coord.state_summary()["attempt_count"] == 1


def test_high_confidence_anpr_ocr_replaces_failed_low_confidence_primary_read():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.96,
            )
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "XYZ9999",
                0.60,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[-1]["ocr_source"] == "anpr_cached"
    assert sink.payloads[-1]["canonical_plate"] == "ABC-1234"


def test_downstream_fallback_can_confirm_when_primary_event_never_arrives():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "f1": [frame("f1", "CAM-03", (1.0, 0.0), PlateReadState.READABLE, "ABC1234", 0.96, "fallback")],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(
        crossing("f1", camera="CAM-03", line="B-IN", direction="B-entry", role=CrossingRole.FALLBACK),
        [b"f"],
    )

    assert result.decision_status == "confirmed"
    assert [payload["status"] for payload in sink.payloads] == ["confirmed"]
    assert sink.payloads[0]["crossing_id"] == "f1"
    assert coord.state_summary()["attempt_count"] == 0


def test_downstream_fallback_reid_uses_cached_anpr_ocr_when_its_ocr_fails():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.96,
            )
        ],
        "f1": [
            frame(
                "f1",
                "CAM-03",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="fallback",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"anpr"])

    result = coord.ingest_crossing(
        crossing(
            "f1",
            camera="CAM-03",
            line="B-IN",
            direction="B-entry",
            role=CrossingRole.FALLBACK,
        ),
        [b"fallback"],
    )

    assert result.decision_status == "confirmed"
    assert sink.payloads[-1]["ocr_source"] == "anpr_cached"


def test_primary_policy_runs_before_higher_scoring_fallback_in_same_evaluation():
    evidence = {
        "p1": [
            frame(
                "p1",
                "CAM-23",
                (0.9, 0.1),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
        "f1": [
            frame(
                "f1",
                "CAM-03",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.96,
                "fallback",
            )
        ],
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_crossing(crossing("p1"), [b"p"])
    coord.ingest_crossing(
        crossing(
            "f1",
            camera="CAM-03",
            line="B-IN",
            direction="B-entry",
            role=CrossingRole.FALLBACK,
        ),
        [b"f"],
    )

    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    assert [payload["crossing_id"] for payload in sink.payloads] == ["p1", "f1"]
    assert [payload["status"] for payload in sink.payloads] == [
        "abstained",
        "confirmed",
    ]
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0


def test_primary_confirmation_compacts_correlated_fallback_before_next_car():
    evidence = {
        "p1": [
            frame(
                "p1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "primary",
            )
        ],
        "f1": [
            frame(
                "f1",
                "CAM-03",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "fallback",
            )
        ],
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_crossing(crossing("p1"), [b"p"])
    coord.ingest_crossing(
        crossing(
            "f1",
            camera="CAM-03",
            line="B-IN",
            direction="B-entry",
            role=CrossingRole.FALLBACK,
        ),
        [b"f"],
    )

    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a1"])

    assert [payload["crossing_id"] for payload in sink.payloads] == ["p1"]
    assert coord.state_summary()["crossing_count"] == 0
    coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"a2"])
    assert len(sink.payloads) == 1
    assert coord.state_summary()["attempt_count"] == 1


def test_uncertain_lookalike_crossing_remains_eligible_for_later_attempt():
    lookalike_vector = (0.85, 0.526782687642637)
    evidence = {
        "p1": [
            frame(
                "p1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
        "p2": [
            frame(
                "p2",
                "CAM-23",
                lookalike_vector,
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [
            frame(
                "a2",
                "ANPR",
                lookalike_vector,
                PlateReadState.READABLE,
                "BBB2222",
                0.99,
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_crossing(crossing("p1"), [b"p1"])
    coord.ingest_crossing(crossing("p2"), [b"p2"])

    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])

    assert [payload["crossing_id"] for payload in sink.payloads] == ["p1"]
    assert coord.state_summary()["crossing_count"] == 1
    confirmed = coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])
    assert confirmed.decision_status == "confirmed"
    assert [payload["crossing_id"] for payload in sink.payloads] == ["p1", "p2"]
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0


def test_unique_assignments_recompute_after_runner_group_finalizes():
    root_three_over_two = 0.8660254037844386
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [
            frame(
                "a2",
                "ANPR",
                (0.5, root_three_over_two),
                PlateReadState.NO_PLATE,
            )
        ],
        # Equally similar to both attempts, so this crossing is initially
        # ambiguous and must wait for a1 to be finalized.
        "c2": [
            frame(
                "c2",
                "CAM-23",
                (root_three_over_two, 0.5),
                PlateReadState.READABLE,
                "BBB2222",
                0.99,
                "primary",
            )
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])
    waiting = coord.ingest_crossing(crossing("c2"), [b"c2"])
    assert waiting.decision_id is None

    coord.ingest_crossing(crossing("c1"), [b"c1"])

    assert [payload["attempt_id"] for payload in sink.payloads] == ["a1", "a2"]
    assert [payload["crossing_id"] for payload in sink.payloads] == ["c1", "c2"]
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0


def test_fractional_camera_confidence_is_rounded_for_integer_pms_contract():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(
        attempt("a1", "ABC-1234", confidence=0.925),
        [b"a"],
    )
    coord.ingest_crossing(crossing("c1"), [b"c"])

    assert sink.payloads[-1]["plate_confidence"] == 92


def test_unreported_plate_correction_needs_independent_camera_consensus():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.94)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.97, "primary")],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    payload = sink.payloads[0]
    assert payload["canonical_plate"] == "XYZ-9999"
    assert payload["corrected"] is True
    assert payload["superseded_plates"] == ["ABC-1234"]
    assert sorted(payload["ocr_evidence_ids"]) == ["a1:0", "c1:0"]


def test_single_crossing_ocr_cannot_rewrite_reported_plate():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.97, "primary")],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[0]["reason"] == "correction_consensus_insufficient"
    assert coord.state_summary()["attempt_count"] == 1


def test_late_correct_anpr_reuses_abstained_crossing_and_supersedes_wrong_plate():
    evidence = {
        "a-wrong": [
            frame(
                "a-wrong",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
            )
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "XYZ9999",
                0.97,
                "primary",
            )
        ],
        "a-correct": [
            frame(
                "a-correct",
                "ANPR-ENTRY",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a-wrong", "ABC-1234"), [b"wrong"])
    first = coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert first.decision_status == "abstained"
    assert coord.state_summary()["crossing_count"] == 1

    second = coord.ingest_attempt(attempt("a-correct", "XYZ-9999"), [b"correct"])

    assert second.decision_status == "confirmed"
    assert [payload["status"] for payload in sink.payloads] == [
        "abstained",
        "confirmed",
    ]
    assert sink.payloads[-1]["canonical_plate"] == "XYZ-9999"
    assert sink.payloads[-1]["superseded_plates"] == ["ABC-1234"]
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0


def test_entry_time_is_physical_crossing_after_long_barrier_wait():
    crossing_at = NOW + timedelta(minutes=10)
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.97,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])
    coord.ingest_crossing(crossing("c1", captured_at=crossing_at), [b"crossing"])

    assert sink.payloads[-1]["entry_camera_id"] == "ANPR-ENTRY"
    assert sink.payloads[-1]["entry_captured_at"] == crossing_at.isoformat()


def test_authoritative_confirmation_hands_embeddings_to_live_registry():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.97,
                "primary",
            )
        ],
    }
    publisher = RecordingPublisher()
    coord, _ = coordinator(evidence, publisher=publisher)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])
    coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert len(publisher.identities) == 1
    identity = publisher.identities[0]
    assert identity.canonical_plate == "ABC-1234"
    assert identity.crossing_camera_id == "CAM-23"
    assert identity.crossing_embeddings == ((1.0, 0.0),)
    assert identity.attempt_embeddings == (("ANPR-ENTRY", (1.0, 0.0)),)


def test_terminal_stale_after_exit_ack_compacts_without_registry_identity():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.97,
                "primary",
            )
        ],
    }
    publisher = RecordingPublisher()
    sink = RecordingSink(
        publish_identity=False,
        session_committed=False,
        ack_result="stale_after_exit",
    )
    coord, _ = coordinator(evidence, sink=sink, publisher=publisher)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    result = coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert result.callback_delivered is True
    assert publisher.identities == []
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["pending_callback_count"] == 0


def test_registry_handoff_failure_backpressures_then_idempotent_retry_succeeds():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.97,
                "primary",
            )
        ],
    }
    publisher = RecordingPublisher(fail=True)
    coord, sink = coordinator(evidence, publisher=publisher)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    with pytest.raises(EntryUnavailable, match="entry_confirmation_delivery_failed"):
        coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert coord.state_summary()["pending_callback_count"] == 1
    assert coord.state_summary()["attempt_count"] == 1
    publisher.fail = False
    duplicate = coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert duplicate.duplicate is True
    assert duplicate.callback_delivered is True
    assert len(sink.payloads) == 2
    assert len(publisher.identities) == 2
    assert coord.state_summary()["attempt_count"] == 0


def test_permanent_callback_failure_stops_admission_without_retry_queue():
    evidence = {
        "a1": [
            frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.97,
                "primary",
            )
        ],
    }

    coord, _ = coordinator(evidence, sink=PermanentFailureSink())
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"attempt"])

    with pytest.raises(
        EntryUnavailable,
        match="entry_confirmation_permanent_failure",
    ):
        coord.ingest_crossing(crossing("c1"), [b"crossing"])

    state = coord.state_summary()
    assert state["pending_callback_count"] == 0
    assert state["permanent_callback_failure_count"] == 1
    assert coord.available is False
    assert coord.unavailable_reason.endswith(":http_401")
    with pytest.raises(EntryUnavailable, match="http_401"):
        coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"next"])


def test_pending_retry_becoming_permanent_is_removed_and_stops_admission():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    coord, _ = coordinator(evidence, sink=RecordingSink(delivered=False))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with pytest.raises(EntryUnavailable, match="delivery_failed"):
        coord.ingest_crossing(crossing("c1"), [b"c"])

    coord._sink = PermanentFailureSink()
    assert list(coord.retry_pending_callbacks().values()) == [False]
    state = coord.state_summary()
    assert state["pending_callback_count"] == 0
    assert state["permanent_callback_failure_count"] == 1
    assert coord.available is False


def test_duplicate_retry_becoming_permanent_fails_closed():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    coord, _ = coordinator(evidence, sink=RecordingSink(delivered=False))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with pytest.raises(EntryUnavailable, match="delivery_failed"):
        coord.ingest_crossing(crossing("c1"), [b"c"])

    coord._sink = PermanentFailureSink()
    with pytest.raises(
        EntryUnavailable,
        match="entry_confirmation_permanent_failure",
    ):
        coord.ingest_crossing(crossing("c1"), [b"c"])

    assert coord.state_summary()["pending_callback_count"] == 0
    assert coord.available is False


@pytest.mark.parametrize(
    ("ocr_confidence", "expected"),
    [(0.75, "confirmed"), (0.749, "abstained")],
)
def test_ocr_confidence_boundary(ocr_confidence, expected):
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "AAA1111", ocr_confidence, "primary")],
    }
    coord, _ = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])
    assert result.decision_status == expected


@pytest.mark.parametrize("garbage", ["9990", "ABCDEFG", "UNKNOWN"])
def test_implausible_ocr_can_never_become_a_confirmed_plate(garbage):
    evidence = {
        "a1": [
            frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)
        ],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                garbage,
                0.99,
                "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", garbage), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[-1]["reason"] == "ocr_plate_implausible"


def test_later_no_plate_crossing_cannot_weaken_prior_readable_conflict():
    evidence = {
        "a1": [
            frame(
                "a1",
                "ANPR",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "ABC1234",
                0.99,
            )
        ],
        "c-conflict": [
            frame(
                "c-conflict",
                "CAM-23",
                (0.8, 0.0),
                PlateReadState.READABLE,
                "XYZ9999",
                0.99,
                "primary",
            )
        ],
        "c-empty": [
            frame(
                "c-empty",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    coord.ingest_crossing(crossing("c-conflict"), [b"conflict"])
    second = coord.ingest_crossing(crossing("c-empty"), [b"empty"])

    assert second.decision_status == "abstained"
    assert sink.payloads[-1]["reason"] == (
        "prior_readable_primary_blocks_weaker_evidence"
    )
    assert coord.state_summary()["attempt_count"] == 1


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.75, "confirmed"), (0.749, None)],
)
def test_reid_absolute_score_boundary_is_observable(score, expected, caplog):
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (score, 0.0), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
    }
    coord, _ = coordinator(evidence)
    caplog.set_level("INFO", logger="src.entry.coordinator")
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == expected
    reid_logs = [
        record.getMessage()
        for record in caplog.records
        if "[EntryV2][ReID]" in record.getMessage()
    ]
    assert len(reid_logs) == 1
    assert "stage=uniqueness_only" in reid_logs[0]
    assert f"score={score:.4f}" in reid_logs[0]
    assert "min_score=0.7500" in reid_logs[0]
    if expected == "confirmed":
        assert "result=accepted reason=accepted" in reid_logs[0]
    else:
        assert (
            "result=rejected reason=score_below_minimum"
            in reid_logs[0]
        )


def test_same_car_wrong_and_correct_attempts_merge_then_supersede_wrong_plate():
    evidence = {
        "a-wrong": [frame("a-wrong", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a-correct": [frame("a-correct", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.96)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.98, "primary")],
    }
    coord, sink = coordinator(evidence)
    first = coord.ingest_attempt(attempt("a-wrong", "ABC-1234"), [b"a1"])
    second = coord.ingest_attempt(attempt("a-correct", "XYZ-9999"), [b"a2"])

    assert first.group_id == second.group_id
    assert coord.state_summary()["group_count"] == 1
    coord.ingest_crossing(crossing("c1"), [b"c"])

    payload = sink.payloads[0]
    assert payload["canonical_plate"] == "XYZ-9999"
    assert payload["superseded_plates"] == ["ABC-1234"]
    assert payload["attempt_id"] == "a-correct"


def test_matching_later_car_does_not_drop_unrelated_earlier_attempt():
    evidence = {
        "earlier": [frame("earlier", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "later": [frame("later", "ANPR-ENTRY", (0.0, 1.0), PlateReadState.NO_PLATE)],
        "c-later": [frame("c-later", "CAM-23", (0.0, 1.0), PlateReadState.READABLE, "BBB2222", 0.98, "primary")],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("earlier", "AAA-1111"), [b"a"])
    coord.ingest_attempt(attempt("later", "BBB-2222"), [b"b"])
    coord.ingest_crossing(crossing("c-later"), [b"c"])

    state = coord.state_summary()
    assert state["attempt_count"] == 1
    remaining = next(iter(state["groups"].values()))
    assert remaining["attempt_ids"] == ["earlier"]
    assert sink.payloads[0]["attempt_id"] == "later"


def test_row_margin_ambiguity_leaves_everything_pending_and_is_logged_once(caplog):
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (0.99, 0.1), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
    }
    coord, sink = coordinator(evidence, cfg=settings(merge_min_score=1.1))
    caplog.set_level("INFO", logger="src.entry.coordinator")
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"b"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])
    with coord._lock:
        coord._rank_pending_matches_locked()
        coord._rank_pending_matches_locked()

    assert result.decision_id is None
    assert sink.payloads == []
    assert coord.state_summary()["attempt_count"] == 2
    assert coord.state_summary()["crossing_count"] == 1
    reid_logs = [
        record.getMessage()
        for record in caplog.records
        if "[EntryV2][ReID]" in record.getMessage()
    ]
    assert len(reid_logs) == 1
    assert "row_runner=" in reid_logs[0]
    assert "row_margin=" in reid_logs[0]
    assert "row_min_margin=0.0800" in reid_logs[0]
    assert "result=rejected reason=row_margin_below_minimum" in reid_logs[0]


def test_column_margin_ambiguity_is_checked_and_logged_when_crossings_arrive_first(
    caplog,
):
    evidence = {
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
        "c2": [frame("c2", "CAM-23", (0.99, 0.1), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, sink = coordinator(evidence)
    caplog.set_level("INFO", logger="src.entry.coordinator")
    coord.ingest_crossing(crossing("c1"), [b"c1"])
    coord.ingest_crossing(crossing("c2"), [b"c2"])
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    assert result.decision_id is None
    assert sink.payloads == []
    assert coord.state_summary()["crossing_count"] == 2
    reid_logs = [
        record.getMessage()
        for record in caplog.records
        if "[EntryV2][ReID]" in record.getMessage()
    ]
    assert len(reid_logs) == 2
    assert all("column_runner=" in message for message in reid_logs)
    assert all("column_margin=" in message for message in reid_logs)
    assert all("column_min_margin=0.0800" in message for message in reid_logs)
    assert all(
        "result=rejected reason=column_margin_below_minimum" in message
        for message in reid_logs
    )


def test_concurrent_crossings_are_all_inserted_before_uniqueness_decision():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
        "c2": [
            frame(
                "c2",
                "CAM-23",
                (0.99, 0.1),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    processor = BarrierProcessor(evidence, {"c1", "c2"})
    sink = RecordingSink()
    coord = EntryCoordinator(settings(), processor, sink)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    errors = []

    def submit(identifier):
        try:
            coord.ingest_crossing(crossing(identifier), [identifier.encode()])
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=submit, args=(item,)) for item in ("c1", "c2")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert sink.payloads == []
    assert coord.state_summary()["attempt_count"] == 1
    assert coord.state_summary()["crossing_count"] == 2
    assert coord.state_summary()["analysis_inflight_count"] == 0


def test_concurrent_attempts_are_all_inserted_before_uniqueness_decision():
    cosine_40 = 0.766044443118978
    sine_40 = 0.6427876096865394
    evidence = {
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
        "a1": [
            frame(
                "a1",
                "ANPR",
                (cosine_40, sine_40),
                PlateReadState.NO_PLATE,
            )
        ],
        "a2": [
            frame(
                "a2",
                "ANPR",
                (cosine_40, -sine_40),
                PlateReadState.NO_PLATE,
            )
        ],
    }
    processor = BarrierProcessor(evidence, {"a1", "a2"})
    sink = RecordingSink()
    coord = EntryCoordinator(settings(), processor, sink)
    coord.ingest_crossing(crossing("c1"), [b"c"])
    errors = []

    def submit(identifier, plate):
        try:
            coord.ingest_attempt(attempt(identifier, plate), [identifier.encode()])
        except Exception as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=submit, args=("a1", "AAA-1111")),
        threading.Thread(target=submit, args=("a2", "BBB-2222")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert not errors
    assert all(not worker.is_alive() for worker in workers)
    assert sink.payloads == []
    assert coord.state_summary()["attempt_count"] == 2
    assert coord.state_summary()["crossing_count"] == 1
    assert coord.state_summary()["analysis_inflight_count"] == 0


def test_capacity_backpressures_instead_of_evicting_pending_attempt():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (0.0, 1.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence, cfg=settings(max_pending_attempts=1))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    with pytest.raises(EntryCapacityExceeded):
        coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"b"])
    state = coord.state_summary()
    assert state["attempt_count"] == 1
    assert next(iter(state["groups"].values()))["attempt_ids"] == ["a1"]


def test_callback_capacity_is_reserved_before_network_io():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (0.0, 1.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
        "c2": [
            frame(
                "c2",
                "CAM-23",
                (0.0, 1.0),
                PlateReadState.READABLE,
                "BBB2222",
                0.99,
                "primary",
            )
        ],
    }
    sink = BlockingSink(delivered=False)
    coord, _ = coordinator(
        evidence,
        sink=sink,
        cfg=settings(max_pending_callbacks=1, merge_min_score=1.1),
    )
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])
    errors = []

    def first_crossing():
        try:
            coord.ingest_crossing(crossing("c1"), [b"c1"])
        except Exception as exc:  # asserted after the worker joins
            errors.append(exc)

    worker = threading.Thread(target=first_crossing)
    worker.start()
    assert sink.started.wait(timeout=2)
    assert coord.state_summary()["reserved_callback_count"] == 1

    with pytest.raises(EntryCapacityExceeded, match="callback_capacity_exceeded"):
        coord.ingest_crossing(crossing("c2"), [b"c2"])

    sink.release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], EntryUnavailable)
    assert coord.state_summary()["pending_callback_count"] == 1


def _two_independent_entry_pairs():
    return {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (0.0, 1.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
        "c2": [
            frame(
                "c2",
                "CAM-23",
                (0.0, 1.0),
                PlateReadState.READABLE,
                "BBB2222",
                0.99,
                "primary",
            )
        ],
    }


def _submit_two_crossings_concurrently(coord):
    errors = []

    def submit(identifier):
        try:
            coord.ingest_crossing(crossing(identifier), [identifier.encode()])
        except Exception as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=submit, args=(identifier,))
        for identifier in ("c1", "c2")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    assert all(not worker.is_alive() for worker in workers)
    return errors


def test_successful_dispatch_drains_next_decision_after_capacity_frees():
    evidence = _two_independent_entry_pairs()
    processor = BarrierProcessor(evidence, {"c1", "c2"})
    sink = RecordingSink(delivered=True)
    coord = EntryCoordinator(
        settings(max_pending_callbacks=1, merge_min_score=1.1),
        processor,
        sink,
    )
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])

    errors = _submit_two_crossings_concurrently(coord)

    assert errors == []
    assert len(sink.payloads) == 2
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0
    assert coord.state_summary()["pending_callback_count"] == 0


def test_callback_retry_drains_next_pending_decision_after_capacity_frees():
    evidence = _two_independent_entry_pairs()
    processor = BarrierProcessor(evidence, {"c1", "c2"})
    sink = RecordingSink(delivered=False)
    coord = EntryCoordinator(
        settings(max_pending_callbacks=1, merge_min_score=1.1),
        processor,
        sink,
    )
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])
    errors = _submit_two_crossings_concurrently(coord)
    assert len(errors) == 1
    assert isinstance(errors[0], EntryUnavailable)
    assert coord.state_summary()["pending_callback_count"] == 1

    sink.should_deliver = True
    outcomes = coord.retry_pending_callbacks()

    assert list(outcomes.values()) == [True]
    assert len(sink.payloads) == 3
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["crossing_count"] == 0
    assert coord.state_summary()["pending_callback_count"] == 0


def test_manual_callback_retry_claim_prevents_duplicate_delivery():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    coord, _ = coordinator(evidence, sink=RecordingSink(delivered=False))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with pytest.raises(EntryUnavailable):
        coord.ingest_crossing(crossing("c1"), [b"c"])

    blocking = BlockingSink(delivered=True)
    coord._sink = blocking
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(coord.retry_pending_callbacks())
    )
    worker.start()
    assert blocking.started.wait(timeout=2)

    assert coord.retry_pending_callbacks() == {}
    assert blocking.calls == 1

    blocking.release.set()
    worker.join(timeout=2)
    assert outcomes and list(outcomes[0].values()) == [True]
    assert coord.state_summary()["pending_callback_count"] == 0


def test_scheduled_retry_before_initial_receipt_reconciles_duplicate_result():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.READABLE,
                "AAA1111",
                0.99,
                "primary",
            )
        ],
    }
    sink = RecordingSink(delivered=False)
    coord, _ = coordinator(evidence, sink=sink)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    reached_receipt = threading.Event()
    release_receipt = threading.Event()
    original_complete = coord._complete_request

    def blocked_complete(*args):
        reached_receipt.set()
        assert release_receipt.wait(timeout=2)
        original_complete(*args)

    coord._complete_request = blocked_complete
    errors = []

    def first_crossing():
        try:
            coord.ingest_crossing(crossing("c1"), [b"c"])
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=first_crossing)
    worker.start()
    assert reached_receipt.wait(timeout=2)
    assert coord.state_summary()["pending_callback_count"] == 1

    sink.should_deliver = True
    assert list(coord.retry_pending_callbacks().values()) == [True]
    release_receipt.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], EntryUnavailable)
    duplicate = coord.ingest_crossing(crossing("c1"), [b"c"])
    assert duplicate.duplicate is True
    assert duplicate.callback_delivered is True
    assert coord.state_summary()["pending_callback_count"] == 0
    assert coord.state_summary()["delivery_reconciliation_count"] == 0


def test_idempotent_duplicate_and_conflicting_reuse():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    original = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"same"])
    duplicate = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"same"])

    assert original.accepted is True
    assert duplicate.duplicate is True
    with pytest.raises(EntryConflict):
        coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"different"])


def test_receive_time_fallback_retry_keeps_same_idempotency_fingerprint():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    first = replace(
        attempt("a1", "AAA-1111"),
        metadata={"timestamp_source": "pms_receive_missing"},
    )
    retry = replace(first, captured_at=first.captured_at + timedelta(seconds=3))

    coord.ingest_attempt(first, [b"same-camera-payload"])
    duplicate = coord.ingest_attempt(retry, [b"same-camera-payload"])

    assert duplicate.duplicate is True


def test_camera_source_timestamp_change_remains_an_id_reuse_conflict():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    first = attempt("a1", "AAA-1111")
    changed = replace(first, captured_at=first.captured_at + timedelta(seconds=3))

    coord.ingest_attempt(first, [b"same-camera-payload"])
    with pytest.raises(EntryConflict, match="id_reused_with_different_payload"):
        coord.ingest_attempt(changed, [b"same-camera-payload"])


def test_live_attempt_receipt_is_pinned_across_lru_churn():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR", (0.0, 1.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(
        evidence,
        cfg=settings(receipt_capacity=1),
    )
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])
    coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"a2"])
    calls_before_retry = len(coord._processor.calls)

    duplicate = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a1"])

    assert duplicate.duplicate is True
    assert len(coord._processor.calls) == calls_before_retry
    with pytest.raises(EntryConflict, match="id_reused_with_different_payload"):
        coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"changed"])
    assert coord.state_summary()["attempt_count"] == 2


def test_live_crossing_receipt_is_pinned_across_lru_churn():
    evidence = {
        "c1": [
            frame(
                "c1",
                "CAM-23",
                (1.0, 0.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
        "c2": [
            frame(
                "c2",
                "CAM-23",
                (0.0, 1.0),
                PlateReadState.NO_PLATE,
                role="primary",
            )
        ],
    }
    coord, _ = coordinator(
        evidence,
        cfg=settings(receipt_capacity=1),
    )
    coord.ingest_crossing(crossing("c1"), [b"c1"])
    coord.ingest_crossing(crossing("c2"), [b"c2"])
    calls_before_retry = len(coord._processor.calls)

    duplicate = coord.ingest_crossing(crossing("c1"), [b"c1"])

    assert duplicate.duplicate is True
    assert len(coord._processor.calls) == calls_before_retry
    with pytest.raises(EntryConflict, match="id_reused_with_different_payload"):
        coord.ingest_crossing(crossing("c1"), [b"changed"])
    assert coord.state_summary()["crossing_count"] == 2


def test_shadow_failed_callback_is_bounded_and_manual_retry_compacts_on_success():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
    }
    sink = RecordingSink(delivered=False)
    coord, _ = coordinator(evidence, sink=sink, cfg=settings(mode=EntryMode.SHADOW))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.callback_delivered is False
    assert coord.state_summary()["pending_callback_count"] == 1
    assert coord.state_summary()["attempt_count"] == 1
    sink.should_deliver = True
    assert list(coord.retry_pending_callbacks().values()) == [True]
    assert coord.state_summary()["attempt_count"] == 0
    assert coord.state_summary()["pending_callback_count"] == 0


def test_shadow_mode_never_sends_confirmed_status_to_pms():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "AAA1111", 0.99, "primary")],
    }
    coord, sink = coordinator(evidence, cfg=settings(mode=EntryMode.SHADOW))
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[0]["status"] == "abstained"
    assert sink.payloads[0]["reason"].startswith("shadow_would_confirm:")


def test_coordinator_state_never_retains_request_image_bytes():
    evidence = {
        "a1": [frame("a1", "ANPR", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    raw = b"unique-raw-camera-image"
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [raw])

    def contains_raw(value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return raw in bytes(value)
        if isinstance(value, dict):
            return any(contains_raw(k) or contains_raw(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            return any(contains_raw(item) for item in value)
        if hasattr(value, "__dict__"):
            return contains_raw(vars(value))
        return False

    assert not contains_raw(coord._attempts)
    assert not contains_raw(coord._groups)
    assert not contains_raw(coord._crossings)
    assert not contains_raw(coord._receipts)
