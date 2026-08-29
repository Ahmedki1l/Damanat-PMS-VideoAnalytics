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


def test_a_later_differently_read_anpr_cannot_rewrite_a_confirmed_entry():
    """Once an entry is confirmed under a plate, a later ANPR read of a
    DIFFERENT plate is a new identity, not a correction to the old one.

    (This replaces a test built on the old behaviour, where a ramp camera's OCR
    conflict held the first crossing pending long enough for a late ANPR read
    to rewrite it. The ramp cameras no longer read plates, so the first
    crossing confirms straight away and there is nothing left pending.)
    """
    evidence = {
        "a-first": [frame("a-first", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
        "a-late": [frame("a-late", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a-first", "ABC-1234"), [b"first"])
    first = coord.ingest_crossing(crossing("c1"), [b"crossing"])

    assert first.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"

    coord.ingest_attempt(attempt("a-late", "XYZ-9999"), [b"late"])

    # The confirmed entry keeps its plate; the late read is its own identity.
    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    assert [p["canonical_plate"] for p in confirmed] == ["ABC-1234"]


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
    [(0.75, "abstained"), (0.749, "confirmed")],
)
def test_ocr_confidence_boundary(ocr_confidence, expected):
    """ocr_min_confidence decides whether OUR reader produced a reading at all.

    The boundary is only observable when the read DISAGREES with ANPR. At or
    above the bar our OCR is a source and contradicts the gate, so there is no
    consensus and nothing opens. Below it our OCR has not produced a reading,
    so ANPR stands alone and the entry confirms.

    (The read is placed on the ANPR image. CAM-23 is not a plate source, so a
    reading there could not move this boundary either way.)
    """
    evidence = {
        "a1": [
            frame(
                "a1", "ANPR", (1.0, 0.0), PlateReadState.READABLE,
                "BBB2222", ocr_confidence,
            )
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
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
    assert sink.payloads[-1]["reason"] == "plate_implausible"


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


def test_two_plate_readings_of_one_car_stay_separate_identities_and_are_marked():
    """POLICY CHANGE (Entry Pipeline v3, stage 2).

    This used to assert that two ANPR reads of the SAME car under DIFFERENT
    plates were merged by Re-ID into one group, and that the later reading
    superseded the earlier one.

    Identity is now keyed by PLATE. Two plate keys are two identities, and
    appearance is not allowed to collapse them: letting Re-ID overrule the plate
    key would put appearance back in charge of who a car is, which is exactly
    what this rewrite removes. The disagreement is recorded rather than
    resolved here — `correction_candidate_of` marks it, and the plate consensus
    over ANPR / HikCentral / our own OCR is what decides which reading is right.

    Note also that CAM-23's OCR (XYZ9999 above) can no longer break the tie: the
    ramp cameras are visual observation sources and read no plates.
    """
    evidence = {
        "a-wrong": [frame("a-wrong", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a-correct": [frame("a-correct", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.96)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE, "XYZ9999", 0.98, "primary")],
    }
    coord, sink = coordinator(evidence)
    first = coord.ingest_attempt(attempt("a-wrong", "ABC-1234"), [b"a1"])
    second = coord.ingest_attempt(attempt("a-correct", "XYZ-9999"), [b"a2"])

    assert first.group_id != second.group_id
    state = coord.state_summary()
    assert state["group_count"] == 2

    groups = state["groups"]
    assert groups[first.group_id]["identity_key"] == "ABC1234"
    assert groups[second.group_id]["identity_key"] == "XYZ9999"
    # Re-ID says it is the same car, so the second identity carries the marker.
    assert groups[second.group_id]["correction_candidate_of"] == first.group_id
    # ...but the marker never merges them.
    assert groups[first.group_id]["correction_candidate_of"] == ""

    # Both identities hold the ANPR witness and the ANPR plate source.
    for group_id in (first.group_id, second.group_id):
        assert groups[group_id]["witnesses"] == ["anpr"]
    assert groups[first.group_id]["plate_sources"]["anpr"] == "ABC-1234"
    assert groups[second.group_id]["plate_sources"]["anpr"] == "XYZ-9999"
    # The second identity's ANPR image also gave OUR reader a look at it, and
    # it agrees - two sources, one answer.
    assert groups[second.group_id]["plate_sources"]["our_ocr"] == "XYZ9999"

    # Two identities of identical appearance leave the crossing unable to pick
    # one, so nothing is confirmed on a guess.
    coord.ingest_crossing(crossing("c1"), [b"c"])
    assert not any(p["status"] == "confirmed" for p in sink.payloads)


def test_a_second_read_of_the_same_plate_and_car_enriches_one_identity():
    """The other half of the rule: same key AND same car is ONE identity."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR-ENTRY", (0.99, 0.01), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a1"])
    second = coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"a2"])

    assert first.group_id == second.group_id
    state = coord.state_summary()
    assert state["group_count"] == 1
    assert state["groups"][first.group_id]["attempt_ids"] == ["a1", "a2"]


def test_same_plate_but_a_different_car_gets_its_own_identity():
    """The appearance guard.

    A key match is not a licence to pool images. If ANPR reads car B's plate as
    car A's, pooling B's images into identity A poisons it: every later crossing
    then matches A perfectly, no crossing can out-margin another on the column
    gate, and nothing ever confirms. Re-ID vetoes the pooling — it does not
    decide the identity, which the plate already did.
    """
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0), PlateReadState.NO_PLATE)],
    }
    coord, _ = coordinator(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a1"])
    second = coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"a2"])

    assert first.group_id != second.group_id
    state = coord.state_summary()
    assert state["group_count"] == 2
    # Both keep the plate key; they are two candidates for one plate, not one
    # candidate with two appearances.
    for group_id in (first.group_id, second.group_id):
        assert state["groups"][group_id]["identity_key"] == "ABC1234"


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


# =========================================================================== #
# Plate consensus (Entry Pipeline v3, stage 4)
#
# POLICY CHANGE. Twelve tests were removed from this file with this change.
# They exercised the CAM-23-versus-CAM-03 OCR arbitration in detail: cached
# ANPR OCR admitted only when the primary had no plate, a readable primary read
# blocking the fallback, low-confidence primary noise unable to veto a reliable
# primary read, arrival order between two reliable primary reads, and so on.
#
# All of it answered one question - "which ramp camera's plate reading wins?" -
# and that question no longer exists. CAM-23 and CAM-03 are visual observation
# sources for Re-ID and read no plates at all. There are exactly three plate
# sources: the gate ANPR system, HikCentral, and our own OCR on an available
# vehicle image. The plate is whatever those agree on.
#
# The tests below assert the rule that replaced them.
# =========================================================================== #


def test_one_available_source_confirms_when_two_witnesses_agree():
    """Consensus is "at least two AGREEING", not "two of exactly three".

    With one source there is nothing to contradict it, and by this point two
    independent observations have already agreed on the physical vehicle.
    HikCentral being unreachable or an image being unusable must not stop
    ordinary traffic; what is refused is a contradiction, not a thin record.
    """
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"
    assert sink.payloads[0]["reason"] == "reid_and_plate_consensus"
    assert sink.payloads[0]["ocr_source"] == "consensus"


def test_two_agreeing_sources_confirm():
    """ANPR reported it and our own OCR of the ANPR image read the same."""
    evidence = {
        "a1": [
            frame(
                "a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "ABC1234", 0.96,
            )
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"


def test_two_sources_that_disagree_have_no_consensus_and_open_nothing():
    """The whole point of the rule. A wrong plate opens a session under another
    person's name, so a contradiction refuses rather than picks."""
    evidence = {
        "a1": [
            frame(
                "a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "XYZ9999", 0.96,
            )
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[0]["reason"] == "plate_no_consensus"
    assert sink.payloads[0]["canonical_plate"] is None


def test_an_implausible_read_is_not_a_source_and_cannot_disagree():
    """A digit-first reading of a letter-first plate is not a rival answer, it
    is an implausible one, and it is filtered before consensus ever sees it.

    (That digit-first and letter-first are DIFFERENT identities is a separate
    invariant, covered in test_entry_v2_domain.py and by the identity tests.)"""
    evidence = {
        "a1": [
            frame(
                "a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "1234ABC", 0.96,
            )
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"


def test_our_ocr_reading_two_images_is_still_one_source():
    """Counting our reader twice would let it reach consensus with itself and
    manufacture a two-source agreement out of one opinion. When it contradicts
    itself the source is dropped entirely, leaving ANPR alone to stand."""
    evidence = {
        "a1": [
            frame(
                "a1:0", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "ABC1234", 0.96,
            ),
            frame(
                "a1:1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "XYZ9999", 0.95,
            ),
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    # our_ocr contradicted itself and is excluded; ANPR is the only source left.
    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"


def test_a_ramp_camera_reading_can_withhold_an_entry_but_never_name_one():
    """The exact limit of what a ramp camera may do to the plate.

    CAM-23 reads XYZ9999 while the sources agree on ABC-1234. It does NOT get
    to make the plate XYZ9999 — it is not a plate source and never names a car.
    What it does is withhold: a reliable read that contradicts the consensus is
    evidence Re-ID matched the wrong identity, so the entry is refused rather
    than opened under a plate one of the two cameras disagrees with.

    Subtractive, like the colour veto. Disable with
    ENTRY_V2_OBSERVATION_PLATE_VETO_ENABLED if the shadow window shows ramp
    reads are too unreliable to withhold on.
    """
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame(
                "c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                "XYZ9999", 0.99, "primary",
            )
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[0]["reason"] == "observation_plate_contradiction"
    # It withheld the entry; it did not rename the car.
    assert sink.payloads[0]["canonical_plate"] is None

    # With the veto off, the ramp read is inert and ANPR stands alone.
    coord2, sink2 = coordinator(
        evidence, cfg=settings(observation_plate_veto_enabled=False)
    )
    coord2.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    assert (
        coord2.ingest_crossing(crossing("c1"), [b"c"]).decision_status == "confirmed"
    )
    assert sink2.payloads[0]["canonical_plate"] == "ABC-1234"


def test_a_low_confidence_read_is_not_a_source_at_all():
    """Below ocr_min_confidence our OCR has not produced a reading, so there is
    nothing to disagree with - not a disagreement to resolve."""
    evidence = {
        "a1": [
            frame(
                "a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.READABLE,
                "XYZ9999", 0.10,
            )
        ],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "confirmed"
    assert sink.payloads[0]["canonical_plate"] == "ABC-1234"


def test_an_implausible_consensus_plate_is_still_refused():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), PlateReadState.NO_PLATE)],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.NO_PLATE, role="primary")
        ],
    }
    coord, sink = coordinator(evidence)
    coord.ingest_attempt(attempt("a1", "UNKNOWN"), [b"a"])
    result = coord.ingest_crossing(crossing("c1"), [b"c"])

    assert result.decision_status == "abstained"
    assert sink.payloads[0]["reason"] == "plate_implausible"
