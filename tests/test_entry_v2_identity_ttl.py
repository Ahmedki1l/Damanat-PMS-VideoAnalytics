"""Stage 2 — plate-keyed identity, the appearance guard, and the TTLs.

Two rules are locked down here, and they pull against each other on purpose:

  * The PLATE is the identity key. A second ANPR read of the same plate
    enriches the live candidate; it does not create a rival for it.
  * A key match is not a licence to pool images. If appearance says the two
    reads are different cars, they stay separate identities under one key.

And the TTLs, which are a LIFETIME bound and never an identity rule: expiry
decides that a candidate has waited long enough to stop being one, never that
two observations belong to the same car.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
    PlateSourceKind,
    WitnessSource,
    witness_for_camera,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


class MovableClock:
    """Wall clock under test control. TTL behaviour must not need sleep()."""

    def __init__(self, start=NOW):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + timedelta(**kwargs)


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
        identity_ttl_minutes=15,
        observation_ttl_minutes=60,
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


def frame(event_id, camera, vector, state=PlateReadState.NO_PLATE, text="",
          confidence=0.0, role="anpr"):
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


def attempt(attempt_id, plate, captured_at=NOW, confidence=0.95):
    return AttemptInput(
        attempt_id=attempt_id,
        source_event_id=f"src-{attempt_id}",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=plate,
        reported_confidence=confidence,
        metadata={},
    )


def crossing(crossing_id, captured_at=NOW + timedelta(seconds=30),
             camera_id="CAM-23", role=CrossingRole.PRIMARY,
             line_id="RAMP-IN", direction="ramp-entry"):
    return CrossingInput(
        crossing_id=crossing_id,
        source_event_id=f"src-{crossing_id}",
        camera_id=camera_id,
        captured_at=captured_at,
        line_id=line_id,
        direction=direction,
        role=role,
        metadata={},
    )


class _Processor:
    def __init__(self, evidence_by_event):
        self.evidence_by_event = evidence_by_event

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        return tuple(self.evidence_by_event[event_id])


class _Sink:
    def __init__(self):
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(True, 1, "", publish_identity=False)


class _CollectingLog:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def of(self, stage=None, result=None):
        return [
            r
            for r in self.records
            if (stage is None or r.get("stage") == stage)
            and (result is None or r.get("result") == result)
        ]


def build(evidence, cfg=None, clock=None, log=None):
    sink = _Sink()
    coord = EntryCoordinator(
        cfg or settings(),
        _Processor(evidence),
        sink,
        decision_log=log,
        clock=clock,
    )
    return coord, sink


# --------------------------------------------------------------------------- #
# The witness / plate-source separation
# --------------------------------------------------------------------------- #
class SeparationTests:
    pass


def test_plate_source_kinds_are_exactly_three():
    # No camera-derived member, ever. If one is added, the ramp cameras have
    # become plate sources and the whole separation is gone.
    assert {k.value for k in PlateSourceKind} == {"anpr", "hik_text", "our_ocr"}


def test_witness_sources_are_the_four_things_that_can_see_a_car():
    assert {w.value for w in WitnessSource} == {"anpr", "hik", "cam23", "cam03"}


def test_ramp_cameras_map_to_witnesses():
    assert witness_for_camera("CAM-23") is WitnessSource.CAM23
    assert witness_for_camera("CAM_03") is WitnessSource.CAM03
    assert witness_for_camera("cam23") is WitnessSource.CAM23


def test_a_non_ramp_camera_is_not_silently_bucketed_as_a_witness():
    assert witness_for_camera("ANPR-ENTRY") is None
    assert witness_for_camera("CAM-99") is None
    assert witness_for_camera("") is None


def test_anpr_records_a_witness_and_a_plate_source_which_are_different_axes():
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    group = coord.state_summary()["groups"][result.group_id]
    assert group["witnesses"] == ["anpr"]            # the image saw a car
    assert group["plate_sources"] == {"anpr": "ABC-1234"}   # the system read a plate


# --------------------------------------------------------------------------- #
# Plate-keyed find-or-create
# --------------------------------------------------------------------------- #
def test_the_same_plate_and_car_is_one_identity():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.98, 0.02))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"b"])

    assert first.group_id == second.group_id
    assert coord.state_summary()["group_count"] == 1


def test_plate_formatting_does_not_create_a_second_identity():
    # plate_key is lossless punctuation-stripping, so these are one plate.
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.99, 0.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "abc 1234"), [b"b"])

    assert first.group_id == second.group_id


def test_digit_first_and_letter_first_readings_stay_different_identities():
    # plate_key never reorders; these are two different plates by design.
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (1.0, 0.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "1234-ABC"), [b"b"])

    assert first.group_id != second.group_id


def test_different_plates_are_different_identities():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])

    assert first.group_id != second.group_id
    assert coord.state_summary()["group_count"] == 2


# --------------------------------------------------------------------------- #
# The appearance guard
# --------------------------------------------------------------------------- #
def test_same_plate_but_a_visually_different_car_is_not_pooled():
    """The poisoning guard. Pooling two cars under one key deadlocks the gate:
    every crossing then matches the identity perfectly, none can out-margin
    another on the column gate, and nothing ever confirms."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"b"])

    assert first.group_id != second.group_id
    groups = coord.state_summary()["groups"]
    # Both still carry the plate: two candidates for one plate, not one
    # candidate with two faces.
    assert groups[first.group_id]["identity_key"] == "ABC1234"
    assert groups[second.group_id]["identity_key"] == "ABC1234"


def test_the_split_is_reported_in_the_decision_log():
    log = _CollectingLog()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"b"])

    records = log.of(stage="anpr_identity")
    assert records[0]["identity"]["same_key_split"] is False
    assert records[1]["identity"]["same_key_split"] is True


def test_a_borderline_appearance_still_enriches_rather_than_splitting():
    # The guard is a veto on gross mismatch, not a demand for a perfect match.
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.95, 0.31))],   # ~0.95 similarity
    }
    coord, _ = build(evidence, cfg=settings(merge_min_score=0.90))
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"b"])

    assert first.group_id == second.group_id


# --------------------------------------------------------------------------- #
# The correction-candidate marker
# --------------------------------------------------------------------------- #
def test_a_misread_plate_on_the_same_car_is_marked_but_never_merged():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (1.0, 0.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])

    groups = coord.state_summary()["groups"]
    assert groups[second.group_id]["correction_candidate_of"] == first.group_id
    # Marked, not merged: appearance never overrules the plate key.
    assert first.group_id != second.group_id
    assert coord.state_summary()["group_count"] == 2


def test_a_visually_unrelated_second_plate_carries_no_marker():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])

    groups = coord.state_summary()["groups"]
    assert groups[second.group_id]["correction_candidate_of"] == ""
    assert groups[first.group_id]["correction_candidate_of"] == ""


# --------------------------------------------------------------------------- #
# Identity TTL
# --------------------------------------------------------------------------- #
def test_an_unconfirmed_identity_expires_after_the_ttl():
    clock = MovableClock()
    log = _CollectingLog()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence, clock=clock, log=log)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    assert coord.state_summary()["group_count"] == 1

    clock.advance(minutes=16)
    # Any later event drives the sweep; there is no timer thread.
    coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])

    state = coord.state_summary()
    assert state["group_count"] == 1
    assert {g["identity_key"] for g in state["groups"].values()} == {"XYZ9999"}

    expired = log.of(stage="ttl_expiry", result="expired")
    assert len(expired) == 1
    assert expired[0]["reason"] == "identity_ttl_expired"
    assert expired[0]["identity"]["identity_key"] == "ABC1234"


def test_an_identity_survives_right_up_to_the_boundary():
    clock = MovableClock()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence, clock=clock)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    clock.advance(minutes=14, seconds=59)
    coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])

    assert coord.state_summary()["group_count"] == 2


def test_a_later_event_cannot_associate_with_an_expired_identity():
    """The point of the TTL: after expiry the old candidate is gone, so a late
    camera event cannot be bound to it."""
    clock = MovableClock()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
    }
    coord, sink = build(evidence, clock=clock)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    clock.advance(minutes=20)
    coord.ingest_crossing(
        crossing("c1", captured_at=NOW + timedelta(minutes=20)), [b"c"]
    )

    assert coord.state_summary()["group_count"] == 0
    assert sink.payloads == []


def test_enrichment_restarts_the_identity_ttl():
    """An ANPR event CREATES OR ACTIVATES a candidate, so a second read of the
    same plate restarts its 15 minutes."""
    clock = MovableClock()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.99, 0.0))],
        "a3": [frame("a3", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence, clock=clock)
    first = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    clock.advance(minutes=10)
    coord.ingest_attempt(attempt("a2", "ABC-1234"), [b"b"])   # re-activates

    clock.advance(minutes=10)   # 20 min after creation, 10 after activation
    coord.ingest_attempt(attempt("a3", "XYZ-9999"), [b"c"])

    state = coord.state_summary()
    assert first.group_id in state["groups"]


def test_an_identity_with_a_callback_in_flight_is_not_expired():
    """Expiring an identity mid-delivery would strand the callback about to
    reference it."""
    clock = MovableClock()
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence, clock=clock)
    result = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    with coord._lock:
        group = coord._groups[result.group_id]
        assert coord._group_has_callback_in_flight_locked(result.group_id) is False
        # Reserve a callback for this identity. Only .group_id is read.
        coord._callback_reservations["d-1"] = SimpleNamespace(
            group_id=result.group_id
        )
        clock.advance(minutes=30)
        coord._expire_stale_locked()
        assert result.group_id in coord._groups


# --------------------------------------------------------------------------- #
# Observation TTL
# --------------------------------------------------------------------------- #
def test_an_unmatched_observation_expires_on_the_longer_ttl():
    clock = MovableClock()
    log = _CollectingLog()
    evidence = {
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
        "a1": [frame("a1", "ANPR-ENTRY", (0.0, 1.0))],
    }
    coord, _ = build(evidence, clock=clock, log=log)
    coord.ingest_crossing(crossing("c1"), [b"c"])
    assert coord.state_summary()["crossing_count"] == 1

    clock.advance(minutes=61)
    coord.ingest_attempt(
        attempt("a1", "XYZ-9999", captured_at=NOW + timedelta(minutes=61)), [b"a"]
    )

    assert coord.state_summary()["crossing_count"] == 0
    expired = log.of(stage="ttl_expiry", result="expired")
    reasons = [r["reason"] for r in expired]
    assert "observation_ttl_expired" in reasons
    record = next(r for r in expired if r["reason"] == "observation_ttl_expired")
    assert record["observation"]["witness"] == "cam23"


def test_an_observation_outlives_the_identity_so_a_late_sweep_can_rescue_it():
    """Why the two TTLs differ at all: an observation that died before its
    identity could never be recovered by a late HikCentral sweep."""
    clock = MovableClock()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (0.0, 1.0), role="primary")],
        "a2": [frame("a2", "ANPR-ENTRY", (0.5, 0.5))],
    }
    coord, _ = build(evidence, clock=clock)
    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    clock.advance(minutes=30)   # past the identity TTL, inside the observation one
    coord.ingest_attempt(
        attempt("a2", "QQQ-1111", captured_at=NOW + timedelta(minutes=30)), [b"b"]
    )

    state = coord.state_summary()
    assert state["crossing_count"] == 1          # observation still alive
    assert {g["identity_key"] for g in state["groups"].values()} == {"QQQ1111"}


def test_expiry_never_touches_a_quarantined_observation():
    # Provisional crossings have their own lifecycle; the TTL sweep must not
    # race it.
    clock = MovableClock()
    evidence = {"c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")]}
    coord, _ = build(evidence, clock=clock)
    coord.ingest_crossing(crossing("c1"), [b"c"])
    with coord._lock:
        record = coord._crossings.pop("c1")
        coord._provisional_crossings["c1"] = (record, None)
        clock.advance(minutes=600)
        coord._expire_stale_locked()
        assert "c1" in coord._provisional_crossings


# --------------------------------------------------------------------------- #
# Fail-closed behaviour
# --------------------------------------------------------------------------- #
def test_an_unstampable_record_is_never_expired():
    """Mixed aware/naive timestamps fail CLOSED. Never delete state you cannot
    reason about — the capacity bounds are the backstop."""
    clock = MovableClock()
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence, clock=clock)
    result = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    with coord._lock:
        # A naive anchor against an aware clock cannot be subtracted.
        coord._groups[result.group_id].last_activity_at = datetime(2020, 1, 1)
        coord._groups[result.group_id].created_at = datetime(2020, 1, 1)
        clock.advance(minutes=600)
        coord._expire_stale_locked()
        assert result.group_id in coord._groups


def test_age_helper_reports_none_rather_than_raising_on_mixed_timezones():
    coord, _ = build({})
    aware = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 8, 29, 12, 0)
    assert coord._age_seconds(naive, aware) is None
    assert coord._age_seconds(None, aware) is None
    assert coord._age_seconds(aware, aware + timedelta(seconds=90)) == 90.0


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def test_ttl_defaults_are_fifteen_and_sixty_minutes():
    base = EntrySettings()
    assert base.identity_ttl_minutes == 15
    assert base.observation_ttl_minutes == 60


def test_an_observation_ttl_below_the_identity_ttl_is_a_configuration_error():
    errors = settings(
        identity_ttl_minutes=30, observation_ttl_minutes=10
    ).configuration_errors()
    assert "observation_ttl_below_identity_ttl" in errors


def test_non_positive_ttls_are_configuration_errors():
    assert "ENTRY_IDENTITY_TTL_MINUTES" in settings(
        identity_ttl_minutes=0
    ).configuration_errors()
    assert "ENTRY_OBSERVATION_TTL_MINUTES" in settings(
        observation_ttl_minutes=0
    ).configuration_errors()
