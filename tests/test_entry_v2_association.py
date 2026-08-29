"""Stage 3 — Re-ID association: colour veto, witnesses, and the two-witness rule.

Three rules are locked down here:

  * Colour is SUBTRACTIVE. It removes a candidate that cannot be this car and
    the margin is recomputed over the survivors. It never adds score, because
    two white sedans agreeing on colour is not evidence they are one car.
  * Column competition is PER CAMERA. CAM-23 and CAM-03 seeing the same car are
    two independent witnesses, not rivals for one identity.
  * An entry needs TWO independent observations of the same physical vehicle,
    at least one of which cleared its gates alone. Two uncertain looks never
    add up to a certainty.

And FIFO is measured against Re-ID and written to the log, where nothing reads
it back: a car can stop on the ramp, so arrival order is a prior and never the
matcher.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.decision import EntryDecisionEngine
from src.entry.domain import (
    AttemptInput,
    CrossingInput,
    CrossingRole,
    EntryMode,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
    WitnessSource,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)

# Mean-HSV signatures. body_colour_compatible treats low saturation OR low
# value as achromatic and then separates on brightness only.
WHITE = (0.0, 5.0, 240.0)
BLACK = (0.0, 5.0, 20.0)
RED = (0.0, 200.0, 200.0)
BLUE = (110.0, 200.0, 200.0)


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
          confidence=0.0, role="anpr", colour=None):
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
        colour_hsv=colour,
    )


def attempt(attempt_id, plate, captured_at=NOW):
    return AttemptInput(
        attempt_id=attempt_id,
        source_event_id=f"src-{attempt_id}",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=plate,
        reported_confidence=0.95,
        metadata={},
    )


def crossing(crossing_id, captured_at=None, camera_id="CAM-23",
             role=CrossingRole.PRIMARY):
    primary = role is CrossingRole.PRIMARY
    return CrossingInput(
        crossing_id=crossing_id,
        source_event_id=f"src-{crossing_id}",
        camera_id=camera_id,
        captured_at=captured_at or (NOW + timedelta(seconds=30)),
        line_id="RAMP-IN" if primary else "B-IN",
        direction="ramp-entry" if primary else "b-entry",
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


def build(evidence, cfg=None, log=None):
    sink = _Sink()
    coord = EntryCoordinator(
        cfg or settings(), _Processor(evidence), sink, decision_log=log
    )
    return coord, sink


def statuses(sink):
    return [p["status"] for p in sink.payloads]


# --------------------------------------------------------------------------- #
# Colour veto
# --------------------------------------------------------------------------- #
def test_colour_removes_an_impostor_and_the_survivor_margin_clears():
    """The tie-break, done subtractively.

    Two identities are almost indistinguishable to Re-ID (0.71 vs 0.70), so the
    row margin fails and nothing can be decided. Colour rules one of them out
    entirely; the margin is then recomputed over what is left and the true
    match clears on its own merit — never because colour 'agreed'.
    """
    evidence = {
        "a-white": [frame("a-white", "ANPR-ENTRY", (1.0, 0.0), colour=WHITE)],
        "a-black": [frame("a-black", "ANPR-ENTRY", (0.99, 0.14), colour=BLACK)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary", colour=WHITE)],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    white = coord.ingest_attempt(attempt("a-white", "AAA-1111"), [b"w"])
    coord.ingest_attempt(attempt("a-black", "BBB-2222"), [b"b"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="reid_evaluation")[-1]
    assert record["reid"]["argmax"] == white.group_id
    assert record["colour"]["vetoed"]          # the black identity was removed
    assert record["reid"]["accepted"] is True


def test_without_the_veto_the_same_pair_is_ambiguous():
    """The control for the test above: colour is what changed the outcome."""
    evidence = {
        "a-white": [frame("a-white", "ANPR-ENTRY", (1.0, 0.0), colour=WHITE)],
        "a-black": [frame("a-black", "ANPR-ENTRY", (0.99, 0.14), colour=BLACK)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary", colour=WHITE)],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, cfg=settings(colour_veto_enabled=False), log=log)
    coord.ingest_attempt(attempt("a-white", "AAA-1111"), [b"w"])
    coord.ingest_attempt(attempt("a-black", "BBB-2222"), [b"b"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="reid_evaluation")[-1]
    assert record["reid"]["accepted"] is False
    assert record["result"] == "ambiguous"
    assert sink.payloads == []


def test_colour_never_rescues_a_car_reid_says_is_wrong():
    """Colour agreeing does not add score. A candidate far below the Re-ID bar
    stays below it — this is the 'two white sedans confirmed at 0.30' hole."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), colour=WHITE)],
        "c1": [frame("c1", "CAM-23", (0.30, 0.95), role="primary", colour=WHITE)],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="reid_evaluation")[-1]
    assert record["reid"]["accepted"] is False
    assert record["reason"] == "score_below_minimum"
    assert sink.payloads == []


def test_a_missing_colour_fails_open():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), colour=None)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary", colour=RED)],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="reid_evaluation")[-1]
    assert record["colour"]["vetoed"] == []
    assert record["reid"]["accepted"] is True


def test_a_vivid_colour_mismatch_is_vetoed():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), colour=RED)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary", colour=BLUE)],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, log=log)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    # Every candidate vetoed means no evaluation at all, so nothing is logged
    # as a match and nothing is confirmed.
    assert sink.payloads == []
    assert result.group_id in coord.state_summary()["groups"]


def test_the_veto_is_recorded_even_when_it_changes_nothing():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0), colour=WHITE)],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary", colour=WHITE)],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    colour = log.of(stage="reid_evaluation")[-1]["colour"]
    assert colour["enabled"] is True
    assert colour["query_hsv"] == [0.0, 5.0, 240.0]


# --------------------------------------------------------------------------- #
# The two-witness rule
# --------------------------------------------------------------------------- #
def test_anpr_plus_cam23_confirms():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    coord, sink = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    assert statuses(sink) == ["confirmed"]
    del result


def test_anpr_plus_cam03_confirms_without_cam23():
    """CAM-03 is a peer witness, not a stage that only runs if CAM-23 failed."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-03", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "fallback")
        ],
    }
    coord, sink = build(evidence)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(
        crossing("c1", camera_id="CAM-03", role=CrossingRole.FALLBACK), [b"c"]
    )

    assert statuses(sink) == ["confirmed"]


def test_a_single_witness_never_confirms():
    """An identity with no camera observation is an ANPR read and nothing more.
    A car that was photographed at the gate has not been shown to have entered.
    """
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, sink = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    assert sink.payloads == []
    group = coord.state_summary()["groups"][result.group_id]
    assert group["witnesses"] == ["anpr"]


def test_a_plateless_identity_with_one_camera_does_not_confirm():
    """The dropped-ANPR case before its second witness arrives: one camera saw
    a car, and there is no independent corroboration that it entered."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    coord, sink = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with coord._lock:
        # Strip the ANPR witness to model an identity built from imagery alone.
        # (The attempts endpoint still requires a reported plate, so a genuinely
        # plateless identity only arrives via the recovery path in a later
        # stage; this reproduces the witness state it will have.)
        coord._groups[result.group_id].witnesses.clear()
    coord.ingest_crossing(crossing("c1"), [b"c"])

    assert statuses(sink) == ["abstained"]
    assert sink.payloads[0]["reason"] == "insufficient_witnesses"


def test_hik_does_not_add_a_second_witness_when_anpr_is_present():
    """A Hik pass record is the platform's log of the same gate event the ANPR
    camera already reported. Counting both counts one observation twice."""
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    with coord._lock:
        group = coord._groups[result.group_id]
        group.witnesses[WitnessSource.HIK] = "guid-1"
        assert group.confirming_witnesses() == {WitnessSource.ANPR}
        assert coord._witness_shortfall_locked(group) == "insufficient_witnesses"


def test_hik_substitutes_for_a_missing_anpr():
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    with coord._lock:
        group = coord._groups[result.group_id]
        group.witnesses.clear()
        group.witnesses[WitnessSource.HIK] = "guid-1"
        group.witnesses[WitnessSource.CAM23] = "c1"
        assert group.confirming_witnesses() == {
            WitnessSource.HIK,
            WitnessSource.CAM23,
        }
        assert coord._witness_shortfall_locked(group) == ""


def test_two_weak_votes_never_confirm():
    """Weak votes let an uncertain camera contribute; they cannot substitute
    for certainty. Two uncertain looks are not one confident one."""
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    with coord._lock:
        group = coord._groups[result.group_id]
        group.witnesses.clear()
        group.weak_votes[WitnessSource.CAM23] = "c1"
        group.weak_votes[WitnessSource.CAM03] = "c2"
        assert len(group.confirming_witnesses()) == 2
        assert coord._witness_shortfall_locked(group) == "no_witness_cleared_gates"


def test_one_strong_and_one_weak_witness_is_enough():
    """The CAM-23-ambiguous / CAM-03-confident case from the design."""
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])

    with coord._lock:
        group = coord._groups[result.group_id]
        group.witnesses.clear()
        group.weak_votes[WitnessSource.CAM23] = "c1"     # ambiguous look
        group.witnesses[WitnessSource.CAM03] = "c2"      # cleared its gates
        assert coord._witness_shortfall_locked(group) == ""


def test_a_strong_vote_replaces_an_earlier_weak_one():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with coord._lock:
        coord._groups[result.group_id].weak_votes[WitnessSource.CAM23] = "c1"
    coord.ingest_crossing(crossing("c1"), [b"c"])

    # The group is resolved and removed on confirmation, so assert via the
    # delivered decision rather than the live state.
    assert coord.state_summary()["group_count"] == 0


# --------------------------------------------------------------------------- #
# Per-camera column competition
# --------------------------------------------------------------------------- #
def test_cam23_and_cam03_are_not_column_competitors():
    """Two views of ONE car must not compete for its identity: making them
    compete means the second camera can only take the identity away."""
    engine = EntryDecisionEngine(settings())
    from src.entry.domain import AttemptGroup, AttemptRecord, CrossingRecord

    group = AttemptGroup(group_id="g1", identity_key="AAA1111")
    group.attempts["a1"] = AttemptRecord(
        request=attempt("a1", "AAA-1111"),
        evidence=(frame("a1", "ANPR-ENTRY", (1.0, 0.0)),),
        group_id="g1",
    )
    primary = CrossingRecord(
        request=crossing("c1"),
        evidence=(frame("c1", "CAM-23", (1.0, 0.0), role="primary"),),
    )
    fallback = CrossingRecord(
        request=crossing("c2", camera_id="CAM-03", role=CrossingRole.FALLBACK),
        evidence=(frame("c2", "CAM-03", (1.0, 0.0), role="fallback"),),
    )

    evaluation = engine.evaluate_unique_match(
        primary, {"g1": group}, [primary, fallback]
    )
    # The CAM-03 row is a different camera, so it is not competition.
    assert evaluation.column_runner == 0.0
    assert evaluation.accepted is True


def test_two_observations_from_one_camera_do_compete():
    """The gate the column margin exists for: two different cars cannot both
    claim one identity from the same viewpoint."""
    engine = EntryDecisionEngine(settings())
    from src.entry.domain import AttemptGroup, AttemptRecord, CrossingRecord

    group = AttemptGroup(group_id="g1", identity_key="AAA1111")
    group.attempts["a1"] = AttemptRecord(
        request=attempt("a1", "AAA-1111"),
        evidence=(frame("a1", "ANPR-ENTRY", (1.0, 0.0)),),
        group_id="g1",
    )
    first = CrossingRecord(
        request=crossing("c1"),
        evidence=(frame("c1", "CAM-23", (1.0, 0.0), role="primary"),),
    )
    second = CrossingRecord(
        request=crossing("c2"),
        evidence=(frame("c2", "CAM-23", (1.0, 0.0), role="primary"),),
    )

    evaluation = engine.evaluate_unique_match(first, {"g1": group}, [first, second])
    assert evaluation.column_runner == 1.0
    assert evaluation.accepted is False
    assert evaluation.reason == "column_margin_below_minimum"


# --------------------------------------------------------------------------- #
# FIFO is measured, never enforced
# --------------------------------------------------------------------------- #
def test_reid_is_followed_when_it_disagrees_with_arrival_order():
    """Cars A then B at the gate; CAM-23 sees B first because A waited on the
    ramp. Re-ID wins, and the disagreement is recorded."""
    evidence = {
        "a-first": [frame("a-first", "ANPR-ENTRY", (1.0, 0.0))],
        "a-second": [frame("a-second", "ANPR-ENTRY", (0.0, 1.0))],
        # The first crossing looks like the SECOND car.
        "c1": [
            frame("c1", "CAM-23", (0.0, 1.0), PlateReadState.READABLE,
                  "BBB2222", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, log=log)
    coord.ingest_attempt(attempt("a-first", "AAA-1111"), [b"1"])
    second = coord.ingest_attempt(
        attempt("a-second", "BBB-2222", captured_at=NOW + timedelta(seconds=5)),
        [b"2"],
    )
    coord.ingest_crossing(crossing("c1"), [b"c"])

    assert statuses(sink) == ["confirmed"]
    assert sink.payloads[0]["canonical_plate"] == "BBB-2222"

    record = log.of(stage="reid_evaluation")[-1]
    assert record["reid"]["argmax"] == second.group_id
    # FIFO expected the FIRST identity; Re-ID chose the second, and the
    # disagreement is written down rather than overruled.
    assert record["fifo"]["agreed"] is False
    assert record["fifo"]["reid_group"] == second.group_id


def test_the_fifo_block_records_agreement_when_order_holds():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    fifo = log.of(stage="reid_evaluation")[-1]["fifo"]
    assert fifo["agreed"] is True
    assert fifo["expected_rank"] == 0


# --------------------------------------------------------------------------- #
# The ranked candidate list
# --------------------------------------------------------------------------- #
def test_the_full_ranked_list_is_recorded_for_later_calibration():
    """A record that only says "0.81, accepted" cannot answer what a different
    threshold would have done. The sweep needs the runners-up."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    first = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    second = coord.ingest_attempt(attempt("a2", "BBB-2222"), [b"b"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    ranked = log.of(stage="reid_evaluation")[-1]["ranked"]
    assert [row[0] for row in ranked] == [first.group_id, second.group_id]
    assert ranked[0][1] > ranked[1][1]


def test_the_ramp_cameras_ocr_is_logged_but_is_not_a_plate_source():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="reid_evaluation")[-1]
    assert record["observed_plate_text"] == "AAA1111"
    # Diagnostic only: it never becomes a plate source on the identity.
    identity_records = log.of(stage="anpr_identity")
    assert identity_records[0]["identity"]["identity_key"] == "AAA1111"
    assert "plate" not in record


def test_the_confirmation_decision_is_recorded_with_its_witness_set():
    """S5 gets its own record: "did two independent observations agree on one
    physical vehicle?" is the whole basis for opening a session."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="physical_confirm")[-1]
    assert record["result"] == "confirmed"
    assert record["reason"] == "witnesses_agree"
    assert record["witnesses"] == ["anpr", "cam23"]
    assert record["identity"]["strong"] == ["anpr", "cam23"]
    assert record["identity"]["weak"] == []
    assert statuses(sink) == ["confirmed"]


def test_a_shortfall_is_recorded_with_the_reason_it_failed():
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [
            frame("c1", "CAM-23", (1.0, 0.0), PlateReadState.READABLE,
                  "AAA1111", 0.99, "primary")
        ],
    }
    log = _CollectingLog()
    coord, sink = build(evidence, log=log)
    result = coord.ingest_attempt(attempt("a1", "AAA-1111"), [b"a"])
    with coord._lock:
        coord._groups[result.group_id].witnesses.clear()
    coord.ingest_crossing(crossing("c1"), [b"c"])

    record = log.of(stage="physical_confirm")[-1]
    assert record["result"] == "abstained"
    assert record["reason"] == "insufficient_witnesses"
    assert statuses(sink) == ["abstained"]
