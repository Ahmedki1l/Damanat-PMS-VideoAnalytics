"""The three defects the 2026-08-30..09-06 shadow corpus exposed.

Every scenario here is a real one, and the numbers are the recorded ones:

  * `_causal_group_projection` rebuilt the identity without `plate_sources`, so
    HikCentral's plate reading was discarded before consensus ran. All 109
    confirmations in the corpus read `available: ["anpr"]` — the two-of-three
    rule never executed once, and every stay would have been opened on a single
    unchecked gate read.

  * Re-ID never opened the durable gallery. A ramp crossing was scored against
    two frontal gate crops of the visit in progress while twenty curated CAM-23
    and CAM-03 views of the same car sat on disk. On 2026-09-06 all 27 confirmed
    cars already had a gallery; 23 of them a full one.

  * A duplicate gate read left a second identity under one plate pending for its
    whole TTL. GGR-9064 was read at 10:09:18 and 10:10:34, the first identity
    took the 10:11:20 crossing, and the leftover then won RGR-6466's 10:24:04
    crossing at 0.636 against RGR-6466's own 0.388 — one car, two witnesses, two
    identities, both confirmed.

The orthogonality tests matter as much as the fixes: a gallery reference must
never make a causally ineligible identity eligible, and a retirement must never
touch a genuine exit-and-re-entry.
"""
import pytest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.decision import EntryDecisionEngine, causal_group_embeddings
from src.entry.domain import (
    AttemptGroup,
    AttemptInput,
    AttemptRecord,
    CrossingInput,
    CrossingRecord,
    CrossingRole,
    EntryMode,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
    PlateReading,
    PlateSourceKind,
    RecordStatus,
)
from src.entry.gallery import GalleryLookup, NullGalleryReferences
from src.entry.settings import EntrySettings


NOW = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)


def settings(**overrides):
    base = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=8,
        max_pending_crossings=8,
        max_pending_callbacks=8,
        receipt_capacity=32,
        max_images_per_event=3,
        max_image_bytes=1024,
        # The thresholds actually shipped on 2026-09-06.
        reid_min_score=0.20,
        reid_row_margin=0.08,
        reid_column_margin=0.08,
        merge_min_score=0.82,
        merge_margin=0.08,
        ocr_min_confidence=0.75,
        colour_veto_enabled=False,
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


def frame(event_id, camera, vector, role="anpr", text="", confidence=0.0):
    state = PlateReadState.READABLE if text else PlateReadState.NO_PLATE
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
        colour_hsv=None,
    )


def attempt(attempt_id, plate, captured_at=NOW, hik=False):
    return AttemptInput(
        attempt_id=attempt_id,
        source_event_id=f"src-{attempt_id}",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=plate,
        reported_confidence=0.96,
        metadata={"evidence_source": "hikcentral"} if hik else {},
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


class _StubGallery:
    """A fixed set of references, keyed by plate exactly as the store is."""

    def __init__(self, by_plate):
        self.by_plate = by_plate
        self.asked = []

    def lookup(self, plate):
        self.asked.append(plate)
        vectors = self.by_plate.get(plate, ())
        return GalleryLookup(
            vectors=tuple(tuple(v) for v in vectors),
            cameras=tuple("CAM-23" for _ in vectors),
            available=len(vectors),
            resolved_plate=plate if vectors else "",
        )


def build(evidence, cfg=None, log=None, gallery=None):
    sink = _Sink()
    coord = EntryCoordinator(
        cfg or settings(),
        _Processor(evidence),
        sink,
        decision_log=log,
        gallery_references=gallery or NullGalleryReferences(),
    )
    return coord, sink


# --------------------------------------------------------------------------- #
# 1. HikCentral's plate reading survives the causal projection
# --------------------------------------------------------------------------- #
def test_hikcentral_plate_reaches_consensus():
    """Two sources agreeing, which the corpus never once recorded.

    The Hik-sourced attempt carries HikCentral's own reading. Before the fix the
    projection dropped `plate_sources` wholesale, so `available_plate_sources`
    saw an empty dict and the gate ANPR read stood alone in every decision.
    """
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a-hik": [frame("a-hik", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
    }
    log = _CollectingLog()
    coord, sink = _build_committing(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "HGD-2926"), [b"a"])
    coord.ingest_attempt(
        attempt("a-hik", "HGD-2926", captured_at=NOW + timedelta(seconds=5), hik=True),
        [b"h"],
    )
    coord.ingest_crossing(crossing("c1"), [b"c"])

    plate = log.of(stage="plate_consensus")[-1]["plate"]
    assert plate["outcome"] == "consensus"
    assert plate["agreeing"] == ["anpr", "hik_text"], plate
    assert plate["available"] == ["anpr", "hik_text"]
    assert sink.payloads[-1]["status"] == "confirmed"


def test_a_hikcentral_pass_after_the_crossing_cannot_name_it():
    """The causality check the storage site asks for, in so many words.

    A Hik record whose pass time follows the crossing describes a LATER journey.
    Restoring `plate_sources` to the projection must not restore it
    unconditionally, or a future pass would name a past entry.
    """
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "a-hik": [frame("a-hik", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "HGD-2926"), [b"a"])
    # Pass time AFTER the crossing at NOW+30s.
    coord.ingest_attempt(
        attempt("a-hik", "HGD-2926", captured_at=NOW + timedelta(seconds=90), hik=True),
        [b"h"],
    )
    coord.ingest_crossing(crossing("c1"), [b"c"])

    plate = log.of(stage="plate_consensus")[-1]["plate"]
    assert plate["available"] == ["anpr"]
    assert "hik_text" not in plate["agreeing"]


# --------------------------------------------------------------------------- #
# 2. The durable gallery as a Re-ID reference set
# --------------------------------------------------------------------------- #
def test_gallery_reference_lifts_a_crossing_the_gate_crop_alone_would_refuse():
    """The 2026-09-06 shape: a weak cross-view gate crop, a strong ramp history.

    The gate crop scores 0.15 against the ramp crossing — below the 0.20 floor,
    which on the real corpus is the whole `score_below_minimum` population. One
    previous CAM-23 view of the same car settles it.
    """
    gate_only = frame("a1", "ANPR-ENTRY", (0.15, 0.9887))
    ramp = frame("c1", "CAM-23", (1.0, 0.0), role="primary")
    evidence = {"a1": [gate_only], "c1": [ramp]}
    log = _CollectingLog()

    # Without the gallery: refused on the absolute floor.
    coord, _ = build(evidence, log=log)
    coord.ingest_attempt(attempt("a1", "ABR-8000"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])
    refused = log.of(stage="reid_evaluation")[-1]
    assert refused["reid"]["accepted"] is False
    assert refused["reason"] == "score_below_minimum"

    # With one previous ramp view of the same car: confirmed.
    log2 = _CollectingLog()
    coord2, sink2 = build(
        evidence,
        cfg=settings(gallery_match_enabled=True),
        log=log2,
        gallery=_StubGallery({"ABR-8000": [(0.74, 0.6726)]}),
    )
    coord2.ingest_attempt(attempt("a1", "ABR-8000"), [b"a"])
    coord2.ingest_crossing(crossing("c1"), [b"c"])
    accepted = log2.of(stage="reid_evaluation")[-1]
    assert accepted["reid"]["accepted"] is True
    assert accepted["reid"]["score"] == pytest.approx(0.74, abs=1e-3)
    # The counterfactual is recorded, so the shadow review can measure the
    # change rather than take it on trust.
    assert accepted["reid"]["attempt_only_score"] == pytest.approx(0.15, abs=1e-3)
    assert accepted["gallery"]["used"] == 1
    assert sink2.payloads[-1]["status"] == "confirmed"


def test_gallery_is_not_consulted_when_the_feature_is_off():
    gallery = _StubGallery({"ABR-8000": [(1.0, 0.0)]})
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (0.15, 0.9887))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
    }
    log = _CollectingLog()
    coord, _ = build(evidence, log=log, gallery=gallery)  # default: disabled
    coord.ingest_attempt(attempt("a1", "ABR-8000"), [b"a"])
    coord.ingest_crossing(crossing("c1"), [b"c"])

    assert gallery.asked == []
    record = log.of(stage="reid_evaluation")[-1]
    assert record["reid"]["accepted"] is False
    assert record["reid"]["attempt_only_score"] is None


def test_gallery_never_makes_a_causally_ineligible_identity_eligible():
    """A gallery ENRICHES a live identity; it never creates a claim.

    The ANPR read here happens AFTER the crossing, so this identity did not
    exist when the car went past. Its previous-visit references must not be
    allowed to claim that crossing — that would let history outvote causality.
    """
    late = AttemptGroup(
        group_id="grp-late",
        attempts={
            "late": AttemptRecord(
                request=attempt("late", "AAA-1111", captured_at=NOW + timedelta(minutes=5)),
                evidence=(frame("late", "ANPR-ENTRY", (1.0, 0.0)),),
                group_id="grp-late",
            )
        },
        status=RecordStatus.PENDING,
        identity_key="AAA1111",
        gallery_embeddings=((1.0, 0.0),),
    )
    observation = CrossingRecord(
        request=crossing("c1", captured_at=NOW),
        evidence=(frame("c1", "CAM-23", (1.0, 0.0), role="primary"),),
        status=RecordStatus.PENDING,
    )
    engine = EntryDecisionEngine(settings(gallery_match_enabled=True))

    assert causal_group_embeddings(late, observation) == ()
    assert engine.evaluate_unique_match(observation, {"grp-late": late}, [observation]) is None


# --------------------------------------------------------------------------- #
# 3. Retiring the superseded duplicate gate read
# --------------------------------------------------------------------------- #
def _ggr_evidence():
    """Two gate reads of one car, too dissimilar to merge at 0.82."""
    return {
        # Read 1 and read 2 of GGR-9064, 0.60 apart — below merge_min_score,
        # which is what made them two identities on the day.
        "ggr-1": [frame("ggr-1", "ANPR-ENTRY", (1.0, 0.0, 0.0))],
        "ggr-2": [frame("ggr-2", "ANPR-ENTRY", (0.60, 0.80, 0.0))],
        "ggr-crossing": [frame("ggr-crossing", "CAM-23", (1.0, 0.0, 0.0), role="primary")],
        # A different car arriving later. Its own gate crop is weak (0.388 on
        # the day); the leftover GGR identity scored 0.636 on its crossing.
        "rgr-1": [frame("rgr-1", "ANPR-ENTRY", (0.388, 0.9217, 0.0))],
        "rgr-crossing": [
            frame("rgr-crossing", "CAM-23", (0.636, 0.0, 0.7717), role="primary")
        ],
    }


def test_a_superseded_duplicate_gate_read_is_retired():
    """GGR-9064, exactly as it happened, with the retirement in place."""
    log = _CollectingLog()
    coord, sink = build(_ggr_evidence(), log=log)
    coord.ingest_attempt(attempt("ggr-1", "GGR-9064"), [b"1"])
    coord.ingest_attempt(
        attempt("ggr-2", "GGR-9064", captured_at=NOW + timedelta(seconds=76)), [b"2"]
    )
    coord.ingest_crossing(
        crossing("ggr-crossing", captured_at=NOW + timedelta(seconds=122)), [b"c"]
    )

    retirements = log.of(stage="same_key_retirement")
    assert len(retirements) == 1
    assert retirements[0]["identity"]["identity_key"] == "GGR9064"
    assert retirements[0]["reason"] == "superseded_by_confirmed_same_key"
    assert [p["status"] for p in sink.payloads].count("confirmed") == 1


def test_the_retired_duplicate_can_no_longer_claim_another_car():
    """The actual harm: without retirement the leftover wins RGR-6466's crossing."""
    log = _CollectingLog()
    coord, sink = build(_ggr_evidence(), log=log)
    coord.ingest_attempt(attempt("ggr-1", "GGR-9064"), [b"1"])
    coord.ingest_attempt(
        attempt("ggr-2", "GGR-9064", captured_at=NOW + timedelta(seconds=76)), [b"2"]
    )
    coord.ingest_crossing(
        crossing("ggr-crossing", captured_at=NOW + timedelta(seconds=122)), [b"c"]
    )
    coord.ingest_attempt(
        attempt("rgr-1", "RGR-6466", captured_at=NOW + timedelta(minutes=12)), [b"r"]
    )
    coord.ingest_crossing(
        crossing("rgr-crossing", captured_at=NOW + timedelta(minutes=13)), [b"rc"]
    )

    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    plates = sorted(p["canonical_plate"] for p in confirmed)
    assert plates == ["GGR-9064", "RGR-6466"], plates
    # The 09-06 failure was GGR-9064 appearing twice.
    assert plates.count("GGR-9064") == 1


def test_retirement_leaves_a_genuine_re_entry_alone():
    """Two live identities may share a plate — that is what a re-entry IS.

    The second read here happens AFTER the first crossing, so it describes a
    later journey. Source time is the whole test; plate equality is not.
    """
    evidence = {
        "v1": [frame("v1", "ANPR-ENTRY", (1.0, 0.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0, 0.0), role="primary")],
        "v2": [frame("v2", "ANPR-ENTRY", (0.0, 1.0, 0.0))],
        "c2": [frame("c2", "CAM-23", (0.0, 1.0, 0.0), role="primary")],
    }
    log = _CollectingLog()
    coord, sink = _build_committing(evidence, log=log)
    coord.ingest_attempt(attempt("v1", "KKR-6294"), [b"1"])
    coord.ingest_crossing(
        crossing("c1", captured_at=NOW + timedelta(seconds=40)), [b"c1"]
    )
    # The car leaves and returns: the second gate read FOLLOWS the first
    # crossing, so it is a new journey and must survive.
    coord.ingest_attempt(
        attempt("v2", "KKR-6294", captured_at=NOW + timedelta(minutes=6)), [b"2"]
    )
    coord.ingest_crossing(
        crossing("c2", captured_at=NOW + timedelta(minutes=7)), [b"c2"]
    )

    assert log.of(stage="same_key_retirement") == []
    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    assert len(confirmed) == 2
    assert {p["canonical_plate"] for p in confirmed} == {"KKR-6294"}


def test_retirement_can_be_switched_off():
    log = _CollectingLog()
    coord, _ = build(
        _ggr_evidence(), cfg=settings(same_key_retirement_enabled=False), log=log
    )
    coord.ingest_attempt(attempt("ggr-1", "GGR-9064"), [b"1"])
    coord.ingest_attempt(
        attempt("ggr-2", "GGR-9064", captured_at=NOW + timedelta(seconds=76)), [b"2"]
    )
    coord.ingest_crossing(
        crossing("ggr-crossing", captured_at=NOW + timedelta(seconds=122)), [b"c"]
    )
    assert log.of(stage="same_key_retirement") == []


# --------------------------------------------------------------------------- #
# 4. The same retirement, pointed forwards: NJS-7894 on 2026-09-07
# --------------------------------------------------------------------------- #
class _CommittingSink(_Sink):
    """A sink that reports a real session, as PMS-AI does on a confirmation.

    The shared `_Sink` answers `publish_identity=False`, which in AUTHORITATIVE
    mode means "no live session was created" and so no journey tombstone is
    kept. These scenarios turn on that tombstone being there, and in the shipped
    shadow configuration it always is: `_should_remember_journey_locked` returns
    True unconditionally outside AUTHORITATIVE mode.
    """

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(
            True, 1, "", publish_identity=True, session_committed=True
        )


def _build_committing(evidence, cfg=None, log=None):
    sink = _CommittingSink()
    coord = EntryCoordinator(
        cfg or settings(),
        _Processor(evidence),
        sink,
        decision_log=log,
        gallery_references=NullGalleryReferences(),
    )
    return coord, sink


def _njs_evidence():
    """One car, one arrival, and our own imagery coming back at us.

    NJS-7894 crossed CAM-23 at 09:27:41 and confirmed at 09:27:55. PMS-AI then
    forwarded the same car's gate snapshot and the HikCentral record for that
    very pass back to VA, which opened a SECOND identity under the same key at
    09:28:23 — the first was RESOLVED and so no longer a find-or-create
    candidate. The spare CAM-03 view of the one crossing was still pending and
    the new identity took it at 0.821.
    """
    return {
        "njs-1": [frame("njs-1", "ANPR-ENTRY", (1.0, 0.0, 0.0))],
        "njs-c23": [frame("njs-c23", "CAM-23", (1.0, 0.0, 0.0), role="primary")],
        # The spare fallback view of the SAME crossing, two seconds later.
        "njs-c03": [frame("njs-c03", "CAM-03", (0.821, 0.571, 0.0), role="fallback")],
        # Our own confirmed imagery, returned to us. It looks like the car
        # because it IS the car.
        "njs-replay": [frame("njs-replay", "ANPR-ENTRY", (1.0, 0.0, 0.0))],
    }


def _njs_sequence(coord):
    """The recorded order: confirm, spare view, then the replay."""
    coord.ingest_attempt(attempt("njs-1", "NJS-7894"), [b"1"])
    coord.ingest_crossing(
        crossing("njs-c23", captured_at=NOW + timedelta(seconds=30)), [b"c23"]
    )
    coord.ingest_crossing(
        crossing(
            "njs-c03",
            captured_at=NOW + timedelta(seconds=32),
            camera_id="CAM-03",
            role=CrossingRole.FALLBACK,
        ),
        [b"c03"],
    )
    # The replayed evidence. Its SOURCE time is the gate read it depicts, which
    # is before the crossing that already confirmed — that is what marks it as
    # this same arrival rather than a new one.
    coord.ingest_attempt(
        attempt(
            "njs-replay",
            "NJS-7894",
            captured_at=NOW + timedelta(seconds=5),
            hik=True,
        ),
        [b"replay"],
    )


def test_replayed_evidence_cannot_confirm_the_same_arrival_twice():
    """NJS-7894, exactly as it happened, with the forward retirement in place."""
    log = _CollectingLog()
    coord, sink = _build_committing(_njs_evidence(), log=log)
    _njs_sequence(coord)

    retirements = log.of(stage="same_key_retirement")
    assert len(retirements) == 1
    assert retirements[0]["reason"] == "superseded_by_resolved_same_key"
    assert retirements[0]["identity"]["identity_key"] == "NJS7894"

    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    assert len(confirmed) == 1, "one arrival must produce exactly one entry"
    assert confirmed[0]["canonical_plate"] == "NJS-7894"


def test_without_the_forward_retirement_the_duplicate_returns():
    """The regression this guards, stated so a revert is not silent."""
    log = _CollectingLog()
    coord, sink = _build_committing(
        _njs_evidence(),
        cfg=settings(late_same_key_retirement_enabled=False),
        log=log,
    )
    _njs_sequence(coord)

    assert log.of(stage="same_key_retirement") == []
    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    assert len(confirmed) == 2, "the duplicate is exactly what the flag re-enables"


def test_a_genuine_re_entry_after_the_crossing_still_opens_a_second_visit():
    """The orthogonality case: sharing a plate key is not the test.

    A car that leaves and comes back is two visits under one key. What must
    separate it from a replay is only WHEN its gate read happened relative to
    the crossing that closed the earlier identity.
    """
    log = _CollectingLog()
    evidence = dict(_njs_evidence())
    evidence["njs-2"] = [frame("njs-2", "ANPR-ENTRY", (1.0, 0.0, 0.0))]
    evidence["njs-c23b"] = [
        frame("njs-c23b", "CAM-23", (1.0, 0.0, 0.0), role="primary")
    ]
    coord, sink = _build_committing(evidence, log=log)

    coord.ingest_attempt(attempt("njs-1", "NJS-7894"), [b"1"])
    coord.ingest_crossing(
        crossing("njs-c23", captured_at=NOW + timedelta(seconds=30)), [b"c23"]
    )
    # A real second arrival: the gate read FOLLOWS the first crossing.
    coord.ingest_attempt(
        attempt("njs-2", "NJS-7894", captured_at=NOW + timedelta(minutes=6)), [b"2"]
    )
    coord.ingest_crossing(
        crossing("njs-c23b", captured_at=NOW + timedelta(minutes=7)), [b"c23b"]
    )

    assert log.of(stage="same_key_retirement") == []
    confirmed = [p for p in sink.payloads if p["status"] == "confirmed"]
    assert len(confirmed) == 2


# --------------------------------------------------------------------------- #
# 5. A crossing nothing could compete for is stated, not silently dropped
# --------------------------------------------------------------------------- #
def test_a_crossing_with_no_eligible_identity_is_recorded():
    """SHR-1198's silence, which cost a three-service log trawl to explain.

    Its own CAM-23 view was ingested and acknowledged and then produced no
    decision record at all, because `evaluate_unique_match` returns None when
    nothing survives to be ranked and that path wrote nothing. The only trace
    was a `ttl_expiry` an hour later, which says when the observation was
    dropped and never why.
    """
    log = _CollectingLog()
    coord, _ = build(
        {"lone-c23": [frame("lone-c23", "CAM-23", (1.0, 0.0), role="primary")]},
        log=log,
    )
    coord.ingest_crossing(crossing("lone-c23"), [b"c"])

    uncontested = [
        r
        for r in log.of(stage="reid_evaluation")
        if r["reason"] == "no_live_identity"
    ]
    assert len(uncontested) == 1
    assert uncontested[0]["uncontested"]["pending_identities"] == 0


def test_a_crossing_older_than_every_identity_says_so():
    """The SHR-1198 shape: identities exist, but none of them can be this car."""
    log = _CollectingLog()
    coord, _ = build(
        {
            "late-read": [frame("late-read", "ANPR-ENTRY", (1.0, 0.0))],
            "early-c23": [frame("early-c23", "CAM-23", (1.0, 0.0), role="primary")],
        },
        log=log,
    )
    # The gate read is recorded as happening AFTER the crossing, so the identity
    # cannot causally own it — the case the matcher used to drop in silence.
    coord.ingest_attempt(
        attempt("late-read", "SHR-1198", captured_at=NOW + timedelta(minutes=5)),
        [b"a"],
    )
    coord.ingest_crossing(crossing("early-c23", captured_at=NOW), [b"c"])

    uncontested = [
        r
        for r in log.of(stage="reid_evaluation")
        if r["reason"] == "no_causally_eligible_identity"
    ]
    assert len(uncontested) == 1
    assert uncontested[0]["uncontested"]["pending_identities"] == 1
    assert uncontested[0]["uncontested"]["causally_eligible"] == 0
    assert uncontested[0]["uncontested"]["identity_keys"] == ["SHR1198"]


def test_a_fresh_gate_read_is_never_retired_as_a_replay():
    """The narrowing, pinned: source time alone must not be the whole test.

    Two genuine visits by one car can BOTH have gate reads that precede either
    crossing — `test_equal_latest_entry_times_close_none_and_retain_one_boundary`
    is that shape, and an earlier draft of this guard deleted the second
    journey. A fresh ANPR read is a car at the barrier now; only evidence we
    went and pulled can be a record of a pass we already confirmed.
    """
    log = _CollectingLog()
    evidence = dict(_njs_evidence())
    evidence["njs-gate2"] = [frame("njs-gate2", "ANPR-ENTRY", (1.0, 0.0, 0.0))]
    coord, _ = _build_committing(evidence, log=log)

    coord.ingest_attempt(attempt("njs-1", "NJS-7894"), [b"1"])
    coord.ingest_crossing(
        crossing("njs-c23", captured_at=NOW + timedelta(seconds=30)), [b"c23"]
    )
    # A second GATE read whose source time also precedes that crossing. Same
    # timing as the replay, but it is not HikCentral-sourced, so it stands.
    coord.ingest_attempt(
        attempt("njs-gate2", "NJS-7894", captured_at=NOW + timedelta(seconds=5)),
        [b"g2"],
    )

    assert log.of(stage="same_key_retirement") == []


# --------------------------------------------------------------------------- #
# 6. The barrier verdict reaches the decision log
# --------------------------------------------------------------------------- #
def test_the_barrier_verdict_is_written_to_the_decision_log():
    """Evidence for the shadow review, carried in on the Hik attempt metadata.

    Nothing branches on it — the point is that a day of
    entry_decisions_gate_*.jsonl can be grouped by what the BARRIER did, beside
    the Re-ID score that actually decided, so an ordering design can be judged
    on real traffic before anything is built on it.
    """
    log = _CollectingLog()
    coord, _ = build(
        {"b-1": [frame("b-1", "ANPR-ENTRY", (1.0, 0.0))]}, log=log
    )
    request = replace(
        attempt("b-1", "GGR-9064", hik=True),
        metadata={
            "evidence_source": "hikcentral",
            "hik_guid": "CA215103F09E4C04A93312F27D3208A4",
            "barrier": {
                "allow_result": 2,
                "allow_type": 3,
                "enter_time": "2026-09-06T10:09:18+03:00",
                "gap_seconds": 1.0,
            },
        },
    )
    coord.ingest_attempt(request, [b"1"])

    created = log.of(stage="anpr_identity")
    assert len(created) == 1
    assert created[0]["barrier"]["allow_result"] == 2
    assert created[0]["barrier"]["allow_type"] == 3


def test_no_verdict_means_no_barrier_block_not_a_refusal():
    """Absence is a missing answer. It must never read as `not allowed`."""
    log = _CollectingLog()
    coord, _ = build(
        {"b-2": [frame("b-2", "ANPR-ENTRY", (1.0, 0.0))]}, log=log
    )
    coord.ingest_attempt(attempt("b-2", "GGR-9064", hik=True), [b"1"])

    created = log.of(stage="anpr_identity")
    assert len(created) == 1
    assert "barrier" not in created[0]
