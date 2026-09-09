"""all_candidates_vetoed must say WHICH colours disagreed.

The colour veto is the only refusal in the matcher that discards a car without
producing a number to argue about: evaluate_unique_match removes the candidate
before scoring, ranked comes back empty, and the coordinator falls through to
all_candidates_vetoed. Until this change the record carried "uncontested" and
nothing else, so there was no way to tell a correct veto from a wrong one.

It is not hypothetical. On 2026-09-09 it fired on 2 of 32 CAM-23 ramp views:
HGD-2926 survived only because the CAM-03 fallback scored 0.735 five seconds
later, and SHR-1198 did not survive at all -- its identity hit the 900s TTL and
its crossing expired unclaimed. Neither record could say what colour anything
was.
"""
import dataclasses
from datetime import datetime, timedelta, timezone

from src.entry.decision import EntryDecisionEngine
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
)
from src.entry.settings import EntrySettings

NOW = datetime(2026, 9, 9, 9, 19, 0, tzinfo=timezone.utc)

# Real HSV pairs. body_colour_compatible reads both as achromatic and vetoes on
# abs(v_a - v_b) >= 90, so this is a dark car against a near-white one.
DARK = (99.2, 20.0, 40.0)
PALE = (99.2, 18.0, 210.0)


def _settings(**overrides):
    base = EntrySettings(
        mode=EntryMode.SHADOW,
        primary_cameras=frozenset({"CAM-23"}),
        primary_lines=frozenset({"Park_Entry"}),
        primary_directions=frozenset({"ramp-entry"}),
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _frame(event_id, vector, colour):
    return FrameEvidence(
        evidence_id=event_id + ":0",
        embedding=tuple(vector),
        plate=PlateEvidence(
            evidence_id=event_id + ":0",
            camera_id="ANPR-ENTRY",
            source_role="anpr",
            state=PlateReadState.NO_PLATE,
            text="",
            confidence=0.0,
        ),
        colour_hsv=colour,
    )


def _identity(group_id, key, colour, vector=(1.0, 0.0), captured_at=None):
    """A PENDING identity whose ANPR read PRECEDES the crossing, so eligible."""
    group = AttemptGroup(group_id=group_id, identity_key=key)
    group.attempts["a-" + group_id] = AttemptRecord(
        request=AttemptInput(
            attempt_id="a-" + group_id,
            source_event_id="src-" + group_id,
            camera_id="ANPR-ENTRY",
            captured_at=captured_at or NOW,
            reported_plate=key,
            reported_confidence=0.95,
            metadata={},
        ),
        evidence=(_frame("a-" + group_id, vector, colour),),
        group_id=group_id,
    )
    return group


def _crossing(colour, vector=(1.0, 0.0), crossing_id="cr-1"):
    request = CrossingInput(
        crossing_id=crossing_id,
        source_event_id="src-" + crossing_id,
        camera_id="CAM-23",
        captured_at=NOW + timedelta(seconds=4),
        line_id="Park_Entry",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
        metadata={},
    )
    return CrossingRecord(
        request=request, evidence=(_frame(crossing_id, vector, colour),)
    )


class _CollectingLog:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _StubProcessor:
    def analyze(self, **kwargs):  # pragma: no cover
        raise AssertionError("no inference here")


class _StubSink:
    def deliver(self, payload):  # pragma: no cover
        raise AssertionError("nothing is delivered here")


def _coordinator(log, **overrides):
    from src.entry.coordinator import EntryCoordinator

    return EntryCoordinator(
        _settings(**overrides), _StubProcessor(), _StubSink(), decision_log=log
    )


def _log_uncontested(log, crossing, groups, **overrides):
    coordinator = _coordinator(log, **overrides)
    coordinator._log_uncontested_crossing_locked(crossing, groups)
    return log.records[-1] if log.records else None


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
def test_a_vetoed_crossing_now_records_both_colours():
    log = _CollectingLog()
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    record = _log_uncontested(log, _crossing(PALE), groups)

    assert record["reason"] == "all_candidates_vetoed"
    colour = record["colour"]
    assert colour["query_hsv"] == [99.2, 18.0, 210.0]
    assert colour["vetoed"] == ["g-1"]
    assert colour["enabled"] is True
    detail = colour["vetoed_detail"][0]
    assert detail["identity_key"] == "SHR1198"
    assert detail["gallery_hsv"] == [99.2, 20.0, 40.0]


def test_every_vetoed_identity_is_named_not_just_the_first():
    log = _CollectingLog()
    groups = {
        "g-1": _identity("g-1", "SHR1198", DARK),
        "g-2": _identity("g-2", "HGD2926", DARK),
    }

    colour = _log_uncontested(log, _crossing(PALE), groups)["colour"]

    assert sorted(colour["vetoed"]) == ["g-1", "g-2"]
    assert sorted(d["identity_key"] for d in colour["vetoed_detail"]) == [
        "HGD2926",
        "SHR1198",
    ]


# --------------------------------------------------------------------------- #
# The record must describe the decision that was actually made
# --------------------------------------------------------------------------- #
def test_the_logged_veto_matches_what_the_matcher_actually_did():
    """The block is recomputed, so it could drift from the real predicate."""
    settings = _settings()
    crossing = _crossing(PALE)
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    # The matcher abstains with nothing ranked - the path that logs the block.
    engine = EntryDecisionEngine(settings)
    assert engine.evaluate_unique_match(crossing, groups, [crossing]) is None

    colour = _log_uncontested(_CollectingLog(), crossing, groups)["colour"]
    assert colour["vetoed"] == ["g-1"]


def test_a_compatible_colour_is_never_reported_as_vetoed():
    log = _CollectingLog()
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    record = _log_uncontested(log, _crossing(DARK), groups)

    # Same colour both sides, so nothing may be named. Asserted unconditionally:
    # a guarded assert here would pass whether or not the block was built.
    assert record["reason"] == "all_candidates_vetoed"
    assert record["colour"]["vetoed"] == []
    assert record["colour"]["vetoed_detail"] == []


def test_the_veto_being_disabled_names_nobody():
    """decision.py gates on colour_veto_enabled BEFORE comparing colours.

    A recomputed block that skips the flag reports a veto the matcher never
    applied - which is worse than no block, because it sends a reader hunting a
    colour veto for an abstention that has some other cause. Caught exactly
    that way: the first version of this fix listed g-1 with the veto off.
    """
    log = _CollectingLog()
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    record = _log_uncontested(
        log, _crossing(PALE), groups, colour_veto_enabled=False
    )

    assert record["reason"] == "all_candidates_vetoed"
    assert record["colour"]["enabled"] is False
    assert record["colour"]["vetoed"] == []
    assert record["colour"]["vetoed_detail"] == []


# --------------------------------------------------------------------------- #
# The other two reasons have no colour to report
# --------------------------------------------------------------------------- #
def test_no_live_identity_carries_no_colour_block():
    record = _log_uncontested(_CollectingLog(), _crossing(PALE), {})

    assert record["reason"] == "no_live_identity"
    assert "colour" not in record


def test_a_causally_ineligible_identity_carries_no_colour_block():
    log = _CollectingLog()
    # ANPR read AFTER the crossing: the car did not exist when it went past.
    late = _identity(
        "g-1", "SHR1198", DARK, captured_at=NOW + timedelta(minutes=5)
    )

    record = _log_uncontested(log, _crossing(PALE), {"g-1": late})

    assert record["reason"] == "no_causally_eligible_identity"
    assert "colour" not in record


# --------------------------------------------------------------------------- #
# Nothing else about the record changed
# --------------------------------------------------------------------------- #
def test_the_uncontested_block_is_untouched():
    log = _CollectingLog()
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    record = _log_uncontested(log, _crossing(PALE), groups)

    assert record["uncontested"] == {
        "pending_identities": 1,
        "causally_eligible": 1,
        "identity_keys": ["SHR1198"],
    }
    assert record["stage"] == "reid_evaluation"
    assert record["result"] == "abstained"
    assert record["observation"]["camera"] == "CAM-23"


def test_the_fingerprint_dedup_still_suppresses_a_repeat():
    log = _CollectingLog()
    coordinator = _coordinator(log)
    crossing = _crossing(PALE)
    groups = {"g-1": _identity("g-1", "SHR1198", DARK)}

    coordinator._log_uncontested_crossing_locked(crossing, groups)
    coordinator._log_uncontested_crossing_locked(crossing, groups)

    assert len(log.records) == 1


def test_a_failing_log_still_cannot_break_the_pipeline():
    class _Explodes:
        def emit(self, record):
            raise RuntimeError("disk on fire")

    coordinator = _coordinator(_Explodes())
    coordinator._log_uncontested_crossing_locked(
        _crossing(PALE), {"g-1": _identity("g-1", "SHR1198", DARK)}
    )
