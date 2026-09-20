"""Entry colour stays observable without filtering candidates."""
import dataclasses
from datetime import datetime, timedelta, timezone

from src.entry.coordinator import EntryCoordinator
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
    return EntryCoordinator(
        _settings(**overrides), _StubProcessor(), _StubSink(), decision_log=log
    )


def _log_uncontested(log, crossing, groups, **overrides):
    coordinator = _coordinator(log, **overrides)
    coordinator._log_uncontested_crossing_locked(crossing, groups)
    return log.records[-1] if log.records else None


def test_colour_mismatch_is_scored_and_logged_without_a_veto(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_COLOUR_VETO_ENABLED", "1")
    log = _CollectingLog()
    cfg = dataclasses.replace(
        EntrySettings.from_env(), mode=EntryMode.SHADOW,
        reid_min_score=0.75, reid_row_margin=0.08, reid_column_margin=0.08,
    )
    coordinator = EntryCoordinator(
        cfg, _StubProcessor(), _StubSink(), decision_log=log
    )
    groups = {"g-1": _identity("g-1", "AAA1111", DARK)}
    crossing = _crossing(PALE)

    match = coordinator._find_unique_match_with_observability_locked(
        crossing, groups, [crossing]
    )

    assert match is not None
    assert match.group_id == "g-1"
    record = log.records[-1]
    assert record["reason"] == "accepted"
    assert record["reid"]["score"] == 1.0
    assert record["colour"] == {
        "query_hsv": [99.2, 18.0, 210.0], "vetoed": [], "enabled": False,
    }


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
    record = _log_uncontested(log, _crossing(PALE), {})

    assert record["uncontested"] == {
        "pending_identities": 0,
        "causally_eligible": 0,
        "identity_keys": [],
    }
    assert record["stage"] == "reid_evaluation"
    assert record["result"] == "abstained"
    assert record["observation"]["camera"] == "CAM-23"


def test_the_fingerprint_dedup_still_suppresses_a_repeat():
    log = _CollectingLog()
    coordinator = _coordinator(log)
    crossing = _crossing(PALE)
    coordinator._log_uncontested_crossing_locked(crossing, {})
    coordinator._log_uncontested_crossing_locked(crossing, {})

    assert len(log.records) == 1


def test_a_failing_log_still_cannot_break_the_pipeline():
    class _Explodes:
        def emit(self, record):
            raise RuntimeError("disk on fire")

    coordinator = _coordinator(_Explodes())
    coordinator._log_uncontested_crossing_locked(
        _crossing(PALE), {}
    )
