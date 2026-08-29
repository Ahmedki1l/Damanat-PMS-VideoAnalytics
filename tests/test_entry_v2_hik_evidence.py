"""Stage 5 — the overlay guard, and what a HikCentral-sourced attempt counts as.

Two things are being kept honest here.

THE OVERLAY GUARD. Hikvision composites its own plate/OSD panel into the corner
of a frame. That panel is a sharp, high-contrast, perfectly rectangular
rendering of a plate, and the detector loves it — it will outscore a real plate
on a car twenty metres away. Reading it is not independent verification, it is a
very expensive echo of Hikvision's own answer.

WHAT A HIK ATTEMPT IS. It arrives because WE queried HikCentral; the platform
pushes nothing. Its plate is HikCentral's reading, not the gate ANPR system's,
and it is a HIK witness rather than an ANPR one. Conflating them would let one
platform's answer be counted as two agreeing sources.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

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
    PlateSourceKind,
    WitnessSource,
)
from src.entry.settings import EntrySettings, _env_regions
from src.ocr.plate_region_detector import OpenVINOPlateRegionDetector


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


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


def attempt(attempt_id, plate, metadata=None, captured_at=NOW):
    return AttemptInput(
        attempt_id=attempt_id,
        source_event_id=f"src-{attempt_id}",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=plate,
        reported_confidence=0.95,
        metadata=metadata or {},
    )


def hik_attempt(attempt_id, plate, guid="guid-1", captured_at=NOW):
    """An attempt that exists because WE asked HikCentral for it."""
    return attempt(
        attempt_id,
        plate,
        metadata={"evidence_source": "hikcentral", "hik_guid": guid},
        captured_at=captured_at,
    )


def crossing(crossing_id, captured_at=None):
    return CrossingInput(
        crossing_id=crossing_id,
        source_event_id=f"src-{crossing_id}",
        camera_id="CAM-23",
        captured_at=captured_at or (NOW + timedelta(seconds=30)),
        line_id="RAMP-IN",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
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


def build(evidence, cfg=None):
    sink = _Sink()
    return EntryCoordinator(cfg or settings(), _Processor(evidence), sink), sink


# --------------------------------------------------------------------------- #
# The overlay guard: box selection
# --------------------------------------------------------------------------- #
def _pick(boxes, regions, width=1000, height=500):
    return OpenVINOPlateRegionDetector._first_box_outside(
        boxes, regions, width=width, height=height
    )


def test_without_regions_the_best_box_wins_exactly_as_before():
    boxes = [(10.0, 10.0, 90.0, 40.0, 0.9), (500.0, 300.0, 580.0, 330.0, 0.5)]
    assert _pick(boxes, None) == boxes[0]
    assert _pick(boxes, ()) == boxes[0]


def test_the_composited_panel_is_skipped_for_the_real_plate():
    """The panel scores higher. It is still not the car."""
    panel = (10.0, 10.0, 90.0, 40.0, 0.97)      # top-left, sharp, fake
    real = (500.0, 300.0, 580.0, 330.0, 0.42)   # a real plate at distance
    top_left = ((0.0, 0.0, 0.20, 0.20),)

    assert _pick([panel, real], top_left) == real


def test_a_real_plate_near_the_panel_is_still_found():
    """Centre-based, not overlap-based, and deliberately so: rejecting on any
    overlap would throw away a legible plate to avoid a false one."""
    real = (150.0, 60.0, 260.0, 95.0, 0.8)   # centre (205, 77) -> (0.205, 0.155)
    top_left = ((0.0, 0.0, 0.20, 0.20),)
    assert _pick([real], top_left) == real


def test_every_candidate_excluded_yields_nothing():
    panel_a = (10.0, 10.0, 90.0, 40.0, 0.97)
    panel_b = (20.0, 20.0, 100.0, 50.0, 0.80)
    assert _pick([panel_a, panel_b], ((0.0, 0.0, 0.20, 0.20),)) is None


def test_multiple_regions_are_all_honoured():
    top_left = (30.0, 30.0, 70.0, 50.0, 0.99)
    bottom_right = (900.0, 450.0, 960.0, 480.0, 0.95)
    real = (500.0, 240.0, 580.0, 270.0, 0.4)
    regions = ((0.0, 0.0, 0.20, 0.20), (0.85, 0.85, 1.0, 1.0))
    assert _pick([top_left, bottom_right, real], regions) == real


# --------------------------------------------------------------------------- #
# The overlay guard: configuration
# --------------------------------------------------------------------------- #
def test_regions_are_parsed_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_OVERLAY_EXCLUDE_REGIONS", "0,0,0.25,0.2;0.8,0,1,0.1")
    assert _env_regions("ENTRY_V2_OVERLAY_EXCLUDE_REGIONS") == (
        (0.0, 0.0, 0.25, 0.2),
        (0.8, 0.0, 1.0, 0.1),
    )


def test_a_malformed_region_yields_no_region_rather_than_a_partial_one(monkeypatch):
    """A guard that silently half-applied would be worse than one that is off:
    it would look configured while protecting the wrong part of the frame."""
    for bad in ("0,0,0.25", "0,0,0.25,abc", "0.5,0,0.2,0.4", "0,0,2,1"):
        monkeypatch.setenv("ENTRY_V2_OVERLAY_EXCLUDE_REGIONS", bad)
        assert _env_regions("ENTRY_V2_OVERLAY_EXCLUDE_REGIONS") == ()


def test_the_guard_is_inert_by_default():
    """The panel's real geometry has to be measured against this facility's own
    frames. A guessed rectangle would reject real plates, which is worse than
    the echo it is meant to prevent."""
    assert EntrySettings().overlay_exclude_regions == ()


# --------------------------------------------------------------------------- #
# The overlay guard: which frames it applies to
# --------------------------------------------------------------------------- #
def _regions_for(source_role, metadata, configured=((0.0, 0.0, 0.2, 0.2),)):
    from src.entry.analyzer import ExistingModelsEvidenceProcessor

    processor = ExistingModelsEvidenceProcessor(
        registry=None, settings=settings(overlay_exclude_regions=configured)
    )
    return processor._overlay_regions_for(source_role, metadata)


def test_the_guard_applies_to_hikcentral_sourced_frames():
    assert _regions_for("anpr", {"evidence_source": "hikcentral"}) != ()


def test_the_guard_applies_to_the_anpr_overview():
    """PMS forwards the bounded FULL frame so we localise the plate ourselves —
    panel and all — so the guard is needed here too."""
    assert _regions_for("anpr", {}) != ()


def test_the_guard_leaves_tight_ramp_crops_alone():
    """A line-crossing crop is already tight around the vehicle: there is no
    panel in it, and no reason to risk excluding part of a real plate."""
    assert _regions_for("primary", {}) == ()
    assert _regions_for("fallback", {}) == ()


def test_nothing_configured_means_no_regions_anywhere():
    assert _regions_for("anpr", {"evidence_source": "hikcentral"}, configured=()) == ()


# --------------------------------------------------------------------------- #
# What a HikCentral-sourced attempt counts as
# --------------------------------------------------------------------------- #
def test_a_hik_attempt_is_a_hik_witness_not_an_anpr_one():
    evidence = {"h1": [frame("h1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(hik_attempt("h1", "ABC-1234"), [b"h"])

    group = coord.state_summary()["groups"][result.group_id]
    assert group["witnesses"] == ["hik"]
    assert "anpr" not in group["witnesses"]


def test_a_hik_attempt_contributes_hik_text_not_the_anpr_source():
    """One platform's answer must not be counted as two agreeing sources."""
    evidence = {"h1": [frame("h1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(hik_attempt("h1", "ABC-1234"), [b"h"])

    with coord._lock:
        group = coord._groups[result.group_id]
        assert PlateSourceKind.HIK_TEXT in group.plate_sources
        assert group.plate_sources[PlateSourceKind.HIK_TEXT].text == "ABC-1234"
        # ANPR is derived from the attempts; a Hik attempt is not a gate read,
        # so nothing here may present it as one.
        sources = EntryDecisionEngine(coord.settings).available_plate_sources(group)
        assert PlateSourceKind.ANPR not in sources


def test_the_consumed_guid_is_remembered_so_a_repeat_query_cannot_double_ingest():
    evidence = {"h1": [frame("h1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(hik_attempt("h1", "ABC-1234", guid="g-77"), [b"h"])

    with coord._lock:
        assert "g-77" in coord._groups[result.group_id].hik_guids_consumed


def test_an_ordinary_gate_attempt_is_untouched_by_any_of_this():
    evidence = {"a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))]}
    coord, _ = build(evidence)
    result = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])

    group = coord.state_summary()["groups"][result.group_id]
    assert group["witnesses"] == ["anpr"]
    assert group["plate_sources"]["anpr"] == "ABC-1234"


def test_a_hik_attempt_attaches_to_the_identity_the_gate_read_created():
    """Plate-keying is the join: no new field on the wire, no new endpoint."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "h1": [frame("h1", "ANPR-ENTRY", (0.99, 0.02))],
    }
    coord, _ = build(evidence)
    gate = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    hik = coord.ingest_attempt(
        hik_attempt("h1", "ABC-1234", captured_at=NOW + timedelta(seconds=2)), [b"h"]
    )

    assert gate.group_id == hik.group_id
    group = coord.state_summary()["groups"][gate.group_id]
    # Both axes now populated, from two different systems.
    assert set(group["witnesses"]) == {"anpr", "hik"}
    assert group["plate_sources"]["anpr"] == "ABC-1234"
    assert group["plate_sources"]["hik_text"] == "ABC-1234"


def test_hik_does_not_become_a_second_witness_alongside_anpr():
    """It is the platform's log of the same gate event the ANPR camera already
    reported, so it substitutes for a MISSING ANPR read and never adds to one."""
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "h1": [frame("h1", "ANPR-ENTRY", (0.99, 0.02))],
    }
    coord, _ = build(evidence)
    gate = coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    coord.ingest_attempt(
        hik_attempt("h1", "ABC-1234", captured_at=NOW + timedelta(seconds=2)), [b"h"]
    )

    with coord._lock:
        group = coord._groups[gate.group_id]
        assert group.confirming_witnesses() == {WitnessSource.ANPR}
        assert coord._witness_shortfall_locked(group) == "insufficient_witnesses"
