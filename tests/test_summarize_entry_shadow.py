"""Stage 7 — the review instrument.

The pipeline raises no alerts, so this tool IS the review. If it under-reports,
a bad run looks clean and gets flipped to authoritative; if it over-reports, a
good run gets held back. Both failures are expensive, so the hard stops are
tested against records produced by the REAL coordinator rather than by hand.
"""
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
from tools.summarize_entry_shadow import hard_stops, load_records, review_sections


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
TOOL = Path(__file__).resolve().parents[1] / "tools" / "summarize_entry_shadow.py"


def settings(**overrides):
    base = EntrySettings(
        mode=EntryMode.SHADOW,
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


class _Processor:
    def __init__(self, evidence):
        self.evidence = evidence

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        return tuple(self.evidence[event_id])


class _Sink:
    def deliver(self, payload):
        return DeliveryResult(True, 1, "", publish_identity=False)


class _Log:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _real_run() -> list[dict]:
    """Drive the actual coordinator so the records are the real shape."""
    log = _Log()
    evidence = {
        "a1": [frame("a1", "ANPR-ENTRY", (1.0, 0.0))],
        "c1": [frame("c1", "CAM-23", (1.0, 0.0), role="primary")],
        "a2": [frame("a2", "ANPR-ENTRY", (0.0, 1.0))],
        "c2": [frame("c2", "CAM-03", (0.0, 1.0), role="fallback")],
    }
    coord = EntryCoordinator(
        settings(), _Processor(evidence), _Sink(), decision_log=log
    )

    def attempt(identifier, plate):
        return AttemptInput(
            attempt_id=identifier,
            source_event_id=f"s-{identifier}",
            camera_id="ANPR-ENTRY",
            captured_at=NOW,
            reported_plate=plate,
            reported_confidence=0.95,
            metadata={},
        )

    def crossing(identifier, camera, role):
        primary = role is CrossingRole.PRIMARY
        return CrossingInput(
            crossing_id=identifier,
            source_event_id=f"s-{identifier}",
            camera_id=camera,
            captured_at=NOW + timedelta(seconds=30),
            line_id="RAMP-IN" if primary else "B-IN",
            direction="ramp-entry" if primary else "b-entry",
            role=role,
            metadata={},
        )

    coord.ingest_attempt(attempt("a1", "ABC-1234"), [b"a"])
    coord.ingest_crossing(crossing("c1", "CAM-23", CrossingRole.PRIMARY), [b"c"])
    coord.ingest_attempt(attempt("a2", "XYZ-9999"), [b"b"])
    coord.ingest_crossing(crossing("c2", "CAM-03", CrossingRole.FALLBACK), [b"d"])
    return log.records


# --------------------------------------------------------------------------- #
# The hard stops, against records the real pipeline produced
# --------------------------------------------------------------------------- #
def test_a_healthy_run_trips_no_hard_stop():
    records = _real_run()
    assert records, "the coordinator produced no records at all"
    for name, count, _ in hard_stops(records):
        assert count == 0, f"{name} tripped on a healthy run: {count}"


def test_the_real_run_produces_the_stages_the_review_reads():
    stages = {r.get("stage") for r in _real_run()}
    # Every section of the report depends on one of these existing.
    assert {"anpr_identity", "reid_evaluation", "physical_confirm"} <= stages


def test_a_single_witness_confirmation_is_caught():
    records = _real_run()
    records.append(
        {
            "stage": "physical_confirm",
            "result": "confirmed",
            "reason": "witnesses_agree",
            "witnesses": ["anpr"],
        }
    )
    tripped = {name: count for name, count, _ in hard_stops(records)}
    assert tripped["confirmations with fewer than two witnesses"] == 1


def test_a_confirmation_without_plate_consensus_is_caught():
    records = _real_run()
    records.append(
        {
            "stage": "plate_consensus",
            "result": "confirmed",
            "reason": "reid_and_plate_consensus",
            "plate": {"outcome": "no_consensus", "available": ["anpr", "hik_text"]},
        }
    )
    tripped = {name: count for name, count, _ in hard_stops(records)}
    assert tripped["confirmations with no plate consensus"] == 1


def test_a_ramp_camera_appearing_as_a_plate_source_is_caught():
    """If this ever fires, the separation the whole design rests on is gone."""
    records = _real_run()
    records.append(
        {
            "stage": "plate_consensus",
            "result": "confirmed",
            "plate": {"outcome": "consensus", "available": ["anpr", "cam23"]},
        }
    )
    tripped = {name: count for name, count, _ in hard_stops(records)}
    assert tripped["plate sources that are not one of the three"] == 1


def test_hik_counted_alongside_anpr_is_caught():
    records = _real_run()
    records.append(
        {
            "stage": "physical_confirm",
            "result": "confirmed",
            "witnesses": ["anpr", "hik"],
        }
    )
    tripped = {name: count for name, count, _ in hard_stops(records)}
    assert tripped["confirmations counting HIK alongside ANPR"] == 1


def test_a_degraded_configuration_is_caught():
    records = _real_run()
    records.append({"stage": "x", "result": "abstained",
                    "reason": "entry_v2_invalid_configuration:PMS_API_URL"})
    tripped = {name: count for name, count, _ in hard_stops(records)}
    assert tripped["entry_v2_invalid_configuration"] == 1


# --------------------------------------------------------------------------- #
# The review sections
# --------------------------------------------------------------------------- #
def test_witness_pairs_are_reported_so_an_unexercised_path_is_visible():
    review = review_sections(_real_run())
    # Both cars confirmed, one through CAM-23 and one through CAM-03.
    assert review["witness_pairs"]["anpr+cam23"] >= 1
    assert review["witness_pairs"]["anpr+cam03"] >= 1


def test_fifo_agreement_is_counted_but_never_judged():
    review = review_sections(_real_run())
    assert review["fifo"]["compared"] > 0
    assert review["fifo"]["agreed"] + review["fifo"]["disagreed"] == (
        review["fifo"]["compared"]
    )


def test_reid_distributions_are_reported_for_the_threshold_sweep():
    review = review_sections(_real_run())
    accepted = review["reid"]["accepted"]
    assert accepted["n"] > 0
    for key in ("min", "p10", "p50", "p90", "max"):
        assert key in accepted


def test_identity_hygiene_is_reported():
    review = review_sections(_real_run())
    assert review["identity"]["created"] == 2
    assert review["identity"]["same_key_splits"] == 0


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #
def test_records_are_read_from_a_directory_of_daily_files(tmp_path):
    records = _real_run()
    for day in ("2026-08-28", "2026-08-29"):
        target = tmp_path / f"entry_decisions_api_{day}.jsonl"
        target.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
    assert len(load_records([tmp_path])) == 2 * len(records)


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path):
    """The writer flushes but never fsyncs, so a copy taken mid-write can clip
    the last line. Losing one record must not lose the run."""
    target = tmp_path / "entry_decisions_api_2026-08-29.jsonl"
    target.write_text(
        json.dumps({"stage": "reid_evaluation", "result": "confirmed"})
        + '\n{"stage": "reid_eval',
        encoding="utf-8",
    )
    assert len(load_records([tmp_path])) == 1


# --------------------------------------------------------------------------- #
# The verdict, end to end
# --------------------------------------------------------------------------- #
def _run_tool(path, *args):
    return subprocess.run(
        [sys.executable, str(TOOL), str(path), *args],
        capture_output=True,
        text=True,
        cwd=str(TOOL.parents[1]),
    )


def test_an_empty_directory_fails_rather_than_reporting_a_clean_run(tmp_path):
    """An absent run is not a clean one, and the difference is the whole gate."""
    result = _run_tool(tmp_path)
    assert result.returncode == 1
    assert "NO RECORDS" in result.stdout
    assert "not a clean run" in result.stdout


def test_a_healthy_run_exits_zero_and_says_it_is_not_sufficient(tmp_path):
    target = tmp_path / "entry_decisions_api_2026-08-29.jsonl"
    target.write_text(
        "\n".join(json.dumps(r) for r in _real_run()) + "\n", encoding="utf-8"
    )
    result = _run_tool(target)

    assert result.returncode == 0, result.stdout
    assert "hard stops clean" in result.stdout
    # A tool that said "PASS" and stopped would be the problem with review gates.
    assert "NOT sufficient" in result.stdout
    assert "needs eyes" in result.stdout.lower()


def test_a_failing_run_exits_nonzero(tmp_path):
    records = _real_run()
    records.append(
        {"stage": "physical_confirm", "result": "confirmed", "witnesses": ["anpr"]}
    )
    target = tmp_path / "entry_decisions_api_2026-08-29.jsonl"
    target.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    result = _run_tool(target)

    assert result.returncode == 1
    assert "NOT READY" in result.stdout
    assert "Do not flip to authoritative" in result.stdout


def test_json_output_is_machine_readable(tmp_path):
    target = tmp_path / "entry_decisions_api_2026-08-29.jsonl"
    target.write_text(
        "\n".join(json.dumps(r) for r in _real_run()) + "\n", encoding="utf-8"
    )
    result = _run_tool(target, "--json")
    payload = json.loads(result.stdout)

    assert payload["records"] > 0
    assert all(count == 0 for count in payload["hard_stops"].values())
