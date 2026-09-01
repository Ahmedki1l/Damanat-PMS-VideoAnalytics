"""The Entry V2 decision-log record schema — one definition, one place.

WHY THIS EXISTS
    The Entry V2 pipeline raises no alerts. Every outcome — confirmed, ambiguous,
    abstained, unreadable, expired, or a degraded HikCentral call — is one JSONL
    record and nothing else. That makes this schema the entire operational
    surface of the pipeline: if a field is not written here, nobody can see it,
    and the shadow review has nothing to review.

    It is also the calibration corpus. ``tools/calibrate_entry_thresholds.py``
    reads these records to sweep the Re-ID score and margin thresholds, so the
    field names are a data contract, not a debug convenience. Change them
    deliberately and bump ``RECORD_VERSION``.

STAGED POPULATION
    The blocks below are filled in by the stage that owns them. A block that its
    stage has not shipped yet is simply absent from the record — never a
    placeholder, so a consumer can always distinguish "this did not happen" from
    "this was not measured".

        stage 1  envelope, observation, reid, result, reason
        stage 2  identity, witnesses, ttl
        stage 3  colour, fifo, ranked candidates
        stage 4  plate
        stage 5  hik

THE HIK BLOCK IS A RECORD OF A CALL WE MADE
    HikCentral is a PULL source: it never pushes into this pipeline and never
    triggers it. Every `hik` block describes a query OUR service issued, in
    response to an event we already had, and what came back:

        "hik": {
          "trigger": "anpr_identity" | "missing_anpr_recovery",
          "queried": true,              # we called the API
          "window": ["<from>", "<to>"], # the search scope we asked for
          "records": 3,                 # what it returned
          "images": 2,                  # of those, how many carried an image
          "reid_matched": ["guid-1"],   # the ones WE associated to this car
          "unmatched": ["guid-2"],      # other cars passing through
          "api_error": null             # 18A: degraded, never fatal
        }

    `trigger` names OUR event, never a HikCentral one, and `queried: false`
    means we chose not to call — never that HikCentral chose not to tell us.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

RECORD_VERSION = 1
EVENT = "entry_decision"

# `result` vocabulary. Every record carries exactly one of these, and the shadow
# review is a group-by over this field, so the set is closed on purpose.
RESULT_CONFIRMED = "confirmed"
RESULT_AMBIGUOUS = "ambiguous"
RESULT_ABSTAINED = "abstained"
RESULT_UNREADABLE = "unreadable"
RESULT_EXPIRED = "expired"
RESULT_HIK_DEGRADED = "hik_degraded"

RESULTS = frozenset(
    {
        RESULT_CONFIRMED,
        RESULT_AMBIGUOUS,
        RESULT_ABSTAINED,
        RESULT_UNREADABLE,
        RESULT_EXPIRED,
        RESULT_HIK_DEGRADED,
    }
)


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


def observation_block(crossing) -> Dict[str, Any]:
    """Describe the observation a decision was made about.

    ``witness`` is the physical-witness identity of the camera, which is a
    different axis from the plate sources — CAM-23 and CAM-03 are witnesses and
    are never plate sources. Kept as a plain string here; the enum arrives with
    stage 2.
    """
    request = crossing.request
    camera_id = request.camera_id
    return {
        "id": request.crossing_id,
        "source_event_id": request.source_event_id,
        "camera": camera_id,
        "witness": _witness_for_camera(camera_id),
        "role": getattr(request.role, "value", str(request.role)),
        "line_id": request.line_id,
        "direction": request.direction,
        "captured_at": _iso(request.captured_at),
        "source": str((request.metadata or {}).get("source", "")),
        # Where captured_at actually came from. PMS-AI has always sent this
        # (`_metadata` -> "timestamp_source": event.trigger_time_source); the
        # record simply never surfaced it, and that is why CAM-03 stamping
        # every event `-08:00` instead of `+03:00` — 11 hours into the future,
        # which silently disabled the causality guard on that camera — went
        # unnoticed for the whole shadow window. "camera" means the device's
        # own offset was trusted; "camera_assumed_facility_timezone" means it
        # sent none; "pms_receive_*" means the time is OUR receipt time, not
        # the camera's, and no causality conclusion should rest on it.
        "captured_at_source": str(
            (request.metadata or {}).get("timestamp_source", "")
        ),
    }


def _witness_for_camera(camera_id: str) -> str:
    """Map a normalised camera id to its witness name.

    Deliberately string-based for stage 1; stage 2 replaces the return with
    ``WitnessSource``. Anything unrecognised reports the raw id rather than
    guessing, so a mis-normalised camera shows up in the log instead of being
    silently bucketed as a known witness.
    """
    normalised = (camera_id or "").upper().replace("-", "").replace("_", "")
    if normalised == "CAM23":
        return "cam23"
    if normalised == "CAM03":
        return "cam03"
    return normalised.lower()


def reid_block(evaluation, settings) -> Dict[str, Any]:
    """The four numbers the thresholds are swept over, plus what they were.

    The thresholds are recorded alongside the scores on purpose: a record read
    six weeks later must say what bar the decision was actually held to, not
    what the bar happens to be at read time.
    """
    return {
        "argmax": evaluation.group_id,
        "score": round(float(evaluation.score), 6),
        "row_runner": round(float(evaluation.row_runner), 6),
        "row_margin": round(float(evaluation.row_margin), 6),
        "column_runner": round(float(evaluation.column_runner), 6),
        "column_margin": round(float(evaluation.column_margin), 6),
        "min_score": float(settings.reid_min_score),
        "min_row_margin": float(settings.reid_row_margin),
        "min_column_margin": float(settings.reid_column_margin),
        "accepted": bool(evaluation.accepted),
    }


def build_record(
    *,
    stage: str,
    result: str,
    reason: str,
    crossing=None,
    evaluation=None,
    settings=None,
    identity: Optional[Mapping[str, Any]] = None,
    witnesses: Optional[Iterable[str]] = None,
    colour: Optional[Mapping[str, Any]] = None,
    fifo: Optional[Mapping[str, Any]] = None,
    hik: Optional[Mapping[str, Any]] = None,
    plate: Optional[Mapping[str, Any]] = None,
    ranked: Optional[Sequence[Tuple[str, float]]] = None,
    observed_plate_text: str = "",
    observed_plate_confidence: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one decision record. Optional blocks are omitted when absent.

    ``observed_plate_*`` is the RAMP CAMERA's own OCR reading. It is diagnostic
    only and must never be read by a decision rule: CAM-23 and CAM-03 are not
    plate-reading sources. It is recorded so the question "could the ramp
    cameras carry plate evidence?" can be answered from data later, without
    letting them carry it now.
    """
    record: Dict[str, Any] = {
        "event": EVENT,
        "record_v": RECORD_VERSION,
        "stage": stage,
        "result": result,
        "reason": reason,
    }
    if crossing is not None:
        record["observation"] = observation_block(crossing)
    if evaluation is not None and settings is not None:
        record["reid"] = reid_block(evaluation, settings)
    if ranked is not None:
        record["ranked"] = [
            [str(group_id), round(float(score), 6)] for group_id, score in ranked
        ]
    if identity is not None:
        record["identity"] = dict(identity)
    if witnesses is not None:
        record["witnesses"] = sorted(str(w) for w in witnesses)
    if colour is not None:
        record["colour"] = dict(colour)
    if fifo is not None:
        record["fifo"] = dict(fifo)
    if hik is not None:
        record["hik"] = dict(hik)
    if plate is not None:
        record["plate"] = dict(plate)
    if observed_plate_text or observed_plate_confidence is not None:
        record["observed_plate_text"] = observed_plate_text
        record["observed_plate_confidence"] = (
            None
            if observed_plate_confidence is None
            else round(float(observed_plate_confidence), 4)
        )
    if extra:
        record.update(dict(extra))
    return record
