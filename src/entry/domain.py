"""Domain types for V2 entry validation.

Only compact metadata, float embeddings, and OCR strings are represented here.
There is intentionally no image/path/blob field: request bytes must stop at the
inference adapter and be released after evidence extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class EntryMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    AUTHORITATIVE = "authoritative"

    @classmethod
    def parse(cls, value: str) -> "EntryMode":
        normalised = (value or "off").strip().lower()
        if normalised in {"on", "enforce"}:
            normalised = cls.AUTHORITATIVE.value
        return cls(normalised)


class CrossingRole(str, Enum):
    PRIMARY = "primary"
    FALLBACK = "fallback"


class PlateReadState(str, Enum):
    NO_PLATE = "no_plate"
    UNREADABLE = "unreadable"
    READABLE = "readable"


class RecordStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class DecisionStatus(str, Enum):
    CONFIRMED = "confirmed"
    ABSTAINED = "abstained"


class WitnessSource(str, Enum):
    """Something that SAW a vehicle. Not a plate reading — see PlateSourceKind.

    These two axes are separate and must stay separate. A source appearing on
    one list does not put it on the other: CAM-23 and CAM-03 are witnesses that
    read no plates, and HikCentral's text-only record is a plate source that
    witnessed nothing we can associate.
    """

    ANPR = "anpr"       # the ANPR vehicle image
    HIK = "hik"         # a HikCentral vehicle image, Re-ID-associated
    CAM23 = "cam23"     # CAM-23 visual observation
    CAM03 = "cam03"     # CAM-03 visual observation

    # DIRECTION OF COMMUNICATION — HikCentral is a PULL source.
    #
    # HikCentral never pushes anything into this pipeline and never triggers
    # it. WE call its API, in response to an event we already have: an ANPR
    # read, or a camera observation with no identity to attach to. It answers
    # with candidate records and images, and we do the deciding.
    #
    # Do not confuse this with the "hikvision" producer string on a crossing.
    # That is the CAMERA webhook, which really is pushed to us by the camera
    # server, and it is a different system from the HikCentral platform we
    # query. Both appear in this codebase; only one of them calls us.


class PlateSourceKind(str, Enum):
    """Something that READ a plate. Exactly three, forever.

    There are only two plate-reading systems in this flow — the gate ANPR and
    HikCentral — plus our own OCR on whatever vehicle image is available
    (primarily the HikCentral vehiclePicUri image, secondarily the ANPR image).

    There is deliberately no camera-derived member. If one is ever added, the
    ramp cameras have become plate sources and the separation above is gone.
    """

    ANPR = "anpr"
    HIK_TEXT = "hik_text"
    OUR_OCR = "our_ocr"


# Cameras whose observations are witnesses. Anything else reports its own
# normalised id rather than being bucketed as a known witness, so a
# mis-normalised camera shows up in the log instead of silently counting.
_WITNESS_BY_CAMERA = {
    "CAM23": WitnessSource.CAM23,
    "CAM03": WitnessSource.CAM03,
}


def witness_for_camera(camera_id: str) -> Optional[WitnessSource]:
    """Map a camera id to its witness, or None when it is not a ramp camera."""
    return _WITNESS_BY_CAMERA.get(norm_camera_id(camera_id))


_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_LETTERS_DIGITS = re.compile(r"^([A-Z]{1,4})([0-9]{1,4})$")


def plate_key(value: str) -> str:
    """Lossless comparison key; never uses fuzzy/edit-distance matching."""
    return _NON_ALNUM.sub("", (value or "").upper())


_DIGIT_RUN = re.compile(r"[0-9]+")


def plate_digit_run(value: str) -> str:
    """The digits of a plate, in order, letters discarded.

    Order-independent by construction, which is the point: the UI renders
    plates digits-first and the DB letters-first, so `7286EED` and `EED7286`
    are the SAME plate written two ways.
    """
    return "".join(_DIGIT_RUN.findall(plate_key(value)))


def plates_contradict(left: str, right: str) -> bool:
    """True only when two reads genuinely name DIFFERENT cars.

    Replaces exact-key inequality, which was measured to be wrong in two ways
    that both WITHHOLD A CORRECT ENTRY:

      * `7286EED` vs `EED7286` — the same plate, digits-first versus
        letters-first. Exact keys call this a contradiction.
      * `7383HAS` vs `AATEIGH7383HAS` — two reads of one car in one window,
        both above the 0.75 confidence gate, the second carrying a
        hallucinated letter prefix. Exact keys call this a contradiction too.

    So the digits decide, and they are compared leniently: identical, or one a
    prefix/suffix of the other, means the same car with characters lost or
    invented off an end. `6951` versus `56951` is one car, not two.

    Deliberately asymmetric in caution. A missed contradiction costs a veto
    that would not have fired; a false contradiction refuses a real entry —
    which is what the producer-family gate did to one real crossing thirteen
    times on 2026-08-30. When either side has no digits at all there is
    nothing to compare, and we do NOT contradict.
    """
    left_key, right_key = plate_key(left), plate_key(right)
    if not left_key or not right_key or left_key == right_key:
        return False
    left_digits, right_digits = plate_digit_run(left), plate_digit_run(right)
    if not left_digits or not right_digits:
        return False
    if left_digits == right_digits:
        return False
    short, long = sorted((left_digits, right_digits), key=len)
    return not (long.startswith(short) or long.endswith(short))


def canonical_plate(value: str) -> str:
    """Stable display form for common Saudi Latin-letter plate readings."""
    key = plate_key(value)
    match = _LETTERS_DIGITS.fullmatch(key)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return key


def norm_camera_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


@dataclass(frozen=True)
class PlateEvidence:
    evidence_id: str
    camera_id: str
    source_role: str
    state: PlateReadState
    text: str = ""
    confidence: float = 0.0

    @property
    def key(self) -> str:
        return plate_key(self.text)


@dataclass(frozen=True)
class PlateReading:
    """One plate-source reading, for the consensus in a later stage.

    Distinct from PlateEvidence, which is an OCR result on one specific image.
    A PlateReading is what a SOURCE says the plate is: the gate's own reported
    value, HikCentral's own reported value, or our OCR's conclusion folded
    across every image it read.

    ``conflicted`` marks a source that contradicted itself — our OCR reading two
    images and disagreeing. Such a source is excluded from consensus entirely
    rather than having its readings counted as two opinions, because our reader
    contradicting itself is evidence of unreliability, not a tie to break.
    """

    source: PlateSourceKind
    text: str = ""
    confidence: float = 0.0
    origin: str = ""          # attempt_id, evidence_id, or HikCentral GUID
    conflicted: bool = False

    @property
    def key(self) -> str:
        return plate_key(self.text)


@dataclass(frozen=True)
class FrameEvidence:
    evidence_id: str
    embedding: Tuple[float, ...]
    plate: PlateEvidence
    # Mean HSV of the crop centre. A LOW-ENTROPY signal: most cars are black,
    # grey, silver or white, so colour can only ever REMOVE a candidate, never
    # add confidence to one. Optional, and every check fails open when absent.
    colour_hsv: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class AttemptInput:
    attempt_id: str
    source_event_id: str
    camera_id: str
    captured_at: datetime
    reported_plate: str
    reported_confidence: Optional[float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CrossingInput:
    crossing_id: str
    source_event_id: str
    camera_id: str
    captured_at: datetime
    line_id: str
    direction: str
    role: CrossingRole
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CancellationInput:
    reason: str
    attempt_id: str = ""
    group_id: str = ""
    crossing_id: str = ""
    exit_plate: str = ""
    exit_captured_at: Optional[datetime] = None


@dataclass
class PlateHypothesis:
    key: str
    display: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    reported_by: set[str] = field(default_factory=set)
    ocr_evidence_ids: set[str] = field(default_factory=set)


@dataclass
class AttemptRecord:
    request: AttemptInput
    evidence: Tuple[FrameEvidence, ...]
    group_id: str

    @property
    def embeddings(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(frame.embedding for frame in self.evidence if frame.embedding)

    @property
    def plate_evidence(self) -> Tuple[PlateEvidence, ...]:
        return tuple(frame.plate for frame in self.evidence)


@dataclass
class AttemptGroup:
    """One entry identity. Keyed by PLATE, not by appearance.

    The ANPR event is the anchor because it supplies both the plate and the
    first vehicle image. A second ANPR read of the same plate enriches this
    identity rather than creating another; HikCentral and ramp-camera evidence
    likewise attach to the identity that already exists.
    """

    group_id: str
    attempts: Dict[str, AttemptRecord] = field(default_factory=dict)
    hypotheses: Dict[str, PlateHypothesis] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.PENDING
    canonical_plate: Optional[str] = None

    # --- identity (stage 2) ------------------------------------------- #
    # plate_key of the anchoring ANPR read. Empty for a PLATELESS identity,
    # which is what the dropped-ANPR recovery path creates: with no plate there
    # is no key to group by, and appearance is the only thing left.
    identity_key: str = ""
    # Wall-clock, not captured_at. The TTL is a lifetime in OUR time; anchoring
    # it on capture would make a backfilled event arrive already expired.
    created_at: Optional[datetime] = None
    # Refreshed whenever an attempt is added: an ANPR event "creates OR
    # activates" the candidate, so enrichment restarts its 15 minutes.
    last_activity_at: Optional[datetime] = None
    # WitnessSource -> the id that produced it. A STRONG vote: this witness
    # named this identity as its Re-ID argmax AND cleared its gates.
    witnesses: Dict["WitnessSource", str] = field(default_factory=dict)
    # WitnessSource -> the id that produced it. A WEAK vote: this witness named
    # this identity as its argmax but could not clear the margin on its own.
    #
    # This is what lets an ambiguous CAM-23 be resolved by a confident CAM-03
    # instead of being forced or discarded. Weak votes count toward the
    # two-witness rule but can never satisfy it alone — a confirmation always
    # needs at least one witness that cleared the bar by itself, so two
    # uncertain looks never add up to a certainty.
    weak_votes: Dict["WitnessSource", str] = field(default_factory=dict)
    # PlateSourceKind -> that source's reading. At most one per source.
    plate_sources: Dict["PlateSourceKind", PlateReading] = field(default_factory=dict)
    # Consumed HikCentral GUIDs, so repeated queries cannot double-ingest.
    hik_guids_consumed: set[str] = field(default_factory=set)
    # Set when this identity's plate differs from a live identity that Re-ID
    # says is the same car — i.e. one of the two reads is probably a misread.
    # A MARKER ONLY: it never merges the identities, because letting appearance
    # override the plate key would put Re-ID back in charge of who a car is.
    correction_candidate_of: str = ""
    # NOTE: the primary_ocr_* / primary_blocks_fallback fields that used to sit
    # here are GONE. They existed to arbitrate a CAM-23 plate read against a
    # CAM-03 plate read — a question that no longer exists, because neither
    # camera is a plate source. Nothing replaced them: the plate is decided by
    # consensus across ANPR, HikCentral and our own OCR.

    @property
    def embeddings(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            embedding
            for attempt in self.attempts.values()
            for embedding in attempt.embeddings
        )

    def confirming_witnesses(self) -> set:
        """The witnesses that count toward the two-witness rule.

        HikCentral substitutes for a MISSING ANPR read; it never adds a second
        witness alongside one. A Hik pass record is the platform's log of the
        same gate event the ANPR camera already reported, so counting both
        would be counting one observation twice.
        """
        names = set(self.witnesses) | set(self.weak_votes)
        if WitnessSource.HIK in names and WitnessSource.ANPR in names:
            names.discard(WitnessSource.HIK)
        return names

    @property
    def colour_hsv(self) -> Optional[Tuple[float, float, float]]:
        """This identity's ground-truth body colour, from its ANPR image."""
        for attempt in self.attempts.values():
            for frame in attempt.evidence:
                if frame.colour_hsv is not None:
                    return frame.colour_hsv
        return None

    @property
    def cached_plate_evidence(self) -> Tuple[PlateEvidence, ...]:
        return tuple(
            item
            for attempt in self.attempts.values()
            for item in attempt.plate_evidence
        )

    @property
    def reported_keys(self) -> set[str]:
        return {
            plate_key(attempt.request.reported_plate)
            for attempt in self.attempts.values()
            if plate_key(attempt.request.reported_plate)
        }


@dataclass
class CrossingRecord:
    request: CrossingInput
    evidence: Tuple[FrameEvidence, ...]
    status: RecordStatus = RecordStatus.PENDING
    matched_group_id: Optional[str] = None
    last_decision_id: Optional[str] = None
    # Wall-clock ingest time, for the observation TTL. Not refreshed: an
    # observation is a point event, not a candidate that can be reactivated.
    created_at: Optional[datetime] = None
    # Highest Re-ID score every identity has ever reached against THIS
    # observation, kept so a margin cannot be won by the pool shrinking.
    #
    # Measured 2026-09-01: observation 3cce0428 ranked ABR8000 0.686 against
    # SNA226 0.627 and was correctly refused as ambiguous, twice. SNA226 then
    # hit its 15-minute identity TTL, and on the very next pass the SAME
    # observation with the SAME score to six decimals cleared the margin and
    # confirmed. Nothing about the evidence changed; the competitor merely got
    # old. Remembering the contest keeps it honest — a candidate that once
    # looked that good does not stop counting because it left the room.
    contested_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def witness(self) -> Optional["WitnessSource"]:
        """Which physical witness this observation is. Never a plate source."""
        return witness_for_camera(self.request.camera_id)

    @property
    def colour_hsv(self) -> Optional[Tuple[float, float, float]]:
        for frame in self.evidence:
            if frame.colour_hsv is not None:
                return frame.colour_hsv
        return None

    @property
    def embeddings(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(frame.embedding for frame in self.evidence if frame.embedding)

    @property
    def plate_evidence(self) -> Tuple[PlateEvidence, ...]:
        return tuple(frame.plate for frame in self.evidence)


@dataclass(frozen=True)
class ReIDMatch:
    group_id: str
    score: float
    row_margin: float
    column_margin: float


@dataclass(frozen=True)
class PlateResolution:
    outcome: str  # confirmed | abstained | pending
    reason: str
    canonical_plate: Optional[str] = None
    ocr_source: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    ocr_evidence_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryDecision:
    decision_id: str
    status: DecisionStatus
    reason: str
    group_id: str
    attempt_id: str
    crossing_id: str
    canonical_plate: Optional[str]
    reported_plate: Optional[str]
    reported_confidence: Optional[float]
    corrected: bool
    superseded_plates: Tuple[str, ...]
    entry_camera_id: str
    entry_captured_at: datetime
    reid_score: float
    reid_row_margin: float
    reid_column_margin: float
    ocr_source: Optional[str]
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    ocr_evidence_ids: Tuple[str, ...]
    finalizes_group: bool = False

    def callback_payload(self, mode: EntryMode) -> Dict[str, Any]:
        status = self.status.value
        reason = self.reason
        if mode == EntryMode.SHADOW and self.status == DecisionStatus.CONFIRMED:
            status = DecisionStatus.ABSTAINED.value
            reason = f"shadow_would_confirm:{reason}"
        return {
            "decision_id": self.decision_id,
            "status": status,
            "canonical_plate": self.canonical_plate,
            "attempt_id": self.attempt_id,
            "crossing_id": self.crossing_id,
            "entry_camera_id": self.entry_camera_id,
            "entry_captured_at": self.entry_captured_at.isoformat(),
            "reported_plate": self.reported_plate,
            # Hikvision/PMS wire confidence is 0..100. The coordinator keeps
            # 0..1 internally so threshold math has one scale.
            "plate_confidence": (
                int(round(self.reported_confidence * 100.0))
                if self.reported_confidence is not None
                else None
            ),
            "reason": reason,
            "corrected": self.corrected,
            "reid_score": self.reid_score,
            "reid_margin": min(self.reid_row_margin, self.reid_column_margin),
            "reid_row_margin": self.reid_row_margin,
            "reid_column_margin": self.reid_column_margin,
            "ocr_source": self.ocr_source,
            "ocr_text": self.ocr_text,
            "ocr_confidence": self.ocr_confidence,
            "ocr_evidence_ids": list(self.ocr_evidence_ids),
            "superseded_plates": list(self.superseded_plates),
        }


@dataclass(frozen=True)
class IngestResult:
    resource_id: str
    accepted: bool
    duplicate: bool
    mode: EntryMode
    evidence_count: int
    group_id: Optional[str] = None
    decision_id: Optional[str] = None
    decision_status: Optional[str] = None
    callback_delivered: Optional[bool] = None


class EntryError(Exception):
    """Base error carrying a stable API reason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class EntryUnavailable(EntryError):
    pass


class EntryCapacityExceeded(EntryError):
    pass


class EntryInvalid(EntryError):
    pass


class EntryConflict(EntryInvalid):
    pass


class EvidenceUnavailable(EntryInvalid):
    pass


EmbeddingSet = Sequence[Tuple[float, ...]]
