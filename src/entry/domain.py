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
    # WitnessSource -> the id that produced it. Two entries confirm an entry.
    witnesses: Dict["WitnessSource", str] = field(default_factory=dict)
    # PlateSourceKind -> that source's reading. At most one per source.
    plate_sources: Dict["PlateSourceKind", PlateReading] = field(default_factory=dict)
    # Consumed HikCentral GUIDs, so repeated queries cannot double-ingest.
    hik_guids_consumed: set[str] = field(default_factory=set)
    # Set when this identity's plate differs from a live identity that Re-ID
    # says is the same car — i.e. one of the two reads is probably a misread.
    # A MARKER ONLY: it never merges the identities, because letting appearance
    # override the plate key would put Re-ID back in charge of who a car is.
    correction_candidate_of: str = ""
    # A reliable/conflicting primary read blocks a downstream fallback. When
    # CAM23 emits no usable event, a separately configured downstream crossing
    # remains independent physical-entry proof; there is no timer-based wait.
    primary_ocr_state: Optional[str] = None
    primary_blocks_fallback: bool = False
    # Reliable CAM23 reads accumulate across distinct crossing events. A later
    # contradictory value must not erase the first one by arrival order.
    primary_reliable_ocr_keys: set[str] = field(default_factory=set)
    primary_ocr_evidence_ids: set[str] = field(default_factory=set)
    primary_ocr_conflicted: bool = False

    @property
    def embeddings(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(
            embedding
            for attempt in self.attempts.values()
            for embedding in attempt.embeddings
        )

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

    @property
    def witness(self) -> Optional["WitnessSource"]:
        """Which physical witness this observation is. Never a plate source."""
        return witness_for_camera(self.request.camera_id)

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
