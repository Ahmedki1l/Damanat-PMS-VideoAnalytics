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
    group_id: str
    attempts: Dict[str, AttemptRecord] = field(default_factory=dict)
    hypotheses: Dict[str, PlateHypothesis] = field(default_factory=dict)
    status: RecordStatus = RecordStatus.PENDING
    canonical_plate: Optional[str] = None
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
