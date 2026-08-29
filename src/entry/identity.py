"""Handoff of a validated entry identity into VA's live tracker registry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Protocol, Tuple

from src.matching.plate_ocr_match import is_plausible_plate
from src.vehicle_registry.errors import ValidatedEntrySupersededByExit

from .decision import OCR_SOURCE_CONSENSUS, REASON_CONSENSUS, REASON_CORRECTION

from .domain import (
    CrossingRole,
    DecisionStatus,
    EntryDecision,
    norm_camera_id,
    plate_key,
)
from .settings import EntrySettings


Embedding = Tuple[float, ...]
CameraEmbedding = Tuple[str, Embedding]


@dataclass(frozen=True)
class GalleryAuthorizationProof:
    """Immutable evidence that a decision cleared durable-gallery policy.

    Presence of this object means the high-confidence entry gates passed. Every
    path still requires a later parked-plate OCR/ReID witness before a registry
    implementation may persist the exact parked crop.
    """

    authorized: bool
    reason: str
    decision_id: str
    authorization_path: Literal["exact_plate", "corrected_plate"]
    canonical_plate: str
    crossing_id: str
    crossing_camera_id: str
    crossing_role: CrossingRole
    reported_plate: Optional[str]
    reported_confidence: Optional[float]
    reid_score: float
    reid_row_margin: float
    reid_column_margin: float
    ocr_source: str
    ocr_text: str
    ocr_confidence: float
    ocr_evidence_ids: Tuple[str, ...]
    corrected: bool
    requires_parked_ocr: bool


def build_gallery_authorization_proof(
    decision: EntryDecision,
    *,
    crossing_camera_id: str,
    crossing_role: CrossingRole,
    settings: EntrySettings,
) -> Optional[GalleryAuthorizationProof]:
    """Return a proof only when the stricter durable-gallery gates pass."""

    if decision.status != DecisionStatus.CONFIRMED:
        return None
    canonical = decision.canonical_plate or ""
    if not canonical or not is_plausible_plate(canonical):
        return None

    # The CAMERA check stands: only an approved ramp camera's crop may be
    # taught to the durable gallery. What is gone is the expectation that the
    # plate came FROM that camera — it never does now. The plate is decided by
    # consensus across ANPR, HikCentral and our own OCR, so ocr_source names
    # the mechanism rather than a camera role.
    camera_id = norm_camera_id(crossing_camera_id)
    if crossing_role == CrossingRole.PRIMARY:
        allowed_camera = camera_id == "CAM23" and camera_id in settings.primary_cameras
    elif crossing_role == CrossingRole.FALLBACK:
        allowed_camera = camera_id == "CAM03" and camera_id in settings.fallback_cameras
    else:  # pragma: no cover - enum currently has exactly two members
        return None
    if not allowed_camera or decision.ocr_source != OCR_SOURCE_CONSENSUS:
        return None

    numeric_evidence = (
        (decision.reid_score, settings.gallery_reid_min_score),
        (decision.reid_row_margin, settings.gallery_reid_row_margin),
        (decision.reid_column_margin, settings.gallery_reid_column_margin),
        (decision.ocr_confidence, settings.gallery_ocr_min_confidence),
    )
    if any(
        value is None or not math.isfinite(value) or value < threshold
        for value, threshold in numeric_evidence
    ):
        return None
    if plate_key(decision.ocr_text or "") != plate_key(canonical):
        return None
    if not decision.ocr_evidence_ids:
        return None

    if decision.reason == REASON_CONSENSUS:
        confidence = decision.reported_confidence
        if (
            confidence is None
            or not math.isfinite(confidence)
            or confidence < settings.gallery_anpr_min_confidence
            or plate_key(decision.reported_plate or "") != plate_key(canonical)
        ):
            return None
        path: Literal["exact_plate", "corrected_plate"] = "exact_plate"
        requires_parked_ocr = True
    elif decision.reason == REASON_CORRECTION:
        if (
            not decision.corrected
            or len(set(decision.ocr_evidence_ids)) < settings.correction_min_evidence
        ):
            return None
        path = "corrected_plate"
        requires_parked_ocr = True
    else:
        # In particular, cached ANPR OCR is valid for live entry confirmation
        # but is not an independent enough witness for durable gallery teaching.
        return None

    return GalleryAuthorizationProof(
        authorized=True,
        reason=decision.reason,
        decision_id=decision.decision_id,
        authorization_path=path,
        canonical_plate=canonical,
        crossing_id=decision.crossing_id,
        crossing_camera_id=crossing_camera_id,
        crossing_role=crossing_role,
        reported_plate=decision.reported_plate,
        reported_confidence=decision.reported_confidence,
        reid_score=decision.reid_score,
        reid_row_margin=decision.reid_row_margin,
        reid_column_margin=decision.reid_column_margin,
        ocr_source=decision.ocr_source or "",
        ocr_text=decision.ocr_text or "",
        ocr_confidence=float(decision.ocr_confidence),
        ocr_evidence_ids=tuple(decision.ocr_evidence_ids),
        corrected=decision.corrected,
        requires_parked_ocr=requires_parked_ocr,
    )


@dataclass(frozen=True)
class ValidatedIdentity:
    """Metadata and embeddings retained after image inference, never pixels."""

    decision_id: str
    canonical_plate: str
    attempt_id: str
    crossing_id: str
    entered_at: datetime
    crossing_camera_id: str
    crossing_embeddings: Tuple[Embedding, ...]
    attempt_embeddings: Tuple[CameraEmbedding, ...]
    gallery_authorization: Optional[GalleryAuthorizationProof] = None


class IdentityPublisher(Protocol):
    def publish(self, identity: ValidatedIdentity) -> None: ...


class IdentitySupersededByExit(Exception):
    """The authoritative exit won the post-ACK registry publication race."""

    def __init__(self, latest_exit_at: datetime):
        super().__init__("validated entry superseded by exit")
        self.latest_exit_at = latest_exit_at


class NullIdentityPublisher:
    def publish(self, identity: ValidatedIdentity) -> None:
        del identity


class RegistryIdentityPublisher:
    """Seed the shared in-process VehicleRegistry without storing an image."""

    def __init__(self, registry):
        self._registry = registry

    def publish(self, identity: ValidatedIdentity) -> None:
        try:
            self._registry.register_validated_entry(
                plate=identity.canonical_plate,
                decision_id=identity.decision_id,
                attempt_id=identity.attempt_id,
                crossing_id=identity.crossing_id,
                timestamp=identity.entered_at,
                crossing_camera_id=identity.crossing_camera_id,
                crossing_feature_vectors=identity.crossing_embeddings,
                attempt_feature_vectors=identity.attempt_embeddings,
                gallery_authorization=identity.gallery_authorization,
            )
        except ValidatedEntrySupersededByExit as exc:
            raise IdentitySupersededByExit(exc.latest_exit_at) from exc
