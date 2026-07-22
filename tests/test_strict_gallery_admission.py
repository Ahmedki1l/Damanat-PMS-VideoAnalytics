"""Fail-closed tests for durable parked-reference gallery admission.

Entry validation and parked-crop validation are independent witnesses.  These
tests pin the second boundary: generic tracking cannot write first, and the
strict writer needs both an authorised Entry V2 session and proof tied to the
exact parked crop.
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import MatchingConfig, load_config
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
from src.entry.identity import GalleryAuthorizationProof, RegistryIdentityPublisher
from src.entry.settings import EntrySettings
from src.vehicle_registry.gallery_store import VehicleGalleryStore, safe_plate
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_identity import (
    ParkedReferenceProof,
    RankedCandidate,
    SlotOcrPlan,
    parked_crop_fingerprint,
)
from src.vehicle_registry.vehicle_registry_models import VehicleSession


PLATE = "ABC-1234"
CAMERA = "CAM-04"
SLOT = "B1-S01"


class DeterministicMatcher:
    """Small cosine matcher whose extracted feature is controlled by a test."""

    backend = "test"

    def __init__(self, vector=(0.8, 0.6), on_extract=None):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.on_extract = on_extract
        self.batch_extract_calls = 0

    def extract_feature(self, image):
        assert image is not None and image.size > 0
        callback, self.on_extract = self.on_extract, None
        if callback is not None:
            callback()
        return self.vector.copy()

    def extract_features_batch(self, images):
        self.batch_extract_calls += 1
        return [self.extract_feature(image) for image in images]

    @staticmethod
    def compute_similarity(left, right):
        left = np.asarray(left, dtype=np.float32).reshape(-1)
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator else 0.0


def _textured_crop() -> np.ndarray:
    """Deterministic, sharp, plausible single-car-shaped image."""

    y, x = np.indices((128, 160))
    base = (((x // 4 + y // 4) % 2) * 180 + 30).astype(np.uint8)
    return np.dstack(
        (
            base,
            np.roll(base, 2, axis=1),
            np.roll(base, 3, axis=0),
        )
    )


def _strict_config() -> MatchingConfig:
    config = MatchingConfig()
    config.decision_log_enabled = False
    config.gallery_persist_enabled = True
    config.gallery_strict_admission_enabled = True
    config.gallery_parked_ocr_min_confidence = 0.90
    config.gallery_parked_reid_min_score = 0.70
    config.gallery_parked_reid_min_margin = 0.15
    config.gallery_parked_min_neighbour_clearance = 0.90
    config.gallery_parked_require_rank_one = True
    config.slot_lpd_enabled = True
    config.slot_lpd_fallback_enabled = False
    config.gallery_min_crop_area = 12_000.0
    config.gallery_min_sharpness = 40.0
    config.gallery_accumulate_min_gt_similarity = 0.45
    config.gallery_dedup_cosine = 0.97
    config.gallery_max_refs_per_car = 8
    config.use_faiss_index = False
    return config


def _session(**overrides) -> VehicleSession:
    values = {
        "session_id": "entryv2-session-1",
        "plate": PLATE,
        "feature_vector": np.asarray((1.0, 0.0), dtype=np.float32),
        "event_id": "decision-1",
        "candidate_id": "crossing-1",
        "status": "parked",
        "linked_slot": SLOT,
        "linked_camera": CAMERA,
        "ocr_confirmed": True,
        "gallery_entry_authorized": True,
        "gallery_entry_authorization_reason": "reid_and_primary_ocr_exact",
        "gallery_entry_proof": {
            "policy": "entry_v2_gallery_v1",
            "decision_id": "decision-1",
            "authorization_path": "exact_plate",
            "canonical_plate": PLATE,
            "crossing_id": "crossing-1",
            "crossing_camera_id": "CAM-23",
            "crossing_role": "primary",
            "reported_plate": PLATE,
            "reported_confidence": 0.95,
            "reid_score": 0.90,
            "reid_row_margin": 0.20,
            "reid_column_margin": 0.20,
            "ocr_source": "primary",
            "ocr_text": PLATE,
            "ocr_confidence": 0.95,
            "ocr_evidence_ids": ["crossing-1:0"],
            "corrected": False,
            "requires_parked_ocr": True,
        },
    }
    values.update(overrides)
    return VehicleSession(**values)


def _registry(tmp_path: Path, *, session=None):
    matcher = DeterministicMatcher()
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    registry._reid_matcher = matcher
    live = session or _session()
    registry._sessions[live.session_id] = live
    registry._parked[SLOT] = live
    return registry, live, matcher


def _proof(*, crop=None, **overrides) -> ParkedReferenceProof:
    proof_crop = _textured_crop() if crop is None else crop
    values = {
        "plate": PLATE,
        "session_id": "entryv2-session-1",
        "slot_id": SLOT,
        "camera_id": CAMERA,
        "crop_sha256": parked_crop_fingerprint(proof_crop),
        "view_quality": 0.95,
        "neighbour_clearance": 0.95,
        "ocr_text": "ABC1234",
        "ocr_confidence": 0.95,
        "reid_score": 0.85,
        "reid_margin": 0.20,
        "reid_rank": 1,
    }
    values.update(overrides)
    return ParkedReferenceProof(**values)


def _entry_authorization() -> GalleryAuthorizationProof:
    return GalleryAuthorizationProof(
        authorized=True,
        reason="reid_and_primary_ocr_exact",
        decision_id="decision-real-handoff",
        authorization_path="exact_plate",
        canonical_plate=PLATE,
        crossing_id="crossing-real-handoff",
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        reported_plate=PLATE,
        reported_confidence=0.95,
        reid_score=0.90,
        reid_row_margin=0.20,
        reid_column_margin=0.20,
        ocr_source="primary",
        ocr_text=PLATE,
        ocr_confidence=0.95,
        ocr_evidence_ids=("crossing-real-handoff:0",),
        corrected=False,
        requires_parked_ocr=True,
    )


def _store_admission(decision_id: str) -> dict:
    entry_proof = dict(_session().gallery_entry_proof)
    entry_proof["decision_id"] = decision_id
    return {
        "verified": True,
        "policy": "entry_v2_parked_v1",
        "entry_decision_id": decision_id,
        "entry_reason": "reid_and_primary_ocr_exact",
        "entry_proof": entry_proof,
        "parked_slot": SLOT,
        "parked_camera": CAMERA,
        "parked_ocr_text": "ABC1234",
        "parked_ocr_confidence": 0.95,
        "parked_reid_score": 0.85,
        "parked_reid_margin": 0.20,
        "parked_reid_rank": 1,
        "parked_view_quality": 0.95,
        "parked_neighbour_clearance": 0.95,
    }


class _EntryEvidenceProcessor:
    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        del images, metadata
        crossing = event_id.startswith("crossing-")
        return (
            FrameEvidence(
                evidence_id=f"{event_id}:0",
                embedding=(1.0, 0.0),
                plate=PlateEvidence(
                    evidence_id=f"{event_id}:0",
                    camera_id=camera_id,
                    source_role=source_role,
                    state=(
                        PlateReadState.READABLE
                        if crossing
                        else PlateReadState.NO_PLATE
                    ),
                    text="ABC1234" if crossing else "",
                    confidence=0.95 if crossing else 0.0,
                ),
            ),
        )


class _DeliveredSink:
    def deliver(self, payload):
        del payload
        return DeliveryResult(True, 1)


def _ranked(plate: str, session_id: str, score: float, rank: int):
    return RankedCandidate(
        plate=plate,
        session_id=session_id,
        score=score,
        same_view_score=score,
        cross_view_score=score,
        warm=True,
        rank=rank,
    )


def _plan(*candidates: RankedCandidate) -> SlotOcrPlan:
    return SlotOcrPlan(
        slot_id=SLOT,
        camera_id=CAMERA,
        candidates=[candidate.plate for candidate in candidates],
        kept=list(candidates),
        rejected=[],
        allow_retry=False,
        decision_ctx={
            "gallery_view_quality": 0.95,
            "gallery_neighbour_clearance": 0.95,
        },
    )


def _meta_path(tmp_path: Path) -> Path:
    return tmp_path / "gallery" / safe_plate(PLATE) / "meta.json"


def _assert_nothing_persisted(tmp_path: Path) -> None:
    assert not _meta_path(tmp_path).exists()


def test_build_proof_records_rank_one_score_and_runner_margin(tmp_path):
    registry, _, _ = _registry(tmp_path)
    plan = _plan(
        _ranked(PLATE, "entryv2-session-1", 0.86, 1),
        _ranked("XYZ-9876", "other-session", 0.61, 2),
    )

    proof = registry.build_parked_reference_proof(
        plan,
        PLATE,
        "ABC1234",
        0.94,
        _textured_crop(),
    )

    assert proof == ParkedReferenceProof(
        plate=PLATE,
        session_id="entryv2-session-1",
        slot_id=SLOT,
        camera_id=CAMERA,
        crop_sha256=parked_crop_fingerprint(_textured_crop()),
        view_quality=0.95,
        neighbour_clearance=0.95,
        ocr_text="ABC1234",
        ocr_confidence=0.94,
        reid_score=0.86,
        reid_margin=pytest.approx(0.25),
        reid_rank=1,
    )


def test_rejects_session_without_entry_authorization(tmp_path):
    registry, session, _ = _registry(
        tmp_path,
        session=_session(gallery_entry_authorized=False),
    )

    assert not registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=_proof()
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_rejects_missing_crop_local_proof(tmp_path):
    registry, session, _ = _registry(tmp_path)

    assert not registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=None
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("slot_lpd_enabled", False),
        ("slot_lpd_fallback_enabled", True),
    ],
)
def test_rejects_parked_ocr_without_fail_closed_plate_roi(
    tmp_path,
    field,
    value,
):
    registry, session, _ = _registry(tmp_path)
    setattr(registry._matching_config, field, value)

    assert not registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(),
    )

    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    "proof",
    [
        _proof(ocr_confidence=0.8999),
        _proof(reid_score=0.6999),
        _proof(reid_margin=0.1499),
        _proof(view_quality=0.8999),
        _proof(neighbour_clearance=0.8999),
        _proof(reid_rank=2),
        _proof(ocr_confidence=float("nan")),
        _proof(reid_score=float("inf")),
        _proof(reid_margin=float("nan")),
        _proof(view_quality=float("inf")),
        _proof(neighbour_clearance=float("nan")),
        _proof(reid_rank=1.0),
        _proof(session_id="different-session"),
        _proof(slot_id="different-slot"),
        _proof(camera_id="CAM-05"),
        _proof(crop_sha256="wrong-crop"),
    ],
    ids=[
        "low-ocr",
        "low-reid-score",
        "low-reid-margin",
        "low-view-quality",
        "low-neighbour-clearance",
        "not-rank-one",
        "nan-ocr",
        "infinite-reid-score",
        "nan-reid-margin",
        "infinite-view-quality",
        "nan-neighbour-clearance",
        "non-integer-rank",
        "wrong-session",
        "wrong-slot",
        "wrong-camera",
        "wrong-crop",
    ],
)
def test_rejects_proof_below_any_strict_parked_threshold(tmp_path, proof):
    registry, session, _ = _registry(tmp_path)

    assert not registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=proof
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_threshold_equality_writes_auditable_reference_and_unlocks_session(tmp_path):
    registry, session, _ = _registry(tmp_path)
    proof = _proof(
        ocr_confidence=0.90,
        reid_score=0.70,
        reid_margin=0.15,
        reid_rank=1,
    )

    assert registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=proof
    )

    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert metadata["plate"] == PLATE
    assert len(metadata["refs"]) == 1
    reference = metadata["refs"][0]
    assert reference["evidence_id"] == (
        "parked:decision-1:B1-S01:CAM-04"
    )
    assert reference["admission"] == {
        "verified": True,
        "policy": "entry_v2_parked_v1",
        "entry_decision_id": "decision-1",
        "entry_reason": "reid_and_primary_ocr_exact",
        "entry_proof": session.gallery_entry_proof,
        "parked_slot": SLOT,
        "parked_camera": CAMERA,
        "parked_ocr_text": "ABC1234",
        "parked_ocr_confidence": 0.90,
        # Recomputed from the exact crop immediately before commit, not copied
        # from the older pre-OCR plan proof.
        "parked_reid_score": pytest.approx(0.80),
        "parked_reid_margin": pytest.approx(0.80),
        "parked_reid_rank": 1,
        "parked_view_quality": 0.95,
        "parked_neighbour_clearance": 0.95,
    }
    plate_dir = _meta_path(tmp_path).parent
    assert (plate_dir / reference["crop"]).is_file()
    assert (plate_dir / reference["vec"]).is_file()
    assert session.gallery_parked_verified is True
    assert session.reference_source_cameras == [CAMERA]


def test_vacate_during_reid_is_revalidated_before_disk_commit(tmp_path):
    registry, session, matcher = _registry(tmp_path)
    matcher.on_extract = lambda: registry.unlink_slot(SLOT)

    assert not registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(),
    )

    assert session.linked_slot is None
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_stronger_candidate_appearing_during_ocr_blocks_stale_reid_proof(tmp_path):
    registry, session, matcher = _registry(tmp_path)

    def add_stronger_candidate():
        other = _session(
            session_id="new-stronger-session",
            plate="XYZ-9876",
            feature_vector=np.asarray((0.8, 0.6), dtype=np.float32),
            event_id="other-decision",
            candidate_id="other-crossing",
            status="confirmed",
            linked_slot=None,
            linked_camera=None,
            ocr_confirmed=False,
            gallery_entry_authorized=False,
            gallery_entry_proof={},
        )
        registry._sessions[other.session_id] = other

    matcher.on_extract = add_stronger_candidate

    assert not registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(),
    )

    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_real_entry_handoff_then_parked_proof_writes_restartable_reference(tmp_path):
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    registry._reid_matcher = DeterministicMatcher()
    session_id = registry.register_validated_entry(
        plate=PLATE,
        decision_id="decision-real-handoff",
        attempt_id="attempt-real-handoff",
        crossing_id="crossing-real-handoff",
        timestamp=datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((1.0, 0.0),),
        attempt_feature_vectors=(),
        gallery_authorization=_entry_authorization(),
    )

    session = registry._sessions[session_id]
    assert session.gallery_entry_authorized is True
    assert session.gallery_entry_proof["policy"] == "entry_v2_gallery_v1"
    assert session.gallery_entry_proof["decision_id"] == "decision-real-handoff"
    assert session.gallery_entry_proof["crossing_role"] == "primary"
    assert registry.bind_plate_to_slot(SLOT, PLATE, CAMERA, source="ocr")

    assert registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(session_id=session.session_id),
    )
    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    admission = metadata["refs"][0]["admission"]
    assert admission["entry_proof"] == session.gallery_entry_proof
    assert admission["policy"] == "entry_v2_parked_v1"


def test_authoritative_ack_to_parked_gallery_to_restart_journey(tmp_path):
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    registry._reid_matcher = DeterministicMatcher()
    settings = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-secret",
        va_process_count=1,
    )
    coordinator = EntryCoordinator(
        settings,
        _EntryEvidenceProcessor(),
        _DeliveredSink(),
        identity_publisher=RegistryIdentityPublisher(registry),
    )
    coordinator.ingest_attempt(
        AttemptInput(
            attempt_id="attempt-e2e",
            source_event_id="source-attempt-e2e",
            camera_id="ANPR-ENTRY",
            captured_at=datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
            reported_plate=PLATE,
            reported_confidence=0.95,
            metadata={},
        ),
        [b"attempt-image"],
    )
    coordinator.ingest_crossing(
        CrossingInput(
            crossing_id="crossing-e2e",
            source_event_id="source-crossing-e2e",
            camera_id="CAM-23",
            captured_at=datetime(2026, 7, 22, 12, 1, tzinfo=timezone.utc),
            line_id="RAMP-IN",
            direction="B-to-A",
            role=CrossingRole.PRIMARY,
            metadata={},
        ),
        [b"crossing-image"],
    )

    session = next(iter(registry._sessions.values()))
    assert session.plate == PLATE
    assert session.gallery_entry_authorized is True
    assert registry.bind_plate_to_slot(SLOT, PLATE, CAMERA, source="ocr")
    assert registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(session_id=session.session_id),
    )

    restarted = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    restarted._reid_matcher = DeterministicMatcher()
    return_coordinator = EntryCoordinator(
        settings,
        _EntryEvidenceProcessor(),
        _DeliveredSink(),
        identity_publisher=RegistryIdentityPublisher(restarted),
    )
    return_coordinator.ingest_attempt(
        AttemptInput(
            attempt_id="attempt-return",
            source_event_id="source-attempt-return",
            camera_id="ANPR-ENTRY",
            captured_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
            reported_plate=PLATE,
            reported_confidence=0.95,
            metadata={},
        ),
        [b"return-attempt-image"],
    )
    return_coordinator.ingest_crossing(
        CrossingInput(
            crossing_id="crossing-return",
            source_event_id="source-crossing-return",
            camera_id="CAM-23",
            captured_at=datetime(2026, 8, 1, 12, 1, tzinfo=timezone.utc),
            line_id="RAMP-IN",
            direction="B-to-A",
            role=CrossingRole.PRIMARY,
            metadata={},
        ),
        [b"return-crossing-image"],
    )

    returned = next(iter(restarted._sessions.values()))
    assert returned.plate == PLATE
    assert np.allclose(returned.feature_vector, (1.0, 0.0))
    assert any(
        np.allclose(reference, (0.8, 0.6))
        for reference in returned.reference_feature_vectors
    )
    assert CAMERA in returned.reference_source_cameras


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_id", "different-decision"),
        ("canonical_plate", "XYZ-9876"),
        ("crossing_id", "different-crossing"),
        ("crossing_camera_id", "CAM-03"),
        ("reported_confidence", float("nan")),
        ("reid_score", 0.84),
        ("reid_row_margin", 0.11),
        ("reid_column_margin", float("inf")),
        ("ocr_source", "anpr_cached"),
        ("ocr_confidence", 0.89),
        ("requires_parked_ocr", False),
    ],
)
def test_entry_authorization_must_bind_to_the_exact_registry_handoff(
    tmp_path,
    field,
    value,
):
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    registry._reid_matcher = DeterministicMatcher()
    authorization = replace(_entry_authorization(), **{field: value})
    session_id = registry.register_validated_entry(
        plate=PLATE,
        decision_id="decision-real-handoff",
        attempt_id="attempt-real-handoff",
        crossing_id="crossing-real-handoff",
        timestamp=datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc),
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((1.0, 0.0),),
        attempt_feature_vectors=(),
        gallery_authorization=authorization,
    )

    session = registry._sessions[session_id]
    assert session.gallery_entry_authorized is False
    assert session.gallery_entry_proof == {}
    assert registry.bind_plate_to_slot(SLOT, PLATE, CAMERA, source="ocr")
    assert not registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(),
    )
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "write_plate"),
    [
        ("plate", "XYZ-9876", "XYZ-9876"),
        ("event_id", "different-decision", PLATE),
    ],
)
def test_mutating_authorized_session_identity_cannot_reuse_old_entry_proof(
    tmp_path,
    field,
    value,
    write_plate,
):
    registry, session, _ = _registry(tmp_path)
    setattr(session, field, value)
    proof = _proof(
        plate=write_plate,
        ocr_text=("XYZ9876" if write_plate != PLATE else "ABC1234"),
    )

    assert not registry.save_parked_reference(
        write_plate,
        _textured_crop(),
        CAMERA,
        proof=proof,
    )
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    "ocr_text",
    ["XYZ9876", "1234"],
    ids=["mismatch", "ambiguous-active-plates"],
)
def test_rejects_ocr_mismatch_or_ambiguity(tmp_path, ocr_text):
    registry, session, _ = _registry(tmp_path)
    if ocr_text == "1234":
        other = _session(
            session_id="other-session",
            plate="XYZ-1234",
            feature_vector=np.asarray((0.0, 1.0), dtype=np.float32),
            status="confirmed",
            linked_slot=None,
            linked_camera=None,
        )
        registry._sessions[other.session_id] = other

    assert not registry.save_parked_reference(
        PLATE,
        _textured_crop(),
        CAMERA,
        proof=_proof(ocr_text=ocr_text),
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    ("session_overrides", "camera_id"),
    [
        ({"status": "confirmed"}, CAMERA),
        ({}, "CAM-05"),
    ],
    ids=["not-parked", "wrong-slot-camera"],
)
def test_rejects_wrong_session_status_or_camera(
    tmp_path,
    session_overrides,
    camera_id,
):
    registry, session, _ = _registry(
        tmp_path,
        session=_session(**session_overrides),
    )

    assert not registry.save_parked_reference(
        PLATE, _textured_crop(), camera_id, proof=_proof()
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize(
    "crop",
    [
        np.tile(_textured_crop(), (1, 3, 1)),
        np.full((128, 160, 3), 120, dtype=np.uint8),
    ],
    ids=["implausibly-wide", "blurred"],
)
def test_rejects_bad_geometry_or_blur(tmp_path, crop):
    registry, session, _ = _registry(tmp_path)

    assert not registry.save_parked_reference(
        PLATE, crop, CAMERA, proof=_proof(crop=crop)
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_rejects_crop_below_ground_truth_similarity_floor(tmp_path):
    registry, session, matcher = _registry(tmp_path)
    matcher.vector = np.asarray((0.0, 1.0), dtype=np.float32)

    assert not registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=_proof()
    )
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)


def test_generic_accumulation_is_always_blocked_in_strict_mode(tmp_path):
    registry, session, _ = _registry(tmp_path)

    assert not registry.accumulate_reference(
        session,
        _textured_crop(),
        CAMERA,
        view_quality=1.0,
    )
    assert session.gallery_entry_authorized is True
    assert session.gallery_parked_verified is False
    _assert_nothing_persisted(tmp_path)

    assert registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=_proof()
    )
    assert session.gallery_parked_verified is True
    assert not registry.accumulate_reference(
        session,
        np.flip(_textured_crop(), axis=1).copy(),
        CAMERA,
        view_quality=1.0,
    )
    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert len(metadata["refs"]) == 1


def test_parked_evidence_is_idempotent_and_conflicting_reuse_fails_closed(tmp_path):
    registry, session, matcher = _registry(tmp_path)
    crop = _textured_crop()
    proof = _proof()

    assert registry.save_parked_reference(PLATE, crop, CAMERA, proof=proof)
    assert not registry.save_parked_reference(PLATE, crop, CAMERA, proof=proof)

    first_metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert len(first_metadata["refs"]) == 1
    assert len(session.reference_feature_vectors) == 1

    changed_crop = crop.copy()
    changed_crop[20:36, 20:36] = 255 - changed_crop[20:36, 20:36]
    matcher.vector = np.asarray((0.6, 0.8), dtype=np.float32)
    assert not registry.save_parked_reference(
        PLATE,
        changed_crop,
        CAMERA,
        proof=replace(proof, reid_score=0.90),
    )

    final_metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert final_metadata == first_metadata
    assert len(session.reference_feature_vectors) == 1
    assert len(list(_meta_path(tmp_path).parent.glob("*.jpg"))) == 1
    assert len(list(_meta_path(tmp_path).parent.glob("*.npy"))) == 1


def test_verified_reference_survives_restart_and_warm_starts_identity(tmp_path):
    registry, _, _ = _registry(tmp_path)
    assert registry.save_parked_reference(
        PLATE, _textured_crop(), CAMERA, proof=_proof()
    )

    restarted = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    restarted._reid_matcher = DeterministicMatcher()

    session_id = restarted.build_session_from_gallery(PLATE)

    assert session_id is not None
    restored = restarted._sessions[session_id]
    assert restored.plate == PLATE
    assert restored.feature_vector is not None
    assert restarted.gallery_store.ref_count(PLATE) == 1


def test_strict_store_quarantines_legacy_unverified_history(tmp_path):
    crop = _textured_crop()
    vector = np.asarray((0.8, 0.6), dtype=np.float32)
    legacy = VehicleGalleryStore(str(tmp_path), "test:model", max_refs=8)
    assert legacy.save_ref(PLATE, crop, vector, quality=999.0) is not None

    strict = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        max_refs=8,
        require_verified_admission=True,
    )
    vectors, _, _ = strict.load_vectors(PLATE)
    assert vectors == []
    assert strict.save_ref(PLATE, crop, vector, quality=1.0) is None

    admission = _store_admission("decision-2")
    assert strict.save_ref(
        PLATE,
        np.flip(crop, axis=1).copy(),
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-2:B1-S01:CAM-04",
        admission=admission,
    ) is not None

    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert len(metadata["refs"]) == 1
    assert metadata["refs"][0]["admission"] == admission
    assert len(metadata["quarantined_refs"]) == 1
    assert strict.ref_count(PLATE) == 1
    active_dir = _meta_path(tmp_path).parent
    quarantine_dir = tmp_path / "gallery_quarantine" / safe_plate(PLATE)
    assert len(list(active_dir.glob("*.jpg"))) == 1
    assert len(list(active_dir.glob("*.npy"))) == 1
    assert len(list(quarantine_dir.glob("*.jpg"))) == 1
    assert len(list(quarantine_dir.glob("*.npy"))) == 1


def test_strict_quarantine_recovers_after_partial_move_interruption(
    tmp_path,
    monkeypatch,
):
    crop = _textured_crop()
    vector = np.asarray((0.8, 0.6), dtype=np.float32)
    legacy = VehicleGalleryStore(str(tmp_path), "test:model", max_refs=8)
    assert legacy.save_ref(PLATE, crop, vector, quality=999.0) is not None

    strict = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        max_refs=8,
        require_verified_admission=True,
    )

    def interrupt_after_first_move(plate, records):
        metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
        assert metadata["refs"] == []
        assert metadata["quarantined_refs"][0]["quarantine_state"] == "pending"
        record = records[0]
        quarantine_dir = tmp_path / "gallery_quarantine" / safe_plate(plate)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        source = _meta_path(tmp_path).parent / record["quarantine_source_files"][
            "crop"
        ]
        source.replace(quarantine_dir / record["crop"])
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(
        strict,
        "_move_quarantine_records",
        interrupt_after_first_move,
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        strict.save_ref(
            PLATE,
            np.flip(crop, axis=1).copy(),
            vector,
            quality=1.0,
            camera_id=CAMERA,
            evidence_id="parked:decision-crash:B1-S01:CAM-04",
            admission=_store_admission("decision-crash"),
        )

    interrupted = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert interrupted["refs"] == []
    assert interrupted["quarantined_refs"][0]["quarantine_state"] == "pending"
    assert strict.load_vectors(PLATE)[0] == []
    assert strict.ref_count(PLATE) == 0

    restarted = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        max_refs=8,
        require_verified_admission=True,
    )
    assert restarted.save_ref(
        PLATE,
        np.flip(crop, axis=1).copy(),
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-crash:B1-S01:CAM-04",
        admission=_store_admission("decision-crash"),
    ) is not None

    recovered = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert recovered["quarantined_refs"][0]["quarantine_state"] == "complete"
    quarantine_dir = tmp_path / "gallery_quarantine" / safe_plate(PLATE)
    assert len(list(quarantine_dir.glob("*.jpg"))) == 1
    assert len(list(quarantine_dir.glob("*.npy"))) == 1
    assert restarted.ref_count(PLATE) == 1


def test_strict_quarantine_recovers_when_completion_metadata_write_fails(
    tmp_path,
    monkeypatch,
):
    crop = _textured_crop()
    vector = np.asarray((0.8, 0.6), dtype=np.float32)
    legacy = VehicleGalleryStore(str(tmp_path), "test:model", max_refs=8)
    assert legacy.save_ref(PLATE, crop, vector, quality=999.0) is not None

    strict = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        max_refs=8,
        require_verified_admission=True,
    )
    original_write_meta = strict._write_meta
    write_calls = 0

    def fail_completion_write(plate, metadata):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            return False
        return original_write_meta(plate, metadata)

    monkeypatch.setattr(strict, "_write_meta", fail_completion_write)
    assert strict.save_ref(
        PLATE,
        np.flip(crop, axis=1).copy(),
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-meta-crash:B1-S01:CAM-04",
        admission=_store_admission("decision-meta-crash"),
    ) is None

    interrupted = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert interrupted["refs"] == []
    assert interrupted["quarantined_refs"][0]["quarantine_state"] == "pending"
    assert strict.ref_count(PLATE) == 0
    active_dir = _meta_path(tmp_path).parent
    assert list(active_dir.glob("*.jpg")) == []
    assert list(active_dir.glob("*.npy")) == []
    quarantine_dir = tmp_path / "gallery_quarantine" / safe_plate(PLATE)
    assert len(list(quarantine_dir.glob("*.jpg"))) == 1
    assert len(list(quarantine_dir.glob("*.npy"))) == 1

    restarted = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        max_refs=8,
        require_verified_admission=True,
    )
    assert restarted.save_ref(
        PLATE,
        np.flip(crop, axis=1).copy(),
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-meta-crash:B1-S01:CAM-04",
        admission=_store_admission("decision-meta-crash"),
    ) is not None
    recovered = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert recovered["quarantined_refs"][0]["quarantine_state"] == "complete"
    assert restarted.ref_count(PLATE) == 1


def test_strict_store_rejects_parked_ocr_for_a_different_plate(tmp_path):
    store = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        require_verified_admission=True,
    )
    admission = _store_admission("decision-wrong-parked-ocr")
    admission["parked_ocr_text"] = "XYZ-9876"

    assert store.save_ref(
        PLATE,
        _textured_crop(),
        np.asarray((0.8, 0.6), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-wrong-parked-ocr:B1-S01:CAM-04",
        admission=admission,
    ) is None
    _assert_nothing_persisted(tmp_path)


@pytest.mark.parametrize("file_kind", ["crop", "vec"])
def test_strict_store_rejects_corrupted_verified_files_on_reload(
    tmp_path,
    file_kind,
):
    store = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        require_verified_admission=True,
    )
    filename = store.save_ref(
        PLATE,
        _textured_crop(),
        np.asarray((0.8, 0.6), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-3:B1-S01:CAM-04",
        admission=_store_admission("decision-3"),
    )
    assert filename is not None
    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    reference = metadata["refs"][0]
    target_path = _meta_path(tmp_path).parent / reference[file_kind]
    target_path.write_bytes(b"corrupted")

    vectors, _, _ = store.load_vectors(PLATE)
    crops, _ = store.load_crops(PLATE)

    assert vectors == []
    assert crops == []


def test_strict_store_idempotency_binds_both_image_and_embedding(tmp_path):
    store = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        require_verified_admission=True,
    )
    crop = _textured_crop()
    evidence_id = "parked:decision-4:B1-S01:CAM-04"
    admission = _store_admission("decision-4")
    first = store.save_ref(
        PLATE,
        crop,
        np.asarray((0.8, 0.6), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id=evidence_id,
        admission=admission,
    )
    assert first is not None
    assert store.save_ref(
        PLATE,
        crop,
        np.asarray((0.8, 0.6), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id=evidence_id,
        admission=admission,
    ) == first
    assert store.save_ref(
        PLATE,
        crop,
        np.asarray((0.6, 0.8), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id=evidence_id,
        admission=admission,
    ) is None
    assert store.ref_count(PLATE) == 1


def test_missing_idempotent_file_is_repaired_without_duplicate_reference(tmp_path):
    store = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        require_verified_admission=True,
    )
    crop = _textured_crop()
    vector = np.asarray((0.8, 0.6), dtype=np.float32)
    evidence_id = "parked:decision-5:B1-S01:CAM-04"
    admission = _store_admission("decision-5")
    assert store.save_ref(
        PLATE,
        crop,
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id=evidence_id,
        admission=admission,
    )
    metadata = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    (_meta_path(tmp_path).parent / metadata["refs"][0]["crop"]).unlink()

    assert store.save_ref(
        PLATE,
        crop,
        vector,
        quality=1.0,
        camera_id=CAMERA,
        evidence_id=evidence_id,
        admission=admission,
    )

    repaired = json.loads(_meta_path(tmp_path).read_text(encoding="utf-8"))
    assert len(repaired["refs"]) == 1
    assert len(list(_meta_path(tmp_path).parent.glob("*.jpg"))) == 1
    assert len(list(_meta_path(tmp_path).parent.glob("*.npy"))) == 1


def test_copied_gallery_cannot_be_loaded_under_a_different_plate(tmp_path):
    store = VehicleGalleryStore(
        str(tmp_path),
        "test:model",
        require_verified_admission=True,
    )
    assert store.save_ref(
        PLATE,
        _textured_crop(),
        np.asarray((0.8, 0.6), dtype=np.float32),
        quality=1.0,
        camera_id=CAMERA,
        evidence_id="parked:decision-6:B1-S01:CAM-04",
        admission=_store_admission("decision-6"),
    )
    other_plate = "XYZ-9876"
    copied_dir = tmp_path / "gallery" / safe_plate(other_plate)
    shutil.copytree(_meta_path(tmp_path).parent, copied_dir)
    copied_meta_path = copied_dir / "meta.json"
    copied_meta = json.loads(copied_meta_path.read_text(encoding="utf-8"))
    copied_meta["plate"] = other_plate
    copied_meta_path.write_text(json.dumps(copied_meta), encoding="utf-8")

    vectors, _, _ = store.load_vectors(other_plate)
    crops, _ = store.load_crops(other_plate)

    assert vectors == []
    assert crops == []


def test_mixed_model_vectors_force_crop_reembedding_on_restart(tmp_path):
    old_store = VehicleGalleryStore(
        str(tmp_path),
        "test:old-model",
        max_refs=8,
        require_verified_admission=True,
    )
    assert old_store.save_ref(
        PLATE,
        _textured_crop(),
        np.asarray((1.0, 0.0), dtype=np.float32),
        quality=0.95,
        camera_id=CAMERA,
        evidence_id="parked:decision-7:B1-S01:CAM-04",
        admission=_store_admission("decision-7"),
    )
    restarted = VehicleRegistry(
        image_dir=str(tmp_path),
        matching_config=_strict_config(),
    )
    matcher = DeterministicMatcher()
    restarted._reid_matcher = matcher
    current_tag = restarted.gallery_store._model_tag
    current_store = VehicleGalleryStore(
        str(tmp_path),
        current_tag,
        max_refs=8,
        require_verified_admission=True,
    )
    assert current_store.save_ref(
        PLATE,
        np.flip(_textured_crop(), axis=1).copy(),
        np.asarray((0.0, 1.0), dtype=np.float32),
        quality=0.96,
        camera_id=CAMERA,
        evidence_id="parked:decision-8:B1-S01:CAM-04",
        admission=_store_admission("decision-8"),
    )
    raw_vectors, stored_tag, _ = current_store.load_vectors(PLATE)
    assert len(raw_vectors) == 2
    assert stored_tag is not None and stored_tag.startswith("mixed:")

    restored = restarted._load_persisted_gallery_references(PLATE)

    assert len(restored) == 2
    assert matcher.batch_extract_calls == 1
    assert all(np.allclose(vector, matcher.vector) for vector, _ in restored)


def test_same_model_path_content_change_forces_crop_reembedding(tmp_path):
    model_dir = tmp_path / "reid-model"
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("<model version='one'/>", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"first-model-weights")
    first_config = _strict_config()
    first_config.reid_openvino_model_dir = str(model_dir)
    first = VehicleRegistry(image_dir=str(tmp_path), matching_config=first_config)
    first._reid_matcher = DeterministicMatcher()
    first_store = first.gallery_store
    first_tag = first_store._model_tag
    assert first_store.save_ref(
        PLATE,
        _textured_crop(),
        np.asarray((1.0, 0.0), dtype=np.float32),
        quality=0.95,
        camera_id=CAMERA,
        evidence_id="parked:decision-9:B1-S01:CAM-04",
        admission=_store_admission("decision-9"),
    )

    (model_dir / "model.bin").write_bytes(b"second-model-weights")
    second_config = _strict_config()
    second_config.reid_openvino_model_dir = str(model_dir)
    second = VehicleRegistry(image_dir=str(tmp_path), matching_config=second_config)
    matcher = DeterministicMatcher()
    second._reid_matcher = matcher
    second_tag = second.gallery_store._model_tag

    restored = second._load_persisted_gallery_references(PLATE)

    assert second_tag != first_tag
    assert len(restored) == 1
    assert matcher.batch_extract_calls == 1


def test_reid_preprocessing_contract_change_invalidates_cached_vectors(tmp_path):
    config = _strict_config()
    first = VehicleRegistry(image_dir=str(tmp_path), matching_config=config)
    first_matcher = DeterministicMatcher()
    first_matcher.input_size = (256, 128)
    first_matcher.norm_mean = (0.485, 0.456, 0.406)
    first_matcher.norm_std = (0.229, 0.224, 0.225)
    first_matcher.preprocessing_config = SimpleNamespace(
        enabled=False,
        clip_limit=2.0,
        grid_size=(8, 8),
    )
    first._reid_matcher = first_matcher
    first_tag = first.gallery_store._model_tag

    second = VehicleRegistry(image_dir=str(tmp_path), matching_config=config)
    second_matcher = DeterministicMatcher()
    second_matcher.input_size = (192, 96)
    second_matcher.norm_mean = first_matcher.norm_mean
    second_matcher.norm_std = first_matcher.norm_std
    second_matcher.preprocessing_config = first_matcher.preprocessing_config
    second._reid_matcher = second_matcher

    assert second.gallery_store._model_tag != first_tag


def test_strict_gallery_yaml_booleans_and_thresholds_are_typed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
matching:
  gallery_strict_admission_enabled: "true"
  gallery_parked_require_rank_one: "true"
  gallery_parked_ocr_min_confidence: 0.91
  gallery_parked_reid_min_score: 0.72
  gallery_parked_reid_min_margin: 0.16
  gallery_parked_min_neighbour_clearance: 0.97
  gallery_neighbour_clearance_enforce: "false"
  gallery_require_slot_authority: "true"
  slot_lpd_enabled: "true"
  slot_lpd_fallback_enabled: "false"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.matching.gallery_strict_admission_enabled is True
    assert config.matching.gallery_parked_require_rank_one is True
    assert config.matching.gallery_parked_ocr_min_confidence == pytest.approx(0.91)
    assert config.matching.gallery_parked_reid_min_score == pytest.approx(0.72)
    assert config.matching.gallery_parked_reid_min_margin == pytest.approx(0.16)
    assert config.matching.gallery_parked_min_neighbour_clearance == pytest.approx(
        0.97
    )
    assert config.matching.gallery_neighbour_clearance_enforce is False
    assert config.matching.gallery_require_slot_authority is True
    assert config.matching.slot_lpd_enabled is True
    assert config.matching.slot_lpd_fallback_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gallery_parked_ocr_min_confidence", 1.01),
        ("gallery_parked_reid_min_score", -1.01),
        ("gallery_parked_reid_min_margin", 2.01),
        ("gallery_parked_min_neighbour_clearance", 1.01),
    ],
)
def test_strict_gallery_yaml_rejects_out_of_range_thresholds(
    tmp_path,
    field,
    value,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"matching:\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gallery_strict_admission_enabled", "false"),
        ("gallery_require_slot_authority", "false"),
        ("gallery_parked_require_rank_one", "false"),
        ("slot_lpd_enabled", "false"),
        ("slot_lpd_fallback_enabled", "true"),
    ],
)
def test_strict_persistent_gallery_rejects_unsafe_ocr_or_rank_mode(
    tmp_path,
    field,
    value,
):
    values = {
        "gallery_persist_enabled": "true",
        "gallery_strict_admission_enabled": "true",
        "gallery_parked_require_rank_one": "true",
        "slot_lpd_enabled": "true",
        "slot_lpd_fallback_enabled": "false",
    }
    values[field] = value
    config_path = tmp_path / "config.yaml"
    body = "\n".join(f'  {key}: "{item}"' for key, item in values.items())
    config_path.write_text(f"matching:\n{body}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_config(str(config_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gallery_parked_ocr_min_confidence", 0.8999),
        ("gallery_parked_reid_min_score", 0.6999),
        ("gallery_parked_reid_min_margin", 0.1499),
        ("gallery_min_view_quality", 0.8999),
        ("gallery_parked_min_neighbour_clearance", 0.8999),
        ("gallery_accumulate_min_gt_similarity", 0.4499),
        ("gallery_min_sharpness", 39.9),
        ("gallery_min_crop_area", 11_999.0),
        ("slot_lpd_confidence", 0.2999),
    ],
)
def test_strict_persistent_gallery_rejects_below_policy_floor(
    tmp_path,
    field,
    value,
):
    values = {
        "gallery_persist_enabled": True,
        "gallery_strict_admission_enabled": True,
        "gallery_parked_require_rank_one": True,
        "slot_lpd_enabled": True,
        "slot_lpd_fallback_enabled": False,
        field: value,
    }
    config_path = tmp_path / "config.yaml"
    body = "\n".join(
        f"  {key}: {str(item).lower()}" for key, item in values.items()
    )
    config_path.write_text(f"matching:\n{body}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_config(str(config_path))


def test_snapshot_path_env_applies_without_output_yaml_block(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    durable_path = tmp_path / "durable-gallery"
    monkeypatch.setenv("SNAPSHOT_PATH", str(durable_path))

    config = load_config(str(config_path))

    assert config.output.snapshot_base_dir == str(durable_path)


def test_snapshot_path_env_overrides_yaml_output_path(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "output:\n  snapshot_base_dir: container-local\n",
        encoding="utf-8",
    )
    durable_path = tmp_path / "durable-gallery"
    monkeypatch.setenv("SNAPSHOT_PATH", str(durable_path))

    config = load_config(str(config_path))

    assert config.output.snapshot_base_dir == str(durable_path)


def test_snapshot_path_env_applies_when_yaml_file_is_missing(tmp_path, monkeypatch):
    durable_path = tmp_path / "durable-gallery"
    monkeypatch.setenv("SNAPSHOT_PATH", str(durable_path))

    config = load_config(str(tmp_path / "missing.yaml"))

    assert config.output.snapshot_base_dir == str(durable_path)
