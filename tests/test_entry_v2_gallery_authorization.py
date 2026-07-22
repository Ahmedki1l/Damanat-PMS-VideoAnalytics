from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import pytest

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    AttemptInput,
    CrossingInput,
    CrossingRole,
    DecisionStatus,
    EntryDecision,
    EntryMode,
    EntryUnavailable,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.identity import (
    GalleryAuthorizationProof,
    RegistryIdentityPublisher,
    build_gallery_authorization_proof,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)


def _settings(**overrides):
    base = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B-IN"}),
        fallback_directions=frozenset({"b-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="secret",
        va_process_count=1,
    )
    return replace(base, **overrides)


def _decision(**overrides):
    base = EntryDecision(
        decision_id="decision-1",
        status=DecisionStatus.CONFIRMED,
        reason="reid_and_primary_ocr_exact",
        group_id="group-1",
        attempt_id="attempt-1",
        crossing_id="crossing-1",
        canonical_plate="ABC-1234",
        reported_plate="ABC-1234",
        reported_confidence=0.90,
        corrected=False,
        superseded_plates=(),
        entry_camera_id="ANPR-ENTRY",
        entry_captured_at=NOW,
        reid_score=0.85,
        reid_row_margin=0.12,
        reid_column_margin=0.12,
        ocr_source="primary",
        ocr_text="ABC-1234",
        ocr_confidence=0.90,
        ocr_evidence_ids=("crossing-1:0",),
        finalizes_group=True,
    )
    return replace(base, **overrides)


def test_gallery_threshold_defaults_and_environment_overrides(monkeypatch):
    names = {
        "ENTRY_V2_GALLERY_ANPR_MIN_CONFIDENCE": "0.91",
        "ENTRY_V2_GALLERY_REID_MIN_SCORE": "0.86",
        "ENTRY_V2_GALLERY_REID_ROW_MARGIN": "0.13",
        "ENTRY_V2_GALLERY_REID_COLUMN_MARGIN": "0.14",
        "ENTRY_V2_GALLERY_OCR_MIN_CONFIDENCE": "0.92",
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)

    defaults = EntrySettings.from_env()
    assert defaults.gallery_anpr_min_confidence == pytest.approx(0.90)
    assert defaults.gallery_reid_min_score == pytest.approx(0.85)
    assert defaults.gallery_reid_row_margin == pytest.approx(0.12)
    assert defaults.gallery_reid_column_margin == pytest.approx(0.12)
    assert defaults.gallery_ocr_min_confidence == pytest.approx(0.90)

    for name, value in names.items():
        monkeypatch.setenv(name, value)
    configured = EntrySettings.from_env()
    assert configured.gallery_anpr_min_confidence == pytest.approx(0.91)
    assert configured.gallery_reid_min_score == pytest.approx(0.86)
    assert configured.gallery_reid_row_margin == pytest.approx(0.13)
    assert configured.gallery_reid_column_margin == pytest.approx(0.14)
    assert configured.gallery_ocr_min_confidence == pytest.approx(0.92)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gallery_anpr_min_confidence", -0.01),
        ("gallery_anpr_min_confidence", 0.8999),
        ("gallery_anpr_min_confidence", 1.01),
        ("gallery_reid_min_score", float("nan")),
        ("gallery_reid_min_score", 0.8499),
        ("gallery_reid_min_score", 1.01),
        ("gallery_reid_row_margin", -0.01),
        ("gallery_reid_row_margin", 0.1199),
        ("gallery_reid_column_margin", 0.1199),
        ("gallery_reid_column_margin", 2.01),
        ("gallery_ocr_min_confidence", 0.8999),
        ("gallery_ocr_min_confidence", float("inf")),
    ],
)
def test_gallery_threshold_validation_rejects_unsafe_values(field, value):
    settings = _settings(**{field: value})

    assert field in settings.configuration_errors()


def test_exact_primary_threshold_equality_authorizes_immutable_proof():
    proof = build_gallery_authorization_proof(
        _decision(),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert isinstance(proof, GalleryAuthorizationProof)
    assert proof.authorized is True
    assert proof.reason == "reid_and_primary_ocr_exact"
    assert proof.authorization_path == "exact_plate"
    assert proof.requires_parked_ocr is True
    with pytest.raises(FrozenInstanceError):
        proof.authorized = False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reported_confidence", 0.8999),
        ("reid_score", 0.8499),
        ("reid_row_margin", 0.1199),
        ("reid_column_margin", 0.1199),
        ("ocr_confidence", 0.8999),
    ],
)
def test_exact_path_rejects_any_value_below_gallery_threshold(field, value):
    proof = build_gallery_authorization_proof(
        _decision(**{field: value}),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert proof is None


def test_exact_path_requires_anpr_confidence_and_rejects_cached_ocr():
    missing_confidence = build_gallery_authorization_proof(
        _decision(reported_confidence=None),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )
    cached = build_gallery_authorization_proof(
        _decision(
            reason="reid_and_anpr_cached_ocr_exact",
            ocr_source="anpr_cached",
        ),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert missing_confidence is None
    assert cached is None


def test_exact_path_rejects_digit_first_reordering_as_a_different_identity():
    proof = build_gallery_authorization_proof(
        _decision(reported_plate="1234ABC"),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert proof is None


def test_configured_cam03_fallback_can_authorize_exact_plate():
    proof = build_gallery_authorization_proof(
        _decision(
            reason="reid_and_fallback_ocr_exact",
            ocr_source="fallback",
        ),
        crossing_camera_id="cam_03",
        crossing_role=CrossingRole.FALLBACK,
        settings=_settings(),
    )

    assert proof is not None
    assert proof.crossing_camera_id == "cam_03"
    assert proof.crossing_role == CrossingRole.FALLBACK


def test_unconfigured_or_wrong_role_crossing_cannot_authorize_gallery():
    unconfigured = build_gallery_authorization_proof(
        _decision(
            reason="reid_and_fallback_ocr_exact",
            ocr_source="fallback",
        ),
        crossing_camera_id="CAM-03",
        crossing_role=CrossingRole.FALLBACK,
        settings=_settings(fallback_cameras=frozenset()),
    )
    wrong_role = build_gallery_authorization_proof(
        _decision(),
        crossing_camera_id="CAM-03",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert unconfigured is None
    assert wrong_role is None


def test_independent_correction_is_deferred_until_parked_ocr():
    proof = build_gallery_authorization_proof(
        _decision(
            reason="reid_and_independent_ocr_correction",
            canonical_plate="XYZ-9999",
            reported_plate="ABC-1234",
            reported_confidence=None,
            corrected=True,
            superseded_plates=("ABC-1234",),
            ocr_text="XYZ-9999",
            ocr_evidence_ids=("attempt-1:0", "crossing-1:0"),
        ),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert proof is not None
    assert proof.authorization_path == "corrected_plate"
    assert proof.requires_parked_ocr is True
    assert proof.reported_confidence is None


def test_correction_without_existing_consensus_proof_is_not_authorized():
    proof = build_gallery_authorization_proof(
        _decision(
            reason="reid_and_independent_ocr_correction",
            canonical_plate="XYZ-9999",
            reported_plate="ABC-1234",
            corrected=True,
            ocr_text="XYZ-9999",
            ocr_evidence_ids=("crossing-1:0",),
        ),
        crossing_camera_id="CAM-23",
        crossing_role=CrossingRole.PRIMARY,
        settings=_settings(),
    )

    assert proof is None


class _Processor:
    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        del images, metadata
        is_crossing = event_id == "crossing-1"
        plate = PlateEvidence(
            evidence_id=f"{event_id}:0",
            camera_id=camera_id,
            source_role=source_role,
            state=(PlateReadState.READABLE if is_crossing else PlateReadState.NO_PLATE),
            text="ABC1234" if is_crossing else "",
            confidence=0.95 if is_crossing else 0.0,
        )
        return (
            FrameEvidence(
                evidence_id=f"{event_id}:0",
                embedding=(1.0, 0.0),
                plate=plate,
            ),
        )


class _Sink:
    def __init__(self, delivered):
        self.delivered = delivered

    def deliver(self, payload):
        del payload
        return DeliveryResult(self.delivered, 1, "" if self.delivered else "down")


class _Registry:
    def __init__(self):
        self.calls = []

    def register_validated_entry(self, **kwargs):
        self.calls.append(kwargs)
        return "session-1"


def _attempt():
    return AttemptInput(
        attempt_id="attempt-1",
        source_event_id="source-attempt-1",
        camera_id="ANPR-ENTRY",
        captured_at=NOW,
        reported_plate="ABC-1234",
        reported_confidence=0.95,
        metadata={},
    )


def _crossing():
    return CrossingInput(
        crossing_id="crossing-1",
        source_event_id="source-crossing-1",
        camera_id="CAM-23",
        captured_at=NOW,
        line_id="RAMP-IN",
        direction="B-to-A",
        role=CrossingRole.PRIMARY,
        metadata={},
    )


@pytest.mark.parametrize("delivered", [True, False])
def test_authorization_is_forwarded_only_after_successful_pms_ack(delivered):
    registry = _Registry()
    coordinator = EntryCoordinator(
        _settings(),
        _Processor(),
        _Sink(delivered),
        identity_publisher=RegistryIdentityPublisher(registry),
    )
    coordinator.ingest_attempt(_attempt(), [b"attempt"])

    if delivered:
        coordinator.ingest_crossing(_crossing(), [b"crossing"])
        assert len(registry.calls) == 1
        proof = registry.calls[0]["gallery_authorization"]
        assert proof.authorized is True
        assert proof.reason == "reid_and_primary_ocr_exact"
    else:
        with pytest.raises(EntryUnavailable, match="delivery_failed"):
            coordinator.ingest_crossing(_crossing(), [b"crossing"])
        assert registry.calls == []
