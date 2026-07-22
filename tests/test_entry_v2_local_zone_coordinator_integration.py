from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    AttemptInput,
    EntryMode,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.local_zone import (
    LocalVehicleCrop,
    LocalZoneCrossingBridge,
    VA_HOST_GRAB_TIMESTAMP_SOURCE,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
PLATE = "ABC-1234"
EMBEDDING = (1.0, 0.0)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def _settings() -> EntrySettings:
    return replace(
        EntrySettings(),
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=8,
        max_pending_crossings=8,
        max_pending_callbacks=8,
        receipt_capacity=32,
        max_images_per_event=4,
        max_image_bytes=4 * 1024 * 1024,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B1_ENTRENCE"}),
        fallback_directions=frozenset({"b-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-key",
    )


def _jpeg(value: int = 120) -> bytes:
    image = np.full((96, 160, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


class _DeterministicEvidenceProcessor:
    """Fake only the model boundary while validating the real JPEG handoff."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def analyze(
        self,
        *,
        event_id: str,
        camera_id: str,
        source_role: str,
        images: Sequence[bytes],
        metadata: Mapping[str, Any],
    ) -> tuple[FrameEvidence, ...]:
        assert images
        for encoded in images:
            decoded = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert decoded is not None and decoded.size > 0

        self.calls.append(
            {
                "event_id": event_id,
                "camera_id": camera_id,
                "source_role": source_role,
                "image_count": len(images),
                "metadata": dict(metadata),
            }
        )
        is_crossing = source_role in {"primary", "fallback"}
        state = PlateReadState.READABLE if is_crossing else PlateReadState.NO_PLATE
        return tuple(
            FrameEvidence(
                evidence_id=f"{event_id}:{index}",
                embedding=EMBEDDING,
                plate=PlateEvidence(
                    evidence_id=f"{event_id}:{index}",
                    camera_id=camera_id,
                    source_role=source_role,
                    state=state,
                    text="ABC1234" if is_crossing else "",
                    confidence=0.99 if is_crossing else 0.0,
                ),
            )
            for index, _ in enumerate(images)
        )


class _RecordingSink:
    """Fake only the PMS transport boundary."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def deliver(self, payload: Mapping[str, Any]) -> DeliveryResult:
        self.payloads.append(dict(payload))
        return DeliveryResult(
            delivered=True,
            attempts=1,
            publish_identity=False,
            session_committed=True,
        )


def _attempt(captured_at: datetime = NOW) -> AttemptInput:
    return AttemptInput(
        attempt_id="attempt-1",
        source_event_id="anpr-event-1",
        camera_id="ANPR-ENTRY",
        captured_at=captured_at,
        reported_plate=PLATE,
        reported_confidence=0.99,
        metadata={"source": "hikvision_anpr"},
    )


def _observe(
    bridge: LocalZoneCrossingBridge,
    *,
    camera_id: str,
    zone_id: str,
    captured_at: datetime,
    track_id: int | None = None,
    pixel_value: int = 120,
) -> None:
    crops = ()
    tracks = ()
    if track_id is not None:
        tracks = (track_id,)
        crops = (
            LocalVehicleCrop(
                track_id=track_id,
                image=np.full((96, 160, 3), pixel_value, dtype=np.uint8),
                quality=float(pixel_value),
            ),
        )
    bridge.observe(
        camera_id=camera_id,
        zone_id=zone_id,
        inside_track_ids=tracks,
        crops=crops,
        captured_at=captured_at,
        timestamp_source=VA_HOST_GRAB_TIMESTAMP_SOURCE,
    )


def _emit_local_crossing(
    bridge: LocalZoneCrossingBridge,
    *,
    camera_id: str,
    zone_id: str,
    captured_at: datetime,
    track_id: int,
) -> None:
    _observe(
        bridge,
        camera_id=camera_id,
        zone_id=zone_id,
        captured_at=captured_at - timedelta(seconds=2),
    )
    _observe(
        bridge,
        camera_id=camera_id,
        zone_id=zone_id,
        captured_at=captured_at - timedelta(seconds=1),
    )
    _observe(
        bridge,
        camera_id=camera_id,
        zone_id=zone_id,
        captured_at=captured_at,
        track_id=track_id,
        pixel_value=120,
    )
    _observe(
        bridge,
        camera_id=camera_id,
        zone_id=zone_id,
        captured_at=captured_at + timedelta(milliseconds=250),
        track_id=track_id,
        pixel_value=180,
    )
    assert bridge.wait_for_idle()


def _assert_no_images(directory: Path) -> None:
    assert not any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def _system() -> tuple[
    EntryCoordinator,
    LocalZoneCrossingBridge,
    _DeterministicEvidenceProcessor,
    _RecordingSink,
]:
    processor = _DeterministicEvidenceProcessor()
    sink = _RecordingSink()
    coordinator = EntryCoordinator(_settings(), processor, sink)
    bridge = LocalZoneCrossingBridge(coordinator)
    return coordinator, bridge, processor, sink


def _assert_no_pending_entry_state(coordinator: EntryCoordinator) -> None:
    state = coordinator.state_summary()
    assert state["attempt_count"] == 0
    assert state["group_count"] == 0
    assert state["crossing_count"] == 0
    assert state["pending_callback_count"] == 0
    assert state["reserved_callback_count"] == 0


def test_anpr_then_real_cam23_bridge_emits_exactly_one_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    coordinator, bridge, processor, sink = _system()
    try:
        attempt_result = coordinator.ingest_attempt(_attempt(), [_jpeg()])
        assert attempt_result.decision_status is None

        _emit_local_crossing(
            bridge,
            camera_id="CAM-23",
            zone_id="Park_Entry",
            captured_at=NOW + timedelta(seconds=5),
            track_id=23,
        )

        assert len(sink.payloads) == 1
        assert sink.payloads[0]["status"] == "confirmed"
        assert sink.payloads[0]["attempt_id"] == "attempt-1"
        assert sink.payloads[0]["entry_camera_id"] == "ANPR-ENTRY"
        assert sink.payloads[0]["crossing_id"].startswith("va-zone-")
        assert sink.payloads[0]["canonical_plate"] == PLATE
        assert [call["source_role"] for call in processor.calls] == [
            "anpr",
            "primary",
        ]
        assert processor.calls[-1]["metadata"]["crossing_source"] == ("va_local_zone")
        assert processor.calls[-1]["camera_id"] == "CAM-23"
        _assert_no_pending_entry_state(coordinator)
        _assert_no_images(tmp_path)
    finally:
        bridge.close(wait=True)


def test_anpr_then_much_later_cam03_fallback_confirms_without_cam23_or_timer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    coordinator, bridge, processor, sink = _system()
    late_crossing_at = NOW + timedelta(hours=6)
    try:
        coordinator.ingest_attempt(_attempt(), [_jpeg(100)])

        _emit_local_crossing(
            bridge,
            camera_id="CAM-03",
            zone_id="B1_Entrence",
            captured_at=late_crossing_at,
            track_id=3,
        )

        assert len(sink.payloads) == 1
        assert sink.payloads[0]["status"] == "confirmed"
        assert sink.payloads[0]["entry_camera_id"] == "ANPR-ENTRY"
        assert sink.payloads[0]["crossing_id"].startswith("va-zone-")
        assert sink.payloads[0]["entry_captured_at"] == late_crossing_at.isoformat()
        assert sink.payloads[0]["ocr_source"] == "fallback"
        assert [call["source_role"] for call in processor.calls] == [
            "anpr",
            "fallback",
        ]
        assert processor.calls[-1]["camera_id"] == "CAM-03"
        _assert_no_pending_entry_state(coordinator)
        _assert_no_images(tmp_path)
    finally:
        bridge.close(wait=True)


def test_delayed_same_car_cam03_after_primary_is_quarantined_without_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    coordinator, bridge, processor, sink = _system()
    try:
        coordinator.ingest_attempt(_attempt(), [_jpeg(100)])
        _emit_local_crossing(
            bridge,
            camera_id="CAM-23",
            zone_id="Park_Entry",
            captured_at=NOW + timedelta(seconds=5),
            track_id=23,
        )
        assert len(sink.payloads) == 1
        first_payload = dict(sink.payloads[0])

        _emit_local_crossing(
            bridge,
            camera_id="CAM-03",
            zone_id="B1_Entrence",
            captured_at=NOW + timedelta(hours=6),
            track_id=3,
        )

        assert sink.payloads == [first_payload]
        assert [call["source_role"] for call in processor.calls] == [
            "anpr",
            "primary",
            "fallback",
        ]
        assert bridge.metrics()["submissions_completed"] == 2
        _assert_no_pending_entry_state(coordinator)
        assert coordinator.state_summary()["finalized_journey_count"] == 1
        _assert_no_images(tmp_path)
    finally:
        bridge.close(wait=True)
