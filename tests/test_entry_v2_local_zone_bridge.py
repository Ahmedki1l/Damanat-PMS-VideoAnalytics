from __future__ import annotations

import threading
import importlib.util
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from src.entry.domain import CrossingRole, EntryMode, IngestResult
from src.entry.local_zone import (
    LocalVehicleCrop,
    LocalZoneCrossingBridge,
    VA_HOST_GRAB_TIMESTAMP_SOURCE,
    local_zone_policy,
)
from src.entry.settings import EntrySettings


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
PRODUCTION_EMPTY_REARM_FRAMES = 8


def _settings(**overrides) -> EntrySettings:
    base = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_concurrent_ingest_requests=2,
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
    return replace(base, **overrides)


class RecordingCoordinator:
    def __init__(self, settings=None, *, fail=False, failures_remaining=0):
        self.settings = settings or _settings()
        self.available = True
        self.fail = fail
        self.failures_remaining = int(failures_remaining)
        self.calls = []
        self._lock = threading.Lock()

    def ingest_crossing(self, request, images):
        with self._lock:
            self.calls.append((request, tuple(images)))
        if self.fail or self.failures_remaining > 0:
            self.failures_remaining = max(0, self.failures_remaining - 1)
            raise RuntimeError("processor unavailable")
        return IngestResult(
            resource_id=request.crossing_id,
            accepted=True,
            duplicate=False,
            mode=self.settings.mode,
            evidence_count=len(images),
            decision_status="confirmed",
            callback_delivered=True,
        )


class BlockingCoordinator(RecordingCoordinator):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def ingest_crossing(self, request, images):
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise RuntimeError("test coordinator was not released")
        return super().ingest_crossing(request, images)


@pytest.fixture
def coordinator():
    return RecordingCoordinator()


@pytest.fixture
def bridge(coordinator):
    item = LocalZoneCrossingBridge(coordinator)
    yield item
    item.close(wait=True)


def _crop(track_id: int, value: int = 120) -> LocalVehicleCrop:
    return LocalVehicleCrop(
        track_id=track_id,
        image=np.full((96, 160, 3), value, dtype=np.uint8),
        quality=float(value),
    )


def _observe(
    bridge,
    camera,
    zone,
    tracks=(),
    crops=(),
    captured_at=NOW,
    outside_tracks=(),
    observation_token=None,
    timestamp_source=VA_HOST_GRAB_TIMESTAMP_SOURCE,
):
    bridge.observe(
        camera_id=camera,
        zone_id=zone,
        inside_track_ids=tracks,
        outside_track_ids=outside_tracks,
        crops=crops,
        captured_at=captured_at,
        timestamp_source=timestamp_source,
        observation_token=observation_token,
    )


def _arm(bridge, camera, zone):
    _observe(bridge, camera, zone)
    _observe(bridge, camera, zone)


def _clear_after_uncertain_episode(bridge, camera, zone):
    for _ in range(PRODUCTION_EMPTY_REARM_FRAMES):
        _observe(bridge, camera, zone)


def test_zone_aliases_resolve_to_canonical_one_way_policies():
    primary = local_zone_policy("CAM_23", "park-entry")
    fallback = local_zone_policy("cam-03", "B1-entry")

    assert primary is not None
    assert primary.line_id == "Park_Entry"
    assert primary.direction == "ramp-entry"
    assert primary.role == CrossingRole.PRIMARY
    assert fallback is not None
    assert fallback.line_id == "B1_Entrence"
    assert fallback.direction == "b-entry"
    assert fallback.role == CrossingRole.FALLBACK


def test_cam23_stable_entry_emits_one_primary_crossing(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7, 100)])
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7],
        [_crop(7, 140)],
        NOW + timedelta(milliseconds=500),
    )
    for index in range(4):
        _observe(
            bridge,
            "CAM-23",
            "Park_Entry",
            [7],
            [_crop(7, 160 + index)],
            NOW + timedelta(seconds=index + 1),
        )

    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1
    request, images = coordinator.calls[0]
    assert request.camera_id == "CAM-23"
    assert request.line_id == "Park_Entry"
    assert request.direction == "ramp-entry"
    assert request.role == CrossingRole.PRIMARY
    assert request.captured_at == NOW
    assert request.metadata["crossing_source"] == "va_local_zone"
    assert request.metadata["timestamp_source"] == VA_HOST_GRAB_TIMESTAMP_SOURCE
    assert request.metadata["direction_evidence"] == (
        "deployment_calibrated_one_way_polygon"
    )
    assert request.metadata["track_id"] == 7
    assert len(images) == 2
    decoded = cv2.imdecode(np.frombuffer(images[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (96, 160)


def test_cam03_fallback_emits_on_physical_transition_without_timer(bridge, coordinator):
    late = NOW + timedelta(minutes=30)
    _arm(bridge, "CAM-03", "B1-entry")
    _observe(bridge, "CAM-03", "B1-entry", [19], [_crop(19)], late)
    _observe(
        bridge,
        "CAM-03",
        "B1-entry",
        [19],
        [_crop(19, 180)],
        late + timedelta(seconds=1),
    )

    assert bridge.wait_for_idle()
    request, _ = coordinator.calls[0]
    assert request.role == CrossingRole.FALLBACK
    assert request.line_id == "B1_Entrence"
    assert request.direction == "b-entry"
    assert request.captured_at == late


def test_startup_with_vehicle_inside_never_fabricates_crossing(bridge, coordinator):
    _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
    _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
    assert bridge.wait_for_idle()
    assert coordinator.calls == []

    _clear_after_uncertain_episode(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [8], [_crop(8)])
    _observe(bridge, "CAM-23", "Park_Entry", [8], [_crop(8)])
    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1


def test_multiple_vehicle_episode_fails_closed_until_zone_clears(bridge, coordinator):
    _arm(bridge, "CAM-03", "B1_Entrence")
    _observe(
        bridge,
        "CAM-03",
        "B1_Entrence",
        [1, 2],
        [_crop(1), _crop(2, 180)],
    )
    # One car disappearing does not retroactively turn an ambiguous episode
    # into a unique physical crossing.
    _observe(bridge, "CAM-03", "B1_Entrence", [1], [_crop(1)])
    _observe(bridge, "CAM-03", "B1_Entrence", [1], [_crop(1)])
    assert bridge.wait_for_idle()
    assert coordinator.calls == []

    _clear_after_uncertain_episode(bridge, "CAM-03", "B1_Entrence")
    _observe(bridge, "CAM-03", "B1_Entrence", [3], [_crop(3)])
    _observe(bridge, "CAM-03", "B1_Entrence", [3], [_crop(3)])
    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1
    assert bridge.metrics()["visits_ambiguous"] == 1


def test_untracked_second_vehicle_is_ambiguous(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7, -1],
        [_crop(7)],
    )
    _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
    _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
    assert bridge.wait_for_idle()
    assert coordinator.calls == []


def test_one_frame_tracker_dropout_does_not_create_second_visit(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5)])
    _observe(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5, 180)])
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5, 190)])

    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1
    assert bridge.metrics()["visits_started"] == 1


def test_two_detector_misses_do_not_rearm_a_stationary_visit(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5)])
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5, 180)])
    assert bridge.wait_for_idle()

    # The reported production failure: two empty YOLO results while the same
    # car is still physically waiting must not become a new outside->inside edge.
    _observe(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5, 190)])
    _observe(bridge, "CAM-23", "Park_Entry", [5], [_crop(5, 200)])

    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1
    assert bridge.metrics()["visits_started"] == 1


def test_duplicate_zone_aliases_cannot_count_one_frame_twice(bridge, coordinator):
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        observation_token="empty-1",
    )
    _observe(
        bridge,
        "CAM-23",
        "park-entry",
        observation_token="empty-2",
    )
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7],
        [_crop(7)],
        observation_token="vehicle-frame-1",
    )
    _observe(
        bridge,
        "CAM-23",
        "park-entry",
        [7],
        [_crop(7, 180)],
        observation_token="vehicle-frame-1",
    )

    assert bridge.wait_for_idle()
    assert coordinator.calls == []
    assert bridge.metrics()["duplicate_frame_observations"] == 1

    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7],
        [_crop(7, 190)],
        observation_token="vehicle-frame-2",
    )
    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1


def test_clean_rearm_creates_new_event_even_if_track_id_is_reused(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    assert bridge.wait_for_idle()

    # Seeing the same track outside the polygon is positive exit evidence, so
    # the bridge can re-arm promptly without waiting for the missing-detection
    # escape hatch.
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        outside_tracks=[9],
    )
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        outside_tracks=[9],
    )
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    assert bridge.wait_for_idle()

    assert len(coordinator.calls) == 2
    assert coordinator.calls[0][0].crossing_id != coordinator.calls[1][0].crossing_id


def test_invalid_crops_fail_closed_without_queuing(bridge, coordinator):
    _arm(bridge, "CAM-03", "B1_Entrence")
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    assert bridge.wait_for_idle()
    assert coordinator.calls == []


def test_policy_must_be_explicitly_enabled_in_entry_settings():
    coordinator = RecordingCoordinator(
        _settings(primary_lines=frozenset({"HIKVISION-LINE-1"}))
    )
    bridge = LocalZoneCrossingBridge(coordinator)
    try:
        _arm(bridge, "CAM-23", "Park_Entry")
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        assert bridge.wait_for_idle()
        assert coordinator.calls == []
    finally:
        bridge.close(wait=True)


def test_timestamp_fault_tracks_each_camera_until_that_camera_recovers(bridge):
    bridge.mark_source_timestamp_invalid("CAM-23")
    bridge.mark_source_timestamp_invalid("CAM-03")

    bridge.mark_source_timestamp_valid("CAM-23")
    partially_recovered = bridge.metrics()
    assert partially_recovered["healthy"] is False
    assert partially_recovered["invalid_timestamp_cameras"] == ["CAM03"]

    bridge.mark_source_timestamp_valid("CAM-03")
    recovered = bridge.metrics()
    assert recovered["healthy"] is True
    assert recovered["source_timestamp_invalid"] is False


def test_worker_failure_is_visible_in_health_metrics():
    coordinator = RecordingCoordinator(fail=True)
    bridge = LocalZoneCrossingBridge(coordinator, max_ingest_retries=0)
    try:
        _arm(bridge, "CAM-23", "Park_Entry")
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        assert bridge.wait_for_idle()
        metrics = bridge.metrics()
        assert metrics["healthy"] is False
        assert metrics["last_error"] == "crossing_ingest_failed"
        assert metrics["submissions_failed"] == 1
    finally:
        bridge.close(wait=True)


def test_transient_worker_failure_retries_same_crossing_and_stays_degraded():
    coordinator = RecordingCoordinator(failures_remaining=1)
    bridge = LocalZoneCrossingBridge(coordinator)
    try:
        _arm(bridge, "CAM-23", "Park_Entry")
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7, 180)])
        assert bridge.wait_for_idle()

        assert len(coordinator.calls) == 2
        first_request, first_images = coordinator.calls[0]
        retry_request, retry_images = coordinator.calls[1]
        assert retry_request.crossing_id == first_request.crossing_id
        assert retry_images == first_images
        metrics = bridge.metrics()
        assert metrics["submissions_retried"] == 1
        assert metrics["submissions_failed"] == 1
        assert metrics["submissions_completed"] == 1
        assert metrics["confirmations"] == 1
        assert metrics["healthy"] is False
        assert metrics["last_error"] == "crossing_ingest_failed"
    finally:
        bridge.close(wait=True)


def test_capacity_rejection_degrades_then_retries_while_visit_remains_inside():
    coordinator = BlockingCoordinator()
    bridge = LocalZoneCrossingBridge(
        coordinator,
        max_ingest_retries=0,
        max_queued=1,
    )
    try:
        _arm(bridge, "CAM-23", "Park_Entry")
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7, 180)])
        assert coordinator.started.wait(timeout=1.0)

        _arm(bridge, "CAM-03", "B1_Entrence")
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19)])
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19, 180)])

        saturated = bridge.metrics()
        assert saturated["healthy"] is False
        assert saturated["queue_saturated"] is True
        assert saturated["last_error"] == "local_zone_queue_capacity_exceeded"
        assert saturated["submissions_capacity_rejected"] == 1

        coordinator.release.set()
        assert bridge.wait_for_idle()
        # The same physical visit is still inside. Its next frame retries the
        # retained evidence instead of needing a fabricated second transition.
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19, 200)])
        assert bridge.wait_for_idle()

        recovered = bridge.metrics()
        assert len(coordinator.calls) == 2
        assert recovered["healthy"] is True
        assert recovered["queue_saturated"] is False
        assert recovered["last_error"] is None
    finally:
        coordinator.release.set()
        bridge.close(wait=True)


class _RecordingBridge:
    def __init__(self):
        self.calls = []
        self.timestamp_rejections = []
        self.timestamp_recovered = 0

    def configured_policy(self, camera_id, zone_id):
        return local_zone_policy(camera_id, zone_id)

    def observe(self, **kwargs):
        self.calls.append(kwargs)

    def mark_source_timestamp_invalid(self, camera_id):
        self.timestamp_rejections.append(camera_id)

    def mark_source_timestamp_valid(self, _camera_id):
        self.timestamp_recovered += 1


def _tracking_engine_harness():
    # Load the tracking mixin without importing the heavyweight engine package;
    # this CPU-only unit environment intentionally has no Ultralytics/Torch.
    detector_stub = types.ModuleType("src.detection.detector")
    detector_stub.is_untracked = lambda track_id: track_id is None or track_id < 0
    prior_detector = sys.modules.get("src.detection.detector")
    sys.modules["src.detection.detector"] = detector_stub
    try:
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "core"
            / "engine"
            / "engine_tracking.py"
        )
        spec = importlib.util.spec_from_file_location(
            "entry_v2_engine_tracking_under_test",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if prior_detector is None:
            sys.modules.pop("src.detection.detector", None)
        else:
            sys.modules["src.detection.detector"] = prior_detector

    class Harness(module.ParkingEngineTrackingMixin):
        pass

    engine = Harness()
    engine._entry_v2_local_bridge = _RecordingBridge()
    return engine


def test_engine_feeds_empty_and_vehicle_frames_with_capture_timestamp():
    from shapely.geometry import Polygon

    from src.models.slot import ParkingSlot

    engine = _tracking_engine_harness()
    zone = ParkingSlot(
        id="park-entry",
        polygon=Polygon([(100, 100), (500, 100), (500, 600), (100, 600)]),
    )
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    detection = type("DetectionStub", (), {})()
    detection.track_id = 11
    detection.bbox = (180.0, 180.0, 420.0, 520.0)
    detection.bottom_center = (300.0, 520.0)
    capture_ts = NOW.timestamp()

    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [],
        {zone.id: zone},
        capture_ts=capture_ts,
    )
    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [detection],
        {zone.id: zone},
        capture_ts=capture_ts,
    )
    detection.bottom_center = (700.0, 520.0)
    detection.bbox = (580.0, 180.0, 820.0, 520.0)
    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [detection],
        {zone.id: zone},
        capture_ts=capture_ts,
    )

    assert engine._entry_v2_local_bridge.calls[0]["inside_track_ids"] == []
    observed = engine._entry_v2_local_bridge.calls[1]
    assert observed["inside_track_ids"] == [11]
    assert observed["captured_at"] == NOW
    assert observed["timestamp_source"] == VA_HOST_GRAB_TIMESTAMP_SOURCE
    assert len(observed["crops"]) == 1
    assert observed["crops"][0].image.shape[:2] == (340, 240)
    assert engine._entry_v2_local_bridge.calls[2]["inside_track_ids"] == []
    assert engine._entry_v2_local_bridge.calls[2]["outside_track_ids"] == [11]


@pytest.mark.parametrize(
    "capture_ts",
    [None, 0.0, -1.0, float("nan"), float("inf"), "not-a-timestamp"],
)
def test_active_local_zone_rejects_invalid_timestamp_without_wall_clock_fallback(
    capture_ts,
):
    from shapely.geometry import Polygon

    from src.models.slot import ParkingSlot

    engine = _tracking_engine_harness()
    zone = ParkingSlot(
        id="park-entry",
        polygon=Polygon([(100, 100), (500, 100), (500, 600), (100, 600)]),
    )
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [],
        {zone.id: zone},
        capture_ts=capture_ts,
    )

    assert engine._entry_v2_local_bridge.calls == []
    assert engine._entry_v2_local_bridge.timestamp_rejections == ["CAM-23"]


def test_cam03_v2_uses_strict_polygon_membership_not_bbox_overlap():
    from shapely.geometry import Polygon

    from src.models.slot import ParkingSlot

    engine = _tracking_engine_harness()
    zone = ParkingSlot(
        id="B1-entry",
        polygon=Polygon([(100, 100), (500, 100), (500, 600), (100, 600)]),
    )
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    detection = type("DetectionStub", (), {})()
    detection.track_id = 11
    detection.bbox = (80.0, 180.0, 180.0, 520.0)
    detection.bottom_center = (90.0, 520.0)

    assert engine._detection_overlaps_zone(detection, zone) is True
    engine._process_entry_v2_local_zones(
        "CAM-03",
        frame,
        [detection],
        {zone.id: zone},
        capture_ts=NOW.timestamp(),
    )

    observed = engine._entry_v2_local_bridge.calls[0]
    assert observed["inside_track_ids"] == []
    assert observed["outside_track_ids"] == [11]
