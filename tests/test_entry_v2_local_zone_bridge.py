from __future__ import annotations

import threading
import importlib.util
import sys
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
from types import SimpleNamespace

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


def _exit(bridge, camera, zone, track_id, captured_at=NOW):
    _observe(
        bridge,
        camera,
        zone,
        outside_tracks=[track_id],
        captured_at=captured_at,
    )
    _observe(
        bridge,
        camera,
        zone,
        outside_tracks=[track_id],
        captured_at=captured_at + timedelta(milliseconds=500),
    )


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
    qualities = [
        100,
        240,
        110,
        220,
        120,
        250,
        130,
        230,
        *range(80, 92),
    ]
    for index, quality in enumerate(qualities):
        _observe(
            bridge,
            "CAM-23",
            "Park_Entry",
            [7],
            [_crop(7, quality)],
            NOW + timedelta(seconds=index),
        )

    # The bridge considers every in-zone frame but cannot declare a physical
    # crossing until it sees the same track beyond the polygon twice.
    assert bridge.wait_for_idle()
    assert coordinator.calls == []
    _exit(bridge, "CAM-23", "Park_Entry", 7, NOW + timedelta(seconds=21))

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
    assert request.metadata["finalization_evidence"] == "tracked_outside_zone"
    assert request.metadata["frames_seen"] == 20
    assert request.metadata["candidate_crops_seen"] == 20
    assert request.metadata["selected_images"] == 4
    assert request.metadata["best_crop_quality"] == 250.0
    assert request.metadata["track_id"] == 7
    assert len(images) == 4
    decoded = [
        cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        for image in images
    ]
    assert all(image.shape[:2] == (96, 160) for image in decoded)
    assert [round(float(image.mean())) for image in decoded] == [240, 220, 250, 230]
    metrics = bridge.metrics()
    assert metrics["candidate_crops_observed"] == 20
    assert metrics["snapshots_selected"] == 4
    assert metrics["visits_finalized"] == 1


def test_duplicate_track_rows_keep_the_best_crop_from_each_frame(
    bridge,
    coordinator,
):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7],
        [_crop(7, 200), _crop(7, 80)],
    )
    _observe(
        bridge,
        "CAM-23",
        "Park_Entry",
        [7],
        [_crop(7, 150)],
        NOW + timedelta(seconds=1),
    )
    _exit(bridge, "CAM-23", "Park_Entry", 7, NOW + timedelta(seconds=2))

    assert bridge.wait_for_idle()
    _, images = coordinator.calls[0]
    decoded = [
        cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        for image in images
    ]
    assert [round(float(image.mean())) for image in decoded] == [200, 150]


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
    _exit(bridge, "CAM-03", "B1-entry", 19, late + timedelta(seconds=2))

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
    _exit(bridge, "CAM-23", "Park_Entry", 8)
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
    _exit(bridge, "CAM-03", "B1_Entrence", 3)
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
    _exit(bridge, "CAM-23", "Park_Entry", 5)

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
    assert coordinator.calls == []
    _exit(bridge, "CAM-23", "Park_Entry", 5)
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
    assert coordinator.calls == []
    _exit(bridge, "CAM-23", "Park_Entry", 7)
    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1


def test_clean_rearm_creates_new_event_even_if_track_id_is_reused(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])

    # Seeing the same track outside the polygon is positive exit evidence, so
    # the bridge can finalize and re-arm promptly without treating YOLO misses
    # as proof of movement.
    _exit(bridge, "CAM-23", "Park_Entry", 9)
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    _observe(bridge, "CAM-23", "Park_Entry", [9], [_crop(9)])
    _exit(bridge, "CAM-23", "Park_Entry", 9)
    assert bridge.wait_for_idle()

    assert len(coordinator.calls) == 2
    assert coordinator.calls[0][0].crossing_id != coordinator.calls[1][0].crossing_id


def test_invalid_crops_fail_closed_without_queuing(bridge, coordinator):
    _arm(bridge, "CAM-03", "B1_Entrence")
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    _observe(bridge, "CAM-03", "B1_Entrence", [4])
    _exit(bridge, "CAM-03", "B1_Entrence", 4)
    assert bridge.wait_for_idle()
    assert coordinator.calls == []
    assert bridge.metrics()["visits_discarded_insufficient"] == 1


def test_one_outside_observation_then_reentry_keeps_collecting_same_visit(
    bridge,
    coordinator,
):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [31], [_crop(31, 100)])
    _observe(bridge, "CAM-23", "Park_Entry", [31], [_crop(31, 110)])
    _observe(bridge, "CAM-23", "Park_Entry", outside_tracks=[31])
    _observe(bridge, "CAM-23", "Park_Entry", [31], [_crop(31, 200)])
    _observe(bridge, "CAM-23", "Park_Entry", outside_tracks=[31])

    assert bridge.wait_for_idle()
    assert coordinator.calls == []

    _observe(bridge, "CAM-23", "Park_Entry", outside_tracks=[31])
    assert bridge.wait_for_idle()
    assert len(coordinator.calls) == 1
    request, images = coordinator.calls[0]
    assert request.metadata["frames_seen"] == 3
    assert request.metadata["candidate_crops_seen"] == 3
    assert len(images) == 3


def test_tracker_loss_discards_visit_without_submitting(bridge, coordinator):
    _arm(bridge, "CAM-23", "Park_Entry")
    _observe(bridge, "CAM-23", "Park_Entry", [41], [_crop(41, 120)])
    _observe(bridge, "CAM-23", "Park_Entry", [41], [_crop(41, 180)])

    for _ in range(PRODUCTION_EMPTY_REARM_FRAMES):
        _observe(bridge, "CAM-23", "Park_Entry")

    assert bridge.wait_for_idle()
    assert coordinator.calls == []
    assert bridge.metrics()["visits_discarded_tracker_loss"] == 1


def test_policy_must_be_explicitly_enabled_in_entry_settings():
    coordinator = RecordingCoordinator(
        _settings(primary_lines=frozenset({"HIKVISION-LINE-1"}))
    )
    bridge = LocalZoneCrossingBridge(coordinator)
    try:
        _arm(bridge, "CAM-23", "Park_Entry")
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _observe(bridge, "CAM-23", "Park_Entry", [7], [_crop(7)])
        _exit(bridge, "CAM-23", "Park_Entry", 7)
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
        _exit(bridge, "CAM-23", "Park_Entry", 7)
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
        _exit(bridge, "CAM-23", "Park_Entry", 7)
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


def test_capacity_rejection_retries_same_confirmed_exit_without_overwrite():
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
        _exit(bridge, "CAM-23", "Park_Entry", 7)
        assert coordinator.started.wait(timeout=1.0)

        _arm(bridge, "CAM-03", "B1_Entrence")
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19)])
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19, 180)])
        _exit(bridge, "CAM-03", "B1_Entrence", 19)

        saturated = bridge.metrics()
        assert saturated["healthy"] is False
        assert saturated["queue_saturated"] is True
        assert saturated["last_error"] == "local_zone_queue_capacity_exceeded"
        assert saturated["submissions_capacity_rejected"] == 1

        coordinator.release.set()
        assert bridge.wait_for_idle()
        # The same completed physical visit remains pending. Its next frame
        # retries prepared evidence instead of allowing a new track to replace
        # it or requiring a fabricated second transition.
        _observe(
            bridge,
            "CAM-03",
            "B1_Entrence",
            [88],
            [_crop(88, 250)],
        )
        assert bridge.wait_for_idle()

        recovered = bridge.metrics()
        assert len(coordinator.calls) == 2
        assert coordinator.calls[1][0].metadata["track_id"] == 19
        assert recovered["healthy"] is True
        assert recovered["queue_saturated"] is False
        assert recovered["last_error"] is None
    finally:
        coordinator.release.set()
        bridge.close(wait=True)


def test_graceful_close_flushes_capacity_blocked_confirmed_exit():
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
        _exit(bridge, "CAM-23", "Park_Entry", 7)
        assert coordinator.started.wait(timeout=1.0)

        _arm(bridge, "CAM-03", "B1_Entrence")
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19)])
        _observe(bridge, "CAM-03", "B1_Entrence", [19], [_crop(19, 180)])
        _exit(bridge, "CAM-03", "B1_Entrence", 19)
        assert bridge.metrics()["queue_saturated"] is True

        coordinator.release.set()
        bridge.close(wait=True)

        assert len(coordinator.calls) == 2
        assert coordinator.calls[1][0].metadata["track_id"] == 19
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
    # The zone crop is PADDED beyond the detection box. YOLO's box stops at the
    # grille on the ramp's top-down view, leaving the number plate outside it —
    # and this is the fallback identity path for cars the gate ANPR never
    # reports, so the plate must be inside the crop. bbox is 240x340, padded by
    # ENTRY_V2_LOCAL_CROP_PADDING_RATIO (0.18) on every side and clamped to the
    # 1280x720 frame: 240 + 2*43 = 326, 340 + 2*61 = 462.
    assert observed["crops"][0].image.shape[:2] == (462, 326)
    assert engine._entry_v2_local_bridge.calls[2]["inside_track_ids"] == []
    assert engine._entry_v2_local_bridge.calls[2]["outside_track_ids"] == [11]


@pytest.mark.parametrize(
    "bbox",
    [
        (0.0, 180.0, 240.0, 520.0),       # border-truncated fragment
        (180.0, 180.0, 880.0, 430.0),     # implausible 2.8:1 sliver
    ],
)
def test_engine_tracks_invalid_whole_car_boxes_without_using_their_crops(bbox):
    from shapely.geometry import Polygon

    from src.models.slot import ParkingSlot

    engine = _tracking_engine_harness()
    zone = ParkingSlot(
        id="park-entry",
        polygon=Polygon([(0, 0), (1000, 0), (1000, 700), (0, 700)]),
    )
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    detection = type("DetectionStub", (), {})()
    detection.track_id = 11
    detection.bbox = bbox
    detection.bottom_center = (
        (bbox[0] + bbox[2]) / 2.0,
        bbox[3],
    )

    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [detection],
        {zone.id: zone},
        capture_ts=NOW.timestamp(),
    )

    observed = engine._entry_v2_local_bridge.calls[0]
    assert observed["inside_track_ids"] == [11]
    assert observed["crops"] == []


@pytest.mark.parametrize(
    ("width", "height"),
    [(264, 293), (566, 585), (432, 523)],
)
def test_realistic_cam23_whole_car_shapes_remain_snapshot_ready(width, height):
    from shapely.geometry import Polygon

    from src.models.slot import ParkingSlot

    engine = _tracking_engine_harness()
    zone = ParkingSlot(
        id="park-entry",
        polygon=Polygon([(100, 100), (1000, 100), (1000, 900), (100, 900)]),
    )
    frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    detection = type("DetectionStub", (), {})()
    detection.track_id = 11
    detection.bbox = (300.0, 200.0, 300.0 + width, 200.0 + height)
    detection.bottom_center = (
        300.0 + width / 2.0,
        200.0 + height,
    )

    engine._process_entry_v2_local_zones(
        "CAM-23",
        frame,
        [detection],
        {zone.id: zone},
        capture_ts=NOW.timestamp(),
    )

    observed = engine._entry_v2_local_bridge.calls[0]
    assert observed["inside_track_ids"] == [11]
    assert len(observed["crops"]) == 1


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


def test_zone_union_crop_reaches_past_the_car_box_for_the_plate():
    """The OCR view must cover the bumper the detector's box excludes.

    On the ramp's top-down view YOLO's box stops at the grille, so the tight
    ReID crop cannot contain the plate. The union with the zone bounds must
    extend past the box; the tight crop must stay tight.
    """
    from shapely.geometry import Polygon
    from src.core.engine.engine_tracking import ParkingEngineTrackingMixin

    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    detection = SimpleNamespace(bbox=(400.0, 200.0, 700.0, 480.0), track_id=7)
    zone = SimpleNamespace(
        polygon=Polygon([(300, 150), (900, 150), (900, 620), (300, 620)])
    )

    engine = ParkingEngineTrackingMixin.__new__(ParkingEngineTrackingMixin)
    union = ParkingEngineTrackingMixin._crop_detection_union_zone(
        engine, frame, detection, zone, padding_ratio=0.0
    )
    tight = ParkingEngineTrackingMixin._crop_detection(engine, frame, detection)

    assert tight.shape[:2] == (280, 300)          # exactly the detection box
    # Union spans zone bounds where they exceed the box: x 300..900, y 150..620.
    assert union.shape[:2] == (470, 600)
    assert union.shape[0] > tight.shape[0]        # reaches BELOW the grille cut
    assert union.shape[1] > tight.shape[1]


def test_zone_union_falls_back_to_the_box_when_zone_bounds_are_unusable():
    """A broken zone must not cost the detection its crop entirely."""
    from src.core.engine.engine_tracking import ParkingEngineTrackingMixin

    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    detection = SimpleNamespace(bbox=(400.0, 200.0, 700.0, 480.0), track_id=7)

    class _BadZone:
        @property
        def polygon(self):
            raise RuntimeError("no geometry")

    engine = ParkingEngineTrackingMixin.__new__(ParkingEngineTrackingMixin)
    crop = ParkingEngineTrackingMixin._crop_detection_union_zone(
        engine, frame, detection, _BadZone(), padding_ratio=0.0
    )
    assert crop is not None and crop.shape[:2] == (280, 300)


# --------------------------------------------------------------------------- #
# Best-plate-view selection
#
# The selector used to rank by _score_snapshot_quality, which is
# `area * (1 + 0.5 * sharpness_factor)`. Area reaches ~10^6 while the sharpness
# term only scales it by 1.0-1.5, so area decides and the biggest frame always
# won. On CAM-23's ramp the biggest frame is the one where the car sits under
# the lens and the plate has rotated out of view: across 29 real captures every
# frame it picked with no plate in it was one of the four largest, and on track
# 1493 it discarded a frame that read `1198 SHR` at 0.980.
# --------------------------------------------------------------------------- #

_PLATE_POLICY = local_zone_policy("CAM-23", "Park_Entry")


def _visit_with(crops):
    """A finished visit holding (quality, frame_no, reid_crop, plate_view)."""
    from src.entry.local_zone import _Visit

    visit = _Visit(
        sequence=1,
        track_id=1493,
        captured_at=NOW,
        timestamp_source=VA_HOST_GRAB_TIMESTAMP_SOURCE,
    )
    visit.frames_seen = len(crops)
    visit.crops = list(crops)
    return visit


def _plate_view(*, size, sharp: bool):
    """A plate view whose 40x18 plate patch is either crisp or flat grey."""
    image = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
    if sharp:
        patch = np.indices((18, 40))[1] % 2 * 255      # 1px stripes -> high variance
        image[40:58, 60:100] = patch[:, :, None].astype(np.uint8)
    return image


class _StubLPD:
    """Reports a plate only for the frames named, at a fixed location."""

    def __init__(self, with_plate):
        self.with_plate = with_plate
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if (image.shape[1], image.shape[0]) not in self.with_plate:
            return []
        return [(60.0, 40.0, 100.0, 58.0, 0.61)]


def test_best_plate_view_prefers_the_legible_plate_over_the_largest_frame(
    bridge, tmp_path, monkeypatch
):
    """The regression: the biggest frame has no plate, a smaller one does."""
    monkeypatch.setenv("ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR", str(tmp_path))

    mid = _plate_view(size=(1203, 1051), sharp=True)     # readable, mid-ramp
    huge = _plate_view(size=(2143, 1169), sharp=True)    # car under the lens
    visit = _visit_with(
        [
            (1_264_353.0, 3, np.zeros((8, 8, 3), np.uint8), mid),
            (2_505_167.0, 9, np.zeros((8, 8, 3), np.uint8), huge),   # wins on area
        ]
    )

    bridge._plate_view_detector = _StubLPD(with_plate={(1203, 1051)})
    bridge._save_best_plate_view(_PLATE_POLICY, "Park_Entry", visit)

    written = list((tmp_path / "entry_zone_captures").glob("*.jpg"))
    assert len(written) == 1
    # frame 3, not the higher-area frame 9
    assert "seq-1best3" in written[0].name
    assert bridge.metrics()["best_plate_view_plate_found"] == 1
    assert bridge.metrics()["best_plate_view_rescued_from_area_pick"] == 1


def test_best_plate_view_ranks_by_plate_sharpness_when_both_have_a_plate(
    bridge, tmp_path, monkeypatch
):
    """Two frames with a plate: the crisper plate wins even though it is smaller.

    Measured on the real captures, plate-crop Laplacian variance separated
    readable from unreadable cleanly (84-233 vs 1405-12180) where vehicle area
    did not.
    """
    monkeypatch.setenv("ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR", str(tmp_path))

    crisp = _plate_view(size=(1121, 1051), sharp=True)
    smeared = _plate_view(size=(1618, 1051), sharp=False)
    visit = _visit_with(
        [
            (1_178_000.0, 2, np.zeros((8, 8, 3), np.uint8), crisp),
            (1_700_000.0, 7, np.zeros((8, 8, 3), np.uint8), smeared),  # wins on area
        ]
    )

    bridge._plate_view_detector = _StubLPD(
        with_plate={(1121, 1051), (1618, 1051)}
    )
    bridge._save_best_plate_view(_PLATE_POLICY, "Park_Entry", visit)

    written = list((tmp_path / "entry_zone_captures").glob("*.jpg"))
    assert len(written) == 1
    assert "seq-1best2" in written[0].name


def test_best_plate_view_falls_back_to_area_when_no_candidate_has_a_plate(
    bridge, tmp_path, monkeypatch
):
    """Writing the old pick still beats writing nothing for field calibration."""
    monkeypatch.setenv("ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR", str(tmp_path))

    visit = _visit_with(
        [
            (900_000.0, 1, np.zeros((8, 8, 3), np.uint8),
             _plate_view(size=(1044, 1051), sharp=False)),
            (2_505_167.0, 6, np.zeros((8, 8, 3), np.uint8),
             _plate_view(size=(2143, 1169), sharp=False)),
        ]
    )

    bridge._plate_view_detector = _StubLPD(with_plate=set())
    bridge._save_best_plate_view(_PLATE_POLICY, "Park_Entry", visit)

    written = list((tmp_path / "entry_zone_captures").glob("*.jpg"))
    assert len(written) == 1
    assert "seq-1best6" in written[0].name            # the old area behaviour
    assert bridge.metrics()["best_plate_view_no_plate_anywhere"] == 1
    assert bridge.metrics()["best_plate_view_rescued_from_area_pick"] == 0


def test_best_plate_view_survives_an_unavailable_plate_detector(
    bridge, tmp_path, monkeypatch
):
    """A missing or broken LPD must degrade to the old pick, never raise."""
    monkeypatch.setenv("ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR", str(tmp_path))

    class _Exploding:
        def detect(self, image):
            raise RuntimeError("openvino unavailable")

    visit = _visit_with(
        [
            (900_000.0, 1, np.zeros((8, 8, 3), np.uint8),
             _plate_view(size=(1044, 1051), sharp=True)),
            (2_505_167.0, 4, np.zeros((8, 8, 3), np.uint8),
             _plate_view(size=(2143, 1169), sharp=True)),
        ]
    )

    bridge._plate_view_detector = _Exploding()
    bridge._save_best_plate_view(_PLATE_POLICY, "Park_Entry", visit)

    written = list((tmp_path / "entry_zone_captures").glob("*.jpg"))
    assert len(written) == 1
    assert "seq-1best4" in written[0].name


def test_best_plate_view_never_builds_the_detector_in_the_hot_path(bridge):
    """_retain_crop runs per frame; it must not touch the LPD."""
    from src.entry.local_zone import _Visit

    visit = _Visit(
        sequence=1,
        track_id=5,
        captured_at=NOW,
        timestamp_source=VA_HOST_GRAB_TIMESTAMP_SOURCE,
    )
    for index in range(12):
        visit.frames_seen = index
        bridge._retain_crop(
            visit,
            LocalVehicleCrop(
                track_id=5,
                image=np.full((96, 160, 3), 120, np.uint8),
                quality=float(1000 * index),
                plate_image=np.full((300, 400, 3), 90, np.uint8),
            ),
        )

    assert bridge._plate_view_detector is None


# --------------------------------------------------------------------------- #
# _score_snapshot_quality
#
# It used to return `area * (1 + 0.5 * sharpness_factor)`. Measured over 29 real
# CAM-23 captures: whole-crop Laplacian variance ran 1075-3746 against a cap of
# 250, so the sharpness factor was pinned at 1.0 on every frame and the formula
# was exactly `area * 1.5` — unbounded area wearing a clarity costume.
# --------------------------------------------------------------------------- #

def _engine():
    from src.core.engine.engine_tracking import ParkingEngineTrackingMixin

    return ParkingEngineTrackingMixin.__new__(ParkingEngineTrackingMixin)


def _detection(x1, y1, x2, y2):
    return SimpleNamespace(bbox=(float(x1), float(y1), float(x2), float(y2)), track_id=1)


def test_snapshot_quality_area_saturates_instead_of_running_away():
    """The real defect: an 8.75x area ratio used to be an 8.75x score ratio.

    Measured on track 905, where the larger frame is the one with no plate in it.
    Bounded growth keeps ordering deterministic without letting size dominate.
    """
    from src.core.engine.engine_tracking import _snapshot_area_score

    frame_h = 1520.0
    small = _snapshot_area_score(146_407.0, frame_h)
    huge = _snapshot_area_score(146_407.0 * 8.75, frame_h)

    assert huge / small < 6.0                  # was exactly 8.75
    assert huge > small                        # still monotonic, never inverted
    # Past the reference the curve is nearly flat: doubling again barely moves it.
    doubled = _snapshot_area_score(146_407.0 * 17.5, frame_h)
    assert doubled - huge < 0.05


def test_snapshot_quality_reference_scales_with_stream_resolution():
    """A fixed pixel gate silently changes meaning when the stream resolution does.

    The same car filling the same fraction of the frame must score the same at
    720p and at 1520p.
    """
    from src.core.engine.engine_tracking import _snapshot_area_score

    at_720 = _snapshot_area_score(100_000.0, 720.0)
    scaled = 100_000.0 * (1520.0 / 720.0) ** 2
    at_1520 = _snapshot_area_score(scaled, 1520.0)
    assert at_720 == pytest.approx(at_1520, rel=1e-6)


def test_snapshot_quality_discounts_a_car_clamped_by_the_frame_edge():
    """A box touching the border holds a fragment of a car, not a car."""
    engine = _engine()
    frame = np.full((1520, 2688, 3), 100, dtype=np.uint8)

    inside = _detection(600, 400, 1400, 1100)           # 800x700, clear of every edge
    clipped = _detection(600, 820, 1400, 1520)          # 800x700, runs off the bottom

    whole = engine._score_snapshot_quality(inside, None, frame)
    fragment = engine._score_snapshot_quality(clipped, None, frame)

    # Same box area, so the only difference is being cut off by the frame.
    assert fragment < whole
    assert fragment == pytest.approx(whole * 0.25, rel=1e-6)


def test_snapshot_quality_without_a_frame_keeps_working():
    """`frame` is optional; older callers must not break, just score bounded area."""
    engine = _engine()
    score = engine._score_snapshot_quality(_detection(0, 0, 400, 300), None)
    assert score > 0.0


def test_snapshot_quality_sharpness_term_still_only_discounts_flat_crops():
    """The cap stays at 250 on purpose.

    On real captures the statistic tracked headlight glare, not legibility: the
    unreadable dark frames scored 2410-3746 while the readable ones scored
    1388-1723. Raising the cap would promote exactly the wrong frames. What it
    must still do is mark a genuinely featureless crop down.
    """
    engine = _engine()
    frame = np.full((1520, 2688, 3), 100, dtype=np.uint8)
    detection = _detection(600, 400, 1400, 1100)

    flat = np.full((200, 300, 3), 128, dtype=np.uint8)
    textured = np.random.default_rng(0).integers(
        0, 255, (200, 300, 3), dtype=np.uint8
    )

    assert engine._score_snapshot_quality(detection, flat, frame) < (
        engine._score_snapshot_quality(detection, textured, frame)
    )
