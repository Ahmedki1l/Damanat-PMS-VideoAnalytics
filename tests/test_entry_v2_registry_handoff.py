import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import AreaEntry, CameraEntry
from src.utils.datetime_helper import normalize_timestamp_for_clock
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import (
    ParkEntryCandidate,
    PendingANPREvent,
)
from src.zoning.area_registry import AreaRegistry


def _minimal_registry(*, clock=None):
    registry = VehicleRegistry.__new__(VehicleRegistry)
    registry._lock = threading.RLock()
    registry._sessions = {}
    registry._last_anpr_entry_at = {}
    registry._last_anpr_exit_at = {}
    registry._clock = clock or datetime.now
    registry._area_registry = None
    registry._area_sessions = defaultdict(set)
    registry._matching_config = SimpleNamespace(
        gallery_max_refs_per_car=8,
        use_faiss_index=False,
    )
    registry._claim_plate_globally = lambda *args, **kwargs: []
    registry._clear_slot_db_binding = lambda slot_id: None
    return registry


class _CosineMatcher:
    @staticmethod
    def compute_similarity(left, right):
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


class _PartiallyFailingGalleryIndex:
    def __init__(self):
        self.session_ids = set()

    def add(self, session_id, feature_vector):
        del feature_vector
        self.session_ids.add(session_id)
        raise RuntimeError("gallery write failed")

    def remove(self, session_id):
        self.session_ids.discard(session_id)


def test_validated_entry_creates_matchable_image_free_session_idempotently():
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = _minimal_registry(clock=lambda: registry_now)
    entered_at = registry_now.astimezone(timezone.utc)

    first = registry.register_validated_entry(
        plate="ABC-1234",
        decision_id="decision-1",
        attempt_id="attempt-1",
        crossing_id="crossing-1",
        timestamp=entered_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(("CAM-ENTRY", (0.0, 2.0)),),
    )
    second = registry.register_validated_entry(
        plate="ABC-1234",
        decision_id="decision-1",
        attempt_id="attempt-1",
        crossing_id="crossing-1",
        timestamp=entered_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(("CAM-ENTRY", (0.0, 2.0)),),
    )

    assert second == first
    assert len(registry._sessions) == 1
    session = registry._sessions[first]
    assert session.plate == "ABC-1234"
    assert session.last_seen_camera == "CAM-23"
    assert session.gate_reference_only is False
    assert np.allclose(session.feature_vector, np.array([0.6, 0.8]))
    assert len(session.reference_feature_vectors) == 1
    assert session.reference_source_cameras == ["CAM-ENTRY"]
    assert session.reference_snapshot_paths == []
    assert session.gate_snapshot_paths == []
    assert session.first_seen_at == registry_now
    assert session.last_seen_at == registry_now
    assert (registry_now - session.last_seen_at).total_seconds() == 0
    assert registry._last_anpr_entry_at["ABC-1234"] == registry_now


def test_validated_entry_is_indexed_and_matchable_in_crossing_camera_area(tmp_path):
    registry_now = datetime(2026, 7, 21, 12, 10)
    config = SimpleNamespace(
        areas=[
            AreaEntry(area_id="B1-ENTRY", floor="B1"),
            AreaEntry(area_id="B2-PARKING", floor="B2"),
        ],
        cameras=[
            CameraEntry(id="CAM-23", floor="B1", area="B1-ENTRY"),
            CameraEntry(id="CAM-04", floor="B1", area="B1-ENTRY"),
            CameraEntry(id="CAM-10", floor="B2", area="B2-PARKING"),
        ],
    )
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        area_registry=AreaRegistry(config),
        clock=lambda: registry_now,
    )
    registry._reid_matcher = _CosineMatcher()
    feature = (3.0, 4.0)

    session_id = registry.register_validated_entry(
        plate="ABC-1234",
        decision_id="decision-area",
        attempt_id="attempt-area",
        crossing_id="crossing-area",
        timestamp=registry_now.astimezone(timezone.utc),
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=(feature,),
        attempt_feature_vectors=(),
    )

    session = registry._sessions[session_id]
    assert session.current_area == "B1-ENTRY"
    assert session.area_entered_at == registry_now
    assert [item.session_id for item in registry.sessions_in_area("B1-ENTRY")] == [
        session_id
    ]
    assert (
        registry.match_global_session(
            np.asarray(feature, dtype=np.float32),
            camera_id="CAM-04",
            area_id="B1-ENTRY",
            similarity_threshold=0.9,
        )
        == session_id
    )
    assert (
        registry.match_global_session(
            np.asarray(feature, dtype=np.float32),
            camera_id="CAM-10",
            area_id="B2-PARKING",
            similarity_threshold=0.9,
        )
        is None
    )


def test_validated_entry_resolves_normalized_camera_alias_to_zoning_id():
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = _minimal_registry(clock=lambda: registry_now)
    config = SimpleNamespace(
        areas=[AreaEntry(area_id="B1-ENTRY", floor="B1")],
        cameras=[CameraEntry(id="CAM-23", floor="B1", area="B1-ENTRY")],
    )
    registry._area_registry = AreaRegistry(config)

    session_id = registry.register_validated_entry(
        plate="ABC-1234",
        decision_id="decision-camera-alias",
        attempt_id="attempt-camera-alias",
        crossing_id="crossing-camera-alias",
        timestamp=registry_now.astimezone(timezone.utc),
        crossing_camera_id="cam_23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )

    session = registry._sessions[session_id]
    assert session.last_seen_camera == "CAM-23"
    assert session.current_area == "B1-ENTRY"
    assert [item.session_id for item in registry.sessions_in_area("B1-ENTRY")] == [
        session_id
    ]


def test_validated_entry_rolls_back_partial_gallery_failure():
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = _minimal_registry(clock=lambda: registry_now)
    config = SimpleNamespace(
        areas=[AreaEntry(area_id="B1-ENTRY", floor="B1")],
        cameras=[CameraEntry(id="CAM-23", floor="B1", area="B1-ENTRY")],
    )
    registry._area_registry = AreaRegistry(config)
    registry._matching_config.use_faiss_index = True
    registry._gallery_index = _PartiallyFailingGalleryIndex()
    claimed = []
    registry._claim_plate_globally = lambda *args, **kwargs: claimed.append(
        (args, kwargs)
    )

    with pytest.raises(RuntimeError, match="gallery write failed"):
        registry.register_validated_entry(
            plate="ABC-1234",
            decision_id="decision-failure",
            attempt_id="attempt-failure",
            crossing_id="crossing-failure",
            timestamp=registry_now.astimezone(timezone.utc),
            crossing_camera_id="CAM-23",
            crossing_feature_vectors=((3.0, 4.0),),
            attempt_feature_vectors=(),
        )

    assert registry._sessions == {}
    assert registry.sessions_in_area("B1-ENTRY") == []
    assert registry._gallery_index.session_ids == set()
    assert registry._last_anpr_entry_at == {}
    assert claimed == []


def test_validated_entry_is_rejected_when_exit_won_before_publication(tmp_path):
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = VehicleRegistry(image_dir=str(tmp_path), clock=lambda: registry_now)
    crossing_at = registry_now.replace(minute=9)

    registry._handle_exit("ABC-1234", registry_now)

    with pytest.raises(ValueError, match="superseded by exit"):
        registry.register_validated_entry(
            plate="ABC-1234",
            decision_id="decision-exit-won",
            attempt_id="attempt-exit-won",
            crossing_id="crossing-exit-won",
            timestamp=crossing_at,
            crossing_camera_id="CAM-23",
            crossing_feature_vectors=((3.0, 4.0),),
            attempt_feature_vectors=(),
        )

    assert registry._sessions == {}
    assert registry._last_anpr_entry_at == {}


def test_delayed_old_exit_before_publication_does_not_block_newer_entry(tmp_path):
    plate = "REENTRY-1"
    exit_at = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    reentry_at = exit_at + timedelta(hours=1)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: reentry_at,
    )

    registry.register_anpr_event(plate, "exit", timestamp=exit_at)
    session_id = registry.register_validated_entry(
        plate=plate,
        decision_id="decision-after-delayed-exit",
        attempt_id="attempt-after-delayed-exit",
        crossing_id="crossing-after-delayed-exit",
        timestamp=reentry_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )

    assert registry._last_anpr_exit_at[plate] == exit_at
    assert registry._sessions[session_id].first_seen_at == reentry_at
    assert registry._last_anpr_entry_at[plate] == reentry_at


def test_delayed_old_exit_after_publication_preserves_new_visit_and_pending_event(
    tmp_path,
):
    plate = "REENTRY-2"
    old_exit_at = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    reentry_at = old_exit_at + timedelta(hours=1)
    pending_at = reentry_at + timedelta(minutes=1)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: pending_at,
    )
    session_id = registry.register_validated_entry(
        plate=plate,
        decision_id="decision-before-delayed-exit",
        attempt_id="attempt-before-delayed-exit",
        crossing_id="crossing-before-delayed-exit",
        timestamp=reentry_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )
    pending = registry.register_anpr_event(
        plate,
        "entry",
        timestamp=pending_at,
        camera_id="CAM-ENTRY",
    )
    registry._gallery_last_add[(plate, "CAM-23")] = 123.0

    registry.register_anpr_event(plate, "exit", timestamp=old_exit_at)

    assert registry._sessions[session_id].status == "confirmed"
    assert registry._pending_events[pending.event_id] is pending
    assert registry._last_anpr_entry_at[plate] == pending_at
    assert registry._last_anpr_exit_at[plate] == old_exit_at
    assert registry._gallery_last_add[(plate, "CAM-23")] == 123.0


def test_older_exit_delivery_does_not_overwrite_newer_tombstone(tmp_path):
    plate = "REENTRY-3"
    old_exit_at = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    newer_exit_at = old_exit_at + timedelta(hours=2)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: newer_exit_at,
    )

    registry.register_anpr_event(plate, "exit", timestamp=newer_exit_at)
    registry.register_anpr_event(plate, "exit", timestamp=old_exit_at)

    assert registry._last_anpr_exit_at[plate] == newer_exit_at


def test_aware_current_exit_closes_older_identity_with_naive_registry_clock(
    tmp_path,
):
    """Production uses a naive datetime.now clock while HTTP source time is aware."""
    plate = "CLOCK-SHAPE-1"
    entry_at = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    exit_at = entry_at + timedelta(hours=1)
    clock_sample = datetime(2026, 7, 21, 12, 0)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: clock_sample,
    )
    session_id = registry.register_validated_entry(
        plate=plate,
        decision_id="decision-naive-clock-current-exit",
        attempt_id="attempt-naive-clock-current-exit",
        crossing_id="crossing-naive-clock-current-exit",
        timestamp=entry_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )

    registry.register_anpr_event(plate, "exit", timestamp=exit_at)

    assert session_id not in registry._sessions
    assert registry._last_anpr_exit_at[plate] == normalize_timestamp_for_clock(
        exit_at,
        clock_sample,
    )


def test_aware_delayed_exit_preserves_newer_identity_with_naive_registry_clock(
    tmp_path,
):
    plate = "CLOCK-SHAPE-2"
    old_exit_at = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    reentry_at = old_exit_at + timedelta(hours=1)
    clock_sample = datetime(2026, 7, 21, 12, 0)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: clock_sample,
    )
    session_id = registry.register_validated_entry(
        plate=plate,
        decision_id="decision-naive-clock-delayed-exit",
        attempt_id="attempt-naive-clock-delayed-exit",
        crossing_id="crossing-naive-clock-delayed-exit",
        timestamp=reentry_at,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )

    registry.register_anpr_event(plate, "exit", timestamp=old_exit_at)

    assert registry._sessions[session_id].status == "confirmed"
    assert registry._last_anpr_exit_at[plate] == normalize_timestamp_for_clock(
        old_exit_at,
        clock_sample,
    )


def test_exit_at_exact_visit_boundary_closes_that_identity(tmp_path):
    """The destructive cutoff is inclusive: state at the exit instant is old."""
    plate = "EXIT-BOUNDARY"
    boundary = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: boundary,
    )
    session_id = registry.register_validated_entry(
        plate=plate,
        decision_id="decision-exit-boundary",
        attempt_id="attempt-exit-boundary",
        crossing_id="crossing-exit-boundary",
        timestamp=boundary,
        crossing_camera_id="CAM-23",
        crossing_feature_vectors=((3.0, 4.0),),
        attempt_feature_vectors=(),
    )

    registry.register_anpr_event(plate, "exit", timestamp=boundary)

    assert session_id not in registry._sessions
    assert registry._last_anpr_exit_at[plate] == boundary


def test_delayed_exit_detaches_newer_candidate_from_older_pending_event(tmp_path):
    """A late binding cannot make a next-visit car look older than the exit."""
    plate = "REENTRY-CANDIDATE"
    old_entry_at = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)
    exit_at = old_entry_at + timedelta(hours=1)
    candidate_at = exit_at + timedelta(hours=1)
    registry = VehicleRegistry(
        image_dir=str(tmp_path),
        clock=lambda: candidate_at,
    )
    event = PendingANPREvent(
        event_id="old-event",
        plate=plate,
        direction="entry",
        timestamp=old_entry_at,
        camera_id="CAM-ENTRY",
        status="provisional",
        candidate_id="new-candidate",
    )
    candidate = ParkEntryCandidate(
        candidate_id="new-candidate",
        camera_id="CAM-23",
        track_id=42,
        entered_at=candidate_at,
        last_seen_at=candidate_at,
        status="provisional",
        bound_event_id=event.event_id,
    )
    registry._pending_events[event.event_id] = event
    registry._pending_event_order.append(event.event_id)
    registry._last_anpr_entry[event.camera_id] = event.event_id
    registry._park_entry_candidates[candidate.candidate_id] = candidate

    registry.register_anpr_event(plate, "exit", timestamp=exit_at)

    assert event.event_id not in registry._pending_events
    assert event.event_id not in registry._pending_event_order
    assert event.camera_id not in registry._last_anpr_entry
    assert registry._park_entry_candidates[candidate.candidate_id] is candidate
    assert candidate.status == "open"
    assert candidate.bound_event_id is None


def test_exit_waiting_on_registry_lock_removes_just_published_identity(
    tmp_path,
    monkeypatch,
):
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = VehicleRegistry(image_dir=str(tmp_path), clock=lambda: registry_now)
    insert_started = threading.Event()
    allow_insert = threading.Event()
    errors = []
    original_upsert = registry._gallery_index_upsert_strict

    def blocked_upsert(session):
        insert_started.set()
        assert allow_insert.wait(timeout=2)
        original_upsert(session)

    monkeypatch.setattr(registry, "_gallery_index_upsert_strict", blocked_upsert)

    def publish():
        try:
            registry.register_validated_entry(
                plate="ABC-1234",
                decision_id="decision-publish-won",
                attempt_id="attempt-publish-won",
                crossing_id="crossing-publish-won",
                timestamp=registry_now.replace(minute=9),
                crossing_camera_id="CAM-23",
                crossing_feature_vectors=((3.0, 4.0),),
                attempt_feature_vectors=(),
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    publish_thread = threading.Thread(target=publish)
    publish_thread.start()
    assert insert_started.wait(timeout=2)

    exit_thread = threading.Thread(
        target=registry._handle_exit,
        args=("ABC-1234", registry_now),
    )
    exit_thread.start()
    allow_insert.set()
    publish_thread.join(timeout=2)
    exit_thread.join(timeout=2)

    assert not publish_thread.is_alive()
    assert not exit_thread.is_alive()
    assert errors == []
    assert registry._sessions == {}
    assert registry._last_anpr_exit_at["ABC-1234"] == registry_now


def test_validated_entry_fails_closed_when_crossing_camera_is_not_zoned():
    registry_now = datetime(2026, 7, 21, 12, 10)
    registry = _minimal_registry(clock=lambda: registry_now)
    config = SimpleNamespace(
        areas=[AreaEntry(area_id="B1-ENTRY", floor="B1")],
        cameras=[CameraEntry(id="CAM-04", floor="B1", area="B1-ENTRY")],
    )
    registry._area_registry = AreaRegistry(config)

    with pytest.raises(
        ValueError,
        match="crossing camera has no configured zoning area",
    ):
        registry.register_validated_entry(
            plate="ABC-1234",
            decision_id="decision-unzoned",
            attempt_id="attempt-unzoned",
            crossing_id="crossing-unzoned",
            timestamp=registry_now.astimezone(timezone.utc),
            crossing_camera_id="CAM-23",
            crossing_feature_vectors=((3.0, 4.0),),
            attempt_feature_vectors=(),
        )

    assert registry._sessions == {}
    assert registry._last_anpr_entry_at == {}
