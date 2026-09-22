from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import sqlite3
import time

import pytest

from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    AttemptInput,
    CrossingInput,
    CrossingRole,
    EntryMode,
    FrameEvidence,
    IngestResult,
    PlateEvidence,
    PlateReadState,
)
from src.entry.durability import EntryDurabilityStore
from src.entry.identity import IdentitySupersededByExit
from src.entry.local_zone import LocalZoneCrossingBridge
from src.entry.settings import EntrySettings


NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def _try_open_store(directory, result_queue):
    try:
        store = EntryDurabilityStore(directory)
    except OSError as exc:
        result_queue.put(type(exc).__name__)
    else:
        store.close()
        result_queue.put("opened")


def _attempt() -> AttemptInput:
    return AttemptInput(
        attempt_id="attempt-durable-1",
        source_event_id="source-durable-1",
        camera_id="CAM23",
        captured_at=NOW,
        reported_plate="ABC-1234",
        reported_confidence=0.99,
        metadata={"producer": "test"},
    )


def _result() -> IngestResult:
    return IngestResult(
        resource_id="attempt-durable-1",
        accepted=True,
        duplicate=False,
        mode=EntryMode.SHADOW,
        evidence_count=1,
    )


def test_exact_retry_survives_sqlite_reopen_and_keeps_pending_input(tmp_path):
    state_dir = tmp_path / "entry-state"
    store = EntryDurabilityStore(str(state_dir))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    assert (state_dir / "entry_v2_durability.sqlite3").exists()

    store.close()
    restarted = EntryDurabilityStore(str(state_dir))
    receipt = restarted.accept("attempt", _attempt(), (b"vehicle-image",))

    assert receipt is not None
    assert receipt.duplicate is True
    assert receipt.receipt_status == "pending"
    recovered = restarted.recovery_inputs()
    assert [(kind, request.attempt_id, images) for kind, request, images in recovered] == [
        ("attempt", "attempt-durable-1", (b"vehicle-image",))
    ]
    restarted.close()


def test_terminal_receipt_is_retained_and_never_reacknowledged_as_active(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    store.record_result("attempt", _result())
    store.mark_terminal("attempt", "attempt-durable-1", "expired", "identity_ttl_expired")
    store.close()

    restarted = EntryDurabilityStore(str(tmp_path))
    receipt = restarted.accept(
        "attempt", _attempt(), (b"vehicle-image",)
    )

    assert receipt is not None
    assert receipt.accepted is False
    assert receipt.receipt_status == "expired"
    assert receipt.duplicate is True
    assert restarted.recovery_inputs() == []
    restarted.close()


def test_decision_callback_payload_is_committed_before_retry_after_restart(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    payload = {
        "decision_id": "decision-durable-1",
        "reported_metadata": {"first": 1, "second": 2},
        "status": "confirmed",
    }
    store.record_decision(
        (("attempt", "attempt-durable-1"), ("crossing", "crossing-not-present")),
        payload,
        None,
        {"mode": "shadow", "group_id": "group-1"},
    )

    with sqlite3.connect(store.path) as connection:
        persisted = connection.execute(
            "SELECT callback_json FROM entry_receipts WHERE kind='attempt'"
        ).fetchone()[0]
    assert persisted == json.dumps(payload, separators=(",", ":"), default=str)
    store.close()

    restarted = EntryDurabilityStore(str(tmp_path))
    assert restarted.pending_callbacks() == [payload]
    restarted.mark_callback("decision-durable-1", True)
    assert restarted.pending_callbacks() == []
    restarted.close()


def test_second_process_cannot_open_live_journal_and_restart_can(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_try_open_store, args=(str(tmp_path), queue))
    child.start()
    child.join(timeout=5)
    assert child.exitcode == 0
    assert queue.get(timeout=1) == "EntryDurabilityLocked"

    store.close()
    restarted = context.Process(target=_try_open_store, args=(str(tmp_path), queue))
    restarted.start()
    restarted.join(timeout=5)
    assert restarted.exitcode == 0
    assert queue.get(timeout=1) == "opened"


def test_closed_or_unwritable_journal_fails_before_an_ingest_ack(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    store.close()
    with pytest.raises(OSError, match="entry_v2_durability_store_closed"):
        store.accept("attempt", _attempt(), (b"vehicle-image",))


def test_shadow_journal_cannot_be_reused_for_authoritative_hydration(tmp_path):
    shadow = EntryCoordinator(
        replace(EntrySettings(), mode=EntryMode.SHADOW),
        _SlowProcessor(),
        _Sink(),
        durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    shadow.durability_store.close()
    authoritative_store = EntryDurabilityStore(str(tmp_path))
    try:
        with pytest.raises(OSError, match="entry_v2_durability_mode_mismatch"):
            EntryCoordinator(
                replace(EntrySettings(), mode=EntryMode.AUTHORITATIVE),
                _SlowProcessor(),
                _Sink(),
                durability_store=authoritative_store,
            )
    finally:
        authoritative_store.close()


def test_post_decision_storage_failure_leaves_recoverable_callback_receipt(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    payload = {"decision_id": "decision-disk-1", "status": "confirmed"}
    store.record_decision(
        (("attempt", "attempt-durable-1"),), payload, None,
        {"mode": "shadow", "group_id": "group-disk"},
    )
    store.close()  # Models a local storage failure after decision commit.
    with pytest.raises(OSError, match="entry_v2_durability_store_closed"):
        store.mark_callback("decision-disk-1", True)
    restarted = EntryDurabilityStore(str(tmp_path))
    assert restarted.pending_callbacks() == [payload]
    restarted.close()


def test_callback_prefix_does_not_mark_another_decision_resolved(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    first = _attempt()
    second = replace(
        first,
        attempt_id="attempt-durable-2",
        source_event_id="source-durable-2",
    )
    assert store.accept("attempt", first, (b"one",)) is None
    assert store.accept("attempt", second, (b"two",)) is None
    store.record_decision(
        (("attempt", first.attempt_id),),
        {"decision_id": "decision-1", "status": "confirmed"},
        None,
        {"mode": "shadow", "group_id": "group-1"},
    )
    store.record_decision(
        (("attempt", second.attempt_id),),
        {"decision_id": "decision-10", "status": "confirmed"},
        None,
        {"mode": "shadow", "group_id": "group-2"},
    )
    store.mark_callback("decision-1", True)
    assert store.pending_callbacks() == [
        {"decision_id": "decision-10", "status": "confirmed"}
    ]
    store.close()


class _SlowProcessor:
    def __init__(self):
        self.calls = []

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        self.calls.append(event_id)
        time.sleep(0.03)
        return (
            FrameEvidence(
                evidence_id=event_id + ":0",
                embedding=(1.0, 0.0),
                plate=PlateEvidence(
                    evidence_id=event_id + ":0",
                    camera_id=camera_id,
                    source_role=source_role,
                    state=PlateReadState.NO_PLATE,
                ),
            ),
        )


class _Sink:
    def deliver(self, payload):
        del payload
        return DeliveryResult(True, 1, "")


class _Publisher:
    def __init__(self):
        self.identities = []

    def publish(self, identity):
        self.identities.append(identity)


class _SupersededPublisher:
    def publish(self, identity):
        del identity
        raise IdentitySupersededByExit(NOW + timedelta(minutes=1))


class _StaleAfterExitSink:
    def __init__(self):
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(
            True,
            1,
            publish_identity=False,
            session_committed=False,
            ack_result="stale_after_exit",
        )


def test_restart_callback_replay_publishes_persisted_identity(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    store.record_decision(
        (("attempt", "attempt-durable-1"),),
        {"decision_id": "decision-restart-1", "status": "confirmed"},
        {
            "decision_id": "decision-restart-1",
            "canonical_plate": "ABC-1234",
            "attempt_id": "attempt-durable-1",
            "crossing_id": "crossing-restart-1",
            "entered_at": NOW.isoformat(),
            "crossing_camera_id": "CAM23",
            "crossing_embeddings": [[1.0, 0.0]],
            "attempt_embeddings": [["CAM23", [1.0, 0.0]]],
            "gallery_authorization": None,
        },
        {"mode": "authoritative", "group_id": "group-1"},
    )
    store.close()
    publisher = _Publisher()
    coordinator = EntryCoordinator(
        replace(EntrySettings(), mode=EntryMode.AUTHORITATIVE),
        _SlowProcessor(),
        _Sink(),
        identity_publisher=publisher,
        durability_store=EntryDurabilityStore(str(tmp_path)),
    )

    assert coordinator.retry_pending_callbacks() == {"decision-restart-1": True}
    assert [item.decision_id for item in publisher.identities] == ["decision-restart-1"]
    coordinator.durability_store.close()


def test_restart_callback_superseded_identity_persists_exit_boundary(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    payload = {"decision_id": "decision-superseded-1", "status": "confirmed"}
    identity = {
        "decision_id": "decision-superseded-1", "canonical_plate": "ABC-1234",
        "attempt_id": "attempt-durable-1", "crossing_id": "crossing-superseded-1",
        "entered_at": NOW.isoformat(), "crossing_camera_id": "CAM23",
        "crossing_embeddings": [[1.0, 0.0]],
        "attempt_embeddings": [["CAM23", [1.0, 0.0]]], "gallery_authorization": None,
    }
    candidate = {
        "decision_id": "decision-superseded-1", "group_id": "group-superseded",
        "decision_status": "confirmed", "canonical_plate_key": "ABC1234",
        "first_attempt_at": NOW.isoformat(), "entry_captured_at": NOW.isoformat(),
        "crossing_role": "primary", "crossing_camera_id": "CAM23",
        "crossing_source": "hikvision", "embeddings": [[1.0, 0.0]],
        "exit_captured_at": None,
    }
    store.record_decision(
        (("attempt", "attempt-durable-1"),), payload, identity,
        {"mode": "authoritative", "group_id": "group-superseded"}, candidate,
    )
    store.close()
    coordinator = EntryCoordinator(
        replace(EntrySettings(), mode=EntryMode.AUTHORITATIVE), _SlowProcessor(), _Sink(),
        identity_publisher=_SupersededPublisher(), durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    assert coordinator.retry_pending_callbacks() == {"decision-superseded-1": True}
    assert coordinator.state_summary()["open_journey_count"] == 0
    persisted = coordinator.durability_store.finalized_journeys()
    assert persisted[0]["exit_captured_at"] == (NOW + timedelta(minutes=1)).isoformat()
    coordinator.durability_store.close()


def test_restart_restores_open_confirmed_identity_but_not_exited_one(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    identity = {
        "decision_id": "decision-open-1", "canonical_plate": "ABC-1234",
        "attempt_id": "attempt-durable-1", "crossing_id": "crossing-open-1",
        "entered_at": NOW.isoformat(), "crossing_camera_id": "CAM23",
        "crossing_embeddings": [[1.0, 0.0]],
        "attempt_embeddings": [["CAM23", [1.0, 0.0]]], "gallery_authorization": None,
    }
    store.record_decision(
        (("attempt", "attempt-durable-1"),),
        {"decision_id": "decision-open-1", "status": "confirmed"},
        identity, {"mode": "authoritative", "group_id": "group-open"},
    )
    store.mark_callback("decision-open-1", True)
    for decision_id, exit_at in (("decision-open-1", None), ("decision-exited-1", NOW.isoformat())):
        store.save_finalized_journey({
            "decision_id": decision_id, "group_id": "group-open",
            "decision_status": "confirmed", "canonical_plate_key": "ABC1234",
            "first_attempt_at": NOW.isoformat(), "entry_captured_at": NOW.isoformat(),
            "crossing_role": "primary", "crossing_camera_id": "CAM23",
            "crossing_source": "hikvision", "embeddings": [[1.0, 0.0]],
            "exit_captured_at": exit_at,
        })
    store.close()
    publisher = _Publisher()
    coordinator = EntryCoordinator(
        replace(EntrySettings(), mode=EntryMode.AUTHORITATIVE), _SlowProcessor(), _Sink(),
        identity_publisher=publisher, durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    coordinator.recover_durable_inputs()
    assert [item.decision_id for item in publisher.identities] == ["decision-open-1"]
    coordinator.durability_store.close()


def test_pms_stale_after_exit_during_va_downtime_does_not_hydrate_identity(tmp_path):
    store = EntryDurabilityStore(str(tmp_path))
    assert store.accept("attempt", _attempt(), (b"vehicle-image",)) is None
    identity = {
        "decision_id": "decision-stale-1", "canonical_plate": "ABC-1234",
        "attempt_id": "attempt-durable-1", "crossing_id": "crossing-stale-1",
        "entered_at": NOW.isoformat(), "crossing_camera_id": "CAM23",
        "crossing_embeddings": [[1.0, 0.0]],
        "attempt_embeddings": [["CAM23", [1.0, 0.0]]], "gallery_authorization": None,
    }
    payload = {"decision_id": "decision-stale-1", "status": "confirmed"}
    store.record_decision(
        (("attempt", "attempt-durable-1"),), payload, identity,
        {"mode": "authoritative", "group_id": "group-stale"},
    )
    store.mark_callback("decision-stale-1", True)
    store.save_finalized_journey({
        "decision_id": "decision-stale-1", "group_id": "group-stale",
        "decision_status": "confirmed", "canonical_plate_key": "ABC1234",
        "first_attempt_at": NOW.isoformat(), "entry_captured_at": NOW.isoformat(),
        "crossing_role": "primary", "crossing_camera_id": "CAM23",
        "crossing_source": "hikvision", "embeddings": [[1.0, 0.0]],
        "exit_captured_at": None,
    })
    store.close()
    publisher, sink = _Publisher(), _StaleAfterExitSink()
    coordinator = EntryCoordinator(
        replace(EntrySettings(), mode=EntryMode.AUTHORITATIVE), _SlowProcessor(), sink,
        identity_publisher=publisher, durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    coordinator.recover_durable_inputs()
    assert publisher.identities == []
    assert sink.payloads == [payload]
    assert coordinator.state_summary()["finalized_journey_count"] == 0
    coordinator.durability_store.close()


def _crossing(value: str) -> CrossingInput:
    return CrossingInput(
        crossing_id=value,
        source_event_id=value,
        camera_id="CAM23",
        captured_at=NOW,
        line_id="PARK_ENTRY",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
        metadata={"source": "va_local_zone"},
    )


def test_saturated_local_bridge_retains_persisted_work_until_worker_drains(tmp_path):
    settings = replace(
        EntrySettings(),
        mode=EntryMode.SHADOW,
        max_pending_attempts=8,
        max_pending_crossings=8,
        max_pending_callbacks=8,
        max_concurrent_ingest_requests=1,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        pms_base_url="http://pms-ai",
        service_key="key",
    )
    processor = _SlowProcessor()
    state = EntryDurabilityStore(str(tmp_path))
    coordinator = EntryCoordinator(settings, processor, _Sink(), durability_store=state)
    bridge = LocalZoneCrossingBridge(coordinator, max_queued=1)
    first, second = _crossing("local-durable-1"), _crossing("local-durable-2")
    assert coordinator.admit_local_crossing(first, (b"one",)) is None
    assert bridge._queue(first, (b"one",), retry_attempt=0)
    assert coordinator.admit_local_crossing(second, (b"two",)) is None
    assert bridge._queue(second, (b"two",), retry_attempt=0)

    assert bridge.wait_for_idle(timeout=2)
    bridge.close(wait=True)
    assert processor.calls == ["local-durable-1", "local-durable-2"]
    assert (tmp_path / "entry_v2_durability.sqlite3").exists()
    assert bridge.metrics()["submissions_capacity_rejected"] == 1


def test_periodic_maintenance_expires_full_pool_without_another_accept(tmp_path):
    now = [NOW]
    settings = replace(
        EntrySettings(),
        mode=EntryMode.SHADOW,
        max_pending_attempts=1,
        max_pending_crossings=1,
        max_pending_callbacks=1,
        identity_ttl_minutes=1,
        observation_ttl_minutes=1,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        pms_base_url="http://pms-ai",
        service_key="key",
    )
    coordinator = EntryCoordinator(
        settings,
        _SlowProcessor(),
        _Sink(),
        clock=lambda: now[0],
        durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    coordinator.ingest_attempt(_attempt(), (b"vehicle-image",))
    now[0] = NOW.replace(minute=NOW.minute + 2)
    coordinator.maintain()

    receipt = coordinator.durability_store.accept(
        "attempt", _attempt(), (b"vehicle-image",)
    )
    assert receipt is not None
    assert receipt.receipt_status == "expired"
    assert receipt.accepted is False


def test_coordinator_restart_replays_only_persisted_undecided_evidence(tmp_path):
    settings = replace(
        EntrySettings(),
        mode=EntryMode.SHADOW,
        max_pending_attempts=8,
        max_pending_crossings=8,
        max_pending_callbacks=8,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        pms_base_url="http://pms-ai",
        service_key="key",
    )
    initial = EntryCoordinator(
        settings,
        _SlowProcessor(),
        _Sink(),
        durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    initial.ingest_attempt(_attempt(), (b"vehicle-image",))
    initial.durability_store.close()

    after_crash_processor = _SlowProcessor()
    restarted = EntryCoordinator(
        settings,
        after_crash_processor,
        _Sink(),
        durability_store=EntryDurabilityStore(str(tmp_path)),
    )
    assert restarted.recover_durable_inputs() == 1
    assert after_crash_processor.calls == ["attempt-durable-1"]
