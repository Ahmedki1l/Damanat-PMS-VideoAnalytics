from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api import create_app
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import EntryMode
from src.entry.settings import EntrySettings


class _UnusedEntryDependency:
    def analyze(self, **kwargs):
        raise AssertionError(kwargs)

    def deliver(self, payload):
        raise AssertionError(payload)


def _entry_coordinator(mode, unavailable_reason=""):
    dependency = _UnusedEntryDependency()
    return EntryCoordinator(
        EntrySettings(mode=mode, service_key="health-test"),
        dependency,
        dependency,
        unavailable_reason=unavailable_reason,
    )


def test_invalid_authoritative_entry_configuration_makes_health_unhealthy(tmp_path):
    coordinator = _entry_coordinator(
        EntryMode.AUTHORITATIVE,
        "entry_v2_invalid_configuration:VA_PROCESS_COUNT",
    )
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert response.json()["entry_v2"] == {
        "mode": "authoritative",
        "available": False,
        "unavailable_reason": "entry_v2_invalid_configuration:VA_PROCESS_COUNT",
        "attempt_count": 0,
        "group_count": 0,
        "crossing_count": 0,
        "provisional_crossing_count": 0,
        "pending_callback_count": 0,
        "reserved_callback_count": 0,
        "analysis_inflight_count": 0,
        "late_ocr_conflict_count": 0,
        "open_journey_count": 0,
        "finalized_journey_count": 0,
        "protected_journey_count": 0,
        "journey_capacity_load": 0,
        "pending_exit_count": 0,
        "ambiguous_exit_count": 0,
        "permanent_callback_failure_count": 0,
        "max_pending_callbacks": 128,
        "max_pending_crossings": 256,
        "max_pending_attempts": 256,
        "journey_capacity": 4096,
    }


def test_entry_v2_off_is_healthy_and_explicit_in_health(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.OFF)
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2"]["mode"] == "off"
    assert response.json()["entry_v2"]["available"] is False


def test_full_callback_backlog_is_unhealthy_and_returns_503(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["pending_callback_count"] = coordinator.settings.max_pending_callbacks
    coordinator.state_summary = lambda: summary
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert response.json()["entry_v2"]["pending_callback_count"] == 128
    assert "entry_v2 callback backlog: 128/128" in response.json()["health_reasons"]


def test_full_provisional_crossing_capacity_is_unhealthy(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["provisional_crossing_count"] = coordinator.settings.max_pending_crossings
    coordinator.state_summary = lambda: summary
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert (
        "entry_v2 crossing capacity exhausted: 256/256"
        in response.json()["health_reasons"]
    )


def test_full_attempt_capacity_is_unhealthy(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["attempt_count"] = coordinator.settings.max_pending_attempts
    coordinator.state_summary = lambda: summary
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert (
        "entry_v2 attempt capacity exhausted: 256/256"
        in response.json()["health_reasons"]
    )


def test_unmatched_exit_boundary_is_degraded_and_visible(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["pending_exit_count"] = 1
    coordinator.state_summary = lambda: summary
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["entry_v2"]["pending_exit_count"] == 1
    assert (
        "entry_v2 unmatched exit boundaries: 1/4096"
        in response.json()["health_reasons"]
    )


def test_exact_operator_exit_cancellation_restores_health(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    exit_at = datetime(2026, 7, 22, 12, 15, tzinfo=timezone.utc)
    assert coordinator.record_exit("ABC-1234", exit_at) == 0
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )
    client = TestClient(app, headers={"X-Service-Key": "health-test"})

    before = client.get("/api/health")
    cancelled = client.post(
        "/api/v2/entry-cancellations",
        json={
            "exit_plate": "ABC-1234",
            "exit_captured_at": exit_at.isoformat(),
            "reason": "operator verified false exit",
        },
    )
    after = client.get("/api/health")

    assert before.json()["status"] == "degraded"
    assert cancelled.status_code == 200
    assert cancelled.json()["removed_pending_exits"] == 1
    assert after.status_code == 200
    assert after.json()["status"] == "ok"
    assert after.json()["entry_v2"]["pending_exit_count"] == 0


def test_full_journey_lifecycle_capacity_is_unhealthy(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["journey_capacity_load"] = coordinator.settings.journey_capacity
    coordinator.state_summary = lambda: summary
    app = create_app(
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path),
    )

    response = TestClient(app).get("/api/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert any(
        "entry_v2 journey lifecycle capacity exhausted" in reason
        for reason in response.json()["health_reasons"]
    )
