from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api import create_app
from src.entry.analyzer import PLATE_CROP_SUBDIRECTORY
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import EntryMode
from src.entry.settings import EntrySettings


class _UnusedEntryDependency:
    def analyze(self, **kwargs):
        raise AssertionError(kwargs)

    def deliver(self, payload):
        raise AssertionError(payload)


def _entry_coordinator(mode, unavailable_reason=""):
    """Entry V2 state can NEVER change the SERVICE health verdict.

    POLICY. Entry V2 conditions used to set the top-level `status`, first
    unconditionally and then behind ENTRY_V2_AFFECTS_SERVICE_HEALTH. Both are
    gone. Entry V2 is one pipeline inside VideoAnalytics and in shadow it is
    observation-only, so it must not report the whole service degraded — the
    gateway aggregates that verdict into a dashboard-wide outage banner and
    operators page on it. `pending_exit_count > 0` did exactly that, holding VA
    amber over 39 exits retained from the two days v3 could not confirm
    anything, while every camera, stream and inference path was healthy.

    The switch that replaced it was worse: `api.py` honoured it while the
    engine's local-zone check degraded from inside `get_engine_status()`,
    upstream of the gate, so a latched `crossing_ingest_failed` reported the
    service degraded for a day while the same payload declared the pipeline
    unlinked. `status` now answers one question — is VideoAnalytics running.

    Every condition below is still asserted. It lands in `entry_v2_status` and
    `entry_v2_reasons`, which is what to page on for the pipeline.
    """
    dependency = _UnusedEntryDependency()
    return EntryCoordinator(
        EntrySettings(mode=mode, service_key="health-test"),
        dependency,
        dependency,
        unavailable_reason=unavailable_reason,
    )


def _health(coordinator, tmp_path):
    return TestClient(
        create_app(
            entry_coordinator=coordinator, snapshot_base_dir=str(tmp_path)
        )
    ).get("/api/health")


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

    # Unlinked by default: the PIPELINE is unavailable, the SERVICE is not.
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert (
        "entry_v2 unavailable: entry_v2_invalid_configuration:VA_PROCESS_COUNT"
        in response.json()["entry_v2_reasons"]
    )
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


def test_entry_plate_crop_diagnostics_are_not_public_snapshots(tmp_path) -> None:
    private_dir = tmp_path / PLATE_CROP_SUBDIRECTORY
    private_dir.mkdir()
    (private_dir / "plate.jpg").write_bytes(b"private-plate")
    public_image = tmp_path / "slot.jpg"
    public_image.write_bytes(b"public-snapshot")
    app = create_app(
        entry_coordinator=_entry_coordinator(EntryMode.OFF),
        snapshot_base_dir=str(tmp_path),
    )
    client = TestClient(app)

    private_response = client.get(
        f"/pms-video-analytics/snapshots/{PLATE_CROP_SUBDIRECTORY}/plate.jpg"
    )
    public_response = client.get("/pms-video-analytics/snapshots/slot.jpg")

    assert private_response.status_code == 404
    assert public_response.status_code == 200
    assert public_response.content == b"public-snapshot"


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

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert response.json()["entry_v2"]["pending_callback_count"] == 128
    assert (
        "entry_v2 callback backlog: 128/128"
        in response.json()["entry_v2_reasons"]
    )


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

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert (
        "entry_v2 crossing capacity exhausted: 256/256"
        in response.json()["entry_v2_reasons"]
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

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert (
        "entry_v2 attempt capacity exhausted: 256/256"
        in response.json()["entry_v2_reasons"]
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

    # The regression this decoupling exists for: one retained exit boundary
    # must not take the whole service amber.
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "degraded"
    assert response.json()["entry_v2"]["pending_exit_count"] == 1
    assert (
        "entry_v2 unmatched exit boundaries: 1/4096"
        in response.json()["entry_v2_reasons"]
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

    assert before.json()["entry_v2_status"] == "degraded"
    assert cancelled.status_code == 200
    assert cancelled.json()["removed_pending_exits"] == 1
    assert after.status_code == 200
    assert after.json()["status"] == "ok"
    assert after.json()["entry_v2_status"] == "ok"
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

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert any(
        "entry_v2 journey lifecycle capacity exhausted" in reason
        for reason in response.json()["entry_v2_reasons"]
    )


# -- the same conditions, with Entry V2 deliberately linked back in ----------


def test_no_entry_v2_condition_can_degrade_the_service(tmp_path) -> None:
    """The whole point: a hard pipeline failure leaves the SERVICE verdict ok.

    There is no longer a setting that changes this, so there is nothing to
    exercise "both ways" — that switch is what let the engine bypass the
    decoupling and report a day-long degrade behind an `unlinked` payload.
    """
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE)
    summary = coordinator.state_summary()
    summary["attempt_count"] = coordinator.settings.max_pending_attempts
    summary["pending_exit_count"] = 1
    coordinator.state_summary = lambda: summary

    response = _health(coordinator, tmp_path)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # No engine status is wired in these tests, so the key is absent rather
    # than empty — either way nothing from entry_v2 reached the service list.
    assert not response.json().get("health_reasons")
    assert response.json()["entry_v2_status"] == "unhealthy"
    assert (
        "entry_v2 attempt capacity exhausted: 256/256"
        in response.json()["entry_v2_reasons"]
    )


def test_the_service_verdict_carries_no_entry_v2_link_field(tmp_path) -> None:
    """The field existed only to describe a switch that no longer exists."""
    response = _health(_entry_coordinator(EntryMode.SHADOW), tmp_path)

    assert "entry_v2_linked_to_service_health" not in response.json()


def test_a_healthy_pipeline_reports_no_entry_v2_reasons(tmp_path) -> None:
    response = _health(_entry_coordinator(EntryMode.SHADOW), tmp_path)

    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "ok"
    assert response.json()["entry_v2_reasons"] == []
