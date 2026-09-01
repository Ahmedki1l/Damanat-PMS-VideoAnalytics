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


def _entry_coordinator(mode, unavailable_reason="", linked=False):
    """`linked` folds Entry V2 state back into the SERVICE health verdict.

    POLICY CHANGE. Entry V2 conditions used to set the top-level `status`
    unconditionally. They no longer do: Entry V2 is one pipeline inside
    VideoAnalytics and in shadow it is observation-only, so it must not be able
    to report the whole service degraded — the gateway aggregates that verdict
    and operators page on it. `pending_exit_count > 0` did exactly that, holding
    VA amber over 39 exits retained from the two days v3 could not confirm
    anything, while every camera, stream and inference path was healthy.

    Every condition below is still asserted. It now lands in `entry_v2_status`
    and `entry_v2_reasons`, and only reaches `status`/`health_reasons` when
    ENTRY_V2_AFFECTS_SERVICE_HEALTH is on — which these tests exercise both ways.
    """
    dependency = _UnusedEntryDependency()
    return EntryCoordinator(
        EntrySettings(
            mode=mode,
            service_key="health-test",
            entry_v2_affects_service_health=linked,
        ),
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


def test_linking_restores_service_level_reporting(tmp_path) -> None:
    """ENTRY_V2_AFFECTS_SERVICE_HEALTH=1 folds the pipeline back into the
    service verdict, for a deployment that wants it that way."""
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE, linked=True)
    summary = coordinator.state_summary()
    summary["pending_exit_count"] = 1
    coordinator.state_summary = lambda: summary

    response = _health(coordinator, tmp_path)

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert (
        "entry_v2 unmatched exit boundaries: 1/4096"
        in response.json()["health_reasons"]
    )


def test_linking_restores_the_503_for_a_hard_failure(tmp_path) -> None:
    coordinator = _entry_coordinator(EntryMode.AUTHORITATIVE, linked=True)
    summary = coordinator.state_summary()
    summary["attempt_count"] = coordinator.settings.max_pending_attempts
    coordinator.state_summary = lambda: summary

    response = _health(coordinator, tmp_path)

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_the_link_state_is_reported_so_an_operator_can_see_it(tmp_path) -> None:
    unlinked = _health(_entry_coordinator(EntryMode.SHADOW), tmp_path)
    linked = _health(_entry_coordinator(EntryMode.SHADOW, linked=True), tmp_path)

    assert unlinked.json()["entry_v2_linked_to_service_health"] is False
    assert linked.json()["entry_v2_linked_to_service_health"] is True


def test_a_healthy_pipeline_reports_no_entry_v2_reasons(tmp_path) -> None:
    response = _health(_entry_coordinator(EntryMode.SHADOW), tmp_path)

    assert response.json()["status"] == "ok"
    assert response.json()["entry_v2_status"] == "ok"
    assert response.json()["entry_v2_reasons"] == []
