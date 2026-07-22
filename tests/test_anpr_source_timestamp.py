"""Source-time contract for the legacy PMS -> VA ANPR endpoint."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import EntryCapacityExceeded, EntryMode
from src.entry.settings import EntrySettings
from src.vehicle_registry.vehicle_registry import VehicleRegistry


@pytest.fixture
def client_and_registry(tmp_path):
    now = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    registry = VehicleRegistry(
        image_dir=str(tmp_path / "registry"),
        clock=lambda: now,
    )
    app = create_app(
        vehicle_registry=registry,
        snapshot_base_dir=str(tmp_path / "snapshots"),
    )
    with TestClient(app) as client:
        yield client, registry


def test_aware_captured_at_is_passed_to_registry_without_delivery_time_substitution(
    client_and_registry,
):
    client, registry = client_and_registry
    original = registry.register_anpr_event
    with patch.object(registry, "register_anpr_event", wraps=original) as register:
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
            },
        )

    assert response.status_code == 200
    supplied = register.call_args.kwargs["timestamp"]
    assert supplied == datetime.fromisoformat("2026-07-21T09:00:00+03:00")
    assert response.json()["timestamp"] == "2026-07-21T09:00:00+03:00"


def test_exit_response_echoes_aware_source_time_when_registry_clock_is_naive(
    tmp_path,
):
    class UnusedDependency:
        def analyze(self, **kwargs):
            raise AssertionError(kwargs)

        def deliver(self, payload):
            raise AssertionError(payload)

    dependency = UnusedDependency()
    coordinator = EntryCoordinator(
        EntrySettings(
            mode=EntryMode.AUTHORITATIVE,
            service_key="pms-va-secret",
        ),
        dependency,
        dependency,
    )
    registry = VehicleRegistry(
        image_dir=str(tmp_path / "registry"),
        clock=lambda: datetime(2026, 7, 21, 12, 0),
    )
    app = create_app(
        vehicle_registry=registry,
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path / "snapshots"),
    )
    source_value = "2026-07-21T09:00:00+03:00"

    with TestClient(app) as client:
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": source_value,
            },
            headers={"X-Service-Key": "pms-va-secret"},
        )

    assert response.status_code == 200
    acknowledged = datetime.fromisoformat(response.json()["timestamp"])
    assert acknowledged == datetime.fromisoformat(source_value)
    assert acknowledged.utcoffset() is not None


@pytest.mark.parametrize(
    "payload,detail",
    [
        (
            {"captured_at": "not-a-time"},
            "captured_at_invalid_iso_timestamp",
        ),
        (
            {"captured_at": "2026-07-21T09:00:00"},
            "captured_at_requires_timezone",
        ),
        ({}, "exit_requires_captured_at"),
    ],
)
def test_exit_fails_closed_without_valid_aware_source_time(
    client_and_registry,
    payload,
    detail,
):
    client, registry = client_and_registry
    with patch.object(registry, "register_anpr_event") as register:
        response = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "exit", **payload},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == detail
    register.assert_not_called()


def test_legacy_timestamp_alias_is_accepted_only_when_aware(client_and_registry):
    client, registry = client_and_registry
    original = registry.register_anpr_event
    with patch.object(registry, "register_anpr_event", wraps=original) as register:
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "timestamp": "2026-07-21T06:00:00Z",
            },
        )

    assert response.status_code == 200
    assert register.call_args.kwargs["timestamp"] == datetime(
        2026,
        7,
        21,
        6,
        0,
        tzinfo=timezone.utc,
    )


def test_conflicting_source_timestamp_fields_are_rejected(client_and_registry):
    client, registry = client_and_registry
    with patch.object(registry, "register_anpr_event") as register:
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
                "timestamp": "2026-07-21T07:00:00Z",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "source_timestamp_conflict"
    register.assert_not_called()


def test_equivalent_source_timestamp_fields_are_accepted(client_and_registry):
    client, registry = client_and_registry
    original = registry.register_anpr_event
    with patch.object(registry, "register_anpr_event", wraps=original) as register:
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
                "timestamp": "2026-07-21T06:00:00Z",
            },
        )

    assert response.status_code == 200
    assert register.call_args.kwargs["timestamp"] == datetime.fromisoformat(
        "2026-07-21T09:00:00+03:00"
    )


def test_legacy_non_exit_without_timestamp_keeps_safe_clock_fallback(
    client_and_registry,
):
    client, registry = client_and_registry
    original = registry.register_anpr_event
    with patch.object(registry, "register_anpr_event", wraps=original) as register:
        response = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "entry"},
        )

    assert response.status_code == 200
    assert register.call_args.kwargs["timestamp"] is None
    assert response.json()["timestamp"] == "2026-07-21T12:00:00+00:00"


def test_upload_exit_cannot_bypass_source_timestamp_contract(client_and_registry):
    client, registry = client_and_registry
    original = registry.register_anpr_event
    with patch.object(registry, "register_anpr_event", wraps=original) as register:
        missing = client.post(
            "/api/anpr/event/upload",
            data={"plate": "ABC-1234", "direction": "exit"},
        )
        accepted = client.post(
            "/api/anpr/event/upload",
            data={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
            },
        )

    assert missing.status_code == 422
    assert missing.json()["detail"] == "exit_requires_captured_at"
    assert accepted.status_code == 200
    assert accepted.json()["timestamp"] == "2026-07-21T09:00:00+03:00"
    register.assert_called_once()
    assert register.call_args.kwargs["timestamp"] == datetime.fromisoformat(
        "2026-07-21T09:00:00+03:00"
    )


def test_authoritative_requires_auth_and_disables_legacy_entry_routes(tmp_path):
    class UnusedDependency:
        def analyze(self, **kwargs):
            raise AssertionError(kwargs)

        def deliver(self, payload):
            raise AssertionError(payload)

    dependency = UnusedDependency()
    coordinator = EntryCoordinator(
        EntrySettings(
            mode=EntryMode.AUTHORITATIVE,
            service_key="pms-va-secret",
        ),
        dependency,
        dependency,
    )
    registry = VehicleRegistry(image_dir=str(tmp_path / "registry"))
    app = create_app(
        vehicle_registry=registry,
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path / "snapshots"),
    )

    with TestClient(app) as client, patch.object(
        registry,
        "register_anpr_event",
        wraps=registry.register_anpr_event,
    ) as register, patch.object(
        coordinator,
        "record_exit",
        wraps=coordinator.record_exit,
    ) as record_exit:
        missing = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "entry"},
        )
        wrong = client.post(
            "/api/anpr/event/upload",
            data={"plate": "ABC-1234", "direction": "entry"},
            headers={"X-Service-Key": "wrong"},
        )
        disabled_entry = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "entry"},
            headers={"X-Service-Key": "pms-va-secret"},
        )
        accepted_exit = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
            },
            headers={"X-Service-Key": "pms-va-secret"},
        )
        disabled_upload = client.post(
            "/api/anpr/event/upload",
            data={"plate": "ABC-1234", "direction": "entry"},
            headers={"X-Service-Key": "pms-va-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert disabled_entry.status_code == 410
    assert disabled_entry.json()["detail"] == (
        "legacy_anpr_entry_disabled_in_entry_v2"
    )
    assert accepted_exit.status_code == 200
    assert disabled_upload.status_code == 410
    assert disabled_upload.json()["detail"] == (
        "legacy_anpr_upload_disabled_in_entry_v2"
    )
    register.assert_called_once()
    record_exit.assert_called_once_with(
        "ABC-1234",
        datetime.fromisoformat("2026-07-21T09:00:00+03:00"),
    )


def test_shadow_keeps_authenticated_legacy_json_entry_for_live_identity(tmp_path):
    class UnusedDependency:
        def analyze(self, **kwargs):
            raise AssertionError(kwargs)

        def deliver(self, payload):
            raise AssertionError(payload)

    dependency = UnusedDependency()
    coordinator = EntryCoordinator(
        EntrySettings(
            mode=EntryMode.SHADOW,
            service_key="pms-va-secret",
        ),
        dependency,
        dependency,
    )
    registry = VehicleRegistry(image_dir=str(tmp_path / "registry"))
    app = create_app(
        vehicle_registry=registry,
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path / "snapshots"),
    )

    with TestClient(app) as client, patch.object(
        registry,
        "register_anpr_event",
        wraps=registry.register_anpr_event,
    ) as register:
        missing = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "entry"},
        )
        accepted = client.post(
            "/api/anpr/event",
            json={"plate": "ABC-1234", "direction": "entry"},
            headers={"X-Service-Key": "pms-va-secret"},
        )

    assert missing.status_code == 401
    assert accepted.status_code == 200
    register.assert_called_once()


def test_exit_lifecycle_capacity_returns_retryable_503(tmp_path):
    class UnusedDependency:
        def analyze(self, **kwargs):
            raise AssertionError(kwargs)

        def deliver(self, payload):
            raise AssertionError(payload)

    dependency = UnusedDependency()
    coordinator = EntryCoordinator(
        EntrySettings(
            mode=EntryMode.AUTHORITATIVE,
            service_key="pms-va-secret",
        ),
        dependency,
        dependency,
    )
    registry = VehicleRegistry(image_dir=str(tmp_path / "registry"))
    app = create_app(
        vehicle_registry=registry,
        entry_coordinator=coordinator,
        snapshot_base_dir=str(tmp_path / "snapshots"),
    )

    with TestClient(app) as client, patch.object(
        coordinator,
        "record_exit",
        side_effect=EntryCapacityExceeded("pending_exit_capacity_exceeded"),
    ):
        response = client.post(
            "/api/anpr/event",
            json={
                "plate": "ABC-1234",
                "direction": "exit",
                "captured_at": "2026-07-21T09:00:00+03:00",
            },
            headers={"X-Service-Key": "pms-va-secret"},
        )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"] == "pending_exit_capacity_exceeded"
