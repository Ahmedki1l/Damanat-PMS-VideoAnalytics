import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
import pytest
import starlette.formparsers as formparsers

from src.entry import router as entry_router
from src.entry.callback import DeliveryResult
from src.entry.coordinator import EntryCoordinator
from src.entry.domain import (
    EntryMode,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
)
from src.entry.router import create_entry_router, install_entry_transport_guard
from src.entry.settings import EntrySettings


class ContractProcessor:
    def __init__(self):
        self.calls = []

    def analyze(self, *, event_id, camera_id, source_role, images, metadata):
        self.calls.append((event_id, tuple(images), dict(metadata)))
        text = "ABC1234" if source_role == "primary" else ""
        state = PlateReadState.READABLE if text else PlateReadState.NO_PLATE
        return (
            FrameEvidence(
                evidence_id=f"{event_id}:0",
                embedding=(1.0, 0.0),
                plate=PlateEvidence(
                    evidence_id=f"{event_id}:0",
                    camera_id=camera_id,
                    source_role=source_role,
                    state=state,
                    text=text,
                    confidence=0.97 if text else 0.0,
                ),
            ),
        )


class ContractSink:
    def __init__(self):
        self.payloads = []

    def deliver(self, payload):
        self.payloads.append(dict(payload))
        return DeliveryResult(True, 1)


class FailThenSucceedSink:
    def __init__(self):
        self.calls = 0
        self.successful_deliveries = 0

    def deliver(self, payload):
        self.calls += 1
        if self.calls == 1:
            return DeliveryResult(False, 3, "pms unavailable")
        self.successful_deliveries += 1
        return DeliveryResult(True, 1)


def make_client(
    *,
    max_attempts=4,
    max_image_bytes=32,
    max_metadata_bytes=16 * 1024,
    sink=None,
):
    cfg = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_pending_attempts=max_attempts,
        max_pending_crossings=4,
        max_pending_callbacks=4,
        max_images_per_event=2,
        max_image_bytes=max_image_bytes,
        max_metadata_bytes=max_metadata_bytes,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"ramp-entry"}),
        fallback_cameras=frozenset({"CAM03"}),
        fallback_lines=frozenset({"B-IN"}),
        fallback_directions=frozenset({"b-entry"}),
        service_key="contract-secret",
    )
    processor = ContractProcessor()
    sink = sink or ContractSink()
    coordinator = EntryCoordinator(cfg, processor, sink)
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    app.include_router(create_entry_router(coordinator))
    return (
        TestClient(
            app,
            headers={
                "X-Service-Key": "contract-secret",
                "X-Entry-V2-Mode": "authoritative",
            },
        ),
        coordinator,
        processor,
        sink,
    )


def attempt_form(identifier="a1", **overrides):
    form = {
        "attempt_id": identifier,
        "source_event_id": f"source-{identifier}",
        "camera_id": "ANPR-ENTRY",
        "captured_at": datetime(2026, 7, 21, tzinfo=timezone.utc).isoformat(),
        "reported_plate": "ABC-1234",
        "reported_confidence": "92",
        "metadata_json": '{"lane":"entry"}',
    }
    form.update(overrides)
    return form


def crossing_form(identifier="c1", **overrides):
    form = {
        "crossing_id": identifier,
        "source_event_id": f"source-{identifier}",
        "camera_id": "CAM-23",
        "captured_at": datetime(2026, 7, 21, tzinfo=timezone.utc).isoformat(),
        "line_id": "RAMP-IN",
        "direction": "ramp-entry",
        "role": "primary",
        "metadata_json": '{"track_id":77}',
    }
    form.update(overrides)
    return form


def image_files(value=b"jpeg"):
    return [("images", ("vehicle.jpg", value, "image/jpeg"))]


def test_attempt_and_crossing_are_multipart_and_return_201_then_callback():
    client, _, processor, sink = make_client()
    attempt_response = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
    )
    crossing_response = client.post(
        "/api/v2/entry-crossings", data=crossing_form(), files=image_files()
    )

    assert attempt_response.status_code == 201
    assert attempt_response.json()["status"] == "accepted"
    assert crossing_response.status_code == 201
    assert crossing_response.json()["decision_status"] == "confirmed"
    assert [call[0] for call in processor.calls] == ["a1", "c1"]
    assert sink.payloads[0]["crossing_id"] == "c1"


def test_large_allowed_vehicle_crop_never_rolls_multipart_part_to_disk(monkeypatch):
    original_spool = formparsers.SpooledTemporaryFile

    class MemoryOnlySpool(original_spool):
        def rollover(self):
            raise AssertionError("entry V2 multipart file rolled to disk")

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", MemoryOnlySpool)
    client, _, _, _ = make_client(max_image_bytes=2 * 1024 * 1024)

    response = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(),
        files=image_files(b"x" * (1024 * 1024 + 1024)),
    )

    assert response.status_code == 201


def test_installed_starlette_parser_contract_bounds_metadata_parts():
    _, coordinator, _, _ = make_client(max_metadata_bytes=4096)

    options = entry_router._multipart_parser_options(coordinator)

    assert options == {
        "max_files": 2,
        "max_fields": 16,
        "max_part_size": 4096,
    }


def test_oversized_metadata_part_is_rejected_by_multipart_parser():
    client, _, processor, _ = make_client(max_metadata_bytes=32)

    response = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(metadata_json="x" * 33),
        files=image_files(),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_multipart"
    assert processor.calls == []


@pytest.mark.parametrize(
    ("parser", "ceiling_attribute"),
    [
        (type("CurrentParser", (), {"spool_max_size": 1})(), "spool_max_size"),
        (type("LegacyParser", (), {"max_file_size": 1})(), "max_file_size"),
    ],
)
def test_supported_starlette_spool_contracts_apply_memory_ceiling(
    parser,
    ceiling_attribute,
):
    entry_router._apply_multipart_spool_ceiling(parser, 4097)

    assert getattr(parser, ceiling_attribute) == 4097


def test_unknown_starlette_spool_contract_fails_closed():
    parser = type("UnknownParser", (), {})()

    with pytest.raises(HTTPException) as raised:
        entry_router._apply_multipart_spool_ceiling(parser, 4097)

    assert raised.value.status_code == 503
    assert raised.value.detail == "entry_v2_multipart_spool_contract_unsupported"


def test_legacy_starlette_constructor_omits_unsupported_part_size(monkeypatch):
    class LegacyMultiPartParser:
        max_file_size = 1024 * 1024

        def __init__(self, headers, stream, *, max_files=1000, max_fields=1000):
            pass

    _, coordinator, _, _ = make_client()
    monkeypatch.setattr(entry_router, "MultiPartParser", LegacyMultiPartParser)

    assert entry_router._multipart_parser_options(coordinator) == {
        "max_files": 2,
        "max_fields": 16,
    }


def test_exact_duplicate_returns_200_without_running_models_twice():
    client, _, processor, _ = make_client()
    first = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
    )
    second = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert len(processor.calls) == 1


def test_receive_time_fallback_retry_is_duplicate_through_http_contract():
    client, _, processor, _ = make_client()
    fallback_metadata = '{"timestamp_source":"pms_receive_missing"}'
    first = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(metadata_json=fallback_metadata),
        files=image_files(),
    )
    retry = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(
            captured_at="2026-07-21T00:00:03+00:00",
            metadata_json=fallback_metadata,
        ),
        files=image_files(),
    )

    assert first.status_code == 201
    assert retry.status_code == 200
    assert retry.json()["duplicate"] is True
    assert len(processor.calls) == 1


def test_same_id_with_changed_image_is_422():
    client, _, _, _ = make_client()
    assert (
        client.post(
            "/api/v2/entry-attempts", data=attempt_form(), files=image_files(b"one")
        ).status_code
        == 201
    )
    changed = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files(b"two")
    )
    assert changed.status_code == 422
    assert changed.json()["detail"] == "id_reused_with_different_payload"


def test_invalid_line_and_direction_are_422_before_inference():
    client, _, processor, _ = make_client()
    response = client.post(
        "/api/v2/entry-crossings",
        data=crossing_form(line_id="WRONG", direction="anything"),
        files=image_files(),
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "crossing_line_not_configured_for_role"
    assert processor.calls == []


def test_caller_cannot_assert_valid_crossing_to_bypass_configuration():
    client, _, processor, _ = make_client()
    data = crossing_form(line_id="WRONG")
    data["valid_crossing"] = "true"
    response = client.post("/api/v2/entry-crossings", data=data, files=image_files())
    assert response.status_code == 422
    assert processor.calls == []


def test_zero_image_json_and_naive_timestamp_are_rejected():
    client, _, _, _ = make_client()
    no_image = client.post("/api/v2/entry-attempts", data=attempt_form())
    json_body = client.post("/api/v2/entry-attempts", json=attempt_form())
    naive = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(captured_at="2026-07-21T12:00:00"),
        files=image_files(),
    )
    assert no_image.status_code == 422
    assert json_body.status_code == 422
    assert naive.status_code == 422
    assert naive.json()["detail"] == "captured_at_requires_timezone"


def test_active_endpoint_requires_constant_contract_service_key():
    client, _, processor, _ = make_client()
    client.headers.pop("X-Service-Key")
    missing = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
    )
    wrong = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(),
        files=image_files(),
        headers={"X-Service-Key": "wrong"},
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert processor.calls == []


def test_mode_mismatch_is_rejected_before_multipart_inference():
    client, _, processor, _ = make_client()

    response = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(),
        files=image_files(),
        headers={"X-Entry-V2-Mode": "shadow"},
    )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json()["detail"] == "entry_v2_mode_mismatch"
    assert processor.calls == []


def test_hikvision_confidence_uses_0_to_100_wire_scale():
    client, coordinator, _, _ = make_client()
    accepted = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(reported_confidence="92"),
        files=image_files(),
    )
    rejected = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form("a2", reported_confidence="101"),
        files=image_files(b"other"),
    )
    assert accepted.status_code == 201
    assert rejected.status_code == 422
    record = next(iter(coordinator._attempts.values()))
    assert record.request.reported_confidence == 0.92


def test_hikvision_confidence_boundaries_zero_and_one_hundred_are_valid():
    for identifier, confidence in (("zero", "0"), ("hundred", "100")):
        client, coordinator, _, _ = make_client()
        response = client.post(
            "/api/v2/entry-attempts",
            data=attempt_form(identifier, reported_confidence=confidence),
            files=image_files(),
        )
        assert response.status_code == 201
        internal = next(
            iter(coordinator._attempts.values())
        ).request.reported_confidence
        assert internal == float(confidence) / 100.0


def test_plate_only_role_or_filename_is_rejected_fail_closed():
    client, _, processor, _ = make_client()
    role_response = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(metadata_json='{"image_roles":["plate"]}'),
        files=image_files(),
    )
    filename_response = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form("a2"),
        files=[("images", ("licensePlatePicture.jpg", b"jpeg", "image/jpeg"))],
    )
    assert role_response.status_code == 422
    assert filename_response.status_code == 422
    assert processor.calls == []


def test_oversize_and_too_many_images_are_422():
    client, _, _, _ = make_client()
    oversize = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files(b"x" * 33)
    )
    too_many = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form(identifier="a2"),
        files=[
            ("images", ("1.jpg", b"1", "image/jpeg")),
            ("images", ("2.jpg", b"2", "image/jpeg")),
            ("images", ("3.jpg", b"3", "image/jpeg")),
        ],
    )
    assert oversize.status_code == 422
    assert oversize.json()["detail"] == "image_too_large"
    assert too_many.status_code == 422
    assert too_many.json()["detail"] == "too_many_images"


def test_bounded_capacity_returns_503_not_fifo_eviction():
    client, coordinator, _, _ = make_client(max_attempts=1)
    first = client.post(
        "/api/v2/entry-attempts", data=attempt_form("a1"), files=image_files()
    )
    second = client.post(
        "/api/v2/entry-attempts",
        data=attempt_form("a2", reported_plate="XYZ-9999"),
        files=image_files(b"other"),
    )
    assert first.status_code == 201
    assert second.status_code == 503
    assert second.json()["detail"] == "attempt_capacity_exceeded"
    remaining = next(iter(coordinator.state_summary()["groups"].values()))
    assert remaining["attempt_ids"] == ["a1"]


def test_authenticated_operator_cancellation_releases_false_pending_attempt():
    client, coordinator, _, _ = make_client()
    accepted = client.post(
        "/api/v2/entry-attempts", data=attempt_form("false-a1"), files=image_files()
    )
    group_id = accepted.json()["group_id"]

    cancelled = client.post(
        "/api/v2/entry-cancellations",
        json={"group_id": group_id, "reason": "operator verified retreat"},
    )
    repeated = client.post(
        "/api/v2/entry-cancellations",
        json={"group_id": group_id, "reason": "idempotent retry"},
    )

    assert cancelled.status_code == 200
    assert cancelled.json() == {
        "status": "cancelled",
        "removed_attempts": 1,
        "removed_groups": 1,
        "removed_crossings": 0,
        "removed_pending_exits": 0,
    }
    assert repeated.json()["status"] == "not_found"
    assert coordinator.state_summary()["attempt_count"] == 0


def test_authenticated_exact_pending_exit_cancellation_releases_capacity(caplog):
    client, coordinator, _, _ = make_client()
    exit_at = datetime(2026, 7, 21, 12, 30, tzinfo=timezone.utc)
    assert coordinator.record_exit("ABC-1234", exit_at) == 0
    assert coordinator.state_summary()["pending_exit_count"] == 1

    payload = {
        "exit_plate": "ABC-1234",
        "exit_captured_at": exit_at.isoformat(),
        "reason": "operator verified legacy or false exit",
    }
    with caplog.at_level("INFO", logger="src.entry.coordinator"):
        cancelled = client.post("/api/v2/entry-cancellations", json=payload)
    repeated = client.post("/api/v2/entry-cancellations", json=payload)

    assert cancelled.status_code == 200
    assert cancelled.json() == {
        "status": "cancelled",
        "removed_attempts": 0,
        "removed_groups": 0,
        "removed_crossings": 0,
        "removed_pending_exits": 1,
    }
    assert repeated.json()["status"] == "not_found"
    assert repeated.json()["removed_pending_exits"] == 0
    assert coordinator.state_summary()["pending_exit_count"] == 0
    assert "reason=operator verified legacy or false exit" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"exit_plate": "ABC-1234", "reason": "missing timestamp"},
        {
            "exit_captured_at": "2026-07-21T12:30:00+00:00",
            "reason": "missing plate",
        },
        {
            "exit_plate": "ABC-1234",
            "exit_captured_at": "2026-07-21T12:30:00",
            "reason": "naive timestamp",
        },
    ],
)
def test_pending_exit_cancellation_requires_complete_aware_exact_key(payload):
    client, _, _, _ = make_client()

    response = client.post("/api/v2/entry-cancellations", json=payload)

    assert response.status_code == 422


def test_cancellation_requires_service_authentication():
    client, _, _, _ = make_client()

    response = client.post(
        "/api/v2/entry-cancellations",
        headers={"X-Service-Key": "wrong"},
        json={"attempt_id": "a1", "reason": "test"},
    )

    assert response.status_code == 401


def test_transport_guard_caps_chunked_body_without_content_length():
    cfg = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_images_per_event=1,
        max_image_bytes=16,
        max_metadata_bytes=16,
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)

    @app.post("/api/v2/entry-attempts")
    async def consume_body(request: Request):
        await request.body()
        return {"status": "unexpected"}

    client = TestClient(app)
    response = client.post(
        "/api/v2/entry-attempts",
        headers={
            "X-Service-Key": "contract-secret",
            "X-Entry-V2-Mode": "authoritative",
        },
        content=(b"x" * 10_000 for _ in range(7)),
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "entry_request_too_large"


def test_transport_guard_rejects_concurrent_ingest_before_route_or_body_work():
    cfg = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_concurrent_ingest_requests=1,
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    entered = threading.Event()
    release = threading.Event()
    route_calls = []

    @app.post("/api/v2/entry-attempts")
    def blocking_ingest():
        route_calls.append("called")
        entered.set()
        assert release.wait(timeout=2)
        return {"status": "ok"}

    headers = {
        "X-Service-Key": "contract-secret",
        "X-Entry-V2-Mode": "authoritative",
    }
    first_result = []

    def send_first():
        with TestClient(app, headers=headers) as client:
            first_result.append(client.post("/api/v2/entry-attempts"))

    first_thread = threading.Thread(target=send_first)
    first_thread.start()
    assert entered.wait(timeout=2)
    with TestClient(app, headers=headers) as client:
        rejected = client.post(
            "/api/v2/entry-attempts",
            content=b"body-that-must-not-reach-the-route",
        )
    release.set()
    first_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert first_result[0].status_code == 200
    assert rejected.status_code == 503
    assert rejected.headers["Retry-After"] == "1"
    assert rejected.json()["detail"] == "entry_v2_ingress_capacity_exceeded"
    assert route_calls == ["called"]


def test_shadow_v2_capacity_does_not_backpressure_legacy_anpr():
    cfg = EntrySettings(
        mode=EntryMode.SHADOW,
        max_concurrent_ingest_requests=2,
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    entered_count = 0
    entered_lock = threading.Lock()
    both_entered = threading.Event()
    release = threading.Event()
    v2_route_calls = []
    legacy_bodies = []

    @app.post("/api/v2/entry-attempts")
    def blocking_v2_ingest():
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
            v2_route_calls.append("called")
            if entered_count == 2:
                both_entered.set()
        assert release.wait(timeout=3)
        return {"status": "ok"}

    @app.post("/api/anpr/event")
    async def consume_legacy_anpr_body(request: Request):
        legacy_bodies.append(await request.body())
        return {"status": "ok"}

    headers = {
        "X-Service-Key": "contract-secret",
        "X-Entry-V2-Mode": "shadow",
    }
    blocked_results = []

    def send_blocked_v2():
        with TestClient(app, headers=headers) as client:
            blocked_results.append(client.post("/api/v2/entry-attempts"))

    blocked_threads = [threading.Thread(target=send_blocked_v2) for _ in range(2)]
    for thread in blocked_threads:
        thread.start()
    assert both_entered.wait(timeout=3)

    with TestClient(app, headers=headers) as client:
        legacy_response = client.post("/api/anpr/event", content=b'{"plate":"A1"}')
        rejected_v2 = client.post("/api/v2/entry-attempts")

    release.set()
    for thread in blocked_threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in blocked_threads)
    assert [response.status_code for response in blocked_results] == [200, 200]
    assert legacy_response.status_code == 200
    assert legacy_bodies == [b'{"plate":"A1"}']
    assert rejected_v2.status_code == 503
    assert rejected_v2.headers["Retry-After"] == "1"
    assert rejected_v2.json()["detail"] == "entry_v2_ingress_capacity_exceeded"
    assert v2_route_calls == ["called", "called"]


def test_off_mode_leaves_concurrent_legacy_anpr_unconstrained_and_v2_disabled():
    cfg = EntrySettings(
        mode=EntryMode.OFF,
        max_concurrent_ingest_requests=1,
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    first_entered = threading.Event()
    release_first = threading.Event()
    route_lock = threading.Lock()
    legacy_route_calls = 0
    v2_route_calls = []

    @app.post("/api/anpr/event")
    def legacy_anpr():
        nonlocal legacy_route_calls
        with route_lock:
            legacy_route_calls += 1
            call_number = legacy_route_calls
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(timeout=3)
        return {"status": "ok"}

    @app.post("/api/v2/entry-attempts")
    def v2_ingest():
        v2_route_calls.append("called")
        return {"status": "unexpected"}

    first_result = []

    def send_first_legacy():
        with TestClient(app) as client:
            first_result.append(client.post("/api/anpr/event"))

    first_thread = threading.Thread(target=send_first_legacy)
    first_thread.start()
    assert first_entered.wait(timeout=3)

    with TestClient(app) as client:
        second_legacy = client.post("/api/anpr/event")
        disabled_v2 = client.post("/api/v2/entry-attempts")

    release_first.set()
    first_thread.join(timeout=3)

    assert not first_thread.is_alive()
    assert first_result[0].status_code == 200
    assert second_legacy.status_code == 200
    assert legacy_route_calls == 2
    assert disabled_v2.status_code == 503
    assert disabled_v2.json()["detail"] == "entry_v2_disabled"
    assert v2_route_calls == []


def test_legacy_anpr_auth_is_rejected_before_body_or_route_work():
    cfg = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_image_bytes=16,
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    route_calls = []

    @app.post("/api/anpr/event")
    async def consume_legacy_anpr_body(request: Request):
        route_calls.append(await request.body())
        return {"status": "unexpected"}

    response = TestClient(app).post(
        "/api/anpr/event",
        headers={"X-Service-Key": "wrong"},
        content=(b"x" * 1024 for _ in range(128)),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_service_key"
    assert route_calls == []


@pytest.mark.parametrize("mode", [EntryMode.SHADOW, EntryMode.AUTHORITATIVE])
def test_unavailable_entry_coordinator_does_not_block_authenticated_exit_bridge(
    mode,
):
    cfg = EntrySettings(mode=mode, service_key="contract-secret")
    coordinator = EntryCoordinator(
        cfg,
        ContractProcessor(),
        ContractSink(),
        unavailable_reason="entry_v2_model_unavailable",
    )
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    route_calls = []

    @app.post("/api/anpr/event")
    async def consume_legacy_anpr_body(request: Request):
        route_calls.append(await request.body())
        return {"status": "ok"}

    response = TestClient(app).post(
        "/api/anpr/event",
        headers={"X-Service-Key": "contract-secret"},
        content=b'{"plate":"ABC-1234","direction":"exit"}',
    )

    assert response.status_code == 200
    assert route_calls == [b'{"plate":"ABC-1234","direction":"exit"}']


def test_invalid_mode_never_falls_back_to_unauthenticated_legacy_ingress():
    cfg = EntrySettings(
        mode=EntryMode.OFF,
        invalid_mode_value="authoratative",
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(
        cfg,
        ContractProcessor(),
        ContractSink(),
        unavailable_reason="entry_v2_invalid_configuration:ENTRY_V2_MODE",
    )
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    route_calls = []

    @app.post("/api/anpr/event")
    async def consume_legacy_anpr_body(request: Request):
        route_calls.append(await request.body())
        return {"status": "unexpected"}

    client = TestClient(app)
    missing = client.post("/api/anpr/event", content=b"must-not-be-read")
    invalid = client.post(
        "/api/anpr/event",
        headers={"X-Service-Key": "contract-secret"},
        content=b"must-not-be-read",
    )

    assert missing.status_code == 401
    assert invalid.status_code == 503
    assert route_calls == []


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        (
            "/api/anpr/event/upload",
            "legacy_anpr_upload_disabled_in_entry_v2",
        ),
        (
            "/api/line-crossing",
            "legacy_line_crossing_disabled_in_entry_v2",
        ),
    ],
)
def test_v2_disables_legacy_image_entry_routes_before_body_work(path, detail):
    cfg = EntrySettings(
        mode=EntryMode.AUTHORITATIVE,
        max_image_bytes=16,
        service_key="contract-secret",
    )
    coordinator = EntryCoordinator(cfg, ContractProcessor(), ContractSink())
    app = FastAPI()
    install_entry_transport_guard(app, coordinator)
    route_calls = []

    @app.post(path)
    async def consume_legacy_body(request: Request):
        route_calls.append(await request.body())
        return {"status": "unexpected"}

    client = TestClient(app)
    wrong = client.post(
        path,
        headers={"X-Service-Key": "wrong"},
        content=(b"x" * 1024 for _ in range(128)),
    )
    disabled = client.post(
        path,
        headers={"X-Service-Key": "contract-secret"},
        content=(b"x" * 1024 for _ in range(128)),
    )

    assert wrong.status_code == 401
    assert disabled.status_code == 410
    assert disabled.json()["detail"] == detail
    assert route_calls == []


def test_authoritative_callback_failure_returns_503_and_same_id_retry_delivers():
    sink = FailThenSucceedSink()
    client, coordinator, _, _ = make_client(sink=sink)
    assert (
        client.post(
            "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
        ).status_code
        == 201
    )

    first = client.post(
        "/api/v2/entry-crossings", data=crossing_form(), files=image_files()
    )
    assert first.status_code == 503
    assert first.json()["detail"] == "entry_confirmation_delivery_failed"
    assert coordinator.state_summary()["pending_callback_count"] == 1
    assert coordinator.state_summary()["attempt_count"] == 1

    retry = client.post(
        "/api/v2/entry-crossings", data=crossing_form(), files=image_files()
    )
    assert retry.status_code == 200
    assert retry.json()["duplicate"] is True
    assert retry.json()["callback_delivered"] is True
    assert sink.successful_deliveries == 1
    assert coordinator.state_summary()["attempt_count"] == 0

    stable_duplicate = client.post(
        "/api/v2/entry-crossings", data=crossing_form(), files=image_files()
    )
    assert stable_duplicate.status_code == 200
    assert sink.calls == 2


def test_off_mode_route_exists_but_is_unavailable_without_reading_image():
    cfg = EntrySettings(mode=EntryMode.OFF, service_key="contract-secret")
    processor = ContractProcessor()
    coordinator = EntryCoordinator(cfg, processor, ContractSink())
    app = FastAPI()
    app.include_router(create_entry_router(coordinator))
    client = TestClient(app, headers={"X-Service-Key": "contract-secret"})

    response = client.post(
        "/api/v2/entry-attempts", data=attempt_form(), files=image_files()
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "entry_v2_disabled"
    assert processor.calls == []


def test_environment_mode_contract_uses_authoritative(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    assert EntrySettings.from_env().mode == EntryMode.AUTHORITATIVE


def test_legacy_enforce_mode_alias_reports_authoritative(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "enforce")
    assert EntrySettings.from_env().mode.value == "authoritative"


def test_invalid_environment_mode_fails_configuration_instead_of_silent_off(
    monkeypatch,
):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoratative")
    cfg = EntrySettings.from_env()
    assert cfg.mode == EntryMode.OFF
    assert cfg.configuration_errors() == ["ENTRY_V2_MODE=authoratative"]


def test_primary_only_configuration_does_not_require_optional_fallback(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_DIRECTIONS", "B-to-A")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.delenv("ENTRY_V2_FALLBACK_CAMERAS", raising=False)
    monkeypatch.delenv("ENTRY_V2_FALLBACK_LINES", raising=False)
    monkeypatch.delenv("ENTRY_V2_FALLBACK_DIRECTIONS", raising=False)

    cfg = EntrySettings.from_env()

    assert cfg.fallback_cameras == frozenset()
    assert "fallback_policy_incomplete" not in cfg.configuration_errors()


@pytest.mark.parametrize("mode", ["shadow", "authoritative"])
def test_active_mode_requires_explicit_calibrated_primary_direction(monkeypatch, mode):
    monkeypatch.setenv("ENTRY_V2_MODE", mode)
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.delenv("ENTRY_V2_PRIMARY_DIRECTIONS", raising=False)
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    if mode == "authoritative":
        monkeypatch.setenv("VA_PROCESS_COUNT", "1")

    cfg = EntrySettings.from_env()

    assert "primary_directions" in cfg.configuration_errors()


@pytest.mark.parametrize(
    "url",
    [
        "[http://pms-ai:8080](http://pms-ai:8080)",
        "http://user:password@pms-ai:8080",
        "http://pms-ai:8080?entry=v2",
        "http://pms-ai:99999",
    ],
)
def test_active_mode_rejects_malformed_callback_base_url(url):
    cfg = EntrySettings(
        mode=EntryMode.SHADOW,
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        pms_base_url=url,
        service_key="secret",
    )

    assert "PMS_API_URL" in cfg.configuration_errors()


def test_authoritative_mode_fails_closed_with_process_local_registries(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.setenv("VA_PROCESS_COUNT", "2")

    cfg = EntrySettings.from_env()

    assert "entry_v2_requires_single_process_va" in cfg.configuration_errors()


@pytest.mark.parametrize("raw_count", [None, "", "two"])
def test_authoritative_mode_requires_explicit_valid_process_count(
    monkeypatch, raw_count
):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    if raw_count is None:
        monkeypatch.delenv("VA_PROCESS_COUNT", raising=False)
    else:
        monkeypatch.setenv("VA_PROCESS_COUNT", raw_count)

    cfg = EntrySettings.from_env()

    assert "VA_PROCESS_COUNT" in cfg.configuration_errors()


def test_authoritative_mode_accepts_explicit_single_process(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.setenv("VA_PROCESS_COUNT", "1")

    cfg = EntrySettings.from_env()

    assert "VA_PROCESS_COUNT" not in cfg.configuration_errors()
    assert "entry_v2_requires_single_process_va" not in cfg.configuration_errors()


def test_receipt_capacity_covers_all_concurrent_delivery_markers(monkeypatch):
    monkeypatch.setenv("ENTRY_V2_MODE", "authoritative")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.setenv("VA_PROCESS_COUNT", "1")
    monkeypatch.setenv("ENTRY_V2_MAX_CONCURRENT_INGEST_REQUESTS", "2")
    monkeypatch.setenv("ENTRY_V2_RECEIPT_CAPACITY", "1")

    cfg = EntrySettings.from_env()

    assert "receipt_capacity_below_ingest_concurrency" in cfg.configuration_errors()


@pytest.mark.parametrize(
    ("name", "value", "expected_error"),
    [
        ("ENTRY_V2_MAX_IMAGES", "four", "max_images_per_event"),
        ("ENTRY_V2_JOURNEY_CAPACITY", "many", "journey_capacity"),
        ("ENTRY_V2_LPD_CONFIDENCE", "high", "lpd_confidence"),
        ("ENTRY_V2_LPD_IOU", "wide", "lpd_iou"),
    ],
)
def test_active_mode_rejects_malformed_numeric_environment_values(
    monkeypatch, name, value, expected_error
):
    monkeypatch.setenv("ENTRY_V2_MODE", "shadow")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "RAMP-IN")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_DIRECTIONS", "B-to-A")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.setenv(name, value)

    cfg = EntrySettings.from_env()

    assert expected_error in cfg.configuration_errors()


def test_local_zone_configuration_requires_two_images():
    cfg = EntrySettings(
        mode=EntryMode.SHADOW,
        max_images_per_event=1,
        primary_cameras=frozenset({"CAM23"}),
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="secret",
    )

    assert "local_zone_requires_two_images" in cfg.configuration_errors()
