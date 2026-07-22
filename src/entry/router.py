"""Multipart HTTP contracts for PMS -> VA V2 evidence ingestion."""

from __future__ import annotations

import inspect
import json
import re
import hmac
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .coordinator import EntryCoordinator
from .domain import (
    AttemptInput,
    CancellationInput,
    CrossingInput,
    CrossingRole,
    EntryCapacityExceeded,
    EntryConflict,
    EntryInvalid,
    EntryUnavailable,
    EvidenceUnavailable,
    IngestResult,
)
from .settings import (
    ATTEMPT_PATH,
    CANCELLATION_PATH,
    CROSSING_PATH,
    MODE_HEADER,
    SERVICE_KEY_HEADER,
)


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
logger = logging.getLogger(__name__)
_LEGACY_ANPR_JSON_PATH = "/api/anpr/event"
_LEGACY_ANPR_UPLOAD_PATH = "/api/anpr/event/upload"
_LEGACY_LINE_CROSSING_PATH = "/api/line-crossing"
_MULTIPART_SPOOL_PARAMETERS = ("spool_max_size", "max_file_size")


class EntryIngestResponse(BaseModel):
    status: str
    id: str
    duplicate: bool
    mode: str
    evidence_count: int
    group_id: Optional[str] = None
    decision_id: Optional[str] = None
    decision_status: Optional[str] = None
    callback_delivered: Optional[bool] = None


class EntryCancellationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: Optional[str] = Field(default=None, max_length=128)
    group_id: Optional[str] = Field(default=None, max_length=128)
    crossing_id: Optional[str] = Field(default=None, max_length=128)
    exit_plate: Optional[str] = Field(default=None, max_length=32)
    exit_captured_at: Optional[datetime] = None
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def require_identifier(self):
        has_exit_plate = bool(self.exit_plate)
        has_exit_timestamp = self.exit_captured_at is not None
        if has_exit_plate != has_exit_timestamp:
            raise ValueError(
                "exit_plate and exit_captured_at must be provided together"
            )
        if not any(
            (
                self.attempt_id,
                self.group_id,
                self.crossing_id,
                has_exit_plate,
            )
        ):
            raise ValueError("at least one cancellation identifier is required")
        return self


class EntryCancellationResponse(BaseModel):
    status: str
    removed_attempts: int
    removed_groups: int
    removed_crossings: int
    removed_pending_exits: int


class _EntryRequestTooLarge(Exception):
    pass


def _entry_request_byte_limit(coordinator: EntryCoordinator) -> int:
    settings = coordinator.settings
    return (
        settings.max_images_per_event * settings.max_image_bytes
        + settings.max_metadata_bytes
        + 64 * 1024
    )


class EntryTransportGuard:
    """Pure ASGI guard, so streamed overflow is caught outside request parsing."""

    def __init__(self, app, coordinator: EntryCoordinator):
        self.app = app
        self.coordinator = coordinator
        self.entry_paths = {ATTEMPT_PATH, CROSSING_PATH, CANCELLATION_PATH}
        self.ingest_paths = {ATTEMPT_PATH, CROSSING_PATH}
        self.legacy_anpr_paths = {
            _LEGACY_ANPR_JSON_PATH,
            _LEGACY_ANPR_UPLOAD_PATH,
            _LEGACY_LINE_CROSSING_PATH,
        }
        self.guarded_paths = self.entry_paths | self.legacy_anpr_paths
        self._ingress_lock = threading.Lock()
        self._active_ingest_requests = 0
        self.max_request_bytes = _entry_request_byte_limit(coordinator)
        self.legacy_anpr_max_request_bytes = (
            (coordinator.settings.max_image_bytes + 2) // 3
        ) * 4 + 64 * 1024

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") not in self.guarded_paths:
            await self.app(scope, receive, send)
            return
        path = scope.get("path")
        is_legacy_anpr = path in self.legacy_anpr_paths
        invalid_mode = bool(self.coordinator.settings.invalid_mode_value)
        auth_required = (
            not is_legacy_anpr
            or self.coordinator.settings.mode.value != "off"
            or invalid_mode
        )
        expected = self.coordinator.settings.service_key
        if auth_required and not expected:
            await self._respond(
                scope,
                receive,
                send,
                503,
                self.coordinator.unavailable_reason
                or "entry_v2_service_authentication_unavailable",
            )
            return

        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        if auth_required:
            supplied = headers.get(SERVICE_KEY_HEADER.lower().encode(), b"").decode(
                "utf-8", errors="replace"
            )
            if not hmac.compare_digest(expected, supplied):
                await self._respond(scope, receive, send, 401, "invalid_service_key")
                return

        if invalid_mode:
            await self._respond(
                scope,
                receive,
                send,
                503,
                self.coordinator.unavailable_reason
                or "entry_v2_invalid_mode_configuration",
            )
            return
        if not is_legacy_anpr and not self.coordinator.available:
            await self._respond(
                scope,
                receive,
                send,
                503,
                self.coordinator.unavailable_reason or "entry_v2_unavailable",
            )
            return

        disabled_legacy_path = {
            _LEGACY_ANPR_UPLOAD_PATH: "legacy_anpr_upload_disabled_in_entry_v2",
            _LEGACY_LINE_CROSSING_PATH: ("legacy_line_crossing_disabled_in_entry_v2"),
        }.get(path)
        if disabled_legacy_path and self.coordinator.settings.mode.value != "off":
            await self._respond(
                scope,
                receive,
                send,
                410,
                disabled_legacy_path,
            )
            return

        if path in self.ingest_paths:
            supplied_mode = headers.get(MODE_HEADER.lower().encode(), b"").decode(
                "utf-8", errors="replace"
            )
            if supplied_mode != self.coordinator.settings.mode.value:
                await self._respond(
                    scope,
                    receive,
                    send,
                    503,
                    "entry_v2_mode_mismatch",
                    headers={"Retry-After": "1"},
                )
                return

        request_limit = (
            self.legacy_anpr_max_request_bytes
            if is_legacy_anpr
            else self.max_request_bytes
        )
        too_large_detail = (
            "anpr_request_too_large" if is_legacy_anpr else "entry_request_too_large"
        )
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                declared_size = int(raw_length)
            except ValueError:
                await self._respond(scope, receive, send, 400, "invalid_content_length")
                return
            if declared_size < 0 or declared_size > request_limit:
                await self._respond(scope, receive, send, 413, too_large_detail)
                return

        ingress_acquired = False
        if path in self.ingest_paths:
            with self._ingress_lock:
                at_capacity = (
                    self._active_ingest_requests
                    >= self.coordinator.settings.max_concurrent_ingest_requests
                )
                if not at_capacity:
                    self._active_ingest_requests += 1
                    ingress_acquired = True
            if at_capacity:
                logger.warning(
                    "[EntryV2] ingress capacity exceeded path=%s active=%d limit=%d",
                    path,
                    self._active_ingest_requests,
                    self.coordinator.settings.max_concurrent_ingest_requests,
                )
                await self._respond(
                    scope,
                    receive,
                    send,
                    503,
                    "entry_v2_ingress_capacity_exceeded",
                    headers={"Retry-After": "1"},
                )
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > request_limit:
                    raise _EntryRequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _EntryRequestTooLarge:
            await self._respond(scope, receive, send, 413, too_large_detail)
        finally:
            if ingress_acquired:
                with self._ingress_lock:
                    self._active_ingest_requests -= 1

    @staticmethod
    async def _respond(
        scope,
        receive,
        send,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        response = JSONResponse(
            status_code=status_code,
            content={"detail": detail},
            headers=headers,
        )
        await response(scope, receive, send)


def install_entry_transport_guard(app, coordinator: EntryCoordinator) -> None:
    """Authenticate and size-check before Starlette parses multipart bodies."""
    app.add_middleware(EntryTransportGuard, coordinator=coordinator)


def create_entry_router(coordinator: EntryCoordinator) -> APIRouter:
    router = APIRouter(tags=["entry-validation-v2"])

    @router.post(ATTEMPT_PATH, response_model=EntryIngestResponse)
    async def ingest_attempt(
        request: Request,
        response: Response,
    ):
        _require_available(coordinator)
        _authenticate(coordinator, request.headers.get(SERVICE_KEY_HEADER))
        _require_mode(coordinator, request.headers.get(MODE_HEADER))
        form = await _parse_memory_only_multipart(request, coordinator)
        try:
            metadata = _parse_metadata(
                _scalar(form, "metadata_json", default="{}"), coordinator
            )
            confidence_raw = _scalar(
                form,
                "reported_confidence",
                default=None,
            )
            try:
                reported_confidence = (
                    float(confidence_raw) if confidence_raw is not None else None
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="invalid_reported_confidence",
                ) from exc
            attempt = AttemptInput(
                attempt_id=_parse_id("attempt_id", _scalar(form, "attempt_id")),
                source_event_id=_parse_id(
                    "source_event_id", _scalar(form, "source_event_id")
                ),
                camera_id=_parse_id("camera_id", _scalar(form, "camera_id")),
                captured_at=_parse_timestamp(_scalar(form, "captured_at")),
                reported_plate=_parse_plate(_scalar(form, "reported_plate")),
                reported_confidence=_parse_reported_confidence(reported_confidence),
                metadata=metadata,
            )
            image_bytes = await _read_images(
                _uploads(form, "images"), coordinator, metadata
            )
            result = await _invoke(
                coordinator.ingest_attempt,
                attempt,
                image_bytes,
            )
        finally:
            await form.close()
        response.status_code = 200 if result.duplicate else 201
        return _response(result)

    @router.post(CROSSING_PATH, response_model=EntryIngestResponse)
    async def ingest_crossing(
        request: Request,
        response: Response,
    ):
        _require_available(coordinator)
        _authenticate(coordinator, request.headers.get(SERVICE_KEY_HEADER))
        _require_mode(coordinator, request.headers.get(MODE_HEADER))
        form = await _parse_memory_only_multipart(request, coordinator)
        try:
            metadata = _parse_metadata(
                _scalar(form, "metadata_json", default="{}"), coordinator
            )
            raw_role = _scalar(form, "role")
            try:
                role = CrossingRole(raw_role)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="invalid_role") from exc
            crossing = CrossingInput(
                crossing_id=_parse_id("crossing_id", _scalar(form, "crossing_id")),
                source_event_id=_parse_id(
                    "source_event_id", _scalar(form, "source_event_id")
                ),
                camera_id=_parse_id("camera_id", _scalar(form, "camera_id")),
                captured_at=_parse_timestamp(_scalar(form, "captured_at")),
                line_id=_parse_id("line_id", _scalar(form, "line_id")),
                direction=_scalar(form, "direction").strip(),
                role=role,
                metadata=metadata,
            )
            image_bytes = await _read_images(
                _uploads(form, "images"), coordinator, metadata
            )
            result = await _invoke(
                coordinator.ingest_crossing,
                crossing,
                image_bytes,
            )
        finally:
            await form.close()
        response.status_code = 200 if result.duplicate else 201
        return _response(result)

    @router.post(CANCELLATION_PATH, response_model=EntryCancellationResponse)
    async def cancel_pending_entry(
        request: EntryCancellationRequest,
        service_key: Optional[str] = Header(None, alias=SERVICE_KEY_HEADER),
    ):
        _require_available(coordinator)
        _authenticate(coordinator, service_key)
        try:
            cancellation = CancellationInput(
                attempt_id=request.attempt_id or "",
                group_id=request.group_id or "",
                crossing_id=request.crossing_id or "",
                exit_plate=request.exit_plate or "",
                exit_captured_at=request.exit_captured_at,
                reason=request.reason,
            )
            removed = await run_in_threadpool(
                coordinator.cancel_pending,
                cancellation,
            )
        except EntryConflict as exc:
            raise HTTPException(status_code=409, detail=exc.reason) from exc
        except EntryInvalid as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from exc
        except EntryUnavailable as exc:
            raise HTTPException(status_code=503, detail=exc.reason) from exc
        return EntryCancellationResponse(
            status="cancelled" if any(removed.values()) else "not_found",
            removed_attempts=removed["attempts"],
            removed_groups=removed["groups"],
            removed_crossings=removed["crossings"],
            removed_pending_exits=removed["pending_exits"],
        )

    return router


async def _parse_memory_only_multipart(
    request: Request,
    coordinator: EntryCoordinator,
) -> FormData:
    """Parse V2 multipart with a route-local RAM spool and strict part counts.

    The outer ASGI guard caps the complete stream. Setting this parser's spool
    ceiling above that cap guarantees even rejected file parts remain in RAM;
    no request can reach the rollover point. The setting is instance-local and
    therefore does not weaken unrelated multipart endpoints in the VA app.
    """
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=422, detail="multipart_form_data_required")
    parser_options = _multipart_parser_options(coordinator)
    parser = MultiPartParser(
        request.headers,
        request.stream(),
        **parser_options,
    )
    if not any(name in parser_options for name in _MULTIPART_SPOOL_PARAMETERS):
        _apply_multipart_spool_ceiling(
            parser,
            _entry_request_byte_limit(coordinator) + 1,
        )
    try:
        return await parser.parse()
    except MultiPartException as exc:
        message = str(exc).lower()
        detail = (
            "too_many_images" if "too many files" in message else "invalid_multipart"
        )
        raise HTTPException(status_code=422, detail=detail) from exc


def _multipart_parser_options(coordinator: EntryCoordinator) -> Dict[str, int]:
    parameters = inspect.signature(MultiPartParser).parameters
    options = {
        "max_files": coordinator.settings.max_images_per_event,
        "max_fields": 16,
    }
    if "max_part_size" in parameters:
        options["max_part_size"] = coordinator.settings.max_metadata_bytes
    for name in _MULTIPART_SPOOL_PARAMETERS:
        if name in parameters:
            options[name] = _entry_request_byte_limit(coordinator) + 1
            break
    return options


def _apply_multipart_spool_ceiling(parser: MultiPartParser, ceiling: int) -> None:
    for name in _MULTIPART_SPOOL_PARAMETERS:
        if not hasattr(parser, name):
            continue
        try:
            setattr(parser, name, ceiling)
            configured_ceiling = getattr(parser, name)
        except (AttributeError, TypeError) as exc:
            raise HTTPException(
                status_code=503,
                detail="entry_v2_multipart_spool_contract_unsupported",
            ) from exc
        if configured_ceiling == ceiling:
            return
    raise HTTPException(
        status_code=503,
        detail="entry_v2_multipart_spool_contract_unsupported",
    )


_MISSING = object()


def _scalar(form: FormData, name: str, *, default=_MISSING):
    values = form.getlist(name)
    if not values:
        if default is not _MISSING:
            return default
        raise HTTPException(status_code=422, detail=f"missing_{name}")
    if len(values) != 1 or isinstance(values[0], UploadFile):
        raise HTTPException(status_code=422, detail=f"invalid_{name}")
    return str(values[0])


def _uploads(form: FormData, name: str) -> List[UploadFile]:
    values = form.getlist(name)
    if not values or any(not isinstance(value, UploadFile) for value in values):
        raise HTTPException(status_code=422, detail="at_least_one_image_is_required")
    return list(values)


async def _invoke(call, request, images: List[bytes]) -> IngestResult:
    try:
        return await run_in_threadpool(call, request, tuple(images))
    except (EntryConflict, EntryInvalid, EvidenceUnavailable) as exc:
        logger.warning(
            "[EntryV2] deterministic ingest rejection id=%s camera=%s reason=%s",
            getattr(request, "attempt_id", None)
            or getattr(request, "crossing_id", None),
            getattr(request, "camera_id", ""),
            exc.reason,
        )
        raise HTTPException(status_code=422, detail=exc.reason) from exc
    except (EntryCapacityExceeded, EntryUnavailable) as exc:
        logger.warning(
            "[EntryV2] retryable ingest failure id=%s camera=%s reason=%s",
            getattr(request, "attempt_id", None)
            or getattr(request, "crossing_id", None),
            getattr(request, "camera_id", ""),
            exc.reason,
        )
        raise HTTPException(status_code=503, detail=exc.reason) from exc
    except Exception as exc:
        # Model/runtime failures are availability failures, never a fabricated
        # success or a half-created attempt.
        logger.exception(
            "[EntryV2] evidence processing failed id=%s camera=%s",
            getattr(request, "attempt_id", None)
            or getattr(request, "crossing_id", None),
            getattr(request, "camera_id", ""),
        )
        raise HTTPException(
            status_code=503, detail="entry_evidence_processing_unavailable"
        ) from exc


def _require_available(coordinator: EntryCoordinator) -> None:
    if not coordinator.available:
        raise HTTPException(
            status_code=503,
            detail=coordinator.unavailable_reason or "entry_v2_unavailable",
        )


def _authenticate(
    coordinator: EntryCoordinator, provided_service_key: Optional[str]
) -> None:
    expected = coordinator.settings.service_key
    supplied = provided_service_key or ""
    if not expected or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="invalid_service_key")


def _require_mode(coordinator: EntryCoordinator, supplied_mode: Optional[str]) -> None:
    if supplied_mode != coordinator.settings.mode.value:
        raise HTTPException(
            status_code=503,
            detail="entry_v2_mode_mismatch",
            headers={"Retry-After": "1"},
        )


def _parse_id(name: str, value: str) -> str:
    value = (value or "").strip()
    if not _ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=422, detail=f"invalid_{name}")
    return value


def _parse_plate(value: str) -> str:
    value = (value or "").strip().upper()
    if not value or len(value) > 32:
        raise HTTPException(status_code=422, detail="invalid_reported_plate")
    return value


def _parse_reported_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not 0.0 <= value <= 100.0:
        raise HTTPException(status_code=422, detail="reported_confidence_out_of_range")
    return value / 100.0


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid_captured_at") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HTTPException(status_code=422, detail="captured_at_requires_timezone")
    return parsed


def _parse_metadata(raw: str, coordinator: EntryCoordinator) -> Dict[str, Any]:
    encoded = (raw or "{}").encode("utf-8")
    if len(encoded) > coordinator.settings.max_metadata_bytes:
        raise HTTPException(status_code=422, detail="metadata_too_large")
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="invalid_metadata_json") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="metadata_must_be_object")
    return value


async def _read_images(
    uploads: List[UploadFile],
    coordinator: EntryCoordinator,
    metadata: Dict[str, Any],
) -> List[bytes]:
    if not uploads:
        raise HTTPException(status_code=422, detail="at_least_one_image_is_required")
    if len(uploads) > coordinator.settings.max_images_per_event:
        for upload in uploads:
            await upload.close()
        raise HTTPException(status_code=422, detail="too_many_images")

    roles = metadata.get("image_roles")
    if roles is not None:
        if not isinstance(roles, list) or len(roles) != len(uploads):
            for upload in uploads:
                await upload.close()
            raise HTTPException(status_code=422, detail="invalid_image_roles")
        if any(str(role).strip().lower() != "vehicle" for role in roles):
            for upload in uploads:
                await upload.close()
            raise HTTPException(
                status_code=422, detail="only_vehicle_crop_images_are_allowed"
            )

    collected: List[bytes] = []
    try:
        for upload in uploads:
            # A role-bearing filename is only a fail-closed safety signal; the
            # positive contract is metadata image_roles or vehicle-only parts.
            # Generic Hikvision filenames are not used to select evidence.
            filename = (upload.filename or "").lower()
            if "licenseplate" in re.sub(r"[^a-z]", "", filename) or filename.startswith(
                "plate"
            ):
                raise HTTPException(
                    status_code=422, detail="plate_only_image_not_allowed"
                )
            content_type = (upload.content_type or "").lower()
            if content_type and not (
                content_type.startswith("image/")
                or content_type == "application/octet-stream"
            ):
                raise HTTPException(
                    status_code=422, detail="image_content_type_required"
                )
            data = await upload.read(coordinator.settings.max_image_bytes + 1)
            if not data:
                raise HTTPException(status_code=422, detail="empty_image")
            if len(data) > coordinator.settings.max_image_bytes:
                raise HTTPException(status_code=422, detail="image_too_large")
            collected.append(bytes(data))
        return collected
    finally:
        for upload in uploads:
            await upload.close()


def _response(result: IngestResult) -> EntryIngestResponse:
    return EntryIngestResponse(
        status="duplicate" if result.duplicate else "accepted",
        id=result.resource_id,
        duplicate=result.duplicate,
        mode=result.mode.value,
        evidence_count=result.evidence_count,
        group_id=result.group_id,
        decision_id=result.decision_id,
        decision_status=result.decision_status,
        callback_delivered=result.callback_delivered,
    )
