"""Authenticated metadata-only PMS callback with bounded retry policy."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Protocol
from urllib import error, request
from urllib.parse import urlsplit

from .settings import SERVICE_KEY_HEADER


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    attempts: int
    error: str = ""
    publish_identity: bool = True
    retryable: bool = True
    # None covers shadow/abstention/test sinks. False is explicit PMS proof
    # that a confirmed decision produced no live session (stale/superseded).
    session_committed: Optional[bool] = None
    ack_result: str = ""
    # Set only when VA's registry proves that a particular source-timestamped
    # exit won after PMS accepted the callback but before identity publication.
    superseding_exit_at: Optional[datetime] = None


class ConfirmationSink(Protocol):
    def deliver(self, payload: Mapping[str, Any]) -> DeliveryResult: ...


class CallbackError(Exception):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


_PLATE_KEY_RE = re.compile(r"[^A-Z0-9]+")
_TERMINAL_NO_PUBLISH_RESULTS = frozenset(
    {
        "stale_after_exit",
        "superseded_by_newer_entry",
    }
)
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429})


class _NoRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


class HttpConfirmationClient:
    """One HTTP attempt; retry scheduling is intentionally separate."""

    def __init__(
        self,
        url: str,
        service_key: str,
        timeout_seconds: float = 5.0,
        expected_mode: str = "authoritative",
        opener=None,
    ):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("PMS callback URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError(
                "PMS callback URL must not contain credentials or a fragment"
            )
        self._url = url
        self._service_key = service_key
        self._timeout = float(timeout_seconds)
        self._expected_mode = expected_mode
        self._open = opener or request.build_opener(_NoRedirectHandler()).open

    def send(self, payload: Mapping[str, Any]) -> str:
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        req = request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                SERVICE_KEY_HEADER: self._service_key,
            },
            method="POST",
        )
        try:
            with self._open(req, timeout=self._timeout) as response:
                status = int(getattr(response, "status", 200))
                response_body = response.read(64 * 1024 + 1)
        except error.HTTPError as exc:
            raise CallbackError(
                f"http_{exc.code}",
                retryable=(300 <= exc.code < 400)
                or exc.code >= 500
                or exc.code in _RETRYABLE_HTTP_STATUSES,
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise CallbackError(str(exc), retryable=True) from exc
        if not 200 <= status < 300:
            raise CallbackError(
                f"http_{status}",
                retryable=(status >= 500 or status in _RETRYABLE_HTTP_STATUSES),
            )
        if len(response_body) > 64 * 1024:
            raise CallbackError("callback_response_too_large", retryable=False)
        return self._validate_ack(payload, response_body)

    def _validate_ack(
        self,
        payload: Mapping[str, Any],
        response_body: bytes,
    ) -> str:
        try:
            ack = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CallbackError("invalid_callback_json", retryable=False) from exc
        if not isinstance(ack, dict):
            raise CallbackError("invalid_callback_ack", retryable=False)
        if ack.get("decision_id") != payload.get("decision_id"):
            raise CallbackError("callback_decision_id_mismatch", retryable=False)
        if ack.get("status") != payload.get("status"):
            raise CallbackError("callback_status_mismatch", retryable=False)

        if self._expected_mode == "shadow":
            expected_results = {"shadowed"}
        elif payload.get("status") == "confirmed":
            expected_results = {"created", "duplicate"} | set(
                _TERMINAL_NO_PUBLISH_RESULTS
            )
        else:
            expected_results = {"abstained"}
        ack_result = ack.get("result")
        if ack_result not in expected_results:
            raise CallbackError("callback_result_mismatch", retryable=False)

        if payload.get("status") == "confirmed":
            expected_plate = _PLATE_KEY_RE.sub(
                "", str(payload.get("canonical_plate") or "").upper()
            )
            actual_plate = _PLATE_KEY_RE.sub(
                "", str(ack.get("plate_number") or "").upper()
            )
            if not expected_plate or actual_plate != expected_plate:
                raise CallbackError("callback_plate_mismatch", retryable=False)
            if ack_result in {"created", "duplicate"} and (
                ack.get("entry_log_id") is None or ack.get("session_id") is None
            ):
                raise CallbackError("callback_ids_missing", retryable=False)
        return str(ack_result)


class RetryingConfirmationSink:
    def __init__(
        self,
        client: HttpConfirmationClient,
        *,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.2,
        max_backoff_seconds: float = 2.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self._client = client
        self._max_attempts = max(1, int(max_attempts))
        self._initial_backoff = max(0.0, float(initial_backoff_seconds))
        self._max_backoff = max(0.0, float(max_backoff_seconds))
        self._sleeper = sleeper

    def deliver(self, payload: Mapping[str, Any]) -> DeliveryResult:
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                ack_result = self._client.send(payload)
                return DeliveryResult(
                    delivered=True,
                    attempts=attempt,
                    publish_identity=(ack_result not in _TERMINAL_NO_PUBLISH_RESULTS),
                    session_committed=(
                        True
                        if ack_result in {"created", "duplicate"}
                        else False
                        if ack_result in _TERMINAL_NO_PUBLISH_RESULTS
                        else None
                    ),
                    ack_result=ack_result,
                )
            except CallbackError as exc:
                last_error = str(exc)
                if not exc.retryable or attempt == self._max_attempts:
                    return DeliveryResult(
                        False,
                        attempt,
                        last_error,
                        retryable=exc.retryable,
                    )
                delay = min(
                    self._max_backoff,
                    self._initial_backoff * (2 ** (attempt - 1)),
                )
                if delay > 0:
                    self._sleeper(delay)
            except Exception as exc:  # defensive boundary for injected clients
                last_error = str(exc)
                if attempt == self._max_attempts:
                    return DeliveryResult(False, attempt, last_error)
        return DeliveryResult(False, self._max_attempts, last_error)
