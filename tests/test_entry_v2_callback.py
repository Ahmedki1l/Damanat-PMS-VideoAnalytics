import json
import threading
from types import SimpleNamespace
from urllib import error as urlerror

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.entry.callback import (
    CallbackError,
    HttpConfirmationClient,
    RetryingConfirmationSink,
)
from src.entry.settings import SERVICE_KEY_HEADER
from src.entry.runtime import entry_callback_retry_lifespan


class SequenceClient:
    def __init__(self, failures, ack_result=None):
        self.failures = list(failures)
        self.ack_result = ack_result
        self.calls = 0

    def send(self, payload):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.ack_result


def test_app_lifecycle_periodically_drains_pending_callbacks():
    called = threading.Event()

    class Coordinator:
        available = True
        settings = SimpleNamespace(callback_retry_interval_seconds=0.01)

        @staticmethod
        def retry_pending_callbacks():
            called.set()
            return {}

    app = FastAPI(lifespan=entry_callback_retry_lifespan(Coordinator()))

    with TestClient(app):
        assert called.wait(timeout=1.0)
        assert app.state.entry_v2_callback_retry_task.done() is False

    assert app.state.entry_v2_callback_retry_task is None


def test_retrying_sink_retries_transient_failure_with_bounded_backoff():
    client = SequenceClient([CallbackError("temporary", retryable=True)])
    sleeps = []
    sink = RetryingConfirmationSink(
        client,
        max_attempts=3,
        initial_backoff_seconds=0.25,
        max_backoff_seconds=1.0,
        sleeper=sleeps.append,
    )
    result = sink.deliver({"decision_id": "d1"})

    assert result.delivered is True
    assert result.attempts == 2
    assert client.calls == 2
    assert sleeps == [0.25]


def test_retrying_sink_does_not_retry_permanent_4xx_equivalent():
    client = SequenceClient([CallbackError("http_401", retryable=False)])
    sink = RetryingConfirmationSink(client, max_attempts=4, sleeper=lambda _: None)
    result = sink.deliver({"decision_id": "d1"})

    assert result.delivered is False
    assert result.attempts == 1
    assert result.retryable is False
    assert client.calls == 1


def test_http_client_sends_json_with_service_key_and_no_image_fields():
    captured = {}

    class Response:
        status = 200

        def read(self, size):
            assert size == 64 * 1024 + 1
            return json.dumps(
                {
                    "decision_id": "d1",
                    "status": "confirmed",
                    "result": "created",
                    "plate_number": "ABC-1234",
                    "entry_log_id": 1,
                    "session_id": 2,
                }
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout):
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data
        captured["timeout"] = timeout
        return Response()

    client = HttpConfirmationClient(
        "http://pms-ai:8080/api/v1/internal/entry-confirmations",
        "shared-secret",
        timeout_seconds=3.0,
        opener=fake_urlopen,
    )
    payload = {
        "decision_id": "d1",
        "status": "confirmed",
        "canonical_plate": "ABC-1234",
    }
    client.send(payload)

    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers[SERVICE_KEY_HEADER.lower()] == "shared-secret"
    assert headers["content-type"] == "application/json"
    assert b"ABC-1234" in captured["data"]
    assert b"image" not in captured["data"]
    assert captured["timeout"] == 3.0


@pytest.mark.parametrize(
    ("ack", "reason"),
    [
        (b"<html>ok</html>", "invalid_callback_json"),
        (
            {
                "decision_id": "other",
                "status": "confirmed",
                "result": "created",
                "plate_number": "ABC-1234",
                "entry_log_id": 1,
                "session_id": 2,
            },
            "callback_decision_id_mismatch",
        ),
        (
            {
                "decision_id": "d1",
                "status": "confirmed",
                "result": "shadowed",
                "plate_number": "ABC-1234",
            },
            "callback_result_mismatch",
        ),
        (
            {
                "decision_id": "d1",
                "status": "confirmed",
                "result": "created",
                "plate_number": "XYZ-9999",
                "entry_log_id": 1,
                "session_id": 2,
            },
            "callback_plate_mismatch",
        ),
    ],
)
def test_http_client_rejects_semantically_invalid_success(ack, reason):
    class Response:
        status = 200

        def read(self, size):
            del size
            return ack if isinstance(ack, bytes) else json.dumps(ack).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    client = HttpConfirmationClient(
        "http://pms/callback",
        "secret",
        opener=lambda req, timeout: Response(),
    )

    with pytest.raises(CallbackError, match=reason) as exc_info:
        client.send(
            {
                "decision_id": "d1",
                "status": "confirmed",
                "canonical_plate": "ABC-1234",
            }
        )
    assert exc_info.value.retryable is False


def test_semantic_callback_failure_is_not_retried():
    class Response:
        status = 200

        def read(self, size):
            del size
            return json.dumps(
                {
                    "decision_id": "wrong-decision",
                    "status": "confirmed",
                    "result": "created",
                    "plate_number": "ABC-1234",
                    "entry_log_id": 1,
                    "session_id": 2,
                }
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    client = HttpConfirmationClient(
        "http://pms/callback",
        "secret",
        opener=lambda req, timeout: Response(),
    )
    sink = RetryingConfirmationSink(client, max_attempts=4, sleeper=lambda _: None)

    result = sink.deliver(
        {
            "decision_id": "d1",
            "status": "confirmed",
            "canonical_plate": "ABC-1234",
        }
    )

    assert result.delivered is False
    assert result.attempts == 1
    assert result.retryable is False


def test_http_client_rejects_invalid_url_and_never_follows_redirect():
    with pytest.raises(ValueError, match="absolute HTTP"):
        HttpConfirmationClient("[http://pms](http://pms)", "secret")

    def redirect(req, timeout):
        del req, timeout
        raise urlerror.HTTPError(
            "http://pms/callback", 302, "redirect", {}, None
        )

    client = HttpConfirmationClient(
        "http://pms/callback", "secret", opener=redirect
    )
    with pytest.raises(CallbackError, match="http_302") as exc_info:
        client.send({"decision_id": "d1", "status": "abstained"})
    assert exc_info.value.retryable is True


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_http_client_classifies_transient_callback_statuses_as_retryable(status):
    def reject(req, timeout):
        del req, timeout
        raise urlerror.HTTPError(
            "http://pms/callback",
            status,
            "temporary",
            {},
            None,
        )

    client = HttpConfirmationClient(
        "http://pms/callback",
        "secret",
        opener=reject,
    )

    with pytest.raises(CallbackError, match=f"http_{status}") as exc_info:
        client.send({"decision_id": "d1", "status": "abstained"})

    assert exc_info.value.retryable is True


@pytest.mark.parametrize(
    "terminal_result",
    ["stale_after_exit", "superseded_by_newer_entry"],
)
def test_terminal_ack_is_delivered_without_identity_publication(terminal_result):
    class Response:
        status = 200

        def read(self, size):
            del size
            return json.dumps(
                {
                    "decision_id": "d-stale",
                    "status": "confirmed",
                    "result": terminal_result,
                    "plate_number": "ABC-1234",
                    "entry_log_id": None,
                    "session_id": None,
                }
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    client = HttpConfirmationClient(
        "http://pms/callback",
        "secret",
        opener=lambda req, timeout: Response(),
    )
    sink = RetryingConfirmationSink(client, max_attempts=3)

    result = sink.deliver(
        {
            "decision_id": "d-stale",
            "status": "confirmed",
            "canonical_plate": "ABC-1234",
        }
    )

    assert result.delivered is True
    assert result.attempts == 1
    assert result.publish_identity is False
