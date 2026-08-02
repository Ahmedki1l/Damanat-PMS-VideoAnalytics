"""Hand recovered slot occupants to PMS-AI so a `parking_session` is opened.

WHY THIS EXISTS
---------------
Naming a slot is only half of a recovery. VA can prove, from its own persisted
gallery, that BHD-9990 is parked in B13 — and did so on 2026-08-02 at score 0.962
/ margin 0.902 against a same-view parked pose. But PMS-AI owns `parking_sessions`,
and until a row exists there the car is uncounted, absent from occupancy, and its
eventual exit still produces `[UC2] No matching entry found`.

Before this module the engine appended every recovery to `_pending_slot_recoveries`
and nothing ever read it: the list was written in one place, drained nowhere, and
grew for the life of the process. The endpoint on the other side was complete,
authenticated and enabled — it was simply never called.

DELIVERY POSTURE
----------------
Recoveries are NOT gate evidence and must never share the entry path's fate:

  * Independent of ENTRY_V2_MODE. This describes a car already inside, not one
    arriving at a barrier, so it works with Entry V2 off (as it is today). The
    only thing borrowed is the service-key boundary.
  * Best-effort with bounded retry. A dropped recovery costs one uncounted car
    until the next attempt; blocking the frame loop to guarantee it would cost
    every camera. Delivery runs on its own thread.
  * A `rejected` result is SUCCESS, not failure. PMS-AI re-reads the live slot
    before writing and refuses when the world moved on (car left, plate cleared,
    another car took the slot). That is the race guard working as designed, and
    re-sending would be asking it to act on evidence it has already judged stale.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib import error, request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

SERVICE_KEY_HEADER = "X-Service-Key"
RECOVERY_PATH = "/api/v1/internal/slot-recoveries"

# 4xx other than these means the claim itself is malformed — a retry sends the same
# bytes and gets the same answer, so it is dropped rather than spun on. 409 is the
# feature being disabled on the PMS-AI side, which a retry cannot fix either.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429})


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """A redirect off an authenticated internal endpoint is a misconfiguration,
    not a hop to follow — following it would leak the service key."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


class SlotRecoverySender:
    """Drains queued recoveries to PMS-AI. One attempt per item per drain."""

    def __init__(
        self,
        base_url: str,
        service_key: str,
        *,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        max_queue: int = 256,
        opener: Optional[Callable] = None,
    ):
        base = (base_url or "").rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("PMS base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("PMS base URL must not carry credentials or a fragment")
        self._url = f"{base}{RECOVERY_PATH}"
        self._service_key = service_key
        self._timeout = float(timeout_seconds)
        self._max_attempts = max(1, int(max_attempts))
        self._max_queue = max(1, int(max_queue))
        self._open = opener or request.build_opener(_NoRedirectHandler()).open
        self._lock = threading.Lock()
        self._queue: List[Dict[str, Any]] = []

    # -- queue -------------------------------------------------------------

    def enqueue(self, payload: Mapping[str, Any]) -> None:
        """Take ownership of one recovery. Bounded: an unreachable PMS-AI must not
        turn a best-effort hand-off into an unbounded memory leak, which is exactly
        what the undrained list it replaces was."""
        with self._lock:
            if len(self._queue) >= self._max_queue:
                dropped = self._queue.pop(0)
                logger.warning(
                    "[recovery] send queue full (%d) — dropping the oldest claim "
                    "%s/%s undelivered",
                    self._max_queue, dropped.get("slot_id"),
                    dropped.get("plate_number"),
                )
            self._queue.append({**payload, "_attempts": 0})

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    # -- delivery ----------------------------------------------------------

    def drain(self) -> int:
        """One attempt for every queued claim. Returns the number settled."""
        with self._lock:
            batch, self._queue = self._queue, []
        settled = 0
        requeue: List[Dict[str, Any]] = []
        for item in batch:
            payload = {k: v for k, v in item.items() if not k.startswith("_")}
            try:
                result, reason = self._post(payload)
            except _Retryable as exc:
                item["_attempts"] += 1
                if item["_attempts"] >= self._max_attempts:
                    logger.warning(
                        "[recovery] GAVE UP handing %s/%s to PMS-AI after %d attempts: "
                        "%s — the slot stays named but the car remains uncounted",
                        payload.get("slot_id"), payload.get("plate_number"),
                        item["_attempts"], exc,
                    )
                    continue
                requeue.append(item)
                continue
            except _Permanent as exc:
                logger.error(
                    "[recovery] REFUSED by PMS-AI for %s/%s: %s — not retrying, the "
                    "claim itself is the problem",
                    payload.get("slot_id"), payload.get("plate_number"), exc,
                )
                continue
            settled += 1
            if result == "created":
                logger.warning(
                    "[recovery] SESSION OPENED for %s in %s — this car entered without "
                    "being recorded and is now counted",
                    payload.get("plate_number"), payload.get("slot_id"),
                )
            elif result == "rejected":
                # The race guard, working. Not an error and not retried.
                logger.info(
                    "[recovery] PMS-AI declined %s/%s: %s",
                    payload.get("slot_id"), payload.get("plate_number"), reason,
                )
            else:
                logger.info(
                    "[recovery] %s/%s -> %s",
                    payload.get("slot_id"), payload.get("plate_number"), result,
                )
        if requeue:
            with self._lock:
                self._queue = requeue + self._queue
        return settled

    def _post(self, payload: Mapping[str, Any]):
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
                raw = response.read(64 * 1024 + 1)
        except error.HTTPError as exc:
            retryable = exc.code >= 500 or exc.code in _RETRYABLE_HTTP_STATUSES
            raise (_Retryable if retryable else _Permanent)(
                f"http_{exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise _Retryable(str(exc)) from exc
        if not 200 <= status < 300:
            retryable = status >= 500 or status in _RETRYABLE_HTTP_STATUSES
            raise (_Retryable if retryable else _Permanent)(f"http_{status}")
        if len(raw) > 64 * 1024:
            raise _Permanent("recovery_response_too_large")
        try:
            ack = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _Permanent("invalid_recovery_json") from exc
        if not isinstance(ack, dict):
            raise _Permanent("invalid_recovery_ack")
        return str(ack.get("result") or "unknown"), str(ack.get("reason") or "")


class _Retryable(Exception):
    pass


class _Permanent(Exception):
    pass
