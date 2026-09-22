"""Crash-safe, local receipts for the opt-in Entry V2 durability pilot.

The store is an admission journal, not a second matching engine.  It keeps the
already validated request bytes and the callback receipt so a restart can
rebuild only work that had no durable decision.  Rows are never pruned by the
matching TTL: expiry is an inspectable terminal state.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import fcntl
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from .domain import AttemptInput, CrossingInput, CrossingRole, EntryConflict, EntryMode, IngestResult


class EntryDurabilityLocked(OSError):
    """Another process owns this single-host journal."""


class EntryDurabilityStore:
    """Single-host SQLite journal.  Every mutating operation is committed."""

    def __init__(self, directory: str):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise OSError("entry_v2_durability_state_dir_not_directory")
        self.path = root / "entry_v2_durability.sqlite3"
        self._lock_file = (root / "entry_v2_durability.lock").open("a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise EntryDurabilityLocked("entry_v2_durability_already_owned") from exc
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS entry_receipts (
                    kind TEXT NOT NULL, resource_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, lifecycle TEXT NOT NULL,
                    request_json TEXT NOT NULL, images_json TEXT NOT NULL,
                    result_json TEXT, callback_json TEXT, callback_delivered INTEGER,
                    decision_id TEXT,
                    identity_json TEXT,
                    terminal_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(kind, resource_id)
                )"""
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(entry_receipts)")
            }
            if "decision_id" not in columns:
                connection.execute("ALTER TABLE entry_receipts ADD COLUMN decision_id TEXT")
            if "identity_json" not in columns:
                connection.execute("ALTER TABLE entry_receipts ADD COLUMN identity_json TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS entry_receipts_decision_id ON entry_receipts(decision_id)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS entry_finalized_journeys (
                    decision_id TEXT PRIMARY KEY, journey_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entry_journal_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        if self._lock_file is None:
            raise OSError("entry_v2_durability_store_closed")
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def close(self) -> None:
        lock_file = getattr(self, "_lock_file", None)
        if lock_file is None:
            return
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        self._lock_file = None

    def bind_mode(self, mode: str) -> None:
        """Refuse a state volume produced under a different decision policy."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM entry_journal_metadata WHERE key='entry_mode'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO entry_journal_metadata(key, value) VALUES ('entry_mode', ?)",
                    (mode,),
                )
            elif row[0] != mode:
                connection.execute("ROLLBACK")
                raise OSError("entry_v2_durability_mode_mismatch")
            connection.execute("COMMIT")

    def __del__(self):  # pragma: no cover - interpreter shutdown is nondeterministic
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    @staticmethod
    def fingerprint(kind: str, request: AttemptInput | CrossingInput, images: Sequence[bytes]) -> str:
        payload = _request_json(kind, request).encode("utf-8")
        digest = hashlib.sha256(kind.encode("ascii") + b"\0" + payload)
        for image in images:
            digest.update(len(image).to_bytes(8, "big"))
            digest.update(image)
        return digest.hexdigest()

    def accept(
        self, kind: str, request: AttemptInput | CrossingInput, images: Sequence[bytes]
    ) -> Optional[IngestResult]:
        """Durably admit once, or return the prior receipt for the exact retry."""
        resource_id = request.attempt_id if kind == "attempt" else request.crossing_id
        fingerprint = self.fingerprint(kind, request, images)
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT fingerprint, result_json, lifecycle, terminal_reason FROM entry_receipts WHERE kind=? AND resource_id=?",
                (kind, resource_id),
            ).fetchone()
            if existing is not None:
                if existing[0] != fingerprint:
                    connection.execute("ROLLBACK")
                    raise EntryConflict("entry_resource_id_reused_with_different_evidence")
                connection.execute("COMMIT")
                return _stored_result(
                    existing[1], resource_id, existing[2], existing[3]
                )
            connection.execute(
                """INSERT INTO entry_receipts
                   (kind, resource_id, fingerprint, lifecycle, request_json, images_json, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)""",
                (kind, resource_id, fingerprint, _request_json(kind, request), _images_json(images), now, now),
            )
            connection.execute("COMMIT")
        return None

    def record_result(self, kind: str, result: IngestResult) -> None:
        self._update_result(kind, result.resource_id, result, lifecycle="pending")

    def record_decision(
        self,
        resources: Sequence[Tuple[str, str]],
        payload: dict[str, Any],
        identity: Optional[dict[str, Any]],
        receipt_metadata: dict[str, Any],
        finalized_journey: Optional[dict[str, Any]] = None,
    ) -> None:
        """Commit the decision before any callback I/O can observe it."""
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        # Preserve field order as originally constructed by EntryDecision.  The
        # callback contract includes optional reported metadata, so replay must
        # not rebuild it from a later domain representation.
        encoded = json.dumps(payload, separators=(",", ":"), default=str)
        decision_id = str(payload["decision_id"])
        identity_json = (
            json.dumps(identity, separators=(",", ":"), default=str)
            if identity is not None
            else None
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for kind, resource_id in resources:
                row = connection.execute(
                    "SELECT result_json, images_json FROM entry_receipts WHERE kind=? AND resource_id=?",
                    (kind, resource_id),
                ).fetchone()
                if row is None:
                    continue
                receipt = json.loads(row[0]) if row[0] else {
                    "resource_id": resource_id,
                    "accepted": True,
                    "duplicate": False,
                    "mode": receipt_metadata["mode"],
                    "evidence_count": len(json.loads(row[1])),
                }
                receipt.update(receipt_metadata)
                receipt.update(
                    decision_id=decision_id,
                    decision_status=payload["status"],
                    callback_delivered=False,
                    receipt_status="callback_pending",
                )
                connection.execute(
                    "UPDATE entry_receipts SET lifecycle='callback_pending', decision_id=?, callback_json=?, identity_json=?, result_json=?, updated_at=? WHERE kind=? AND resource_id=?",
                    (decision_id, encoded, identity_json, json.dumps(receipt, sort_keys=True), now, kind, resource_id),
                )
            if finalized_journey is not None:
                candidate = dict(finalized_journey)
                candidate["delivery_state"] = "callback_pending"
                connection.execute(
                    "INSERT OR REPLACE INTO entry_finalized_journeys(decision_id, journey_json, updated_at) VALUES (?, ?, ?)",
                    (decision_id, json.dumps(candidate, separators=(",", ":")), now),
                )
            connection.execute("COMMIT")

    def mark_callback(
        self,
        decision_id: str,
        delivered: bool,
        reason: str = "",
        retryable: bool = True,
        finalized_journey: Optional[dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        lifecycle = "resolved" if delivered else ("callback_pending" if retryable else "unresolved")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT kind, resource_id, result_json FROM entry_receipts WHERE decision_id=?",
                (decision_id,),
            ).fetchall()
            for kind, resource_id, encoded in rows:
                result = _stored_result(encoded, resource_id, lifecycle, reason)
                result = replace(
                    result,
                    accepted=False,
                    callback_delivered=delivered,
                    receipt_status=lifecycle,
                )
                self._write_result(connection, kind, resource_id, result, lifecycle, reason, now)
            if delivered:
                if finalized_journey is None:
                    connection.execute(
                        "DELETE FROM entry_finalized_journeys WHERE decision_id=?",
                        (decision_id,),
                    )
                else:
                    finalized = dict(finalized_journey)
                    finalized["delivery_state"] = "resolved"
                    connection.execute(
                        "INSERT OR REPLACE INTO entry_finalized_journeys(decision_id, journey_json, updated_at) VALUES (?, ?, ?)",
                        (decision_id, json.dumps(finalized, separators=(",", ":")), now),
                    )
            connection.execute("COMMIT")

    def mark_terminal(self, kind: str, resource_id: str, status: str, reason: str) -> None:
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result_json FROM entry_receipts WHERE kind=? AND resource_id=?", (kind, resource_id)
            ).fetchone()
            if row is not None:
                result = _stored_result(row[0], resource_id, status, reason)
                result = replace(result, accepted=False, receipt_status=status)
                self._write_result(connection, kind, resource_id, result, status, reason, now)
            connection.execute("COMMIT")

    def recovery_inputs(self) -> list[Tuple[str, AttemptInput | CrossingInput, Tuple[bytes, ...]]]:
        """Return only undecided active work, oldest capture first."""
        return [
            (kind, request, images)
            for kind, _, request in self.recovery_candidates(limit=256)
            for loaded in [self.load_input(kind, request)]
            if loaded is not None
            for _, _, images in [loaded]
        ]

    def recovery_candidates(
        self,
        *,
        limit: int,
    ) -> list[Tuple[str, str, AttemptInput | CrossingInput]]:
        """Fetch bounded metadata only; callers load images after claiming it."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT kind, resource_id, request_json FROM entry_receipts WHERE lifecycle='pending' ORDER BY created_at, kind, resource_id LIMIT ?",
                (limit,),
            ).fetchall()
        candidates = [
            (kind, resource_id, _request_from_json(kind, request_json))
            for kind, resource_id, request_json in rows
        ]
        return sorted(candidates, key=lambda item: (item[2].captured_at, item[0]))

    def load_input(
        self,
        kind: str,
        request: AttemptInput | CrossingInput,
    ) -> Optional[Tuple[str, AttemptInput | CrossingInput, Tuple[bytes, ...]]]:
        resource_id = request.attempt_id if kind == "attempt" else request.crossing_id
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT images_json FROM entry_receipts WHERE kind=? AND resource_id=? AND lifecycle='pending'",
                (kind, resource_id),
            ).fetchone()
        if row is None:
            return None
        return kind, request, _images_from_json(row[0])

    def pending_callbacks(self) -> list[dict[str, Any]]:
        return [payload for payload, _, _ in self.pending_callback_records()]

    def pending_callback_records(self) -> list[
        Tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]
    ]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT receipt.callback_json, receipt.identity_json, journey.journey_json
                   FROM entry_receipts AS receipt
                   LEFT JOIN entry_finalized_journeys AS journey ON journey.decision_id=receipt.decision_id
                   WHERE receipt.lifecycle='callback_pending' AND receipt.callback_json IS NOT NULL"""
            ).fetchall()
        return [
            (
                json.loads(payload),
                json.loads(identity) if identity else None,
                json.loads(journey) if journey else None,
            )
            for payload, identity, journey in rows
        ]

    def is_pending(self, kind: str, resource_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT lifecycle FROM entry_receipts WHERE kind=? AND resource_id=?",
                (kind, resource_id),
            ).fetchone()
        return row is not None and row[0] == "pending"

    def pending_local_crossings(self) -> list[Tuple[CrossingInput, Tuple[bytes, ...]]]:
        """Read persisted local-zone work without retaining crop bytes in RAM."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT request_json FROM entry_receipts WHERE kind='crossing' AND lifecycle='pending' AND result_json IS NULL ORDER BY created_at LIMIT 1"
            ).fetchall()
        pending = []
        for (request_json,) in rows:
            crossing = _request_from_json("crossing", request_json)
            if crossing.metadata.get("source") == "va_local_zone":
                loaded = self.load_input("crossing", crossing)
                if loaded is not None:
                    pending.append((crossing, loaded[2]))
        return sorted(pending, key=lambda item: item[0].captured_at)

    def save_finalized_journey(self, journey: dict[str, Any]) -> None:
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO entry_finalized_journeys(decision_id, journey_json, updated_at) VALUES (?, ?, ?)",
                (journey["decision_id"], json.dumps(journey, separators=(",", ":")), now),
            )

    def finalized_journeys(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT journey_json FROM entry_finalized_journeys ORDER BY updated_at"
            ).fetchall()
        return [
            record
            for row in rows
            for record in [json.loads(row[0])]
            if record.get("delivery_state") != "callback_pending"
        ]

    def close_finalized_journey(self, decision_id: str, reason: str) -> None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT journey_json FROM entry_finalized_journeys WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
            if row is None:
                return
            journey = json.loads(row[0])
            journey["lifecycle"] = "closed_without_exit_timestamp"
            journey["close_reason"] = reason
            connection.execute(
                "UPDATE entry_finalized_journeys SET journey_json=?, updated_at=? WHERE decision_id=?",
                (
                    json.dumps(journey, separators=(",", ":")),
                    datetime.utcnow().isoformat(timespec="microseconds") + "Z",
                    decision_id,
                ),
            )

    def identity_for_decision(self, decision_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT identity_json FROM entry_receipts WHERE decision_id=? AND lifecycle='resolved' AND identity_json IS NOT NULL LIMIT 1",
                (decision_id,),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def callback_for_decision(self, decision_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT callback_json FROM entry_receipts WHERE decision_id=? AND lifecycle='resolved' AND callback_json IS NOT NULL LIMIT 1",
                (decision_id,),
            ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def _update_result(self, kind: str, resource_id: str, result: IngestResult, *, lifecycle: str) -> None:
        now = datetime.utcnow().isoformat(timespec="microseconds") + "Z"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT lifecycle, terminal_reason FROM entry_receipts WHERE kind=? AND resource_id=?",
                (kind, resource_id),
            ).fetchone()
            if existing is not None and existing[0] != "pending":
                lifecycle, reason = existing
            else:
                reason = ""
            self._write_result(connection, kind, resource_id, result, lifecycle, reason, now)
            connection.execute("COMMIT")

    @staticmethod
    def _write_result(connection, kind, resource_id, result, lifecycle, reason, now) -> None:
        connection.execute(
            "UPDATE entry_receipts SET lifecycle=?, result_json=?, terminal_reason=?, callback_delivered=?, updated_at=? WHERE kind=? AND resource_id=?",
            (lifecycle, json.dumps(_result_dict(result), sort_keys=True), reason, result.callback_delivered, now, kind, resource_id),
        )


def _request_json(kind: str, request: AttemptInput | CrossingInput) -> str:
    payload: dict[str, Any] = dict(request.metadata)
    common = {"source_event_id": request.source_event_id, "camera_id": request.camera_id, "captured_at": request.captured_at.isoformat(), "metadata": payload}
    if kind == "attempt":
        return json.dumps({"attempt_id": request.attempt_id, "reported_plate": request.reported_plate, "reported_confidence": request.reported_confidence, **common}, sort_keys=True, separators=(",", ":"), default=str)
    crossing = request
    return json.dumps({"crossing_id": crossing.crossing_id, "line_id": crossing.line_id, "direction": crossing.direction, "role": crossing.role.value, **common}, sort_keys=True, separators=(",", ":"), default=str)


def _request_from_json(kind: str, value: str) -> AttemptInput | CrossingInput:
    raw = json.loads(value)
    common = dict(source_event_id=raw["source_event_id"], camera_id=raw["camera_id"], captured_at=datetime.fromisoformat(raw["captured_at"]), metadata=raw["metadata"])
    if kind == "attempt":
        return AttemptInput(attempt_id=raw["attempt_id"], reported_plate=raw["reported_plate"], reported_confidence=raw["reported_confidence"], **common)
    return CrossingInput(crossing_id=raw["crossing_id"], line_id=raw["line_id"], direction=raw["direction"], role=CrossingRole(raw["role"]), **common)


def _images_json(images: Sequence[bytes]) -> str:
    return json.dumps([image.hex() for image in images], separators=(",", ":"))


def _images_from_json(value: str) -> Tuple[bytes, ...]:
    return tuple(bytes.fromhex(encoded) for encoded in json.loads(value))


def _result_dict(result: IngestResult) -> dict[str, Any]:
    return {"resource_id": result.resource_id, "accepted": result.accepted, "duplicate": result.duplicate, "mode": result.mode.value, "evidence_count": result.evidence_count, "group_id": result.group_id, "decision_id": result.decision_id, "decision_status": result.decision_status, "callback_delivered": result.callback_delivered, "receipt_status": result.receipt_status}


def _stored_result(value: Optional[str], resource_id: str, lifecycle: str, reason: str) -> IngestResult:
    raw = json.loads(value) if value else {}
    status = raw.get("receipt_status") or lifecycle
    active = lifecycle == "pending" and status == "pending"
    return IngestResult(resource_id=raw.get("resource_id", resource_id), accepted=bool(raw.get("accepted", active)) if active else False, duplicate=True, mode=EntryMode(raw.get("mode", EntryMode.OFF.value)), evidence_count=int(raw.get("evidence_count", 0)), group_id=raw.get("group_id"), decision_id=raw.get("decision_id"), decision_status=raw.get("decision_status"), callback_delivered=raw.get("callback_delivered"), receipt_status=status)
