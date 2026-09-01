"""Bounded, RAM-only coordinator for attempt/crossing validation."""

from __future__ import annotations

import itertools
import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from . import decision_record
from .analyzer import EvidenceProcessor
from .callback import ConfirmationSink, DeliveryResult
from .decision import (
    EntryDecisionEngine,
    ReIDMatchEvaluation,
    causal_group_embeddings,
    causally_eligible_attempts,
    max_similarity,
)
from .identity import (
    IdentitySupersededByExit,
    IdentityPublisher,
    NullIdentityPublisher,
    ValidatedIdentity,
    build_gallery_authorization_proof,
)
from .domain import (
    AttemptGroup,
    AttemptInput,
    AttemptRecord,
    CancellationInput,
    CrossingInput,
    CrossingRecord,
    CrossingRole,
    DecisionStatus,
    EntryCapacityExceeded,
    EntryConflict,
    EntryDecision,
    EntryInvalid,
    EntryMode,
    EntryUnavailable,
    EvidenceUnavailable,
    FrameEvidence,
    HypothesisStatus,
    IngestResult,
    PlateHypothesis,
    PlateReading,
    PlateSourceKind,
    ReIDMatch,
    RecordStatus,
    WitnessSource,
    canonical_plate,
    norm_camera_id,
    plate_key,
    plates_contradict,
)
from .settings import EntrySettings


logger = logging.getLogger(__name__)


def _is_hik_sourced_attempt(request) -> bool:
    """Did these bytes come from a HikCentral query we issued?

    HikCentral is a PULL source. An attempt marked this way exists because our
    service asked HikCentral for a record and got an image back — not because
    HikCentral told us anything. The marker changes what the attempt counts as:
    HikCentral's plate reading rather than the gate ANPR system's, and a HIK
    witness rather than an ANPR one.
    """
    try:
        metadata = request.metadata or {}
        return str(metadata.get("evidence_source", "")).lower() == "hikcentral"
    except Exception:  # pragma: no cover - defensive
        return False


def _round_hsv(value):
    return None if value is None else [round(float(v), 1) for v in value]


def _best_observed_plate_read(crossing):
    """The ramp camera's own OCR reading and its confidence — DIAGNOSTIC ONLY.

    CAM-23 and CAM-03 are visual observation sources and read no plates. This
    is recorded so the question "could the ramp cameras carry plate evidence?"
    can be answered from real traffic later, and it is deliberately not
    returned in any shape a decision rule could consume.

    Returns (text, confidence). The confidence used to be computed here and
    then dropped on the floor, which left `observed_plate_confidence` null in
    every record ever written — so the one field that exists to answer "are
    ramp reads good enough to veto on?" could not answer it. That question
    gates re-enabling the observation plate veto, so the number has to survive
    the return.
    """
    best = ""
    best_confidence = None
    for item in crossing.plate_evidence:
        if item.text and (best_confidence is None or item.confidence > best_confidence):
            best, best_confidence = item.text, item.confidence
    return best, best_confidence


@dataclass(frozen=True)
class _FinalizedJourney:
    """Compact, bounded tombstone for source-ordered crossing deduplication."""

    decision_id: str
    group_id: str
    decision_status: str
    canonical_plate_key: str
    first_attempt_at: datetime
    entry_captured_at: datetime
    crossing_role: CrossingRole
    crossing_camera_id: str
    crossing_source: str
    embeddings: Tuple[Tuple[float, ...], ...]
    exit_captured_at: Optional[datetime] = None


class EntryCoordinator:
    """Correlates evidence without TTL, FIFO, image persistence, or a DB.

    Pending attempts are removed only after a strict confirmed decision is
    delivered.  An unrelated earlier attempt is never displaced because a
    later car matched.  All public mutators are safe to call from API worker
    threads; expensive inference and callback I/O happen outside the lock.
    """

    def __init__(
        self,
        settings: EntrySettings,
        evidence_processor: EvidenceProcessor,
        confirmation_sink: ConfirmationSink,
        identity_publisher: Optional[IdentityPublisher] = None,
        *,
        unavailable_reason: str = "",
        decision_log: Optional[Any] = None,
        clock: Optional[Any] = None,
    ):
        # The ONLY wall clock in the coordinator, and it exists solely to age
        # state out. It is never consulted to decide whether two observations
        # are the same vehicle — that is Re-ID's job, and a clock cannot answer
        # it. Injectable so TTL behaviour is testable without sleeping.
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.settings = settings
        self._processor = evidence_processor
        self._sink = confirmation_sink
        self._identity_publisher = identity_publisher or NullIdentityPublisher()
        # Optional by design: every test constructs a coordinator directly, and
        # none of them should have to know about a log. None means "record
        # nothing", never "fail".
        self._decision_log = decision_log
        self._engine = EntryDecisionEngine(settings)
        self._unavailable_reason = unavailable_reason
        self._lock = threading.RLock()
        self._attempts: Dict[str, AttemptRecord] = {}
        self._groups: Dict[str, AttemptGroup] = {}
        self._crossings: Dict[str, CrossingRecord] = {}
        self._receipts: "OrderedDict[Tuple[str, str], Tuple[str, IngestResult]]" = (
            OrderedDict()
        )
        self._inflight: set[Tuple[str, str]] = set()
        self._analysis_inflight: set[Tuple[str, str]] = set()
        self._pending_callbacks: Dict[str, Tuple[EntryDecision, Dict[str, Any]]] = {}
        self._callback_reservations: Dict[str, EntryDecision] = {}
        self._callback_retries_inflight: set[str] = set()
        self._delivered_callbacks_before_receipt: "OrderedDict[str, None]" = (
            OrderedDict()
        )
        self._finalized_journeys: "OrderedDict[str, _FinalizedJourney]" = OrderedDict()
        self._provisional_crossings: Dict[
            str, Tuple[CrossingRecord, _FinalizedJourney]
        ] = {}
        self._quarantined_crossings_before_receipt: Dict[str, _FinalizedJourney] = {}
        self._released_crossings_before_receipt: Dict[str, Optional[str]] = {}
        self._decisions_before_receipt: Dict[
            Tuple[str, str], Tuple[EntryDecision, bool]
        ] = {}
        self._pending_exits: "OrderedDict[Tuple[str, datetime], None]" = OrderedDict()
        self._applied_exits: "OrderedDict[Tuple[str, datetime], None]" = OrderedDict()
        self._ambiguous_exit_keys: "OrderedDict[Tuple[str, datetime], None]" = (
            OrderedDict()
        )
        self._fatal_delivery_error = ""
        self._permanent_callback_failure_count = 0
        self._late_ocr_conflict_count = 0
        self._late_ocr_conflict_ids: "OrderedDict[str, None]" = OrderedDict()
        self._ambiguous_exit_count = 0
        self._reid_evaluation_log_cache: "OrderedDict[str, Tuple[Any, ...]]" = (
            OrderedDict()
        )

    @property
    def available(self) -> bool:
        return (
            self.settings.mode != EntryMode.OFF
            and not self._unavailable_reason
            and not self._fatal_delivery_error
        )

    @property
    def unavailable_reason(self) -> str:
        if self._unavailable_reason:
            return self._unavailable_reason
        if self._fatal_delivery_error:
            return "entry_v2_callback_permanent_failure:" + self._fatal_delivery_error
        if self.settings.mode == EntryMode.OFF:
            return "entry_v2_disabled"
        return ""

    def ingest_attempt(
        self, request: AttemptInput, images: Sequence[bytes]
    ) -> IngestResult:
        self._ensure_available()
        self._validate_attempt(request, images)
        fingerprint = self._fingerprint("attempt", request, images)
        key = ("attempt", request.attempt_id)
        duplicate = self._preflight(
            key,
            fingerprint,
            maximum=self.settings.max_pending_attempts,
        )
        if duplicate is not None:
            return replace(duplicate, duplicate=True, accepted=False)

        try:
            evidence = tuple(
                self._processor.analyze(
                    event_id=request.attempt_id,
                    camera_id=request.camera_id,
                    source_role="anpr",
                    images=images,
                    metadata=request.metadata,
                )
            )
            self._require_single_vehicle_event(evidence)
            with self._lock:
                group = self._add_attempt_locked(request, evidence)
                decisions = self._finish_analysis_locked(key)
            delivered = self._dispatch(decisions)
            with self._lock:
                live_attempt = self._attempts.get(request.attempt_id)
                result_group_id = (
                    live_attempt.group_id
                    if live_attempt is not None
                    else group.group_id
                )
            related = next(
                (
                    decision
                    for decision in decisions
                    if decision.attempt_id == request.attempt_id
                ),
                self._related_decision(decisions, result_group_id),
            )
            blocking = self._failed_authoritative_confirmation(decisions, delivered)
            reported = blocking or related
            result = IngestResult(
                resource_id=request.attempt_id,
                accepted=True,
                duplicate=False,
                mode=self.settings.mode,
                evidence_count=len(evidence),
                group_id=result_group_id,
                decision_id=reported.decision_id if reported else None,
                decision_status=self._effective_status(reported) if reported else None,
                callback_delivered=(
                    delivered.get(reported.decision_id) if reported else None
                ),
            )
            self._complete_request(key, fingerprint, result)
            if blocking is not None:
                raise EntryUnavailable(
                    "entry_confirmation_permanent_failure"
                    if self._fatal_delivery_error
                    else "entry_confirmation_delivery_failed"
                )
            return result
        except Exception:
            self._dispatch(self._release_inflight(key))
            raise

    def ingest_crossing(
        self, request: CrossingInput, images: Sequence[bytes]
    ) -> IngestResult:
        self._ensure_available()
        self._validate_crossing(request, images)
        fingerprint = self._fingerprint("crossing", request, images)
        key = ("crossing", request.crossing_id)
        duplicate = self._preflight(
            key,
            fingerprint,
            maximum=self.settings.max_pending_crossings,
        )
        if duplicate is not None:
            return replace(duplicate, duplicate=True, accepted=False)

        try:
            evidence = tuple(
                self._processor.analyze(
                    event_id=request.crossing_id,
                    camera_id=request.camera_id,
                    source_role=request.role.value,
                    images=images,
                    metadata=request.metadata,
                )
            )
            self._require_single_vehicle_event(evidence)
            finalized_duplicate = None
            match_kind = ""
            with self._lock:
                finalized_match = self._find_finalized_journey_locked(
                    request,
                    evidence,
                )
                if finalized_match is None:
                    crossing = CrossingRecord(
                        request=request,
                        evidence=tuple(evidence),
                        created_at=self._clock(),
                    )
                    self._crossings[request.crossing_id] = crossing
                else:
                    finalized_duplicate, match_kind = finalized_match
                    if match_kind == "provisional":
                        # Quarantined rows have their own lifecycle (they are
                        # released or retired by journey reconciliation), so
                        # they are stamped for the log but never TTL-swept.
                        self._provisional_crossings[request.crossing_id] = (
                            CrossingRecord(
                                request=request,
                                evidence=tuple(evidence),
                                created_at=self._clock(),
                            ),
                            finalized_duplicate,
                        )
                decisions = self._finish_analysis_locked(key)
            delivered = self._dispatch(decisions)
            if finalized_match is None:
                related = next(
                    (
                        decision
                        for decision in decisions
                        if decision.crossing_id == request.crossing_id
                    ),
                    None,
                )
            else:
                related = None
            blocking = self._failed_authoritative_confirmation(decisions, delivered)
            reported = blocking or related
            if finalized_duplicate is not None and blocking is None:
                result = self._finalized_duplicate_result(
                    request.crossing_id,
                    len(evidence),
                    finalized_duplicate,
                )
                logger.info(
                    "[EntryV2] %s late crossing=%s finalized_decision=%s "
                    "group=%s captured_at=%s entry_captured_at=%s",
                    (
                        "provisionally quarantined"
                        if match_kind == "provisional"
                        else "quarantined"
                    ),
                    request.crossing_id,
                    finalized_duplicate.decision_id,
                    finalized_duplicate.group_id,
                    request.captured_at.isoformat(),
                    finalized_duplicate.entry_captured_at.isoformat(),
                )
            else:
                result = IngestResult(
                    resource_id=request.crossing_id,
                    accepted=True,
                    duplicate=False,
                    mode=self.settings.mode,
                    evidence_count=len(evidence),
                    group_id=reported.group_id if reported else None,
                    decision_id=reported.decision_id if reported else None,
                    decision_status=(
                        self._effective_status(reported) if reported else None
                    ),
                    callback_delivered=(
                        delivered.get(reported.decision_id) if reported else None
                    ),
                )
            result = self._complete_request(key, fingerprint, result)
            if blocking is not None:
                raise EntryUnavailable(
                    "entry_confirmation_permanent_failure"
                    if self._fatal_delivery_error
                    else "entry_confirmation_delivery_failed"
                )
            return result
        except Exception:
            self._dispatch(self._release_inflight(key))
            raise

    def retry_pending_callbacks(self) -> Dict[str, bool]:
        """Retry the bounded exhaustion queue without holding the state lock."""
        with self._lock:
            pending = []
            for decision, payload in self._pending_callbacks.values():
                if decision.decision_id in self._callback_retries_inflight:
                    continue
                self._callback_retries_inflight.add(decision.decision_id)
                pending.append((decision, payload))
        outcomes: Dict[str, bool] = {}
        capacity_freed = False
        for decision, payload in pending:
            try:
                result = self._deliver_decision(decision, payload)
                outcomes[decision.decision_id] = result.delivered
                if result.delivered:
                    with self._lock:
                        self._pending_callbacks.pop(decision.decision_id, None)
                        self._compact_delivered_locked(
                            decision,
                            remember_journey=self._should_remember_journey_locked(
                                decision,
                                result,
                            ),
                            superseding_exit_at=result.superseding_exit_at,
                        )
                        self._reconcile_pending_exits_locked(
                            plate_key(decision.canonical_plate or "")
                        )
                        self._mark_decision_delivered_locked(decision.decision_id)
                        capacity_freed = True
                elif not result.retryable:
                    with self._lock:
                        self._pending_callbacks.pop(decision.decision_id, None)
                        self._record_permanent_delivery_failure_locked(result)
                        self._reconcile_pending_exits_locked(
                            plate_key(decision.canonical_plate or "")
                        )
            finally:
                with self._lock:
                    self._callback_retries_inflight.discard(decision.decision_id)
        if capacity_freed and not self._fatal_delivery_error:
            with self._lock:
                decisions = (
                    [] if self._analysis_inflight else self._evaluate_pending_locked()
                )
            self._dispatch(decisions)
        return outcomes

    def state_summary(self) -> Dict[str, Any]:
        """Metadata-only diagnostic snapshot used by tests/health tooling."""
        with self._lock:
            return {
                "mode": self.settings.mode.value,
                "attempt_count": len(self._attempts),
                "group_count": len(self._groups),
                "crossing_count": len(self._crossings),
                "pending_callback_count": len(self._pending_callbacks),
                "reserved_callback_count": len(self._callback_reservations),
                "delivery_reconciliation_count": len(
                    self._delivered_callbacks_before_receipt
                ),
                "finalized_journey_count": len(self._finalized_journeys),
                "open_journey_count": sum(
                    journey.exit_captured_at is None
                    for journey in self._finalized_journeys.values()
                ),
                "protected_journey_count": self._protected_journey_count_locked(),
                "journey_capacity_load": self._journey_capacity_load_locked(),
                "pending_exit_count": len(self._pending_exits),
                "provisional_crossing_count": len(self._provisional_crossings),
                "analysis_inflight_count": len(self._analysis_inflight),
                "late_ocr_conflict_count": self._late_ocr_conflict_count,
                "ambiguous_exit_count": self._ambiguous_exit_count,
                "permanent_callback_failure_count": (
                    self._permanent_callback_failure_count
                ),
                "groups": {
                    group_id: {
                        "attempt_ids": sorted(group.attempts),
                        "status": group.status.value,
                        "identity_key": group.identity_key,
                        "correction_candidate_of": group.correction_candidate_of,
                        "witnesses": sorted(w.value for w in group.witnesses),
                        "plate_sources": {
                            source.value: reading.text
                            for source, reading in (
                                self._engine.available_plate_sources(group).items()
                            )
                        },
                        "hypotheses": {
                            hypothesis.display: hypothesis.status.value
                            for hypothesis in group.hypotheses.values()
                        },
                    }
                    for group_id, group in self._groups.items()
                },
            }

    def record_exit(self, plate: str, captured_at: datetime) -> int:
        """Apply or retain one authoritative, source-ordered exit boundary."""
        normalized_plate = plate_key(plate)
        if not normalized_plate:
            raise EntryInvalid("exit_plate_is_required")
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise EntryInvalid("exit_captured_at_timezone_required")

        with self._lock:
            exit_key = (normalized_plate, captured_at)
            if exit_key in self._applied_exits or any(
                journey.canonical_plate_key == normalized_plate
                and journey.exit_captured_at == captured_at
                for journey in self._finalized_journeys.values()
            ):
                self._remember_applied_exit_locked(exit_key)
                return 0
            if exit_key in self._pending_exits:
                self._pending_exits.move_to_end(exit_key)
            closed_decision_ids = self._apply_exit_boundary_locked(exit_key)
            if not closed_decision_ids:
                self._remember_pending_exit_locked(*exit_key)
            decisions = (
                [] if self._analysis_inflight else self._evaluate_pending_locked()
            )

        self._dispatch(decisions)
        return len(closed_decision_ids)

    def _apply_exit_boundary_locked(
        self,
        exit_key: Tuple[str, datetime],
    ) -> set[str]:
        """Close the unique latest journey once no eligible callback is active."""
        normalized_plate, captured_at = exit_key
        if self._has_active_eligible_decision_locked(normalized_plate, captured_at):
            return set()
        eligible = self._eligible_open_journeys_locked(exit_key)
        if not eligible:
            return set()
        selected = self._select_exit_journey_locked(exit_key, eligible)
        if selected is None:
            return set()
        return self._close_journey_for_exit_locked(exit_key, selected)

    def _eligible_open_journeys_locked(
        self,
        exit_key: Tuple[str, datetime],
    ) -> list[_FinalizedJourney]:
        normalized_plate, captured_at = exit_key
        eligible = [
            journey
            for journey in self._finalized_journeys.values()
            if journey.canonical_plate_key == normalized_plate
            and journey.exit_captured_at is None
            and self._timestamp_strictly_after(
                journey.entry_captured_at,
                captured_at,
            )
        ]
        return sorted(
            eligible,
            key=lambda journey: (
                journey.entry_captured_at,
                journey.decision_id,
            ),
        )

    def _select_exit_journey_locked(
        self,
        exit_key: Tuple[str, datetime],
        eligible: Sequence[_FinalizedJourney],
    ) -> Optional[_FinalizedJourney]:
        if len(eligible) > 1:
            self._mark_ambiguous_exit_locked(exit_key)
        latest_entry_at = eligible[-1].entry_captured_at
        latest = [
            journey
            for journey in eligible
            if journey.entry_captured_at == latest_entry_at
        ]
        if len(latest) > 1:
            self._log_equal_latest_exit(exit_key, latest)
            return None
        selected = latest[0]
        if len(eligible) > 1:
            self._log_selected_exit(exit_key, eligible, selected)
        return selected

    @staticmethod
    def _log_equal_latest_exit(
        exit_key: Tuple[str, datetime],
        latest: Sequence[_FinalizedJourney],
    ) -> None:
        logger.warning(
            "[EntryV2] exit is ambiguous between equally-timed open journeys; "
            "retaining boundary plate=%s exit=%s eligible=%s",
            exit_key[0],
            exit_key[1].isoformat(),
            ",".join(item.decision_id for item in latest),
        )

    @staticmethod
    def _log_selected_exit(
        exit_key: Tuple[str, datetime],
        eligible: Sequence[_FinalizedJourney],
        selected: _FinalizedJourney,
    ) -> None:
        logger.warning(
            "[EntryV2] multiple open journeys eligible for one exit; closing "
            "latest source-ordered journey plate=%s exit=%s eligible=%s selected=%s",
            exit_key[0],
            exit_key[1].isoformat(),
            ",".join(item.decision_id for item in eligible),
            selected.decision_id,
        )

    def _close_journey_for_exit_locked(
        self,
        exit_key: Tuple[str, datetime],
        selected: _FinalizedJourney,
    ) -> set[str]:
        self._finalized_journeys[selected.decision_id] = replace(
            selected,
            exit_captured_at=exit_key[1],
        )
        self._pending_exits.pop(exit_key, None)
        self._remember_applied_exit_locked(exit_key)
        closed_decision_ids = {selected.decision_id}
        self._release_crossings_after_closed_journeys_locked(
            closed_decision_ids,
        )
        self._trim_finalized_journeys_locked()
        return closed_decision_ids

    def _has_active_eligible_decision_locked(
        self,
        normalized_plate: str,
        captured_at: datetime,
    ) -> bool:
        """Protect source ordering while a same-plate confirmation can finalize."""
        active = list(self._callback_reservations.values()) + [
            decision for decision, _ in self._pending_callbacks.values()
        ]
        return any(
            decision.finalizes_group
            and plate_key(decision.canonical_plate or "") == normalized_plate
            and self._timestamp_strictly_after(
                decision.entry_captured_at,
                captured_at,
            )
            for decision in active
        )

    def _reconcile_pending_exits_locked(
        self,
        normalized_plate: str,
    ) -> set[str]:
        """Apply retained boundaries after callbacks for one plate settle."""
        closed_decision_ids: set[str] = set()
        matching_keys = sorted(
            (key for key in self._pending_exits if key[0] == normalized_plate),
            key=lambda key: key[1],
        )
        for exit_key in matching_keys:
            closed_decision_ids.update(self._apply_exit_boundary_locked(exit_key))
        return closed_decision_ids

    def _mark_ambiguous_exit_locked(
        self,
        exit_key: Tuple[str, datetime],
    ) -> None:
        if exit_key in self._ambiguous_exit_keys:
            self._ambiguous_exit_keys.move_to_end(exit_key)
            return
        self._ambiguous_exit_keys[exit_key] = None
        self._ambiguous_exit_count += 1
        while len(self._ambiguous_exit_keys) > self.settings.journey_capacity:
            self._ambiguous_exit_keys.popitem(last=False)

    def _remember_pending_exit_locked(
        self,
        normalized_plate: str,
        captured_at: datetime,
    ) -> None:
        """Retain an exit that raced ahead of journey finalization."""
        key = (normalized_plate, captured_at)
        if key in self._pending_exits:
            self._pending_exits.move_to_end(key)
            return
        if len(self._pending_exits) >= self.settings.journey_capacity:
            raise EntryCapacityExceeded("pending_exit_capacity_exceeded")
        self._pending_exits[key] = None
        logger.info(
            "[EntryV2] retained unmatched exit plate=%s captured_at=%s",
            normalized_plate,
            captured_at.isoformat(),
        )

    def _remember_applied_exit_locked(
        self,
        key: Tuple[str, datetime],
    ) -> None:
        self._applied_exits[key] = None
        self._applied_exits.move_to_end(key)
        while len(self._applied_exits) > self.settings.journey_capacity:
            self._applied_exits.popitem(last=False)

    def _protected_journey_count_locked(self) -> int:
        referenced = {
            journey.decision_id for _, journey in self._provisional_crossings.values()
        }
        return sum(
            journey.exit_captured_at is None or journey.decision_id in referenced
            for journey in self._finalized_journeys.values()
        )

    def _journey_capacity_load_locked(self) -> int:
        inflight_attempts = self._unmaterialized_inflight_count_locked("attempt")
        return (
            self._protected_journey_count_locked()
            + len(self._groups)
            + inflight_attempts
        )

    def _unmaterialized_inflight_count_locked(self, kind: str) -> int:
        """Count reservations not already represented by retained state."""
        count = 0
        for key_kind, resource_id in self._inflight:
            if key_kind != kind:
                continue
            key = (key_kind, resource_id)
            if key in self._decisions_before_receipt:
                continue
            if kind == "attempt" and resource_id in self._attempts:
                continue
            if kind == "crossing" and (
                resource_id in self._crossings
                or resource_id in self._provisional_crossings
            ):
                continue
            count += 1
        return count

    def _trim_finalized_journeys_locked(self) -> None:
        """Evict only closed tombstones with no retained evidence reference."""
        while len(self._finalized_journeys) > self.settings.journey_capacity:
            referenced = {
                journey.decision_id
                for _, journey in self._provisional_crossings.values()
            }
            evictable = next(
                (
                    decision_id
                    for decision_id, journey in self._finalized_journeys.items()
                    if journey.exit_captured_at is not None
                    and decision_id not in referenced
                ),
                None,
            )
            if evictable is None:
                logger.error(
                    "[EntryV2] protected finalized journeys exceed capacity "
                    "count=%s capacity=%s",
                    len(self._finalized_journeys),
                    self.settings.journey_capacity,
                )
                return
            self._finalized_journeys.pop(evictable, None)

    def cancel_pending(self, cancellation: CancellationInput) -> Dict[str, int]:
        """Explicitly remove false/retreated RAM state; never time-expire it."""
        attempt_id = cancellation.attempt_id
        group_id = cancellation.group_id
        crossing_id = cancellation.crossing_id
        identifiers = {value for value in (attempt_id, group_id, crossing_id) if value}
        exit_key = self._validated_cancellation_exit_key(cancellation)
        if not identifiers and exit_key is None:
            raise EntryInvalid("cancellation_identifier_required")
        with self._lock:
            if (attempt_id and ("attempt", attempt_id) in self._inflight) or (
                crossing_id and ("crossing", crossing_id) in self._inflight
            ):
                raise EntryUnavailable("entry_evidence_is_currently_processing")

            target_groups = {group_id} if group_id else set()
            attempt = self._attempts.get(attempt_id) if attempt_id else None
            if attempt is not None:
                target_groups.add(attempt.group_id)
            crossing = self._crossings.get(crossing_id) if crossing_id else None
            if crossing is None and crossing_id:
                provisional = self._provisional_crossings.get(crossing_id)
                crossing = provisional[0] if provisional is not None else None
            if crossing is not None and crossing.matched_group_id:
                target_groups.add(crossing.matched_group_id)

            active_decisions = list(self._callback_reservations.values()) + [
                decision for decision, _ in self._pending_callbacks.values()
            ]
            if any(
                decision.group_id in target_groups
                or crossing_id
                and decision.crossing_id == crossing_id
                for decision in active_decisions
            ):
                raise EntryConflict("entry_decision_delivery_in_progress")

            removed_attempts = 0
            removed_groups = 0
            removed_crossings = 0
            removed_pending_exits = 0
            for target_group_id in target_groups:
                group = self._groups.get(target_group_id)
                if group is None:
                    continue
                if group.status != RecordStatus.PENDING:
                    raise EntryConflict("resolved_entry_group_cannot_be_cancelled")
                self._groups.pop(target_group_id, None)
                removed_groups += 1
                for member_id in group.attempts:
                    if self._attempts.pop(member_id, None) is not None:
                        removed_attempts += 1
                for member_crossing_id, member_crossing in list(
                    self._crossings.items()
                ):
                    if member_crossing.matched_group_id == target_group_id:
                        self._crossings.pop(member_crossing_id, None)
                        removed_crossings += 1

            if crossing_id and self._crossings.pop(crossing_id, None) is not None:
                removed_crossings += 1
            if (
                crossing_id
                and self._provisional_crossings.pop(crossing_id, None) is not None
            ):
                removed_crossings += 1
            if exit_key is not None:
                removed_pending_exits = self._cancel_pending_exit_locked(
                    exit_key,
                    cancellation.reason,
                )
            self._trim_receipts_locked()
            removed = {
                "attempts": removed_attempts,
                "groups": removed_groups,
                "crossings": removed_crossings,
                "pending_exits": removed_pending_exits,
            }
            if any(removed.values()) and not removed_pending_exits:
                logger.info(
                    "[EntryV2] operator cancelled pending evidence "
                    "attempts=%s groups=%s crossings=%s reason=%s",
                    removed_attempts,
                    removed_groups,
                    removed_crossings,
                    cancellation.reason,
                )
            return removed

    @staticmethod
    def _validated_cancellation_exit_key(
        cancellation: CancellationInput,
    ) -> Optional[Tuple[str, datetime]]:
        has_plate = bool(cancellation.exit_plate)
        has_timestamp = cancellation.exit_captured_at is not None
        if has_plate != has_timestamp:
            raise EntryInvalid("exit_plate_and_captured_at_are_required_together")
        if not has_plate:
            return None
        normalized_plate = plate_key(cancellation.exit_plate)
        if not normalized_plate:
            raise EntryInvalid("exit_plate_is_required")
        captured_at = cancellation.exit_captured_at
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise EntryInvalid("exit_captured_at_timezone_required")
        return normalized_plate, captured_at

    def _cancel_pending_exit_locked(
        self,
        exit_key: Tuple[str, datetime],
        reason: str,
    ) -> int:
        if exit_key not in self._pending_exits:
            return 0
        self._pending_exits.pop(exit_key)
        logger.info(
            "[EntryV2] operator cancelled unmatched exit plate=%s "
            "captured_at=%s reason=%s",
            exit_key[0],
            exit_key[1].isoformat(),
            reason,
        )
        return 1

    def _ensure_available(self) -> None:
        if not self.available:
            raise EntryUnavailable(self.unavailable_reason or "entry_v2_unavailable")

    def _validate_attempt(self, request: AttemptInput, images: Sequence[bytes]) -> None:
        if not request.attempt_id or not request.source_event_id:
            raise EntryInvalid("attempt_id_and_source_event_id_are_required")
        if not request.camera_id:
            raise EntryInvalid("camera_id_is_required")
        if not plate_key(request.reported_plate):
            raise EntryInvalid("reported_plate_is_required")
        if request.reported_confidence is not None and not (
            0.0 <= request.reported_confidence <= 1.0
        ):
            raise EntryInvalid("reported_confidence_out_of_range")
        self._validate_images_and_metadata(images, request.metadata)

    def _validate_crossing(
        self, request: CrossingInput, images: Sequence[bytes]
    ) -> None:
        if not request.crossing_id or not request.source_event_id:
            raise EntryInvalid("crossing_id_and_source_event_id_are_required")
        cameras, lines, directions = self.settings.crossing_policy(request.role)
        if norm_camera_id(request.camera_id) not in cameras:
            raise EntryInvalid("crossing_camera_not_configured_for_role")
        if request.line_id.strip().upper() not in lines:
            raise EntryInvalid("crossing_line_not_configured_for_role")
        if request.direction.strip().lower() not in directions:
            raise EntryInvalid("crossing_direction_not_configured_for_role")
        self._validate_images_and_metadata(images, request.metadata)

    def _validate_images_and_metadata(
        self, images: Sequence[bytes], metadata: Mapping[str, Any]
    ) -> None:
        if not images:
            raise EvidenceUnavailable("at_least_one_image_is_required")
        if len(images) > self.settings.max_images_per_event:
            raise EntryInvalid("too_many_images")
        if any(not image for image in images):
            raise EvidenceUnavailable("empty_image")
        if any(len(image) > self.settings.max_image_bytes for image in images):
            raise EntryInvalid("image_too_large")
        try:
            encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise EntryInvalid("metadata_must_be_json") from exc
        if len(encoded) > self.settings.max_metadata_bytes:
            raise EntryInvalid("metadata_too_large")
        if self._contains_binary(metadata):
            raise EntryInvalid("metadata_must_not_contain_binary_data")

    @classmethod
    def _contains_binary(cls, value: Any) -> bool:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return True
        if isinstance(value, Mapping):
            return any(cls._contains_binary(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._contains_binary(v) for v in value)
        return False

    def _require_single_vehicle_event(self, evidence: Iterable[FrameEvidence]) -> None:
        """Fail closed when one multipart event contains more than one car.

        OCR and ReID are paired per frame by the analyzer, but correlation uses
        all frames in an event. Requiring every frame embedding to describe the
        same vehicle prevents ReID from selecting car B while OCR selects car A
        in a tailgate/mixed Hikvision burst.
        """
        frames = tuple(evidence)
        if not frames or any(not frame.embedding for frame in frames):
            raise EvidenceUnavailable("reid_embedding_unavailable")

        normalized = []
        for frame in frames:
            if not all(math.isfinite(value) for value in frame.embedding):
                raise EvidenceUnavailable("reid_embedding_unavailable")
            norm = math.sqrt(math.fsum(value * value for value in frame.embedding))
            if not math.isfinite(norm) or norm <= 0.0:
                raise EvidenceUnavailable("reid_embedding_unavailable")
            normalized.append(tuple(value / norm for value in frame.embedding))

        for left_index, left in enumerate(normalized):
            for right in normalized[left_index + 1 :]:
                if len(left) != len(right):
                    raise EvidenceUnavailable("mixed_vehicle_evidence")
                similarity = math.fsum(
                    left_value * right_value
                    for left_value, right_value in zip(left, right)
                )
                if similarity < self.settings.event_consistency_min_score:
                    raise EvidenceUnavailable("mixed_vehicle_evidence")

    def _preflight(
        self,
        key: Tuple[str, str],
        fingerprint: str,
        *,
        maximum: int,
    ) -> Optional[IngestResult]:
        retry_item = None
        with self._lock:
            receipt = self._receipts.get(key)
            if receipt is not None:
                prior_fingerprint, result = receipt
                if prior_fingerprint != fingerprint:
                    raise EntryConflict("id_reused_with_different_payload")
                if (
                    self.settings.mode == EntryMode.AUTHORITATIVE
                    and result.decision_status == DecisionStatus.CONFIRMED.value
                    and result.callback_delivered is False
                    and result.decision_id
                ):
                    if result.decision_id in self._callback_retries_inflight:
                        raise EntryUnavailable("entry_confirmation_retry_in_progress")
                    retry_item = self._pending_callbacks.get(result.decision_id)
                    if retry_item is None:
                        raise EntryUnavailable("entry_confirmation_delivery_pending")
                    self._callback_retries_inflight.add(result.decision_id)
                else:
                    self._receipts.move_to_end(key)
                    return result
            if retry_item is not None:
                # Network I/O must happen after releasing the state lock.
                pass
            elif key in self._inflight:
                raise EntryUnavailable("same_id_is_currently_processing")
            elif (
                key[0] == "attempt"
                and self._journey_capacity_load_locked()
                >= self.settings.journey_capacity
            ):
                raise EntryCapacityExceeded("journey_capacity_exceeded")
            elif (
                len(self._attempts)
                if key[0] == "attempt"
                else len(self._crossings) + len(self._provisional_crossings)
            ) + self._unmaterialized_inflight_count_locked(key[0]) >= maximum:
                raise EntryCapacityExceeded(f"{key[0]}_capacity_exceeded")
            elif (
                len(self._pending_callbacks) + len(self._callback_reservations)
                >= self.settings.max_pending_callbacks
            ):
                raise EntryCapacityExceeded("callback_capacity_exceeded")
            else:
                self._inflight.add(key)
                self._analysis_inflight.add(key)
                return None

        assert retry_item is not None
        decision, payload = retry_item
        delivery = self._deliver_decision(decision, payload)
        with self._lock:
            self._callback_retries_inflight.discard(decision.decision_id)
            if not delivery.delivered:
                if not delivery.retryable:
                    self._pending_callbacks.pop(decision.decision_id, None)
                    self._record_permanent_delivery_failure_locked(delivery)
                    self._reconcile_pending_exits_locked(
                        plate_key(decision.canonical_plate or "")
                    )
                    raise EntryUnavailable("entry_confirmation_permanent_failure")
                raise EntryUnavailable("entry_confirmation_delivery_failed")
            self._pending_callbacks.pop(decision.decision_id, None)
            self._compact_delivered_locked(
                decision,
                remember_journey=self._should_remember_journey_locked(
                    decision,
                    delivery,
                ),
                superseding_exit_at=delivery.superseding_exit_at,
            )
            self._reconcile_pending_exits_locked(
                plate_key(decision.canonical_plate or "")
            )
            self._mark_decision_delivered_locked(decision.decision_id)
            updated_receipt = self._receipts.get(key)
            if updated_receipt is None:
                raise EntryUnavailable("entry_confirmation_receipt_unavailable")
            self._receipts.move_to_end(key)
            updated_result = updated_receipt[1]
            decisions = (
                [] if self._analysis_inflight else self._evaluate_pending_locked()
            )
        self._dispatch(decisions)
        return updated_result

    def _add_attempt_locked(
        self, request: AttemptInput, evidence: Sequence[FrameEvidence]
    ) -> AttemptGroup:
        now = self._clock()
        embeddings = [item.embedding for item in evidence if item.embedding]
        identity_key = plate_key(request.reported_plate)

        # PLATE-KEYED find-or-create. The plate is the identity, so a second
        # ANPR read of the same plate enriches the live candidate instead of
        # creating a rival for it.
        group = self._identity_for_plate_locked(identity_key, embeddings)
        same_key_split = bool(
            group is None
            and identity_key
            and any(
                other.status == RecordStatus.PENDING
                and other.identity_key == identity_key
                for other in self._groups.values()
            )
        )
        correction_of = ""
        if group is None:
            if identity_key:
                # A plated attempt that matches no live identity by key. Before
                # creating one, ask Re-ID whether a live identity is the same
                # CAR under a different plate string — an ANPR misread. We do
                # NOT merge on that answer: letting appearance override the
                # plate key would put Re-ID back in charge of who a car is, and
                # the plate consensus is what resolves a misread. Record the
                # suspicion and let the shadow log say how often it fires.
                correction_of = (
                    self._engine.find_merge_group(embeddings, self._groups.values())
                    or ""
                )
            else:
                # PLATELESS. There is no key to group by, so appearance is all
                # there is — this is the dropped-ANPR recovery path, and the
                # Re-ID merge gate is retained for exactly this case.
                merged = self._engine.find_merge_group(
                    embeddings,
                    [g for g in self._groups.values() if not g.identity_key],
                )
                if merged:
                    group = self._groups.get(merged)

        if group is None:
            group_id = self._group_id(request.attempt_id)
            group = AttemptGroup(
                group_id=group_id,
                identity_key=identity_key,
                created_at=now,
                last_activity_at=now,
                correction_candidate_of=correction_of,
            )
            self._groups[group_id] = group
            created = True
        else:
            group_id = group.group_id
            # "Creates OR activates": enrichment restarts the candidate's TTL.
            group.last_activity_at = now
            created = False

        # The image is a WITNESS; the reported plate is a PLATE SOURCE. Two
        # different axes for one event — neither implies the other.
        #
        # WHICH witness and WHICH source depends on where the bytes came from.
        # A HikCentral-sourced attempt exists because WE went and asked for it
        # (HikCentral is a pull source and never pushes); its plate is
        # HikCentral's own reading, not the gate ANPR system's, and conflating
        # the two would let one platform's answer be counted as two sources.
        hik_sourced = _is_hik_sourced_attempt(request)
        if embeddings:
            group.witnesses.setdefault(
                WitnessSource.HIK if hik_sourced else WitnessSource.ANPR,
                request.attempt_id,
            )
        if hik_sourced and identity_key:
            # HikCentral's own plate reading is STORED, unlike ANPR's and our
            # OCR's, which the engine derives from the attempts. It has to be:
            # it did not come from a camera event, so there is no attempt whose
            # capture time places it before or after the crossing. Stage 6 must
            # give it a causality check of its own if it ever attaches records
            # that could postdate the observation they are used to explain.
            group.plate_sources.setdefault(
                PlateSourceKind.HIK_TEXT,
                PlateReading(
                    source=PlateSourceKind.HIK_TEXT,
                    text=request.reported_plate,
                    confidence=float(request.reported_confidence or 0.0),
                    origin=str(
                        (request.metadata or {}).get("hik_guid", request.attempt_id)
                    ),
                ),
            )
            guid = str((request.metadata or {}).get("hik_guid", ""))
            if guid:
                group.hik_guids_consumed.add(guid)
        # The gate's reported plate is not stored: the engine derives it from
        # the attempts, so the causal projection cannot be handed a plate read
        # from after the car went past.

        self._emit_decision_record(
            stage="anpr_identity",
            result=decision_record.RESULT_ABSTAINED,
            reason="identity_created" if created else "identity_enriched",
            identity={
                "group_id": group_id,
                "identity_key": identity_key,
                "attempt_id": request.attempt_id,
                "created": created,
                "correction_candidate_of": group.correction_candidate_of,
                # The plate key already existed but appearance disagreed, so a
                # second identity now shares it. Rare and worth counting: a
                # rising rate means ANPR is misreading, or merge_min_score is
                # too tight for this facility's gate imagery.
                "same_key_split": same_key_split,
                "hik_sourced": hik_sourced,
                "attempt_count": len(group.attempts) + 1,
            },
            witnesses=[w.value for w in group.witnesses],
        )
        record = AttemptRecord(
            request=request,
            evidence=tuple(evidence),
            group_id=group_id,
        )
        group.attempts[request.attempt_id] = record
        self._attempts[request.attempt_id] = record
        self._add_hypothesis(
            group,
            request.reported_plate,
            reported_by=request.attempt_id,
        )
        for frame in evidence:
            if frame.plate.key:
                self._add_hypothesis(
                    group,
                    frame.plate.text,
                    ocr_evidence_id=frame.plate.evidence_id,
                )
        self._retire_provisional_crossings_before_attempt_locked(record)
        self._release_provisional_crossings_locked()
        return group

    def _retire_provisional_crossings_before_attempt_locked(
        self,
        new_attempt: AttemptRecord,
    ) -> None:
        """Retire old-journey evidence that cannot follow the new attempt."""
        retired = []
        for crossing_id, (crossing, finalized) in list(
            self._provisional_crossings.items()
        ):
            try:
                new_attempt_is_after_crossing = (
                    new_attempt.request.captured_at > crossing.request.captured_at
                )
            except TypeError:
                continue
            if not new_attempt_is_after_crossing:
                continue
            if finalized.exit_captured_at is None:
                continue
            duplicate_score_floor = max(
                self.settings.merge_min_score,
                self.settings.producer_pair_min_reid_score,
            )
            if (
                max_similarity(crossing.embeddings, new_attempt.embeddings)
                < duplicate_score_floor
            ):
                continue
            has_causal_pending_attempt = any(
                self._timestamp_between(
                    finalized.exit_captured_at,
                    attempt.request.captured_at,
                    crossing.request.captured_at,
                )
                for group in self._groups.values()
                if group.status == RecordStatus.PENDING
                for attempt in group.attempts.values()
            )
            if has_causal_pending_attempt:
                continue
            self._provisional_crossings.pop(crossing_id, None)
            retired.append(crossing_id)

        if retired:
            self._trim_receipts_locked()
            logger.info(
                "[EntryV2] retired causally obsolete provisional crossings=%s "
                "new_attempt=%s captured_at=%s",
                ",".join(sorted(retired)),
                new_attempt.request.attempt_id,
                new_attempt.request.captured_at.isoformat(),
            )

    def _release_crossings_after_closed_journeys_locked(
        self,
        closed_decision_ids: set[str],
    ) -> None:
        if not closed_decision_ids:
            return
        released = []
        finalized_duplicates = []
        for crossing_id, (crossing, journey) in tuple(
            self._provisional_crossings.items()
        ):
            if journey.decision_id not in closed_decision_ids:
                continue
            closed_journey = self._finalized_journeys.get(journey.decision_id)
            if closed_journey is None or closed_journey.exit_captured_at is None:
                continue
            try:
                follows_exit = (
                    crossing.request.captured_at > closed_journey.exit_captured_at
                )
            except TypeError:
                continue
            if not follows_exit:
                # The source timestamp proves this observation belongs to the
                # journey that just closed. Its duplicate receipt is already
                # durable, so discard the retained inference evidence now.
                self._provisional_crossings.pop(crossing_id, None)
                finalized_duplicates.append(crossing_id)
                continue
            self._provisional_crossings.pop(crossing_id, None)
            self._crossings[crossing_id] = crossing
            self._mark_crossing_receipt_released_locked(crossing_id)
            released.append(crossing_id)

        if released or finalized_duplicates:
            self._trim_receipts_locked()
        if finalized_duplicates:
            logger.info(
                "[EntryV2] finalized pre-exit duplicate crossings=%s",
                ",".join(sorted(finalized_duplicates)),
            )
        if released:
            logger.info(
                "[EntryV2] released post-exit crossings=%s",
                ",".join(sorted(released)),
            )

    def _mark_crossing_receipt_released_locked(
        self,
        crossing_id: str,
        *,
        group_id: Optional[str] = None,
    ) -> None:
        """Remove the old-journey decision from post-exit evidence metadata."""
        key = ("crossing", crossing_id)
        receipt = self._receipts.get(key)
        if receipt is None:
            if key in self._inflight:
                self._released_crossings_before_receipt[crossing_id] = group_id
            return
        fingerprint, result = receipt
        self._receipts[key] = (
            fingerprint,
            replace(
                result,
                accepted=True,
                duplicate=False,
                group_id=group_id,
                decision_id=None,
                decision_status=None,
                callback_delivered=None,
            ),
        )
        self._receipts.move_to_end(key)

    def _release_provisional_crossings_locked(self) -> None:
        """Release post-exit re-entry evidence through normal ReID gates."""
        for crossing_id, (crossing, finalized) in list(
            self._provisional_crossings.items()
        ):
            group_id = self._newer_unique_group_locked(
                crossing.request,
                crossing.evidence,
                finalized,
            )
            if group_id is None:
                continue
            self._provisional_crossings.pop(crossing_id, None)
            self._crossings[crossing_id] = crossing
            self._mark_crossing_receipt_released_locked(
                crossing_id,
                group_id=group_id,
            )
            logger.info(
                "[EntryV2] released provisional crossing=%s group=%s captured_at=%s",
                crossing_id,
                group_id,
                crossing.request.captured_at.isoformat(),
            )

    def _finish_analysis_locked(self, key: Tuple[str, str]) -> list[EntryDecision]:
        """Evaluate only after every concurrently reserved input is inserted."""
        self._analysis_inflight.discard(key)
        if self._analysis_inflight:
            return []
        return self._evaluate_pending_locked()

    @staticmethod
    def _add_hypothesis(
        group: AttemptGroup,
        value: str,
        *,
        reported_by: str = "",
        ocr_evidence_id: str = "",
    ) -> None:
        key = plate_key(value)
        if not key:
            return
        hypothesis = group.hypotheses.get(key)
        if hypothesis is None:
            hypothesis = PlateHypothesis(key=key, display=canonical_plate(value))
            group.hypotheses[key] = hypothesis
        if reported_by:
            hypothesis.reported_by.add(reported_by)
        if ocr_evidence_id:
            hypothesis.ocr_evidence_ids.add(ocr_evidence_id)

    def _causal_group_projection(
        self,
        group: AttemptGroup,
        crossing: CrossingRecord,
    ) -> AttemptGroup:
        """Project attempt-derived identity to source-causal evidence only."""
        projected = AttemptGroup(
            group_id=group.group_id,
            attempts={
                attempt.request.attempt_id: attempt
                for attempt in causally_eligible_attempts(group, crossing)
            },
            status=group.status,
        )
        for attempt in projected.attempts.values():
            self._add_hypothesis(
                projected,
                attempt.request.reported_plate,
                reported_by=attempt.request.attempt_id,
            )
            for frame in attempt.evidence:
                if frame.plate.key:
                    self._add_hypothesis(
                        projected,
                        frame.plate.text,
                        ocr_evidence_id=frame.plate.evidence_id,
                    )
        for key, hypothesis in projected.hypotheses.items():
            original = group.hypotheses.get(key)
            if original is not None:
                hypothesis.status = original.status
        return projected

    def _partition_group_for_confirmation_locked(
        self,
        group: AttemptGroup,
        confirmed: AttemptGroup,
    ) -> None:
        """Keep source-future attempts pending as a separate journey."""
        confirmed_ids = set(confirmed.attempts)
        future_attempts = [
            attempt
            for attempt_id, attempt in group.attempts.items()
            if attempt_id not in confirmed_ids
        ]
        group.attempts = dict(confirmed.attempts)
        group.hypotheses = confirmed.hypotheses
        group.canonical_plate = confirmed.canonical_plate
        if not future_attempts:
            return

        future_attempts.sort(key=lambda item: item.request.attempt_id)
        seed = future_attempts[0].request.attempt_id
        future_group_id = self._group_id(seed)
        if future_group_id == group.group_id or future_group_id in self._groups:
            future_group_id = self._group_id(
                "future:"
                + group.group_id
                + ":"
                + "|".join(attempt.request.attempt_id for attempt in future_attempts)
            )
        future_group = AttemptGroup(group_id=future_group_id)
        for attempt in future_attempts:
            attempt.group_id = future_group_id
            future_group.attempts[attempt.request.attempt_id] = attempt
            self._add_hypothesis(
                future_group,
                attempt.request.reported_plate,
                reported_by=attempt.request.attempt_id,
            )
            for frame in attempt.evidence:
                if frame.plate.key:
                    self._add_hypothesis(
                        future_group,
                        frame.plate.text,
                        ocr_evidence_id=frame.plate.evidence_id,
                    )
            receipt_key = ("attempt", attempt.request.attempt_id)
            receipt = self._receipts.get(receipt_key)
            if receipt is not None:
                fingerprint, result = receipt
                self._receipts[receipt_key] = (
                    fingerprint,
                    replace(result, group_id=future_group_id),
                )
        self._groups[future_group_id] = future_group
        self._release_provisional_crossings_locked()

    def _evaluate_pending_locked(self) -> list[EntryDecision]:
        self._expire_stale_locked()
        available_callbacks = max(
            0,
            self.settings.max_pending_callbacks
            - len(self._pending_callbacks)
            - len(self._callback_reservations),
        )
        if available_callbacks == 0:
            return []

        decisions: list[EntryDecision] = []
        while len(decisions) < available_callbacks:
            # Recompute after every new decision. Finalizing one group can
            # remove the runner-up that made another crossing ambiguous, so a
            # one-shot candidate snapshot can otherwise strand a valid entry
            # forever (there is intentionally no business TTL or timer).
            candidates = self._rank_pending_matches_locked()
            if not candidates:
                break

            produced_decision = False
            for _, _, crossing_id, match, resolution_crossing in candidates:
                crossing = self._crossings.get(crossing_id)
                group = self._groups.get(match.group_id)
                if (
                    crossing is None
                    or group is None
                    or crossing.status != RecordStatus.PENDING
                    or group.status != RecordStatus.PENDING
                ):
                    continue
                causal_group = self._causal_group_projection(
                    group,
                    resolution_crossing,
                )
                if not causal_group.attempts:
                    continue
                resolution = self._engine.resolve_plate(
                    causal_group,
                    resolution_crossing,
                    self._primary_crossings_correlated_to_group_locked(
                        group.group_id,
                        resolution_crossing,
                    ),
                )
                if resolution.outcome == "pending":
                    continue
                # This crossing cleared its gates on this identity, so it is a
                # STRONG witness for it.
                witness = crossing.witness
                if witness is not None:
                    group.witnesses.setdefault(
                        witness, crossing.request.crossing_id
                    )
                    group.weak_votes.pop(witness, None)
                if resolution.outcome == "confirmed":
                    shortfall = self._witness_shortfall_locked(group)
                    # S5, recorded in its own right: this is where "did two
                    # independent observations agree on one physical vehicle?"
                    # is answered, and the answer is the whole basis for
                    # opening a session.
                    self._emit_decision_record(
                        stage="physical_confirm",
                        result=(
                            decision_record.RESULT_ABSTAINED
                            if shortfall
                            else decision_record.RESULT_CONFIRMED
                        ),
                        reason=shortfall or "witnesses_agree",
                        crossing=crossing,
                        identity={
                            "group_id": group.group_id,
                            "identity_key": group.identity_key,
                            "strong": sorted(w.value for w in group.witnesses),
                            "weak": sorted(w.value for w in group.weak_votes),
                        },
                        witnesses=[
                            w.value for w in group.confirming_witnesses()
                        ],
                    )
                    if shortfall:
                        # Re-ID and the plate agree, but only ONE independent
                        # observation places this car at the entry. A single
                        # source must never open a session, so hold it: another
                        # witness may still arrive, and no TTL has run out yet.
                        resolution = replace(
                            resolution,
                            outcome="abstained",
                            reason=shortfall,
                        )
                # The plate decision, recorded in its own right. The review
                # gate asks "were there confirmations with no plate consensus?"
                # and "did any plate source come from a ramp camera?", and
                # neither question can be answered from a decision status.
                consensus = self._engine.plate_consensus(causal_group)
                _observed = _best_observed_plate_read(crossing)
                self._emit_decision_record(
                    stage="plate_consensus",
                    result=(
                        decision_record.RESULT_CONFIRMED
                        if resolution.outcome == "confirmed"
                        else decision_record.RESULT_UNREADABLE
                        if consensus.outcome != "consensus"
                        else decision_record.RESULT_ABSTAINED
                    ),
                    reason=resolution.reason,
                    crossing=crossing,
                    identity={
                        "group_id": group.group_id,
                        "identity_key": group.identity_key,
                    },
                    plate=consensus.as_record(),
                    witnesses=[w.value for w in group.confirming_witnesses()],
                    observed_plate_text=_observed[0],
                    observed_plate_confidence=_observed[1],
                )
                decision = self._build_decision(
                    causal_group,
                    crossing,
                    match,
                    resolution,
                )
                crossing.matched_group_id = group.group_id
                # Abstention means that the current evidence is insufficient.
                # Keep the strict crossing available for a later same-car ANPR
                # correction, while suppressing an identical callback until
                # the facts change.
                if decision.decision_id == crossing.last_decision_id:
                    continue
                crossing.last_decision_id = decision.decision_id
                if decision.finalizes_group:
                    self._partition_group_for_confirmation_locked(
                        group,
                        causal_group,
                    )
                    # Only the selected crossing and a strict, source-bounded
                    # producer counterpart are proven to be this journey.
                    # Threshold-similar rows remain tentative until compaction
                    # can quarantine them for a source-ordered later attempt.
                    for related in self._crossing_family_members_locked(crossing):
                        related.matched_group_id = group.group_id
                        related.status = RecordStatus.RESOLVED
                    group.status = RecordStatus.RESOLVED
                self._callback_reservations[decision.decision_id] = decision
                decisions.append(decision)
                produced_decision = True
                break

            if not produced_decision:
                break
        return decisions

    def _rank_pending_matches_locked(self):
        """Return the current strict assignments in semantic policy order."""
        candidates = []
        families = self._pending_crossing_families_locked()
        family_rows = []
        for members in families:
            earliest = min(
                members,
                key=lambda member: member.request.captured_at,
            )
            family_rows.append(
                CrossingRecord(
                    request=earliest.request,
                    evidence=tuple(
                        frame for member in members for frame in member.evidence
                    ),
                )
            )
        for members, family_row in zip(families, family_rows):
            if self._family_has_reliable_ocr_conflict(members):
                # Two producers reading DIFFERENT plates are evidence they are
                # not reporting the same physical crossing after all, and
                # pooling them would put two cars' embeddings in one row — the
                # poisoning that makes every later match ambiguous. Keep the
                # family pending for explicit correction/cancellation.
                #
                # This is the ONE use of a ramp camera's OCR that survives, and
                # it is not plate evidence: the question is "are these two
                # notifications the same event?", never "what is the plate?".
                # Subtractive, like the colour veto — it can withhold a row, it
                # can never name a car.
                #
                # Applies to BOTH roles now. It used to guard only CAM-03,
                # because a conflicting CAM-23 read was caught by the primary
                # OCR policy instead; that policy is gone, so the guard has to
                # cover what it used to.
                #
                # Withholding is silent to PMS — the rows simply stay pending
                # and are retried on the next event — so it MUST be recorded
                # here, or a fail-closed refusal would leave no trace anywhere.
                self._emit_decision_record(
                    stage="producer_family",
                    result=decision_record.RESULT_AMBIGUOUS,
                    reason="producer_family_ocr_conflict",
                    crossing=family_row,
                    extra={
                        "family": [
                            member.request.crossing_id for member in members
                        ],
                        "observed_plate_texts": sorted(
                            {
                                item.text
                                for member in members
                                for item in member.plate_evidence
                                if item.text
                            }
                        ),
                    },
                )
                continue
            match = self._find_unique_match_with_observability_locked(
                family_row,
                self._groups,
                family_rows,
            )
            if match is None:
                continue
            group = self._groups.get(match.group_id)
            if group is None:
                continue
            group_embeddings = causal_group_embeddings(group, family_row)
            crossing = max(
                members,
                key=lambda member: (
                    max_similarity(member.embeddings, group_embeddings),
                    member.request.crossing_id,
                ),
            )
            resolution_crossing = CrossingRecord(
                # The aggregate is causally eligible only when the attempt
                # precedes both producer notifications, so use the family's
                # earliest source timestamp rather than the chosen best view.
                request=family_row.request,
                evidence=family_row.evidence,
            )
            # Resolve source-causal physical stages in capture order, not HTTP
            # arrival order. A primary may govern a later fallback only when it
            # was actually observed first; a CAM23 event hours after CAM03 can
            # belong to a later journey and must not steal the earlier entry.
            # Role priority is only a deterministic tie-break at one instant.
            role_priority = 1 if crossing.request.role.value == "primary" else 0
            candidates.append(
                (
                    (
                        -family_row.request.captured_at.timestamp(),
                        role_priority,
                    ),
                    match.score,
                    crossing.request.crossing_id,
                    match,
                    resolution_crossing,
                )
            )
        candidates.sort(reverse=True)
        return candidates

    def _find_unique_match_with_observability_locked(
        self,
        crossing: CrossingRecord,
        groups: Mapping[str, AttemptGroup],
        crossings: Iterable[CrossingRecord],
    ) -> Optional[ReIDMatch]:
        """Evaluate and log the ReID uniqueness gate without log flooding."""
        evaluation = self._engine.evaluate_unique_match(
            crossing,
            groups,
            crossings,
        )
        if evaluation is None:
            return None
        self._log_reid_evaluation_locked(crossing, evaluation)
        return evaluation.match

    def _log_reid_evaluation_locked(
        self,
        crossing: CrossingRecord,
        evaluation: ReIDMatchEvaluation,
    ) -> None:
        crossing_id = crossing.request.crossing_id
        fingerprint = (
            evaluation.group_id,
            evaluation.score,
            evaluation.row_runner,
            evaluation.row_margin,
            evaluation.column_runner,
            evaluation.column_margin,
            evaluation.reason,
            evaluation.accepted,
        )
        if self._reid_evaluation_log_cache.get(crossing_id) == fingerprint:
            self._reid_evaluation_log_cache.move_to_end(crossing_id)
            return
        self._reid_evaluation_log_cache[crossing_id] = fingerprint
        self._reid_evaluation_log_cache.move_to_end(crossing_id)
        while (
            len(self._reid_evaluation_log_cache)
            > max(1, self.settings.max_pending_crossings)
        ):
            self._reid_evaluation_log_cache.popitem(last=False)

        logger.info(
            "[EntryV2][ReID] stage=uniqueness_only crossing=%s camera=%s "
            "role=%s best_group=%s score=%.4f min_score=%.4f "
            "row_runner=%.4f row_margin=%.4f row_min_margin=%.4f "
            "column_runner=%.4f column_margin=%.4f column_min_margin=%.4f "
            "result=%s reason=%s",
            crossing_id,
            crossing.request.camera_id,
            crossing.request.role.value,
            evaluation.group_id,
            evaluation.score,
            self.settings.reid_min_score,
            evaluation.row_runner,
            evaluation.row_margin,
            self.settings.reid_row_margin,
            evaluation.column_runner,
            evaluation.column_margin,
            self.settings.reid_column_margin,
            "accepted" if evaluation.accepted else "rejected",
            evaluation.reason,
        )
        self._record_weak_vote_locked(crossing, evaluation)

        group = self._groups.get(evaluation.group_id)
        if evaluation.accepted:
            result = decision_record.RESULT_CONFIRMED
        elif evaluation.cleared_absolute_score:
            # Cleared the absolute bar but not the margin: two plausible cars,
            # not one weak look. Recorded as AMBIGUOUS so the review can tell
            # "more evidence would help" from "more evidence will not".
            result = decision_record.RESULT_AMBIGUOUS
        else:
            result = decision_record.RESULT_ABSTAINED

        _observed = _best_observed_plate_read(crossing)
        self._emit_decision_record(
            stage="reid_evaluation",
            result=result,
            reason=evaluation.reason,
            crossing=crossing,
            evaluation=evaluation,
            ranked=evaluation.ranked,
            colour={
                "query_hsv": _round_hsv(crossing.colour_hsv),
                "vetoed": list(evaluation.vetoed),
                "enabled": self.settings.colour_veto_enabled,
            },
            fifo=self._fifo_block_locked(crossing, evaluation),
            identity=(
                {
                    "group_id": evaluation.group_id,
                    "identity_key": group.identity_key,
                }
                if group is not None
                else None
            ),
            witnesses=(
                [w.value for w in group.confirming_witnesses()]
                if group is not None
                else None
            ),
            observed_plate_text=_observed[0],
            observed_plate_confidence=_observed[1],
        )

    def _identity_for_plate_locked(self, identity_key: str, embeddings):
        """The live identity for this plate key that this attempt may enrich.

        Derived by scan rather than kept as an index. `_groups` is bounded by
        max_pending_attempts (256), so the scan is trivial, and an index would
        have to be kept in sync across three insertion sites and two removal
        sites — a stale entry there would bind an attempt to the WRONG
        identity, which is the one failure this whole design exists to prevent.
        Cheaper to be certain.

        THE APPEARANCE GUARD. The plate is the identity key, but a key match is
        not a licence to pool images. ANPR misreads car B's plate as car A's,
        and pooling B's images into identity A poisons it: every later crossing
        then matches A perfectly, no crossing can out-margin any other on the
        column gate, and NOTHING ever confirms. That is a deadlock, and it is
        the same gallery-poisoning failure this codebase has fought before.

        So an attempt enriches a same-key identity only when Re-ID agrees it is
        the same car, at `merge_min_score` — already documented as the
        "very-high-confidence same-car" bar and comfortably above the measured
        ~0.66 same-view similarity between DIFFERENT cars. When appearance
        disagrees, the attempt gets its own identity under the same key.

        This is not Re-ID deciding identity: the plate already decided that.
        Re-ID is vetoing the pooling of contradictory evidence, exactly as the
        colour check does. Two live identities may therefore share a plate key
        — which is also what lets a genuine exit-and-re-entry stay two visits.
        """
        if not identity_key:
            return None
        candidates = [
            group
            for group in self._groups.values()
            if group.status == RecordStatus.PENDING
            and group.identity_key == identity_key
        ]
        if not candidates:
            return None
        if not embeddings:
            return candidates[0]

        best, best_score = None, -1.0
        for group in candidates:
            score = max_similarity(embeddings, group.embeddings)
            if score > best_score:
                best, best_score = group, score
        # A candidate with no embeddings at all scores -1.0 and is still the
        # right thing to enrich — there is nothing to contradict.
        if best_score < 0.0:
            return best
        if best_score >= self.settings.merge_min_score:
            return best
        return None

    def _age_seconds(self, anchor: Optional[datetime], now: datetime):
        """Age in seconds, or None when it cannot be established.

        Mixed aware/naive timestamps fail CLOSED — returning None means "do not
        expire this". Never delete state you cannot reason about; the capacity
        bounds are still there as the backstop.
        """
        if anchor is None:
            return None
        try:
            return (now - anchor).total_seconds()
        except TypeError:
            return None

    def _expire_stale_locked(self) -> None:
        """Age out unconfirmed identities and unmatched observations.

        Runs on every evaluation pass rather than on a timer thread: the
        coordinator is already woken by every event, and a timer would need its
        own lock discipline for no benefit.

        This is a LIFETIME bound, not an identity rule. Expiry never decides
        that two observations belong to the same car; it only decides that a
        candidate has waited long enough to stop being one.
        """
        now = self._clock()
        identity_ttl = self.settings.identity_ttl_minutes * 60
        observation_ttl = self.settings.observation_ttl_minutes * 60

        for group_id, group in list(self._groups.items()):
            if group.status != RecordStatus.PENDING:
                continue
            # An identity with a decision in flight is not idle — expiring it
            # would strand the callback that is about to reference it.
            if self._group_has_callback_in_flight_locked(group_id):
                continue
            age = self._age_seconds(group.last_activity_at or group.created_at, now)
            if age is None or age < identity_ttl:
                continue
            self._groups.pop(group_id, None)
            for attempt_id in group.attempts:
                self._attempts.pop(attempt_id, None)
            self._emit_decision_record(
                stage="ttl_expiry",
                result=decision_record.RESULT_EXPIRED,
                reason="identity_ttl_expired",
                identity={
                    "group_id": group_id,
                    "identity_key": group.identity_key,
                    "age_seconds": round(age, 1),
                    "ttl_seconds": identity_ttl,
                    "attempt_count": len(group.attempts),
                },
                witnesses=[w.value for w in group.witnesses],
            )

        for crossing_id, crossing in list(self._crossings.items()):
            if crossing.status != RecordStatus.PENDING:
                continue
            age = self._age_seconds(crossing.created_at, now)
            if age is None or age < observation_ttl:
                continue
            self._crossings.pop(crossing_id, None)
            self._emit_decision_record(
                stage="ttl_expiry",
                result=decision_record.RESULT_EXPIRED,
                reason="observation_ttl_expired",
                crossing=crossing,
                extra={
                    "age_seconds": round(age, 1),
                    "ttl_seconds": observation_ttl,
                },
            )

    def _witness_shortfall_locked(self, group) -> str:
        """"" when the two-witness rule is satisfied, else the failing reason.

        An entry is confirmed only when at least TWO independent observations
        agree on the same physical vehicle. The accepted pairs are
        ANPR+CAM-23, ANPR+CAM-03, CAM-23+CAM-03, and — only when ANPR is
        missing — CAM-23+Hik or CAM-03+Hik.

        At least one of them must have cleared its gates on its own. Weak votes
        exist so an ambiguous CAM-23 can be resolved by a confident CAM-03,
        not so that two uncertain looks can add up to a certainty.
        """
        witnesses = group.confirming_witnesses()
        if len(witnesses) < 2:
            return "insufficient_witnesses"
        if not (set(group.witnesses) & witnesses):
            return "no_witness_cleared_gates"
        return ""

    def _record_weak_vote_locked(self, crossing, evaluation) -> None:
        """Record that a witness named this identity but could not prove it.

        This is the CAM-23-ambiguous / CAM-03-confident path: the uncertain
        camera still says which identity it thinks this is, and that agreement
        is what a later confident witness combines with. It can never confirm
        on its own — see _witness_shortfall_locked.
        """
        witness = crossing.witness
        if witness is None or evaluation is None or evaluation.accepted:
            return
        group = self._groups.get(evaluation.group_id)
        if group is None or group.status != RecordStatus.PENDING:
            return
        if witness in group.witnesses:
            return
        group.weak_votes[witness] = crossing.request.crossing_id

    def _fifo_block_locked(self, crossing, evaluation) -> Dict[str, Any]:
        """Compare the expected ramp ordering against what Re-ID actually did.

        The ramp is narrow, so cars are EXPECTED to arrive in order. That is a
        prior, not a matcher: a car can stop or wait, and forcing the expected
        order onto an out-of-order pair is exactly how the wrong plate gets
        bound. So this is measured and written to the log, and nothing reads it
        back. If the shadow data later shows FIFO and Re-ID agree almost
        always, that is an argument for using it as a tie-break — made from
        evidence rather than assumption.
        """
        pending = sorted(
            (
                other
                for other in self._crossings.values()
                if other.status == RecordStatus.PENDING
                and norm_camera_id(other.request.camera_id)
                == norm_camera_id(crossing.request.camera_id)
            ),
            key=lambda item: item.request.captured_at,
        )
        try:
            fifo_rank = [
                item.request.crossing_id for item in pending
            ].index(crossing.request.crossing_id)
        except ValueError:
            fifo_rank = -1

        groups_by_age = sorted(
            (
                group
                for group in self._groups.values()
                if group.status == RecordStatus.PENDING
            ),
            key=lambda item: min(
                (a.request.captured_at for a in item.attempts.values()),
                default=item.created_at,
            ),
        )
        expected = (
            groups_by_age[fifo_rank].group_id
            if 0 <= fifo_rank < len(groups_by_age)
            else ""
        )
        return {
            "expected_rank": fifo_rank,
            "expected_group": expected,
            "reid_group": evaluation.group_id,
            # None, not False, when FIFO had NO expectation to offer — the rank
            # fell outside the live group list, so there is nothing to agree or
            # disagree with. Writing False there conflates "FIFO was wrong" with
            # "FIFO was silent" and buries the measurement this field exists to
            # take: on 2026-09-01, 346 of 451 records had no expectation at all,
            # making agreement read as 13% when it was 58% (61/105) among the
            # records that actually had one.
            "agreed": (expected == evaluation.group_id) if expected else None,
        }

    def _group_has_callback_in_flight_locked(self, group_id: str) -> bool:
        for decision, _payload in self._pending_callbacks.values():
            if decision.group_id == group_id:
                return True
        for decision in self._callback_reservations.values():
            if decision.group_id == group_id:
                return True
        return False

    def _emit_decision_record(self, **kwargs: Any) -> None:
        """Queue one decision-log record. Never raises, never blocks.

        Called under the coordinator lock, so the write MUST stay on the
        DecisionLog queue — a synchronous disk write here would hold the lock
        across I/O and stall every camera behind it. DecisionLog.emit drops on a
        full queue rather than waiting, which is the behaviour this relies on.

        A failure to build or queue a record is swallowed: the log exists to
        observe the pipeline, and it must never be able to break it.
        """
        log = self._decision_log
        if log is None:
            return
        try:
            log.emit(
                decision_record.build_record(settings=self.settings, **kwargs)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[EntryV2] decision record dropped: %r", exc)

    def _pending_crossing_families_locked(
        self,
    ) -> list[list[CrossingRecord]]:
        """Collapse proven same-stage notifications for column assignment.

        Hikvision and the local VA zone can independently report the same
        physical CAM23 crossing. Treating both rows as column competitors makes
        their equal ReID scores fail the uniqueness margin and strands the car.
        Families use complete-linkage: every member must be a strict same-car
        match to every other member. This prevents a similarity chain from
        merging two genuinely different vehicles through an intermediate row.

        Evidence is aggregated into one assignment row after grouping. Plate
        resolution therefore sees every family member; reliable fallback OCR
        disagreement is held pending before resolution, while primary conflict
        follows the primary policy's durable abstention state.
        """
        families: list[list[CrossingRecord]] = []
        pending = sorted(
            (
                crossing
                for crossing in self._crossings.values()
                if crossing.status == RecordStatus.PENDING
            ),
            key=lambda crossing: crossing.request.crossing_id,
        )
        for crossing in pending:
            family = next(
                (
                    members
                    for members in families
                    if all(
                        self._crossings_share_family(crossing, member)
                        for member in members
                    )
                ),
                None,
            )
            if family is None:
                families.append([crossing])
            else:
                family.append(crossing)
        return families

    def _crossings_share_family(
        self,
        left: CrossingRecord,
        right: CrossingRecord,
    ) -> bool:
        if left.request.role != right.request.role:
            return False
        if norm_camera_id(left.request.camera_id) != norm_camera_id(
            right.request.camera_id
        ):
            return False
        if {
            self._crossing_source(left),
            self._crossing_source(right),
        } != {"hikvision", "va_local_zone"}:
            # Similarity alone cannot prove that two independent event IDs are
            # duplicate notifications; two real cars can be visually close.
            # Collapse only the explicitly independent Hikvision/local producer
            # pair. Retries within either producer remain ID-idempotent.
            return False
        try:
            producer_skew_seconds = abs(
                (left.request.captured_at - right.request.captured_at).total_seconds()
            )
        except TypeError:
            return False
        if producer_skew_seconds > self.settings.producer_pair_max_skew_seconds:
            return False
        if (
            max_similarity(left.embeddings, right.embeddings)
            < self.settings.producer_pair_min_reid_score
        ):
            return False
        return True

    def _family_has_reliable_ocr_conflict(
        self,
        members: Sequence[CrossingRecord],
    ) -> bool:
        """True when this producer family holds two reads of DIFFERENT cars.

        Was `len({keys}) > 1`, i.e. any two distinct plate strings. Exact keys
        treat a digits-first/letters-first swap and a hallucinated letter
        prefix as disagreements, so this gate refused one real crossing
        thirteen times on 2026-08-30 (`7383HAS` against `AATEIGH7383HAS`, both
        above the confidence gate, one car). `plates_contradict` compares digit
        runs instead, so only a genuine difference blocks the family.
        """
        reliable = [
            item.text
            for member in members
            for item in member.plate_evidence
            if item.key and item.confidence >= self.settings.ocr_min_confidence
        ]
        return any(
            plates_contradict(left, right)
            for left, right in itertools.combinations(reliable, 2)
        )

    @staticmethod
    def _crossing_source(crossing: CrossingRecord) -> str:
        return EntryCoordinator._request_crossing_source(crossing.request)

    @staticmethod
    def _request_crossing_source(request: CrossingInput) -> str:
        raw = str(request.metadata.get("crossing_source") or "")
        normalized = raw.strip().lower().replace("-", "_")
        if normalized in {"hikvision", "va_local_zone"}:
            return normalized
        event_type = str(request.metadata.get("event_type") or "").strip().lower()
        if event_type == "linedetection":
            return "hikvision"
        return ""

    def _crossing_family_members_locked(
        self,
        selected: CrossingRecord,
    ) -> list[CrossingRecord]:
        """Return only the selected crossing's proven producer family."""
        selected_id = selected.request.crossing_id
        for members in self._pending_crossing_families_locked():
            if any(member.request.crossing_id == selected_id for member in members):
                return members
        return [selected]

    def _primary_crossings_correlated_to_group_locked(
        self,
        group_id: str,
        resolution_crossing: CrossingRecord,
    ) -> list[CrossingRecord]:
        """Return primary evidence with an absolute and row-unique ReID link.

        Column uniqueness is intentionally omitted here: multiple CAM23 events
        for one physical car compete in that column, but all of their reliable
        OCR must be considered before any one event can confirm the plate.
        """
        pending_groups = [
            group
            for group in self._groups.values()
            if group.status == RecordStatus.PENDING and group.embeddings
        ]
        correlated = []
        for crossing in self._crossings.values():
            if (
                crossing.status != RecordStatus.PENDING
                or crossing.request.role != CrossingRole.PRIMARY
            ):
                continue
            try:
                if (
                    crossing.request.captured_at
                    > resolution_crossing.request.captured_at
                ):
                    continue
            except TypeError:
                continue
            ranked = []
            for group in pending_groups:
                group_embeddings = causal_group_embeddings(group, crossing)
                if not group_embeddings:
                    continue
                ranked.append(
                    (
                        max_similarity(crossing.embeddings, group_embeddings),
                        group.group_id,
                    )
                )
            ranked.sort(reverse=True)
            if not ranked:
                continue
            best_score, best_group_id = ranked[0]
            runner_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if (
                best_group_id == group_id
                and best_score >= self.settings.reid_min_score
                and best_score - runner_score >= self.settings.reid_row_margin
            ):
                correlated.append(crossing)
        return correlated

    def _build_decision(self, group, crossing, match, resolution) -> EntryDecision:
        canonical_key = plate_key(resolution.canonical_plate or "")
        eligible_attempts = sorted(
            causally_eligible_attempts(group, crossing),
            key=lambda item: item.request.attempt_id,
        )
        if not eligible_attempts:
            raise EntryUnavailable("entry_causal_attempt_unavailable")
        matching_attempts = sorted(
            (
                attempt
                for attempt in eligible_attempts
                if plate_key(attempt.request.reported_plate) == canonical_key
            ),
            key=lambda item: item.request.attempt_id,
        )
        selected_attempt = (matching_attempts or eligible_attempts)[0]

        superseded = []
        if resolution.outcome == "confirmed":
            for key, hypothesis in group.hypotheses.items():
                if key == canonical_key:
                    hypothesis.status = HypothesisStatus.CONFIRMED
                else:
                    hypothesis.status = HypothesisStatus.SUPERSEDED
                    superseded.append(hypothesis.display)
            group.canonical_plate = resolution.canonical_plate

        status = (
            DecisionStatus.CONFIRMED
            if resolution.outcome == "confirmed"
            else DecisionStatus.ABSTAINED
        )
        corrected = bool(
            canonical_key
            and any(
                plate_key(attempt.request.reported_plate) != canonical_key
                for attempt in eligible_attempts
            )
        )
        decision_seed = "|".join(
            [
                group.group_id,
                crossing.request.crossing_id,
                status.value,
                canonical_key,
                resolution.reason,
            ]
        )
        decision_id = (
            "dec-" + hashlib.sha256(decision_seed.encode("utf-8")).hexdigest()[:24]
        )
        return EntryDecision(
            decision_id=decision_id,
            status=status,
            reason=resolution.reason,
            group_id=group.group_id,
            attempt_id=selected_attempt.request.attempt_id,
            crossing_id=crossing.request.crossing_id,
            canonical_plate=resolution.canonical_plate,
            reported_plate=selected_attempt.request.reported_plate,
            reported_confidence=selected_attempt.request.reported_confidence,
            corrected=corrected,
            superseded_plates=tuple(sorted(set(superseded))),
            # Keep the trusted gate camera as source provenance, but start the
            # visit when the validated inward crossing physically occurs. A car
            # may wait at the barrier for minutes before being admitted.
            entry_camera_id=selected_attempt.request.camera_id,
            entry_captured_at=crossing.request.captured_at,
            reid_score=match.score,
            reid_row_margin=match.row_margin,
            reid_column_margin=match.column_margin,
            ocr_source=resolution.ocr_source,
            ocr_text=resolution.ocr_text,
            ocr_confidence=resolution.ocr_confidence,
            ocr_evidence_ids=resolution.ocr_evidence_ids,
            finalizes_group=status == DecisionStatus.CONFIRMED,
        )

    def _dispatch(self, decisions: Sequence[EntryDecision]) -> Dict[str, bool]:
        delivered: Dict[str, bool] = {}
        current = list(decisions)
        while current:
            capacity_freed = False
            for decision in current:
                payload = decision.callback_payload(self.settings.mode)
                logger.info(
                    "[EntryV2] decision=%s status=%s reason=%s attempt=%s "
                    "crossing=%s plate=%s reid=%.4f row_margin=%.4f "
                    "column_margin=%.4f ocr_source=%s ocr_confidence=%s",
                    decision.decision_id,
                    payload["status"],
                    payload["reason"],
                    decision.attempt_id,
                    decision.crossing_id,
                    decision.canonical_plate or "",
                    decision.reid_score,
                    decision.reid_row_margin,
                    decision.reid_column_margin,
                    decision.ocr_source or "",
                    decision.ocr_confidence,
                )
                result = self._deliver_decision(decision, payload)
                log = logger.info if result.delivered else logger.warning
                log(
                    "[EntryV2] callback decision=%s delivered=%s attempts=%s "
                    "publish_identity=%s retryable=%s error=%s",
                    decision.decision_id,
                    result.delivered,
                    result.attempts,
                    result.publish_identity,
                    result.retryable,
                    result.error,
                )
                delivered[decision.decision_id] = result.delivered
                with self._lock:
                    self._callback_reservations.pop(decision.decision_id, None)
                    if decision.finalizes_group and not result.delivered:
                        self._reconcile_decision_resources_locked(
                            decision,
                            callback_delivered=False,
                        )
                    if result.delivered:
                        self._compact_delivered_locked(
                            decision,
                            remember_journey=self._should_remember_journey_locked(
                                decision,
                                result,
                            ),
                            superseding_exit_at=result.superseding_exit_at,
                        )
                        capacity_freed = True
                    elif not result.retryable:
                        # Permanent callback/auth/contract failures cannot heal
                        # by replay. Stop admission and expose unhealthy health
                        # rather than silently exhausting the bounded queue.
                        self._record_permanent_delivery_failure_locked(result)
                    else:
                        self._pending_callbacks[decision.decision_id] = (
                            decision,
                            payload,
                        )
                    self._reconcile_pending_exits_locked(
                        plate_key(decision.canonical_plate or "")
                    )

            if not capacity_freed or self._fatal_delivery_error:
                break
            with self._lock:
                # Never decide while another event is still being analyzed;
                # that event's completion will perform the quiescent re-drive.
                current = (
                    [] if self._analysis_inflight else self._evaluate_pending_locked()
                )
        return delivered

    def _record_permanent_delivery_failure_locked(
        self,
        result: DeliveryResult,
    ) -> None:
        """Stop admission after a callback error that replay cannot repair."""
        self._fatal_delivery_error = result.error or "permanent_error"
        self._permanent_callback_failure_count += 1

    def _deliver(self, payload: Mapping[str, Any]) -> DeliveryResult:
        try:
            return self._sink.deliver(payload)
        except Exception as exc:
            return DeliveryResult(False, 1, str(exc))

    def _deliver_decision(
        self,
        decision: EntryDecision,
        payload: Mapping[str, Any],
    ) -> DeliveryResult:
        result = self._deliver(payload)
        if not result.delivered:
            return result
        if (
            self.settings.mode != EntryMode.AUTHORITATIVE
            or decision.status != DecisionStatus.CONFIRMED
            or not result.publish_identity
        ):
            return result
        try:
            identity = self._validated_identity(decision)
            self._identity_publisher.publish(identity)
            return result
        except IdentitySupersededByExit as exc:
            if not self._timestamp_at_or_after(
                decision.entry_captured_at,
                exc.latest_exit_at,
            ):
                logger.error(
                    "[EntryV2] registry returned an invalid superseding exit "
                    "decision=%s entry=%r exit=%r",
                    decision.decision_id,
                    decision.entry_captured_at,
                    exc.latest_exit_at,
                )
                return DeliveryResult(
                    False,
                    result.attempts,
                    "identity_superseded_exit_timestamp_invariant",
                    publish_identity=False,
                    retryable=False,
                    session_committed=False,
                    ack_result="identity_superseded_by_exit",
                )
            logger.info(
                "[EntryV2] identity publication superseded by exit decision=%s",
                decision.decision_id,
            )
            return DeliveryResult(
                True,
                result.attempts,
                str(exc),
                publish_identity=False,
                retryable=False,
                session_committed=False,
                ack_result="identity_superseded_by_exit",
                superseding_exit_at=exc.latest_exit_at,
            )
        except Exception as exc:
            # The PMS callback is idempotent, so retaining this delivery and
            # returning 503 lets the camera retry both steps safely. Never
            # compact the evidence while VA's live tracker is still plate-blind.
            return DeliveryResult(
                False,
                result.attempts,
                f"identity_publish:{exc}",
                publish_identity=True,
                session_committed=result.session_committed,
                ack_result=result.ack_result,
            )

    def _validated_identity(self, decision: EntryDecision) -> ValidatedIdentity:
        with self._lock:
            group = self._groups.get(decision.group_id)
            crossing = self._crossings.get(decision.crossing_id)
            if group is None or crossing is None:
                raise EntryUnavailable("validated_identity_evidence_unavailable")
            crossing_embeddings = tuple(crossing.embeddings)
            attempt_embeddings = tuple(
                (attempt.request.camera_id, embedding)
                for attempt in group.attempts.values()
                for embedding in attempt.embeddings
            )
        if not crossing_embeddings:
            raise EntryUnavailable("validated_identity_crossing_embedding_unavailable")
        return ValidatedIdentity(
            decision_id=decision.decision_id,
            canonical_plate=decision.canonical_plate or "",
            attempt_id=decision.attempt_id,
            crossing_id=decision.crossing_id,
            entered_at=decision.entry_captured_at,
            crossing_camera_id=crossing.request.camera_id,
            crossing_embeddings=crossing_embeddings,
            attempt_embeddings=attempt_embeddings,
            gallery_authorization=build_gallery_authorization_proof(
                decision,
                crossing_camera_id=crossing.request.camera_id,
                crossing_role=crossing.request.role,
                settings=self.settings,
            ),
        )

    def _should_remember_journey_locked(
        self,
        decision: EntryDecision,
        result: DeliveryResult,
    ) -> bool:
        """Create lifecycle state only for a real or source-closed session."""
        if self.settings.mode != EntryMode.AUTHORITATIVE:
            return True
        if result.session_committed is True:
            return True
        canonical_key = plate_key(decision.canonical_plate or "")
        if result.superseding_exit_at is not None:
            return (
                result.ack_result == "identity_superseded_by_exit"
                and self._timestamp_at_or_after(
                    decision.entry_captured_at,
                    result.superseding_exit_at,
                )
            )
        has_matching_exit = any(
            pending_plate == canonical_key
            and self._timestamp_strictly_after(
                decision.entry_captured_at,
                exit_captured_at,
            )
            for pending_plate, exit_captured_at in self._pending_exits
        )
        if result.session_committed is False:
            return has_matching_exit and result.ack_result in {
                "stale_after_exit",
                "identity_superseded_by_exit",
            }
        # Backward-compatible injected sinks must opt in to a committed session
        # before suppressing identity publication; otherwise a terminal ACK can
        # silently mint a fake open lifecycle.
        return result.publish_identity

    def _compact_delivered_locked(
        self,
        decision: EntryDecision,
        *,
        remember_journey: bool = True,
        superseding_exit_at: Optional[datetime] = None,
    ) -> None:
        if decision.finalizes_group:
            self._reconcile_decision_resources_locked(
                decision,
                callback_delivered=True,
            )
            finalized = (
                self._remember_finalized_journey_locked(
                    decision,
                    superseding_exit_at=superseding_exit_at,
                )
                if remember_journey
                else None
            )
            for crossing_id, crossing in list(self._crossings.items()):
                assigned_to_decision = crossing_id == decision.crossing_id or (
                    crossing.status == RecordStatus.RESOLVED
                    and crossing.matched_group_id == decision.group_id
                )
                if not assigned_to_decision and crossing.status != RecordStatus.PENDING:
                    # Another reserved confirmation owns this evidence. An
                    # older callback finishing first must never steal it.
                    continue
                finalized_match_kind = (
                    self._crossing_finalized_match_kind_locked(
                        crossing.request,
                        crossing.evidence,
                        finalized,
                    )
                    if not assigned_to_decision and finalized is not None
                    else None
                )
                if assigned_to_decision or finalized_match_kind is not None:
                    self._crossings.pop(crossing_id, None)
                    if finalized_match_kind == "provisional":
                        crossing.matched_group_id = None
                        self._provisional_crossings[crossing_id] = (
                            crossing,
                            finalized,
                        )
                        self._reconcile_quarantined_crossing_locked(
                            crossing_id,
                            finalized,
                        )
                    elif finalized_match_kind == "definitive":
                        self._reconcile_quarantined_crossing_locked(
                            crossing_id,
                            finalized,
                        )
            group = self._groups.pop(decision.group_id, None)
            if group is not None:
                for attempt_id in group.attempts:
                    self._attempts.pop(attempt_id, None)
            self._trim_receipts_locked()

    def _remember_finalized_journey_locked(
        self,
        decision: EntryDecision,
        *,
        superseding_exit_at: Optional[datetime] = None,
    ) -> Optional[_FinalizedJourney]:
        """Retain enough non-image evidence to consume delayed duplicates.

        This history is capacity-bounded and has no time-based expiry. Source
        ordering distinguishes an old notification from a later genuine
        journey by the same (or visually similar) vehicle.
        """
        group = self._groups.get(decision.group_id)
        crossing = self._crossings.get(decision.crossing_id)
        if group is None or crossing is None or not group.attempts:
            return None

        selected_attempt = group.attempts.get(decision.attempt_id)
        if selected_attempt is None:
            return None
        embeddings = tuple(selected_attempt.embeddings) + tuple(crossing.embeddings)
        if not embeddings:
            return None

        canonical_key = plate_key(decision.canonical_plate or "")
        if superseding_exit_at is not None:
            exit_captured_at = superseding_exit_at
            self._pending_exits.pop((canonical_key, exit_captured_at), None)
            self._remember_applied_exit_locked((canonical_key, exit_captured_at))
        else:
            # A retained boundary is assigned only after every already-reserved
            # same-plate callback settles. Otherwise sequential callback order
            # can close an older journey before a newer eligible one finalizes.
            exit_captured_at = None
        finalized = _FinalizedJourney(
            decision_id=decision.decision_id,
            group_id=decision.group_id,
            decision_status=self._effective_status(decision),
            canonical_plate_key=canonical_key,
            first_attempt_at=min(
                attempt.request.captured_at for attempt in group.attempts.values()
            ),
            entry_captured_at=decision.entry_captured_at,
            crossing_role=crossing.request.role,
            crossing_camera_id=crossing.request.camera_id,
            crossing_source=self._crossing_source(crossing),
            embeddings=embeddings,
            exit_captured_at=exit_captured_at,
        )
        self._finalized_journeys[decision.decision_id] = finalized
        self._finalized_journeys.move_to_end(decision.decision_id)
        self._trim_finalized_journeys_locked()
        return finalized

    def _find_finalized_journey_locked(
        self,
        request: CrossingInput,
        evidence: Sequence[FrameEvidence],
    ) -> Optional[Tuple[_FinalizedJourney, str]]:
        embeddings = tuple(frame.embedding for frame in evidence if frame.embedding)
        candidates = []
        for finalized in self._finalized_journeys.values():
            if not self._crossing_is_in_journey(request, finalized):
                continue
            score = max_similarity(embeddings, finalized.embeddings)
            match_kind = self._finalized_evidence_match_kind(
                request,
                evidence,
                finalized,
                score,
            )
            if match_kind is None:
                continue
            if (
                match_kind == "provisional"
                and self._newer_unique_group_locked(
                    request,
                    evidence,
                    finalized,
                )
                is not None
            ):
                continue
            candidates.append(
                (
                    match_kind == "definitive",
                    score,
                    finalized.decision_id,
                    finalized,
                    match_kind,
                )
            )
        if not candidates:
            return None
        _, _, _, finalized, match_kind = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return finalized, match_kind

    def _crossing_finalized_match_kind_locked(
        self,
        request: CrossingInput,
        evidence: Sequence[FrameEvidence],
        finalized: _FinalizedJourney,
    ) -> Optional[str]:
        if not self._crossing_is_in_journey(request, finalized):
            return None
        embeddings = tuple(frame.embedding for frame in evidence if frame.embedding)
        score = max_similarity(embeddings, finalized.embeddings)
        match_kind = self._finalized_evidence_match_kind(
            request,
            evidence,
            finalized,
            score,
        )
        if match_kind is None:
            return None
        if (
            match_kind == "provisional"
            and self._newer_unique_group_locked(
                request,
                evidence,
                finalized,
            )
            is not None
        ):
            return None
        return match_kind

    def _crossing_is_in_journey(
        self,
        request: CrossingInput,
        finalized: _FinalizedJourney,
    ) -> bool:
        """Check trusted topology and the exit-bounded source-time interval."""
        trusted_sources = {"hikvision", "va_local_zone"}
        if (
            self._request_crossing_source(request) not in trusted_sources
            or finalized.crossing_source not in trusted_sources
        ):
            return False

        try:
            if request.captured_at < finalized.first_attempt_at:
                return False
            if (
                finalized.exit_captured_at is not None
                and request.captured_at > finalized.exit_captured_at
            ):
                return False
        except TypeError:
            return False

        same_stage = request.role == finalized.crossing_role and norm_camera_id(
            request.camera_id
        ) == norm_camera_id(finalized.crossing_camera_id)
        if same_stage:
            return True
        if (
            finalized.crossing_role == CrossingRole.PRIMARY
            and request.role == CrossingRole.FALLBACK
        ):
            try:
                return request.captured_at >= finalized.entry_captured_at
            except TypeError:
                return False
        if (
            finalized.crossing_role == CrossingRole.FALLBACK
            and request.role == CrossingRole.PRIMARY
        ):
            try:
                return request.captured_at <= finalized.entry_captured_at
            except TypeError:
                return False
        return False

    def _finalized_evidence_match_kind(
        self,
        request: CrossingInput,
        evidence: Sequence[FrameEvidence],
        finalized: _FinalizedJourney,
        reid_score: float,
    ) -> Optional[str]:
        reliable_plate_keys = {
            frame.plate.key
            for frame in evidence
            if frame.plate.key
            and frame.plate.confidence >= self.settings.ocr_min_confidence
        }
        ordinary_floor = max(
            self.settings.reid_min_score,
            self.settings.merge_min_score,
        )
        if reid_score < ordinary_floor:
            return None
        if reliable_plate_keys and reliable_plate_keys != {
            finalized.canonical_plate_key
        }:
            self._record_late_ocr_conflict_locked(
                request.crossing_id,
                finalized,
                reliable_plate_keys,
            )
            return None
        if finalized.exit_captured_at is None:
            if self._is_strict_open_journey_counterpart(
                request,
                finalized,
                reid_score,
            ):
                return "definitive"
            # Until the authoritative exit source timestamp arrives, even an
            # exact plate may be a same-car re-entry whose HTTP delivery raced
            # ahead of the exit webhook. Retain the evidence so record_exit()
            # can classify it without a timeout or arrival-order assumption.
            return "provisional"
        if reliable_plate_keys:
            return "definitive"
        if reid_score >= self.settings.producer_pair_min_reid_score:
            return "definitive"
        return "provisional"

    def _is_strict_open_journey_counterpart(
        self,
        request: CrossingInput,
        finalized: _FinalizedJourney,
        reid_score: float,
    ) -> bool:
        """Recognize only the independent twin of the confirming event."""
        if (
            request.role != finalized.crossing_role
            or norm_camera_id(request.camera_id)
            != norm_camera_id(finalized.crossing_camera_id)
            or {
                self._request_crossing_source(request),
                finalized.crossing_source,
            }
            != {"hikvision", "va_local_zone"}
            or reid_score < self.settings.producer_pair_min_reid_score
        ):
            return False
        try:
            skew_seconds = abs(
                (request.captured_at - finalized.entry_captured_at).total_seconds()
            )
        except TypeError:
            return False
        return skew_seconds <= self.settings.producer_pair_max_skew_seconds

    def _record_late_ocr_conflict_locked(
        self,
        crossing_id: str,
        finalized: _FinalizedJourney,
        observed_plate_keys: set[str],
    ) -> None:
        """Count each conflicting crossing once despite multiple tombstones."""
        if crossing_id in self._late_ocr_conflict_ids:
            return
        self._late_ocr_conflict_ids[crossing_id] = None
        self._late_ocr_conflict_ids.move_to_end(crossing_id)
        while len(self._late_ocr_conflict_ids) > self.settings.receipt_capacity:
            self._late_ocr_conflict_ids.popitem(last=False)
        self._late_ocr_conflict_count += 1
        logger.warning(
            "[EntryV2] late crossing OCR conflicts with finalized journey "
            "crossing=%s decision=%s expected=%s observed=%s",
            crossing_id,
            finalized.decision_id,
            finalized.canonical_plate_key,
            ",".join(sorted(observed_plate_keys)),
        )

    def _newer_unique_group_locked(
        self,
        request: CrossingInput,
        evidence: Sequence[FrameEvidence],
        finalized: _FinalizedJourney,
    ) -> Optional[str]:
        """Use normal assignment gates after an authoritative exit boundary."""
        if finalized.exit_captured_at is None:
            return None
        candidate = CrossingRecord(
            request=request,
            evidence=tuple(evidence),
        )
        competitors = list(self._crossings.values()) + [
            record
            for crossing_id, (record, _) in self._provisional_crossings.items()
            if crossing_id != request.crossing_id
        ]
        match = self._find_unique_match_with_observability_locked(
            candidate,
            self._groups,
            competitors,
        )
        if match is None:
            return None
        group = self._groups.get(match.group_id)
        if group is None or group.status != RecordStatus.PENDING:
            return None
        if not any(
            self._timestamp_between(
                finalized.exit_captured_at,
                attempt.request.captured_at,
                request.captured_at,
            )
            for attempt in group.attempts.values()
        ):
            return None
        return group.group_id

    @staticmethod
    def _timestamp_between(
        lower_exclusive: datetime,
        value: datetime,
        upper_inclusive: datetime,
    ) -> bool:
        try:
            return lower_exclusive < value <= upper_inclusive
        except TypeError:
            return False

    @staticmethod
    def _timestamp_strictly_after(lower_exclusive: datetime, value: datetime) -> bool:
        try:
            return lower_exclusive < value
        except TypeError:
            return False

    @staticmethod
    def _timestamp_at_or_after(lower_inclusive: datetime, value: datetime) -> bool:
        try:
            return lower_inclusive <= value
        except TypeError:
            return False

    def _reconcile_decision_resources_locked(
        self,
        decision: EntryDecision,
        *,
        callback_delivered: bool,
    ) -> None:
        """Attach an asynchronously produced decision to retry receipts."""
        group = self._groups.get(decision.group_id)
        keys = {
            ("crossing", crossing_id)
            for crossing_id, crossing in self._crossings.items()
            if crossing_id == decision.crossing_id
            or (
                crossing.status == RecordStatus.RESOLVED
                and crossing.matched_group_id == decision.group_id
            )
        }
        if group is not None:
            keys.update(("attempt", attempt_id) for attempt_id in group.attempts)
        for key in keys:
            receipt = self._receipts.get(key)
            if receipt is None:
                if key in self._inflight:
                    self._decisions_before_receipt[key] = (
                        decision,
                        callback_delivered,
                    )
                continue
            fingerprint, result = receipt
            updates = {
                "group_id": decision.group_id,
                "decision_id": decision.decision_id,
                "decision_status": self._effective_status(decision),
                "callback_delivered": callback_delivered,
            }
            if key[0] == "crossing":
                updates.update(accepted=True, duplicate=False)
            self._receipts[key] = (
                fingerprint,
                replace(result, **updates),
            )
            self._receipts.move_to_end(key)

    def _reconcile_quarantined_crossing_locked(
        self,
        crossing_id: str,
        finalized: _FinalizedJourney,
    ) -> None:
        key = ("crossing", crossing_id)
        receipt = self._receipts.get(key)
        if receipt is not None:
            fingerprint, result = receipt
            self._receipts[key] = (
                fingerprint,
                self._finalized_duplicate_result(
                    crossing_id,
                    result.evidence_count,
                    finalized,
                ),
            )
            self._receipts.move_to_end(key)
        elif key in self._inflight:
            self._quarantined_crossings_before_receipt[crossing_id] = finalized

    def _finalized_duplicate_result(
        self,
        crossing_id: str,
        evidence_count: int,
        finalized: _FinalizedJourney,
    ) -> IngestResult:
        return IngestResult(
            resource_id=crossing_id,
            accepted=False,
            duplicate=True,
            mode=self.settings.mode,
            evidence_count=evidence_count,
            group_id=finalized.group_id,
            decision_id=finalized.decision_id,
            decision_status=finalized.decision_status,
            callback_delivered=True,
        )

    def _mark_decision_delivered_locked(self, decision_id: str) -> None:
        updated = False
        for key, (fingerprint, result) in list(self._receipts.items()):
            if result.decision_id == decision_id and result.callback_delivered is False:
                self._receipts[key] = (
                    fingerprint,
                    replace(result, callback_delivered=True),
                )
                updated = True
        if updated:
            self._delivered_callbacks_before_receipt.pop(decision_id, None)
            return

        # A scheduled retry can win the narrow race after the initial dispatch
        # queued its failure but before that ingest stores its receipt. Retain a
        # bounded metadata marker so _complete_request cannot subsequently
        # write a permanently-false receipt after the evidence was compacted.
        self._delivered_callbacks_before_receipt[decision_id] = None
        self._delivered_callbacks_before_receipt.move_to_end(decision_id)
        while (
            len(self._delivered_callbacks_before_receipt)
            > self.settings.receipt_capacity
        ):
            self._delivered_callbacks_before_receipt.popitem(last=False)

    def _complete_request(
        self,
        key: Tuple[str, str],
        fingerprint: str,
        result: IngestResult,
    ) -> IngestResult:
        with self._lock:
            self._inflight.discard(key)
            if key[0] == "crossing":
                finalized = self._quarantined_crossings_before_receipt.pop(
                    key[1],
                    None,
                )
                if finalized is not None:
                    result = self._finalized_duplicate_result(
                        key[1],
                        result.evidence_count,
                        finalized,
                    )
                if key[1] in self._released_crossings_before_receipt:
                    released_group_id = self._released_crossings_before_receipt.pop(
                        key[1]
                    )
                    result = replace(
                        result,
                        accepted=True,
                        duplicate=False,
                        group_id=released_group_id,
                        decision_id=None,
                        decision_status=None,
                        callback_delivered=None,
                    )
            reconciled = self._decisions_before_receipt.pop(key, None)
            if reconciled is not None:
                decision, callback_delivered = reconciled
                updates = {
                    "group_id": decision.group_id,
                    "decision_id": decision.decision_id,
                    "decision_status": self._effective_status(decision),
                    "callback_delivered": callback_delivered,
                }
                if key[0] == "crossing":
                    updates.update(accepted=True, duplicate=False)
                result = replace(result, **updates)
            if (
                result.decision_id
                and result.decision_id in self._delivered_callbacks_before_receipt
            ):
                if result.callback_delivered is False:
                    result = replace(result, callback_delivered=True)
                self._delivered_callbacks_before_receipt.pop(result.decision_id, None)
            self._receipts[key] = (fingerprint, result)
            self._receipts.move_to_end(key)
            self._trim_receipts_locked()
            return result

    def _trim_receipts_locked(self) -> None:
        """Bound completed receipts while pinning every still-live evidence ID."""
        while len(self._receipts) > self.settings.receipt_capacity:
            evictable = next(
                (
                    key
                    for key in self._receipts
                    if not self._receipt_key_is_live_locked(key)
                ),
                None,
            )
            if evictable is None:
                # Live records are independently bounded by the attempt and
                # crossing capacities. Their fingerprints must remain pinned
                # until confirmation/cancellation so an LRU churn cannot mutate
                # an active identity by reusing its ID.
                break
            self._receipts.pop(evictable, None)

    def _receipt_key_is_live_locked(self, key: Tuple[str, str]) -> bool:
        kind, resource_id = key
        if kind == "attempt":
            return resource_id in self._attempts
        if kind == "crossing":
            return (
                resource_id in self._crossings
                or resource_id in self._provisional_crossings
            )
        return False

    def _release_inflight(self, key: Tuple[str, str]) -> list[EntryDecision]:
        with self._lock:
            self._inflight.discard(key)
            self._decisions_before_receipt.pop(key, None)
            if key[0] == "crossing":
                self._released_crossings_before_receipt.pop(key[1], None)
                self._quarantined_crossings_before_receipt.pop(key[1], None)
            was_analyzing = key in self._analysis_inflight
            self._analysis_inflight.discard(key)
            if (
                was_analyzing
                and not self._analysis_inflight
                and not self._fatal_delivery_error
            ):
                return self._evaluate_pending_locked()
            return []

    def _effective_status(self, decision: EntryDecision) -> str:
        if (
            self.settings.mode == EntryMode.SHADOW
            and decision.status == DecisionStatus.CONFIRMED
        ):
            return DecisionStatus.ABSTAINED.value
        return decision.status.value

    def _failed_authoritative_confirmation(
        self,
        decisions: Sequence[EntryDecision],
        delivered: Mapping[str, bool],
    ) -> Optional[EntryDecision]:
        if self.settings.mode != EntryMode.AUTHORITATIVE:
            return None
        return next(
            (
                decision
                for decision in decisions
                if decision.status == DecisionStatus.CONFIRMED
                and not delivered.get(decision.decision_id, False)
            ),
            None,
        )

    @staticmethod
    def _related_decision(
        decisions: Sequence[EntryDecision], group_id: str
    ) -> Optional[EntryDecision]:
        return next((item for item in decisions if item.group_id == group_id), None)

    @staticmethod
    def _group_id(attempt_id: str) -> str:
        return "grp-" + hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _fingerprint(kind: str, request, images: Sequence[bytes]) -> str:
        request_data = {
            key: value for key, value in vars(request).items() if key != "metadata"
        }
        timestamp_source = str(request.metadata.get("timestamp_source") or "")
        if timestamp_source.startswith("pms_receive_"):
            # PMS receive time is a provenance-marked fallback, not immutable
            # camera evidence. The stable source/event/image IDs still protect
            # idempotency when the camera retries identical bytes later.
            request_data.pop("captured_at", None)
        else:
            request_data["captured_at"] = request.captured_at.isoformat()
        if hasattr(request_data.get("role"), "value"):
            request_data["role"] = request_data["role"].value
        digest = hashlib.sha256()
        digest.update(kind.encode("utf-8"))
        digest.update(
            json.dumps(
                request_data,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(
            json.dumps(request.metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        for image in images:
            digest.update(hashlib.sha256(image).digest())
        return digest.hexdigest()
