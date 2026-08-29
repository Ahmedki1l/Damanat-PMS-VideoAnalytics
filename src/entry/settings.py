"""Environment-backed settings for the feature-gated V2 entry path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import FrozenSet, List
from urllib.parse import urlsplit

from .domain import CrossingRole, EntryMode, norm_camera_id


ATTEMPT_PATH = "/api/v2/entry-attempts"
CROSSING_PATH = "/api/v2/entry-crossings"
CANCELLATION_PATH = "/api/v2/entry-cancellations"
CONFIRMATION_PATH = "/api/v1/internal/entry-confirmations"
SERVICE_KEY_HEADER = "X-Service-Key"
MODE_HEADER = "X-Entry-V2-Mode"
_ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _csv(value: str) -> FrozenSet[str]:
    return frozenset(item.strip() for item in (value or "").split(",") if item.strip())


def _csv_cameras(value: str) -> FrozenSet[str]:
    return frozenset(norm_camera_id(item) for item in _csv(value))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        # Every integer-backed Entry V2 setting is validated as positive (or
        # against a stricter positive floor).  Returning an invalid sentinel
        # makes an explicitly malformed deployment value fail configuration
        # instead of silently changing policy by falling back to a default.
        return 0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        # All float-backed settings are checked for finiteness below.  NaN is
        # therefore a fail-closed sentinel for an explicitly malformed value.
        return math.nan


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _ENV_TRUE_VALUES


@dataclass(frozen=True)
class EntrySettings:
    mode: EntryMode = EntryMode.OFF
    invalid_mode_value: str = ""
    max_pending_attempts: int = 256
    max_pending_crossings: int = 256
    max_pending_callbacks: int = 128
    max_concurrent_ingest_requests: int = 2
    receipt_capacity: int = 4096
    journey_capacity: int = 4096
    max_images_per_event: int = 4
    max_image_bytes: int = 4 * 1024 * 1024
    max_decoded_image_pixels: int = 12_000_000
    max_decoded_image_dimension: int = 8192
    max_metadata_bytes: int = 16 * 1024

    reid_min_score: float = 0.75
    reid_row_margin: float = 0.08
    reid_column_margin: float = 0.08
    merge_min_score: float = 0.82
    merge_margin: float = 0.08
    event_consistency_min_score: float = 0.82
    producer_pair_max_skew_seconds: float = 5.0
    producer_pair_min_reid_score: float = 0.95
    ocr_min_confidence: float = 0.75
    correction_min_evidence: int = 2
    correction_min_cameras: int = 2

    # Durable gallery admission is intentionally stricter than live entry
    # confirmation. These thresholds authorize only the metadata handoff; the
    # registry remains responsible for any later persistence policy.
    gallery_anpr_min_confidence: float = 0.90
    gallery_reid_min_score: float = 0.85
    gallery_reid_row_margin: float = 0.12
    gallery_reid_column_margin: float = 0.12
    gallery_ocr_min_confidence: float = 0.90

    primary_cameras: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"CAM23"})
    )
    primary_lines: FrozenSet[str] = field(default_factory=frozenset)
    primary_directions: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"ramp-entry"})
    )
    fallback_cameras: FrozenSet[str] = field(default_factory=frozenset)
    fallback_lines: FrozenSet[str] = field(default_factory=frozenset)
    fallback_directions: FrozenSet[str] = field(default_factory=frozenset)

    pms_base_url: str = ""
    service_key: str = ""
    callback_timeout_seconds: float = 5.0
    callback_max_attempts: int = 1
    callback_initial_backoff_seconds: float = 0.2
    callback_max_backoff_seconds: float = 2.0
    callback_retry_interval_seconds: float = 5.0

    lpd_model_dir: str = "models/yolo11n_lpd_openvino_model"
    lpd_confidence: float = 0.30
    lpd_iou: float = 0.45
    lpd_threads: int = 2
    ocr_model_dir: str = ""
    # The decision log. Empty disables it, matching
    # ENTRY_V2_LOCAL_CAPTURE_DEBUG_DIR's convention. Point it at the volume the
    # vehicle images live on: every record cites image paths on that volume, so
    # log and evidence must be copied, archived and rotated as one unit.
    decision_log_dir: str = ""
    decision_log_retention_days: int = 30
    decision_log_queue_max: int = 2000
    # Colour is a VETO and a tie-break, never confirming weight. Uses the
    # already-tuned HSV compatibility check, which costs a mean over a centre
    # crop — no second model on the gate path, because VA is CPU-starved and a
    # learned classifier there would compete with the detector for frames.
    colour_veto_enabled: bool = True
    # A ramp camera is not a plate source, but a reliable read that contradicts
    # the consensus plate is evidence Re-ID matched the wrong identity, and
    # refusing on that is not the same as naming a plate with it. Subtractive:
    # it can withhold an entry, never create one.
    observation_plate_veto_enabled: bool = True
    # How long an UNCONFIRMED entry identity stays eligible for correlation.
    # This is a LIFETIME, not an identity-matching rule: it never says two
    # observations are the same vehicle, it only bounds how long a candidate
    # exists. Re-ID remains the only thing that associates a car.
    #
    # It is far longer than the legacy path's 10s FIFO bind window, and safe
    # only because nothing here binds a plate by arrival order. If FIFO ever
    # returns to the binding path, this must come down with it.
    identity_ttl_minutes: int = 15
    # Observations outlive identities on purpose, so a late HikCentral sweep can
    # still rescue a dropped entry after the ANPR side is gone.
    observation_ttl_minutes: int = 60
    va_process_count: int = 1
    invalid_va_process_count: str = ""
    va_single_process: bool = False

    @classmethod
    def from_env(cls) -> "EntrySettings":
        raw_mode = os.getenv("ENTRY_V2_MODE", "off")
        try:
            mode = EntryMode.parse(raw_mode)
            invalid_mode_value = ""
        except ValueError:
            mode = EntryMode.OFF
            invalid_mode_value = raw_mode
        raw_process_count = os.getenv("VA_PROCESS_COUNT")
        if raw_process_count is None or not raw_process_count.strip():
            va_process_count = 1
            invalid_va_process_count = "missing"
        else:
            try:
                va_process_count = int(raw_process_count)
                invalid_va_process_count = ""
            except ValueError:
                va_process_count = 1
                invalid_va_process_count = raw_process_count
        return cls(
            mode=mode,
            invalid_mode_value=invalid_mode_value,
            max_pending_attempts=_env_int("ENTRY_V2_MAX_PENDING_ATTEMPTS", 256),
            max_pending_crossings=_env_int("ENTRY_V2_MAX_PENDING_CROSSINGS", 256),
            max_pending_callbacks=_env_int("ENTRY_V2_MAX_PENDING_CALLBACKS", 128),
            max_concurrent_ingest_requests=_env_int(
                "ENTRY_V2_MAX_CONCURRENT_INGEST_REQUESTS", 2
            ),
            receipt_capacity=_env_int("ENTRY_V2_RECEIPT_CAPACITY", 4096),
            journey_capacity=_env_int("ENTRY_V2_JOURNEY_CAPACITY", 4096),
            max_images_per_event=_env_int("ENTRY_V2_MAX_IMAGES", 4),
            max_image_bytes=_env_int("ENTRY_V2_MAX_IMAGE_BYTES", 4 * 1024 * 1024),
            max_decoded_image_pixels=_env_int(
                "ENTRY_V2_MAX_DECODED_IMAGE_PIXELS", 12_000_000
            ),
            max_decoded_image_dimension=_env_int(
                "ENTRY_V2_MAX_DECODED_IMAGE_DIMENSION", 8192
            ),
            max_metadata_bytes=_env_int("ENTRY_V2_MAX_METADATA_BYTES", 16 * 1024),
            reid_min_score=_env_float("ENTRY_V2_REID_MIN_SCORE", 0.75),
            reid_row_margin=_env_float("ENTRY_V2_REID_ROW_MARGIN", 0.08),
            reid_column_margin=_env_float("ENTRY_V2_REID_COLUMN_MARGIN", 0.08),
            merge_min_score=_env_float("ENTRY_V2_MERGE_MIN_SCORE", 0.82),
            merge_margin=_env_float("ENTRY_V2_MERGE_MARGIN", 0.08),
            event_consistency_min_score=_env_float(
                "ENTRY_V2_EVENT_CONSISTENCY_MIN_SCORE", 0.82
            ),
            producer_pair_max_skew_seconds=_env_float(
                "ENTRY_V2_PRODUCER_PAIR_MAX_SKEW_SECONDS", 5.0
            ),
            producer_pair_min_reid_score=_env_float(
                "ENTRY_V2_PRODUCER_PAIR_MIN_REID_SCORE", 0.95
            ),
            ocr_min_confidence=_env_float("ENTRY_V2_OCR_MIN_CONFIDENCE", 0.75),
            correction_min_evidence=_env_int("ENTRY_V2_CORRECTION_MIN_EVIDENCE", 2),
            correction_min_cameras=_env_int("ENTRY_V2_CORRECTION_MIN_CAMERAS", 2),
            gallery_anpr_min_confidence=_env_float(
                "ENTRY_V2_GALLERY_ANPR_MIN_CONFIDENCE", 0.90
            ),
            gallery_reid_min_score=_env_float("ENTRY_V2_GALLERY_REID_MIN_SCORE", 0.85),
            gallery_reid_row_margin=_env_float(
                "ENTRY_V2_GALLERY_REID_ROW_MARGIN", 0.12
            ),
            gallery_reid_column_margin=_env_float(
                "ENTRY_V2_GALLERY_REID_COLUMN_MARGIN", 0.12
            ),
            gallery_ocr_min_confidence=_env_float(
                "ENTRY_V2_GALLERY_OCR_MIN_CONFIDENCE", 0.90
            ),
            primary_cameras=_csv_cameras(
                os.getenv("ENTRY_V2_PRIMARY_CAMERAS", "CAM-23")
            ),
            primary_lines=frozenset(
                v.upper() for v in _csv(os.getenv("ENTRY_V2_PRIMARY_LINES", ""))
            ),
            # No active-mode default: this must be the calibrated raw Hikvision
            # inward value (or an explicit one-way fallback), never a guessed
            # semantic label.
            primary_directions=frozenset(
                v.lower() for v in _csv(os.getenv("ENTRY_V2_PRIMARY_DIRECTIONS", ""))
            ),
            fallback_cameras=_csv_cameras(os.getenv("ENTRY_V2_FALLBACK_CAMERAS", "")),
            fallback_lines=frozenset(
                v.upper() for v in _csv(os.getenv("ENTRY_V2_FALLBACK_LINES", ""))
            ),
            fallback_directions=frozenset(
                v.lower() for v in _csv(os.getenv("ENTRY_V2_FALLBACK_DIRECTIONS", ""))
            ),
            pms_base_url=os.getenv("PMS_API_URL", "").rstrip("/"),
            service_key=os.getenv("ENTRY_V2_SERVICE_KEY", ""),
            callback_timeout_seconds=_env_float(
                "ENTRY_V2_CALLBACK_TIMEOUT_SECONDS", 5.0
            ),
            callback_max_attempts=_env_int("ENTRY_V2_CALLBACK_MAX_ATTEMPTS", 1),
            callback_initial_backoff_seconds=_env_float(
                "ENTRY_V2_CALLBACK_INITIAL_BACKOFF_SECONDS", 0.2
            ),
            callback_max_backoff_seconds=_env_float(
                "ENTRY_V2_CALLBACK_MAX_BACKOFF_SECONDS", 2.0
            ),
            callback_retry_interval_seconds=_env_float(
                "ENTRY_V2_CALLBACK_RETRY_INTERVAL_SECONDS", 5.0
            ),
            lpd_model_dir=os.getenv(
                "ENTRY_V2_LPD_MODEL_DIR", "models/yolo11n_lpd_openvino_model"
            ),
            lpd_confidence=_env_float("ENTRY_V2_LPD_CONFIDENCE", 0.30),
            lpd_iou=_env_float("ENTRY_V2_LPD_IOU", 0.45),
            lpd_threads=_env_int("ENTRY_V2_LPD_THREADS", 2),
            ocr_model_dir=os.getenv("ENTRY_V2_OCR_MODEL_DIR", ""),
            decision_log_dir=os.getenv("ENTRY_V2_DECISION_LOG_DIR", "").strip(),
            decision_log_retention_days=_env_int(
                "ENTRY_V2_DECISION_LOG_RETENTION_DAYS", 30
            ),
            decision_log_queue_max=_env_int("ENTRY_V2_DECISION_LOG_QUEUE_MAX", 2000),
            colour_veto_enabled=os.getenv(
                "ENTRY_V2_COLOUR_VETO_ENABLED", "1"
            ).strip().lower() in _ENV_TRUE_VALUES,
            observation_plate_veto_enabled=os.getenv(
                "ENTRY_V2_OBSERVATION_PLATE_VETO_ENABLED", "1"
            ).strip().lower() in _ENV_TRUE_VALUES,
            identity_ttl_minutes=_env_int("ENTRY_IDENTITY_TTL_MINUTES", 15),
            observation_ttl_minutes=_env_int("ENTRY_OBSERVATION_TTL_MINUTES", 60),
            va_process_count=va_process_count,
            invalid_va_process_count=invalid_va_process_count,
            va_single_process=_env_true("VA_SINGLE_PROCESS"),
        )

    @property
    def callback_url(self) -> str:
        return f"{self.pms_base_url}{CONFIRMATION_PATH}" if self.pms_base_url else ""

    def crossing_policy(self, role: CrossingRole):
        if role == CrossingRole.PRIMARY:
            return self.primary_cameras, self.primary_lines, self.primary_directions
        return self.fallback_cameras, self.fallback_lines, self.fallback_directions

    def configuration_errors(self) -> List[str]:
        if self.invalid_mode_value:
            return [f"ENTRY_V2_MODE={self.invalid_mode_value}"]
        if self.mode == EntryMode.OFF:
            return []
        errors: List[str] = []
        if self.mode == EntryMode.AUTHORITATIVE:
            if self.invalid_va_process_count:
                errors.append("VA_PROCESS_COUNT")
            elif self.va_process_count != 1:
                errors.append("entry_v2_requires_single_process_va")
        positive = {
            "max_pending_attempts": self.max_pending_attempts,
            "max_pending_crossings": self.max_pending_crossings,
            "max_pending_callbacks": self.max_pending_callbacks,
            "max_concurrent_ingest_requests": self.max_concurrent_ingest_requests,
            "receipt_capacity": self.receipt_capacity,
            "journey_capacity": self.journey_capacity,
            "max_images_per_event": self.max_images_per_event,
            "max_image_bytes": self.max_image_bytes,
            "max_decoded_image_pixels": self.max_decoded_image_pixels,
            "max_decoded_image_dimension": self.max_decoded_image_dimension,
            "max_metadata_bytes": self.max_metadata_bytes,
            "callback_max_attempts": self.callback_max_attempts,
            "lpd_threads": self.lpd_threads,
            "va_process_count": self.va_process_count,
        }
        errors.extend(name for name, value in positive.items() if value <= 0)
        local_zone_enabled = (
            "CAM23" in self.primary_cameras and "PARK_ENTRY" in self.primary_lines
        ) or ("CAM03" in self.fallback_cameras and "B1_ENTRENCE" in self.fallback_lines)
        if local_zone_enabled and self.max_images_per_event < 2:
            # The RTSP bridge requires two stable, independently processed
            # frames. Accepting a one-image limit would leave the configured
            # local path silently enabled but physically unable to emit.
            errors.append("local_zone_requires_two_images")
        if local_zone_enabled and not self.va_single_process:
            errors.append("entry_v2_local_zone_requires_va_single_process")
        if self.receipt_capacity < self.max_concurrent_ingest_requests:
            errors.append("receipt_capacity_below_ingest_concurrency")
        if self.identity_ttl_minutes <= 0:
            errors.append("ENTRY_IDENTITY_TTL_MINUTES")
        if self.observation_ttl_minutes <= 0:
            errors.append("ENTRY_OBSERVATION_TTL_MINUTES")
        elif self.observation_ttl_minutes < self.identity_ttl_minutes:
            # An observation that died before its identity could never be
            # rescued by a late sweep, which is the only reason the two TTLs
            # differ at all.
            errors.append("observation_ttl_below_identity_ttl")
        if self.decision_log_dir:
            # Only checked when the log is actually configured. A malformed
            # integer arrives here as 0 (see _env_int), and 0 would silently mean
            # "never prune" on the same volume that holds vehicle imagery — so a
            # typo must fail configuration rather than quietly fill a disk.
            if self.decision_log_retention_days <= 0:
                errors.append("ENTRY_V2_DECISION_LOG_RETENTION_DAYS")
            if self.decision_log_queue_max <= 0:
                errors.append("ENTRY_V2_DECISION_LOG_QUEUE_MAX")
        if not self.primary_cameras:
            errors.append("primary_cameras")
        if not self.primary_lines:
            errors.append("primary_lines")
        if not self.primary_directions:
            errors.append("primary_directions")
        if not self.callback_url:
            errors.append("PMS_API_URL")
        else:
            try:
                parsed_url = urlsplit(self.callback_url)
                _ = parsed_url.port
            except ValueError:
                errors.append("PMS_API_URL")
            else:
                if (
                    parsed_url.scheme not in {"http", "https"}
                    or not parsed_url.hostname
                    or parsed_url.username
                    or parsed_url.password
                    or parsed_url.query
                    or parsed_url.fragment
                ):
                    errors.append("PMS_API_URL")
        if not self.service_key.strip():
            errors.append("ENTRY_V2_SERVICE_KEY")
        if (
            not math.isfinite(self.callback_timeout_seconds)
            or self.callback_timeout_seconds <= 0
        ):
            errors.append("callback_timeout_seconds")
        if (
            not math.isfinite(self.callback_initial_backoff_seconds)
            or self.callback_initial_backoff_seconds < 0
        ):
            errors.append("callback_initial_backoff_seconds")
        if (
            not math.isfinite(self.callback_max_backoff_seconds)
            or self.callback_max_backoff_seconds < 0
        ):
            errors.append("callback_max_backoff_seconds")
        if (
            not math.isfinite(self.callback_retry_interval_seconds)
            or self.callback_retry_interval_seconds <= 0
        ):
            errors.append("callback_retry_interval_seconds")
        for name, value in {
            "ocr_min_confidence": self.ocr_min_confidence,
            "gallery_anpr_min_confidence": self.gallery_anpr_min_confidence,
            "gallery_ocr_min_confidence": self.gallery_ocr_min_confidence,
            "lpd_confidence": self.lpd_confidence,
            "lpd_iou": self.lpd_iou,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(name)
        for name, value in {
            "reid_min_score": self.reid_min_score,
            "merge_min_score": self.merge_min_score,
            "event_consistency_min_score": self.event_consistency_min_score,
            "producer_pair_min_reid_score": self.producer_pair_min_reid_score,
            "gallery_reid_min_score": self.gallery_reid_min_score,
        }.items():
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                errors.append(name)
        if (
            math.isfinite(self.event_consistency_min_score)
            and -1.0 <= self.event_consistency_min_score <= 1.0
            and self.event_consistency_min_score < max(0.0, self.reid_min_score)
        ):
            errors.append("event_consistency_min_score")
        if (
            not math.isfinite(self.producer_pair_max_skew_seconds)
            or self.producer_pair_max_skew_seconds <= 0
        ):
            errors.append("producer_pair_max_skew_seconds")
        producer_pair_floor_inputs = (
            self.event_consistency_min_score,
            self.merge_min_score,
        )
        if (
            "producer_pair_min_reid_score" not in errors
            and all(
                math.isfinite(value) and -1.0 <= value <= 1.0
                for value in producer_pair_floor_inputs
            )
            and self.producer_pair_min_reid_score
            < max(0.90, *producer_pair_floor_inputs)
        ):
            errors.append("producer_pair_min_reid_score")
        for name, value in {
            "reid_row_margin": self.reid_row_margin,
            "reid_column_margin": self.reid_column_margin,
            "merge_margin": self.merge_margin,
            "gallery_reid_row_margin": self.gallery_reid_row_margin,
            "gallery_reid_column_margin": self.gallery_reid_column_margin,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 2.0:
                errors.append(name)
        gallery_safety_floors = {
            "gallery_anpr_min_confidence": (
                self.gallery_anpr_min_confidence,
                0.90,
            ),
            "gallery_reid_min_score": (
                self.gallery_reid_min_score,
                max(0.85, self.reid_min_score),
            ),
            "gallery_reid_row_margin": (
                self.gallery_reid_row_margin,
                max(0.12, self.reid_row_margin),
            ),
            "gallery_reid_column_margin": (
                self.gallery_reid_column_margin,
                max(0.12, self.reid_column_margin),
            ),
            "gallery_ocr_min_confidence": (
                self.gallery_ocr_min_confidence,
                max(0.90, self.ocr_min_confidence),
            ),
        }
        for name, (value, minimum) in gallery_safety_floors.items():
            if math.isfinite(value) and value < minimum and name not in errors:
                errors.append(name)
        if self.correction_min_evidence < 2:
            errors.append("correction_min_evidence")
        if self.correction_min_cameras < 2:
            errors.append("correction_min_cameras")
        fallback_parts = (
            bool(self.fallback_cameras),
            bool(self.fallback_lines),
            bool(self.fallback_directions),
        )
        if any(fallback_parts) and not all(fallback_parts):
            errors.append("fallback_policy_incomplete")
        return errors
