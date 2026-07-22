"""Per-camera frame-difference hints for the single-process scheduler.

Motion decides only whether a fresh frame deserves YOLO now. Occupancy remains
owned exclusively by successful detector observations and the slot state
machines.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, Iterable, Optional

import cv2
import numpy as np

from src.config import MotionSchedulingConfig, norm_camera_id


def validate_motion_runtime(mode: str, *, scheduler_available: bool) -> None:
    """Reject a configured motion policy when this engine path cannot run it."""
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in MotionScheduler._VALID_MODES:
        raise ValueError(f"unsupported motion scheduler mode: {mode!r}")
    if normalized_mode != "legacy" and not scheduler_available:
        raise ValueError(
            "motion scheduler shadow/enforce modes require "
            "VA_SINGLE_PROCESS=1 in multi-camera mode"
        )


class MotionSignal(Enum):
    ACTIVE = "ACTIVE"
    QUIET = "QUIET"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MotionFrame:
    camera_id: str
    image: np.ndarray
    sequence: int
    stream_epoch: int
    age_seconds: float


@dataclass(frozen=True)
class MotionDecision:
    signal: MotionSignal
    should_infer: bool
    reason: str
    frame_is_valid: bool = True
    would_gate: bool = False


def motion_iteration_plan(
    *,
    mode: str,
    bypass: bool,
    target_due: bool,
    analysis_due: bool,
) -> tuple[bool, bool]:
    """Choose whether the engine reads a frame and runs frame-difference work.

    Shadow samples motion at its own cadence but preserves every legacy
    target-due YOLO opportunity. Enforce may sample between YOLO opportunities
    so it can notice a transition promptly.
    """
    if mode == "legacy" or bypass:
        return target_due, target_due
    if mode == "shadow":
        return target_due or analysis_due, analysis_due
    return analysis_due, analysis_due


@dataclass
class _CameraMotionState:
    stream_epoch: int = -1
    baseline: Optional[np.ndarray] = None
    signal: MotionSignal = MotionSignal.UNKNOWN
    transition_latched: bool = False
    last_motion_at: Optional[float] = None
    last_analysis_at: Optional[float] = None
    last_inferred_at: Optional[float] = None
    last_examined_sequence: int = -1
    counters: Dict[str, int] = field(default_factory=dict)


class MotionScheduler:
    """Own one independent motion baseline and sentinel clock per camera."""

    _VALID_MODES = {"legacy", "shadow", "enforce"}

    def __init__(
        self,
        config: MotionSchedulingConfig,
        bypass_cameras: Iterable[str] = (),
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = monotonic_clock
        self._lock = threading.RLock()
        self._bypass_cameras = {norm_camera_id(c) for c in bypass_cameras}
        self._states: Dict[str, _CameraMotionState] = {}
        self._camera_configs: Dict[str, MotionSchedulingConfig] = {}
        self._validate_config(config)

    @property
    def mode(self) -> str:
        return self.config.mode

    def analysis_due(self, camera_id: str) -> bool:
        with self._lock:
            if self.mode == "legacy" or self.is_bypassed(camera_id):
                return True
            state = self._state(camera_id)
            interval = 1.0 / self._camera_config(camera_id).analysis_fps
            return (
                state.last_analysis_at is None
                or self._clock() - state.last_analysis_at >= interval
            )

    def is_bypassed(self, camera_id: str) -> bool:
        with self._lock:
            policy = self._camera_config(camera_id)
            return (
                policy.always_infer
                or norm_camera_id(camera_id) in self._bypass_cameras
            )

    def validate_target_rate(
        self,
        target_fps_per_camera: float,
        camera_ids: Iterable[str],
    ) -> None:
        """Fail closed when enforce mode cannot inspect every target FPS tick."""
        if self.mode != "enforce":
            return
        try:
            target_fps = float(target_fps_per_camera)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "processing.target_fps_per_camera must be numeric"
            ) from exc
        if not math.isfinite(target_fps):
            raise ValueError(
                "processing.target_fps_per_camera must be finite"
            )
        if target_fps <= 0.0:
            return

        invalid = []
        for camera_id in camera_ids:
            if self.is_bypassed(camera_id):
                continue
            analysis_fps = self._camera_config(camera_id).analysis_fps
            if analysis_fps < target_fps:
                invalid.append(f"{camera_id}:{analysis_fps:g}")
        if invalid:
            raise ValueError(
                "motion analysis_fps must be >= "
                "processing.target_fps_per_camera in enforce mode for every "
                "non-bypass camera "
                f"(target={target_fps:g}, invalid={','.join(sorted(invalid))})"
            )

    def examine(self, sample: MotionFrame) -> MotionDecision:
        with self._lock:
            return self._examine_locked(sample)

    def shadow_passthrough(self, sample: MotionFrame) -> MotionDecision:
        """Preserve a legacy YOLO tick between lower-rate motion samples."""
        with self._lock:
            if self.mode != "shadow":
                raise ValueError("shadow passthrough requires shadow mode")
            state = self._state(sample.camera_id)
            if sample.stream_epoch != state.stream_epoch:
                self._reset_for_epoch(state, sample.stream_epoch)
            policy = self._camera_config(sample.camera_id)
            if sample.age_seconds > policy.stale_frame_seconds:
                self._mark_unknown_state(state, "stale_frame")
                return self._decision_for_invalid_frame(state, "stale_frame")
            if sample.image.size == 0:
                self._mark_unknown_state(state, "empty_frame")
                return self._decision_for_invalid_frame(state, "empty_frame")
            self._count(state, "shadow_passthrough")
            return MotionDecision(
                state.signal,
                True,
                "shadow_passthrough",
            )

    def _examine_locked(self, sample: MotionFrame) -> MotionDecision:
        state = self._state(sample.camera_id)
        now = self._clock()
        policy = self._camera_config(sample.camera_id)
        if sample.stream_epoch != state.stream_epoch:
            self._reset_for_epoch(state, sample.stream_epoch)
        if sample.age_seconds > policy.stale_frame_seconds:
            state.last_analysis_at = now
            self._mark_unknown_state(state, "stale_frame")
            return self._decision_for_invalid_frame(state, "stale_frame")
        if sample.image.size == 0:
            state.last_analysis_at = now
            self._mark_unknown_state(state, "empty_frame")
            return self._decision_for_invalid_frame(state, "empty_frame")
        if (
            sample.sequence <= state.last_examined_sequence
        ):
            # The scheduler still spent this analysis opportunity discovering
            # that the grabber has not published a new frame. Advance only the
            # cadence clock so a slow/frozen stream cannot be recopied and
            # re-examined every feeder-loop tick; baseline and occupancy hints
            # remain untouched.
            state.last_analysis_at = now
            self._count(state, "duplicate_sequence")
            if self.mode == "shadow":
                self._count(state, "shadow_would_gate")
                return MotionDecision(
                    state.signal,
                    True,
                    "shadow_duplicate_sequence",
                    would_gate=True,
                )
            return MotionDecision(state.signal, False, "duplicate_sequence")

        state.last_analysis_at = now
        state.last_examined_sequence = sample.sequence
        self._count(state, "examined")

        if self.mode == "legacy":
            return MotionDecision(MotionSignal.UNKNOWN, True, "legacy")
        if self.is_bypassed(sample.camera_id):
            self._count(state, "bypass")
            return MotionDecision(MotionSignal.UNKNOWN, True, "bypass")

        try:
            gray = self._low_resolution_gray(sample.image, policy.analysis_width)
        except cv2.error:
            self._mark_unknown_state(state, "motion_error")
            return MotionDecision(MotionSignal.UNKNOWN, True, "motion_error")

        if state.baseline is None or state.baseline.shape != gray.shape:
            state.baseline = gray
            self._mark_unknown_state(state, "baseline_reset", keep_baseline=True)
            return MotionDecision(MotionSignal.UNKNOWN, True, "baseline_reset")

        changed_ratio = self._changed_ratio(
            state.baseline,
            gray,
            policy.pixel_delta,
        )
        state.baseline = gray
        raw_active = changed_ratio >= policy.changed_ratio
        if raw_active:
            state.last_motion_at = now

        active = raw_active or self._within_active_hold(state, policy, now)
        next_signal = MotionSignal.ACTIVE if active else MotionSignal.QUIET
        self._set_signal(state, next_signal)
        self._count(state, next_signal.value.lower())
        return self._scheduling_decision(state, next_signal, sample.camera_id, now)

    def mark_submitted(self, camera_id: str, reason: str) -> None:
        with self._lock:
            state = self._state(camera_id)
            state.last_inferred_at = self._clock()
            state.transition_latched = False
            self._count(state, f"submitted_{reason}")

    def mark_unknown(self, camera_id: str, stream_epoch: int, reason: str) -> None:
        with self._lock:
            state = self._state(camera_id)
            if stream_epoch != state.stream_epoch:
                self._reset_for_epoch(state, stream_epoch)
            self._mark_unknown_state(state, reason)

    def completion_is_valid(
        self,
        camera_id: str,
        stream_epoch: int,
        frame_age_seconds: float,
    ) -> bool:
        with self._lock:
            state = self._state(camera_id)
            policy = self._camera_config(camera_id)
            stale = (
                stream_epoch != state.stream_epoch
                or frame_age_seconds > policy.stale_frame_seconds
            )
            if not stale:
                return True
            self._count(state, "stale_completion")
            if self.mode == "shadow":
                self._count(state, "shadow_stale_completion")
            return False

    def metrics(self) -> Dict[str, Dict[str, object]]:
        with self._lock:
            now = self._clock()
            metrics: Dict[str, Dict[str, object]] = {}
            for camera_id, state in self._states.items():
                last_infer_age = None
                if state.last_inferred_at is not None:
                    last_infer_age = max(0.0, now - state.last_inferred_at)
                metrics[camera_id] = {
                    "signal": state.signal.value,
                    "stream_epoch": state.stream_epoch,
                    "last_examined_sequence": state.last_examined_sequence,
                    "last_infer_age_seconds": last_infer_age,
                    "transition_latched": state.transition_latched,
                    "bypass": self.is_bypassed(camera_id),
                    **state.counters,
                }
            return metrics

    def _state(self, camera_id: str) -> _CameraMotionState:
        return self._states.setdefault(camera_id, _CameraMotionState())

    def _camera_config(self, camera_id: str) -> MotionSchedulingConfig:
        existing = self._camera_configs.get(camera_id)
        if existing is not None:
            return existing
        override = self.config.camera_overrides.get(norm_camera_id(camera_id), {})
        resolved = replace(self.config, camera_overrides={}, **override)
        self._validate_config(resolved)
        self._camera_configs[camera_id] = resolved
        return resolved

    def _validate_config(self, config: MotionSchedulingConfig) -> None:
        if config.mode not in self._VALID_MODES:
            raise ValueError("motion scheduler mode must be legacy, shadow, or enforce")
        if not math.isfinite(config.analysis_fps) or config.analysis_fps <= 0:
            raise ValueError("motion analysis_fps must be positive")
        if config.analysis_width < 16:
            raise ValueError("motion analysis_width must be at least 16 pixels")
        if not 0 <= config.pixel_delta <= 255:
            raise ValueError("motion pixel_delta must be between 0 and 255")
        if not math.isfinite(config.changed_ratio) or not 0 <= config.changed_ratio <= 1:
            raise ValueError("motion changed_ratio must be between 0 and 1")
        if (
            not math.isfinite(config.active_hold_seconds)
            or config.active_hold_seconds < 0
        ):
            raise ValueError("motion active_hold_seconds cannot be negative")
        if (
            not math.isfinite(config.sentinel_interval_seconds)
            or config.sentinel_interval_seconds <= 0
        ):
            raise ValueError("motion sentinel_interval_seconds must be positive")
        if (
            not math.isfinite(config.stale_frame_seconds)
            or config.stale_frame_seconds <= 0
        ):
            raise ValueError("motion stale_frame_seconds must be positive")

    def _reset_for_epoch(self, state: _CameraMotionState, stream_epoch: int) -> None:
        state.stream_epoch = stream_epoch
        state.baseline = None
        state.signal = MotionSignal.UNKNOWN
        state.transition_latched = True
        state.last_motion_at = None
        state.last_examined_sequence = -1
        self._count(state, "epoch_reset")

    def _mark_unknown_state(
        self,
        state: _CameraMotionState,
        reason: str,
        keep_baseline: bool = False,
    ) -> None:
        if not keep_baseline:
            state.baseline = None
        state.signal = MotionSignal.UNKNOWN
        state.transition_latched = True
        self._count(state, reason)
        self._count(state, "unknown")

    def _decision_for_invalid_frame(
        self, state: _CameraMotionState, reason: str
    ) -> MotionDecision:
        would_gate = self.mode == "shadow"
        return MotionDecision(
            signal=state.signal,
            should_infer=self.mode == "shadow",
            reason=reason,
            frame_is_valid=False,
            would_gate=would_gate,
        )

    def _scheduling_decision(
        self,
        state: _CameraMotionState,
        signal: MotionSignal,
        camera_id: str,
        now: float,
    ) -> MotionDecision:
        should_infer, reason = self._raw_schedule(state, signal, camera_id, now)
        if self.mode == "shadow":
            if not should_infer:
                self._count(state, "shadow_would_gate")
            return MotionDecision(
                signal,
                True,
                f"shadow_{reason}",
                would_gate=not should_infer,
            )
        if not should_infer:
            self._count(state, "gated_quiet")
        return MotionDecision(signal, should_infer, reason, would_gate=not should_infer)

    def _raw_schedule(
        self,
        state: _CameraMotionState,
        signal: MotionSignal,
        camera_id: str,
        now: float,
    ) -> tuple[bool, str]:
        if self.is_bypassed(camera_id):
            return True, "bypass"
        if signal == MotionSignal.ACTIVE:
            return True, "active"
        if state.transition_latched:
            return True, "transition"
        policy = self._camera_config(camera_id)
        if (
            state.last_inferred_at is None
            or now - state.last_inferred_at >= policy.sentinel_interval_seconds
        ):
            return True, "sentinel"
        return False, "quiet"

    @staticmethod
    def _low_resolution_gray(frame: np.ndarray, width: int) -> np.ndarray:
        height = max(16, round(frame.shape[0] * width / frame.shape[1]))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    @staticmethod
    def _changed_ratio(
        previous: np.ndarray,
        current: np.ndarray,
        pixel_delta: int,
    ) -> float:
        difference = cv2.absdiff(previous, current)
        _, changed = cv2.threshold(
            difference,
            pixel_delta,
            255,
            cv2.THRESH_BINARY,
        )
        return cv2.countNonZero(changed) / float(changed.size)

    @staticmethod
    def _within_active_hold(
        state: _CameraMotionState,
        policy: MotionSchedulingConfig,
        now: float,
    ) -> bool:
        return (
            state.last_motion_at is not None
            and now - state.last_motion_at <= policy.active_hold_seconds
        )

    @staticmethod
    def _set_signal(state: _CameraMotionState, signal: MotionSignal) -> None:
        if signal != state.signal:
            state.signal = signal
            state.transition_latched = True

    @staticmethod
    def _count(state: _CameraMotionState, name: str) -> None:
        state.counters[name] = state.counters.get(name, 0) + 1


def rotated_camera_ids(camera_ids: list[str], start_index: int) -> list[str]:
    """Return one fair scheduler pass and rotate who gets first access next."""
    if not camera_ids:
        return []
    offset = start_index % len(camera_ids)
    return camera_ids[offset:] + camera_ids[:offset]


def motion_bypass_cameras(
    special_zones: dict,
    entry_cameras: Iterable[str] = (),
) -> set[str]:
    """Entry-confirmation cameras must never be quiet-frame gated."""
    # Keep the site's legacy entry cameras safe while also honoring every
    # primary/fallback camera configured for Entry V2. The scheduler normalizes
    # this set, so configured spellings such as CAM24 and CAM-24 are equivalent.
    bypass = {"CAM-23", "CAM-03", *entry_cameras}
    for camera_id, zones in special_zones.items():
        for zone_id in zones:
            normalized = str(zone_id).lower().replace("_", "").replace("-", "")
            if any(
                marker in normalized
                for marker in ("parkentry", "b1entry", "entrance", "entrence")
            ):
                bypass.add(camera_id)
                break
    return bypass
