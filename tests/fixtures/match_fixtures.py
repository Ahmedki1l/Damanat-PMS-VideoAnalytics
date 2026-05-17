"""
tests.fixtures.match_fixtures — Factory helpers for matching e2e tests.

Builds the three core matching dataclasses (``PendingANPREvent``,
``ParkEntryCandidate``, ``VehicleSession``) and a fully wired
``VehicleRegistry`` instance with stub ReID and image matchers so no real
model is loaded. Uses the Phase-0 DI seams (``db_checker``, ``clock``)
unmodified.

Public surface
--------------

* ``make_pending_anpr_event(plate, ...)`` — :class:`PendingANPREvent` factory.
* ``make_park_entry_candidate(camera_id, track_id, ...)`` —
  :class:`ParkEntryCandidate` factory. Auto-generates a tiny uniform-coloured
  crop when no image is supplied.
* ``make_vehicle_session(plate, ...)`` — :class:`VehicleSession` factory.
* ``make_test_registry(db_checker=..., clock=..., matching_config=...)`` —
  fully-wired :class:`VehicleRegistry` with stub matchers attached.
* ``StubReIDMatcher`` / ``StubImageMatcher`` — deterministic stand-ins for
  the real ReID and image matchers.

These helpers are pure-Python: no GPU, no network, no real ANPR / DB.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.vehicle_registry_models import (
    ParkEntryCandidate,
    PendingANPREvent,
    VehicleSession,
)

from tests.fixtures.fake_clock import FakeClock
from tests.fixtures.fake_db import FakeDBChecker


# --------------------------------------------------------------------------- #
# Stub matchers — keep all "real model" calls out of tests.
# --------------------------------------------------------------------------- #


class StubReIDMatcher:
    """Deterministic stand-in for :mod:`src.reid_matcher`.

    Feature vectors are derived from the image's mean pixel value so two
    crops produced by the same factory yield byte-identical features —
    perfect for testing the matching logic without loading torchreid /
    OpenVINO.

    Tests that want to control similarity directly can call
    :meth:`set_pair_score` to pin the cosine for a specific ``(plate_a,
    plate_b)`` pair.
    """

    def __init__(self):
        self.batch_calls: List[int] = []
        self.feature_calls: int = 0
        # Optional override: maps a tuple of "feature-vector identity"
        # (i.e. the bytes of feature_vector_a.tobytes(), feature_vector_b.tobytes())
        # to a similarity score. Mainly used by tests that want to force
        # specific cosines independent of image content.
        self._pinned_scores: dict = {}

    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        self.feature_calls += 1
        return self._feature_from_image(image)

    def extract_features_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        self.batch_calls.append(len(images))
        return [self._feature_from_image(image) for image in images]

    def compute_similarity(
        self,
        feat_a: Optional[np.ndarray],
        feat_b: Optional[np.ndarray],
    ) -> float:
        if feat_a is None or feat_b is None:
            return 0.0
        # If the caller pinned a score for this exact pair, honour it.
        key = (feat_a.tobytes(), feat_b.tobytes())
        reverse = (feat_b.tobytes(), feat_a.tobytes())
        if key in self._pinned_scores:
            return float(self._pinned_scores[key])
        if reverse in self._pinned_scores:
            return float(self._pinned_scores[reverse])

        # Default: cosine on the two 2-D unit vectors.
        a = feat_a.astype(np.float32)
        b = feat_b.astype(np.float32)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def pin_similarity(
        self,
        feat_a: np.ndarray,
        feat_b: np.ndarray,
        score: float,
    ) -> None:
        """Force :meth:`compute_similarity` to return ``score`` for this pair."""
        self._pinned_scores[(feat_a.tobytes(), feat_b.tobytes())] = float(score)

    @staticmethod
    def _feature_from_image(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        # Use mean R / mean B channel as a 2-D feature so different colours
        # produce different vectors and the same colour produces the same.
        if image.ndim == 3 and image.shape[2] >= 3:
            r = float(np.mean(image[:, :, 2]) / 255.0)
            b = float(np.mean(image[:, :, 0]) / 255.0)
        else:
            r = float(np.mean(image) / 255.0)
            b = 1.0 - r
        return np.array([r, b], dtype=np.float32)


class StubImageMatcher:
    """Deterministic stand-in for :class:`src.image_matcher.VehicleImageMatcher`."""

    def __init__(self, dominant_score: float = 1.0, legacy_score: float = 0.0):
        self._dominant = float(dominant_score)
        self._legacy = float(legacy_score)

    def _compare_dominant_colors(
        self,
        _image_a: np.ndarray,
        _image_b: np.ndarray,
    ) -> float:
        return self._dominant

    def compare(
        self,
        _image_a: np.ndarray,
        _image_b: np.ndarray,
    ) -> float:
        return self._legacy


# --------------------------------------------------------------------------- #
# Image helpers — tiny solid-colour BGR crops keep tests fast and predictable.
# --------------------------------------------------------------------------- #


def make_color_crop(
    bgr: Tuple[int, int, int] = (128, 128, 128),
    size: Tuple[int, int] = (24, 24),
) -> np.ndarray:
    """Build a small ``(H, W, 3)`` BGR uint8 crop filled with ``bgr``."""
    h, w = size
    return np.full((h, w, 3), bgr, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Dataclass factories
# --------------------------------------------------------------------------- #


def make_pending_anpr_event(
    plate: str,
    *,
    direction: str = "entry",
    timestamp: Optional[datetime] = None,
    camera_id: Optional[str] = "CAM-01",
    status: str = "pending",
    event_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    detected_color: Optional[str] = None,
    vehicle_type: Optional[str] = None,
) -> PendingANPREvent:
    """Build a :class:`PendingANPREvent` with sensible defaults.

    The timestamp defaults to the FakeClock default (2026-05-13T12:00:00) when
    not provided. Callers that already have a clock instance should pass
    ``clock()`` so the event lines up with the registry's notion of "now".
    """
    return PendingANPREvent(
        event_id=event_id or f"anpr_{uuid.uuid4().hex[:12]}",
        plate=plate,
        direction=direction,
        timestamp=timestamp if timestamp is not None else datetime(2026, 5, 13, 12, 0, 0),
        camera_id=camera_id,
        status=status,
        candidate_id=candidate_id,
        detected_color=detected_color,
        vehicle_type=vehicle_type,
    )


def make_park_entry_candidate(
    *,
    camera_id: str = "CAM-03",
    track_id: int = 1,
    timestamp: Optional[datetime] = None,
    image: Optional[np.ndarray] = None,
    feature_vector: Optional[np.ndarray] = None,
    quality_score: float = 1.0,
    status: str = "open",
    bound_event_id: Optional[str] = None,
    source: str = "zone_crop",
    color_class: Optional[str] = None,
    color_class_conf: float = 0.0,
    type_class: Optional[str] = None,
    type_class_conf: float = 0.0,
    ocr_plate: Optional[str] = None,
    ocr_plate_conf: float = 0.0,
) -> ParkEntryCandidate:
    """Build a :class:`ParkEntryCandidate` ready to be inserted into the
    registry's ``_park_entry_candidates`` map.

    ``image`` may be an array OR omitted (a uniform-coloured stub is generated
    from a hash of the camera/track so distinct tracks get distinct features).
    """
    now = timestamp if timestamp is not None else datetime(2026, 5, 13, 12, 0, 0)
    if image is None:
        # Pick a deterministic colour from the camera+track so different
        # candidates produce different feature vectors. The values are
        # bounded to (0..255).
        seed = (hash((camera_id, track_id)) & 0xFFFFFF)
        b = seed & 0xFF
        g = (seed >> 8) & 0xFF
        r = (seed >> 16) & 0xFF
        image = make_color_crop(bgr=(b, g, r))

    if feature_vector is None:
        feature_vector = StubReIDMatcher._feature_from_image(image)

    return ParkEntryCandidate(
        candidate_id=f"cand_{uuid.uuid4().hex[:12]}",
        camera_id=camera_id,
        track_id=track_id,
        entered_at=now,
        last_seen_at=now,
        snapshot_image=image.copy(),
        feature_vector=feature_vector,
        feature_vectors=[feature_vector],
        snapshot_paths=[],
        snapshot_images=[image.copy()],
        quality_score=float(quality_score),
        status=status,
        bound_event_id=bound_event_id,
        source=source,
        color_class=color_class,
        color_class_conf=color_class_conf,
        type_class=type_class,
        type_class_conf=type_class_conf,
        ocr_plate=ocr_plate,
        ocr_plate_conf=ocr_plate_conf,
    )


def make_vehicle_session(
    plate: Optional[str],
    *,
    timestamp: Optional[datetime] = None,
    feature_vector: Optional[np.ndarray] = None,
    last_seen_camera: str = "CAM-03",
    last_seen_track_id: Optional[int] = 1,
    status: str = "confirmed",
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    color_class: Optional[str] = None,
    type_class: Optional[str] = None,
    ocr_plate_variants: Optional[List[str]] = None,
) -> VehicleSession:
    """Build a :class:`VehicleSession`. ``feature_vector`` defaults to a
    deterministic 2-D vector derived from the plate so two sessions for the
    same plate get the same vector."""
    now = timestamp if timestamp is not None else datetime(2026, 5, 13, 12, 0, 0)
    if feature_vector is None:
        # Stable per-plate vector derived from hash bits.
        digest = hash(plate or "") & 0xFFFF
        x = (digest & 0xFF) / 255.0
        y = ((digest >> 8) & 0xFF) / 255.0
        feature_vector = np.array([x, y], dtype=np.float32)

    return VehicleSession(
        session_id=session_id or f"sess_{uuid.uuid4().hex[:12]}",
        plate=plate,
        feature_vector=feature_vector,
        first_seen_at=now,
        last_seen_at=now,
        last_seen_camera=last_seen_camera,
        last_seen_track_id=last_seen_track_id,
        event_id=event_id,
        candidate_id=candidate_id,
        status=status,
        color_class=color_class,
        type_class=type_class,
        ocr_plate_variants=list(ocr_plate_variants or []),
    )


# --------------------------------------------------------------------------- #
# Wired test-registry factory
# --------------------------------------------------------------------------- #


def make_test_registry(
    *,
    db_checker: Optional[Callable[[Optional[str]], bool]] = None,
    clock: Optional[Callable[[], datetime]] = None,
    matching_config: Optional[MatchingConfig] = None,
    image_dir: Optional[str] = None,
    reid_matcher: Optional[object] = None,
    image_matcher: Optional[object] = None,
) -> VehicleRegistry:
    """Build a :class:`VehicleRegistry` for e2e tests.

    The returned registry is wired with:

      * ``db_checker`` — defaults to a fresh :class:`FakeDBChecker` (always_true).
      * ``clock`` — defaults to a fresh :class:`FakeClock` (2026-05-13T12:00:00).
      * ``matching_config`` — defaults to :class:`MatchingConfig` (Phase-0
        threshold defaults).
      * Stub matchers (:class:`StubReIDMatcher`, :class:`StubImageMatcher`)
        attached so no real model is ever loaded.

    The ``image_dir`` defaults to a fresh ``tempfile.mkdtemp`` so each test
    has an isolated snapshot directory.
    """
    if image_dir is None:
        image_dir = tempfile.mkdtemp(prefix="match_e2e_")

    registry = VehicleRegistry(
        image_dir=image_dir,
        matching_config=matching_config or MatchingConfig(),
        db_checker=db_checker if db_checker is not None else FakeDBChecker(),
        clock=clock if clock is not None else FakeClock(),
    )
    registry._reid_matcher = reid_matcher or StubReIDMatcher()
    registry._matcher = image_matcher or StubImageMatcher()
    return registry
