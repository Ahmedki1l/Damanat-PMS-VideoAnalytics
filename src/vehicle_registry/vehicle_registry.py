"""
Public VehicleRegistry entrypoint.

The heavy implementation is split into focused helper modules so this file can
stay easy to navigate while preserving the existing import path.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from src.config import MatchingConfig, ReIDPreprocessingConfig
from src.matching import MatchDecision
from src.vehicle_registry.vehicle_registry_core import VehicleRegistryCoreMixin
from src.vehicle_registry.vehicle_registry_identity import VehicleRegistryIdentityMixin
from src.vehicle_registry.vehicle_registry_models import (
    ParkEntryCandidate,
    PendingANPREvent,
    VehicleSession,
)
from src.vehicle_registry.vehicle_registry_queries import VehicleRegistryQueryMixin

logger = logging.getLogger(__name__)


class VehicleRegistry(
    VehicleRegistryCoreMixin,
    VehicleRegistryIdentityMixin,
    VehicleRegistryQueryMixin,
):
    """
    Central registry for ANPR -> Park_Entry -> B1_Entrance -> parking slot flow.
    """

    PENDING_ANPR_EXPIRY_SECONDS = 30
    CANDIDATE_EXPIRY_SECONDS = 30
    # Effectively infinite; track mappings are now primarily cleared via the ANPR Exit event.
    TRACK_MAPPING_EXPIRY_SECONDS = 86400 * 30
    SESSION_HANDOFF_GUARD_SECONDS = 10
    _GC_INTERVAL_SECONDS = 5

    def __init__(
        self,
        image_dir: str = "vehicle_images",
        reid_preprocessing_config: ReIDPreprocessingConfig = None,
        public_base_url: str = "",
        snapshot_url_prefix: str = "/pms-video-analytics/snapshots",
        gateway_path_prefix: str = "",
        matching_config: Optional[MatchingConfig] = None,
        db_checker: Optional[Callable[[Optional[str]], bool]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        match_decision: Optional[MatchDecision] = None,
    ):
        self._lock = threading.RLock()
        self._matcher_lock = threading.Lock()

        self._pending_event_order: List[str] = []
        self._pending_events: Dict[str, PendingANPREvent] = {}
        self._park_entry_candidates: Dict[str, ParkEntryCandidate] = {}
        self._sessions: Dict[str, VehicleSession] = {}
        self._track_session_map: Dict[Tuple[str, int], str] = {}
        self._track_last_seen: Dict[Tuple[str, int], datetime] = {}
        self._parked: Dict[str, VehicleSession] = {}
        self._history: List[VehicleSession] = []

        self._matcher = None
        self._reid_matcher = None
        self._image_dir = image_dir
        self._public_base_url = public_base_url
        self._snapshot_url_prefix = snapshot_url_prefix.strip("/")
        self._gateway_path_prefix = gateway_path_prefix.strip("/")
        self._reid_preprocessing_config = (
            reid_preprocessing_config or ReIDPreprocessingConfig()
        )

        # Matching configuration + decision chokepoint. When not injected,
        # fall back to defaults that preserve historical behaviour. The
        # NoopColorClassifier inside MatchDecision needs the legacy image
        # matcher for ``_compare_dominant_colors``; pass a zero-arg factory
        # so we don't force its construction at __init__ time (the matcher
        # property is intentionally lazy).
        self._matching_config: MatchingConfig = matching_config or MatchingConfig()
        if match_decision is not None:
            self._match_decision = match_decision
        else:
            self._match_decision = self._build_match_decision(
                self._matching_config,
                lambda: self.matcher,
            )

        # DI seams (T0.5). ``db_checker`` lets tests replace the SQL probe in
        # is_plate_inside; ``clock`` lets tests advance time deterministically.
        # Both default to None — callsites still use the real
        # implementations exactly as before.
        self._db_checker: Optional[Callable[[Optional[str]], bool]] = db_checker
        self._clock: Callable[[], datetime] = clock or datetime.now

        os.makedirs(image_dir, exist_ok=True)

        self._gc_thread = threading.Thread(
            target=self._gc_loop,
            name="VehicleRegistry-GC",
            daemon=True,
        )
        self._gc_thread.start()
        logger.debug(
            "Background GC thread started (interval=%ds)",
            self._GC_INTERVAL_SECONDS,
        )

    @property
    def matching_config(self) -> MatchingConfig:
        """Active matching configuration (thresholds + feature flags)."""
        return self._matching_config

    @property
    def match_decision(self) -> MatchDecision:
        """Centralised match decision chokepoint."""
        return self._match_decision

    # ------------------------------------------------------------------ #
    # Plugin instantiation (Phase 2 T2.1)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_match_decision(
        config: MatchingConfig,
        image_matcher_factory,
    ) -> MatchDecision:
        """Construct a :class:`MatchDecision` with real plugins when their
        model files exist on disk; otherwise fall back to Noop plugins.

        Path resolution is explicit (no magic auto-discovery): each plugin's
        path is taken from ``MatchingConfig`` and the file is probed with
        ``os.path.exists`` before the heavy import happens. Missing models
        produce a log line and a Noop instance — never a hard import error
        at boot. This keeps test bootstrap fast and prevents a fresh checkout
        from crashing before any model is trained.
        """
        from src.matching import NoopColorClassifier, NoopPlateOCR, NoopTypeClassifier

        # --- Color classifier -------------------------------------------- #
        color_plugin = None
        color_path = (config.color_classifier_model or "").strip()
        if color_path and os.path.exists(color_path):
            try:
                from src.classifiers.color_classifier import OpenVINOColorClassifier

                color_plugin = OpenVINOColorClassifier(
                    model_path=color_path,
                    config=config,
                )
                logger.info(
                    "[MatchDecision] color classifier loaded from %s",
                    color_path,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[MatchDecision] failed to load color classifier at %s: %r — "
                    "falling back to NoopColorClassifier",
                    color_path,
                    exc,
                )
                color_plugin = None
        if color_plugin is None:
            color_plugin = NoopColorClassifier(
                image_matcher=image_matcher_factory,
                hsv_h_tol=config.hsv_h_tol,
                hsv_s_tol=config.hsv_s_tol,
                hsv_v_tol=config.hsv_v_tol,
            )

        # --- Type classifier --------------------------------------------- #
        type_plugin = None
        type_path = (config.type_classifier_model or "").strip()
        if type_path and os.path.exists(type_path):
            try:
                from src.classifiers.type_classifier import OpenVINOTypeClassifier

                type_plugin = OpenVINOTypeClassifier(model_path=type_path)
                logger.info(
                    "[MatchDecision] type classifier loaded from %s",
                    type_path,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[MatchDecision] failed to load type classifier at %s: %r — "
                    "falling back to NoopTypeClassifier",
                    type_path,
                    exc,
                )
                type_plugin = None
        if type_plugin is None:
            type_plugin = NoopTypeClassifier()

        # --- Plate OCR ---------------------------------------------------- #
        # The OCR plugin is heavy (PaddleOCR pulls ~50 MB of weights on first
        # use); only instantiate when ``plate_ocr_model`` is configured non-
        # empty so a fresh checkout boots without paddle installed.
        ocr_plugin = None
        ocr_path = (config.plate_ocr_model or "").strip()
        if ocr_path:
            try:
                from src.ocr.plate_ocr import PaddlePlateOCR

                ocr_plugin = PaddlePlateOCR(model_dir=ocr_path or None)
                logger.info(
                    "[MatchDecision] plate OCR plugin loaded (model_dir=%s)",
                    ocr_path or "<default>",
                )
            except Exception as exc:
                logger.warning(
                    "[MatchDecision] failed to load plate OCR plugin (%r); "
                    "falling back to NoopPlateOCR",
                    exc,
                )
                ocr_plugin = None
        if ocr_plugin is None:
            ocr_plugin = NoopPlateOCR()

        return MatchDecision(
            config,
            color_classifier=color_plugin,
            type_classifier=type_plugin,
            plate_ocr=ocr_plugin,
        )
