"""
Public VehicleRegistry entrypoint.

The heavy implementation is split into focused helper modules so this file can
stay easy to navigate while preserving the existing import path.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Dict, List, Tuple

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
    TRACK_MAPPING_EXPIRY_SECONDS = 8
    SESSION_HANDOFF_GUARD_SECONDS = 10
    _GC_INTERVAL_SECONDS = 5

    def __init__(self, image_dir: str = "vehicle_images"):
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
