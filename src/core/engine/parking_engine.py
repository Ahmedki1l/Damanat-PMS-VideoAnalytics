from typing import Dict, List, Tuple
from src.detection.tracker import TrackedDetector
from src.events.event_bus import EventBus
from src.models.slot import ParkingSlot
from src.core.engine import CameraPipeline



class ParkingEngine:
    def __init__(self, config, vehicle_registry=None, db_manager=None):
        self.config = config
        self.vehicle_registry = vehicle_registry
        self.db_manager = db_manager

        # --- Shared detector (one YOLO model for all cameras) ---
        self.detector = TrackedDetector(
            detector_config=config.detector,
            tracker_config=config.tracker,
        )

        # --- Event bus ---
        self.event_bus = EventBus(log_file=config.output.log_file)

        # --- Per-camera pipelines ---
        self.pipelines: Dict[str, CameraPipeline] = {}
        # [NEW] Tracking zones mapping: cam_id -> zone_id -> ParkingSlot
        self.special_zones: Dict[str, Dict[str, ParkingSlot]] = {}

        # --- [NEW] Tracking State for Zones ---
        self._park_entry_track_to_candidate: Dict[int, str] = {}
        # Maps (cam_id, zone_id, track_id) to the last time it was seen (for entry/exit detection)
        # Using a simple set of track_ids currently "inside" for simpler logic
        self._tracks_inside_zones: Dict[Tuple[str, str], set] = {}

        # --- Violation Alert State ---
        self._recent_violators: List[Dict] = []  # List of {crop, timestamp, camera_id}
        self._violation_match_threshold = 0.4
        self._violation_history_limit = 30 # seconds

        # --- Frame counter for perf logging ---
        self._frame_count = 0
        self._start_time = 0.0
    
