"""
config.py — YAML configuration loader for multi-camera setup.

Loads config.yaml and provides typed access to all settings.
Supports both single-camera (legacy) and multi-camera configurations.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class CameraEntry:
    """Configuration for a single camera."""
    id: str = ""
    name: str = ""
    floor: str = ""
    ip: str = ""
    user: str = ""
    password: str = ""
    slots_file: str = ""


@dataclass
class ProcessingConfig:
    """Multi-camera processing settings."""
    mode: str = "round_robin"
    target_fps_per_camera: int = 1
    stream_channel: int = 102  # 101=main(4K), 102=sub(720p)
    # Resolution that slot polygons (slots/*.json) were drawn at.
    # The runtime will scale polygon coords from this to the actual stream resolution.
    slot_ref_width: int = 640
    slot_ref_height: int = 360


@dataclass
class DetectorConfig:
    """YOLO detector settings."""
    model_path: str = "models/yolo11n.pt"
    confidence: float = 0.35
    classes: List[int] = field(default_factory=lambda: [2])  # 2 = car
    imgsz: int = 480
    device: str = "cuda"  # "auto", "cpu", or "cuda"


@dataclass
class TrackerConfig:
    """Tracker settings."""
    type: str = "bytetrack"


@dataclass
class StateMachineConfig:
    """State machine debounce thresholds."""
    confirm_enter_frames: int = 5
    confirm_leave_frames: int = 8


@dataclass
class AssignerConfig:
    """Slot assigner settings."""
    overlap_threshold: float = 0.3


@dataclass
class DetectorPreprocessingConfig:
    """Luminance normalization settings for the detector path."""
    enabled: bool = True
    clip_limit: float = 2.0
    grid_size: tuple = (8, 8)
    gamma_correction: bool = True
    dark_threshold: int = 65


@dataclass
class ReIDPreprocessingConfig:
    """Luminance normalization settings for the ReID path."""
    enabled: bool = True
    clip_limit: float = 2.0
    grid_size: tuple = (8, 8)


@dataclass
class PreprocessingConfig:
    """Preprocessing settings for both detector and ReID."""
    detector: DetectorPreprocessingConfig = field(default_factory=DetectorPreprocessingConfig)
    reid: ReIDPreprocessingConfig = field(default_factory=ReIDPreprocessingConfig)


@dataclass
class OutputConfig:
    """Output/logging settings."""
    log_file: str = ""
    show_video: bool = False
    show_camera: str = ""  # Which camera to visualize (e.g., "CAM_04")
    snapshot_base_dir: str = "vehicle_images"
    # Externally-reachable origin used to build full snapshot URLs (e.g.
    # "http://localhost:8000"). Empty → emits site-relative URLs (legacy behaviour).
    public_base_url: str = "http://localhost:8000"
    # URL path prefix used for serving and referencing snapshot images
    # (e.g. "/snapshots" → URLs become "{public_base_url}/snapshots/{file}").
    snapshot_url_prefix: str = "/snapshots"
    # Optional gateway routing prefix prepended to all generated URLs.
    # Only needed when the service sits behind a reverse-proxy / API gateway
    # (e.g. "/pms-ai" → URLs become "/pms-ai/snapshots/{file}").
    # Leave empty when accessing the service directly.
    gateway_path_prefix: str = ""

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = ""

@dataclass
class AppConfig:
    """Root configuration container."""
    cameras: List[CameraEntry] = field(default_factory=list)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    assigner: AssignerConfig = field(default_factory=AssignerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Legacy single-camera support
    video_source: str = ""
    video_target_fps: int = 2
    slots_file: str = "parking_slots.json"

    @property
    def is_multi_camera(self) -> bool:
        """True if config has multiple cameras defined."""
        return len(self.cameras) > 0


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file.

    Supports both multi-camera format and legacy single-camera format.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Populated AppConfig instance.
    """
    config = AppConfig()


    if not os.path.exists(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Using defaults.")
        return config

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    if "database" in raw:
        config.database.url = raw["database"].get("DATABASE_URL", "")

    # --- Cameras (multi-camera mode) ---
    if "cameras" in raw:
        for cam_raw in raw["cameras"]:
            cam = CameraEntry(
                id=cam_raw.get("id", ""),
                name=cam_raw.get("name", ""),
                floor=cam_raw.get("floor", ""),
                ip=cam_raw.get("ip", ""),
                user=cam_raw.get("user", ""),
                password=cam_raw.get("password", ""),
                slots_file=cam_raw.get("slots_file", ""),
            )
            config.cameras.append(cam)

    # --- Processing ---
    if "processing" in raw:
        p = raw["processing"]
        config.processing.mode = p.get("mode", config.processing.mode)
        config.processing.target_fps_per_camera = p.get(
            "target_fps_per_camera", config.processing.target_fps_per_camera
        )
        config.processing.stream_channel = p.get(
            "stream_channel", config.processing.stream_channel
        )
        config.processing.slot_ref_width = p.get(
            "slot_ref_width", config.processing.slot_ref_width
        )
        config.processing.slot_ref_height = p.get(
            "slot_ref_height", config.processing.slot_ref_height
        )

    # --- Legacy single-camera support ---
    if "video" in raw:
        v = raw["video"]
        config.video_source = v.get("source", "")
        config.video_target_fps = v.get("target_fps", 2)
    if "slots" in raw:
        config.slots_file = raw["slots"].get("file", "parking_slots.json")

    # --- Detector ---
    if "detector" in raw:
        d = raw["detector"]
        config.detector.model_path = d.get("model_path", config.detector.model_path)
        config.detector.confidence = d.get("confidence", config.detector.confidence)
        config.detector.classes = d.get("classes", config.detector.classes)
        config.detector.imgsz = d.get("imgsz", config.detector.imgsz)
        config.detector.device = d.get("device", config.detector.device)

    # --- Tracker ---
    if "tracker" in raw:
        config.tracker.type = raw["tracker"].get("type", config.tracker.type)

    # --- State Machine ---
    if "state_machine" in raw:
        sm = raw["state_machine"]
        config.state_machine.confirm_enter_frames = sm.get(
            "confirm_enter_frames", config.state_machine.confirm_enter_frames
        )
        config.state_machine.confirm_leave_frames = sm.get(
            "confirm_leave_frames", config.state_machine.confirm_leave_frames
        )

    # --- Assigner ---
    if "assigner" in raw:
        config.assigner.overlap_threshold = raw["assigner"].get(
            "overlap_threshold", config.assigner.overlap_threshold
        )

    # --- Preprocessing ---
    if "preprocessing" in raw:
        pp = raw["preprocessing"]
        if "detector" in pp:
            d_pp = pp["detector"]
            config.preprocessing.detector.enabled = d_pp.get(
                "enabled", config.preprocessing.detector.enabled
            )
            config.preprocessing.detector.clip_limit = d_pp.get(
                "clip_limit", config.preprocessing.detector.clip_limit
            )
            gs = d_pp.get("grid_size", None)
            if gs is not None:
                config.preprocessing.detector.grid_size = tuple(gs)
            config.preprocessing.detector.gamma_correction = d_pp.get(
                "gamma_correction", config.preprocessing.detector.gamma_correction
            )
            config.preprocessing.detector.dark_threshold = d_pp.get(
                "dark_threshold", config.preprocessing.detector.dark_threshold
            )
        if "reid" in pp:
            r_pp = pp["reid"]
            config.preprocessing.reid.enabled = r_pp.get(
                "enabled", config.preprocessing.reid.enabled
            )
            config.preprocessing.reid.clip_limit = r_pp.get(
                "clip_limit", config.preprocessing.reid.clip_limit
            )
            gs = r_pp.get("grid_size", None)
            if gs is not None:
                config.preprocessing.reid.grid_size = tuple(gs)

    # --- Output ---
    if "output" in raw:
        o = raw["output"]
        config.output.log_file = o.get("log_file", config.output.log_file)
        config.output.show_video = o.get("show_video", config.output.show_video)
        config.output.show_camera = o.get("show_camera", config.output.show_camera)
        config.output.snapshot_base_dir = o.get(
            "snapshot_base_dir",
            os.environ.get("SNAPSHOT_PATH", config.output.snapshot_base_dir)
        )
        config.output.public_base_url = o.get(
            "public_base_url",
            os.environ.get("PUBLIC_BASE_URL", config.output.public_base_url),
        )
        config.output.snapshot_url_prefix = o.get(
            "snapshot_url_prefix",
            config.output.snapshot_url_prefix,
        )
        config.output.gateway_path_prefix = o.get(
            "gateway_path_prefix",
            os.environ.get("GATEWAY_PATH_PREFIX", config.output.gateway_path_prefix),
        )

    return config
