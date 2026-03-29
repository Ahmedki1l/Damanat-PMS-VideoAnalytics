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


@dataclass
class DetectorConfig:
    """YOLO detector settings."""
    model_path: str = "models/yolo11n.pt"
    confidence: float = 0.35
    classes: List[int] = field(default_factory=lambda: [2])  # 2 = car
    imgsz: int = 480
    device: str = "auto"  # "auto", "cpu", or "cuda"


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
class OutputConfig:
    """Output/logging settings."""
    log_file: str = ""
    show_video: bool = False
    show_camera: str = ""  # Which camera to visualize (e.g., "CAM_04")

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

    # --- Output ---
    if "output" in raw:
        o = raw["output"]
        config.output.log_file = o.get("log_file", config.output.log_file)
        config.output.show_video = o.get("show_video", config.output.show_video)
        config.output.show_camera = o.get("show_camera", config.output.show_camera)

    return config
