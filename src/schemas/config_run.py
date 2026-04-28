import enum


class ProcessingModeEnum(str, enum.Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    SINGLE_STREAM = "single_stream"


class DeviceEnum(str, enum.Enum):
    CPU = "cpu"
    CUDA = "cuda"


class TrackerTypeEnum(str, enum.Enum):
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"


class ScopeEnum(str, enum.Enum):
    DETECTOR = "detector"
    REID = "reid"


class StreamChannelEnum(int, enum.Enum):
    MAIN = 101
    SUB = 102


# =========================
# Pydantic Schemas
# =========================
from typing import List, Optional
from pydantic import BaseModel, Field


class PreprocessingSchema(BaseModel):
    id: Optional[int] = None
    scope: ScopeEnum
    enabled: bool = True
    clip_limit: float = 2.0
    grid_w: int = 8
    grid_h: int = 8
    gamma_correction: Optional[bool] = None
    dark_threshold: Optional[int] = None

    class Config:
        from_attributes = True


class ConfigSchema(BaseModel):
    id: Optional[int] = None
    target_fps: int = 20
    processing_mode: ProcessingModeEnum = ProcessingModeEnum.ROUND_ROBIN
    stream_channel: StreamChannelEnum = StreamChannelEnum.SUB
    slot_ref_width: int = 640
    slot_ref_height: int = 360
    
    device: DeviceEnum = DeviceEnum.CPU
    model_path: str
    confidence: float = 0.20
    imgsz: int = 640
    classes: str = "2,5,7"
    
    tracker_type: TrackerTypeEnum = TrackerTypeEnum.BYTETRACK
    
    confirm_enter_frames: int = 10
    confirm_leave_frames: int = 40
    overlap_threshold: float = 0.3
    
    show_video: bool = False
    show_camera: Optional[str] = None
    log_file: Optional[str] = None
    
    preprocessing: List[PreprocessingSchema] = []

    class Config:
        from_attributes = True


class PreprocessingGroupSchema(BaseModel):
    """Utility schema for grouping preprocessing by scope."""
    detector: PreprocessingSchema
    reid: PreprocessingSchema


class CameraSchema(BaseModel):
    """Schema for individual camera configuration."""
    id: str
    name: str
    ip: str
    user: str
    password: str
    slots_file: str
    floor: int