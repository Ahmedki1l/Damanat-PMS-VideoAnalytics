from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Enum
from sqlalchemy.orm import relationship
from src.database import Base

from src.schemas import (
    ProcessingModeEnum,
    DeviceEnum,
    TrackerTypeEnum,
    ScopeEnum,
    StreamChannelEnum,
)


class Config(Base):
    __tablename__ = "config"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # =========================
    # Processing
    # =========================
    target_fps = Column(Integer, nullable=False, default=20)

    processing_mode = Column(
        Enum(ProcessingModeEnum),
        nullable=False,
        default=ProcessingModeEnum.ROUND_ROBIN
    )

    stream_channel = Column(
        Enum(StreamChannelEnum),
        nullable=False,
        default=StreamChannelEnum.SUB
    )

    # Slot reference resolution
    slot_ref_width = Column(Integer, nullable=False, default=1280)
    slot_ref_height = Column(Integer, nullable=False, default=720)

    # =========================
    # Detector
    # =========================
    device = Column(
        Enum(DeviceEnum),
        nullable=False,
        default=DeviceEnum.CPU
    )

    model_path = Column(String(255), nullable=False)

    confidence = Column(Float, nullable=False, default=0.20)
    imgsz = Column(Integer, nullable=False, default=640)

    classes = Column(String(50), nullable=False, default="2,5,7")

    # =========================
    # Tracker
    # =========================
    tracker_type = Column(
        Enum(TrackerTypeEnum),
        nullable=False,
        default=TrackerTypeEnum.BYTETRACK
    )

    # =========================
    # State Machine
    # =========================
    confirm_enter_frames = Column(Integer, nullable=False, default=10)
    confirm_leave_frames = Column(Integer, nullable=False, default=40)

    # =========================
    # Assigner
    # =========================
    overlap_threshold = Column(Float, nullable=False, default=0.3)

    # =========================
    # Output
    # =========================
    show_video = Column(Boolean, nullable=False, default=False)
    show_camera = Column(String(50), nullable=True)
    log_file = Column(String(255), nullable=True)

    # =========================
    # Relationships
    # =========================
    preprocessing = relationship(
        "PreprocessingConfig",
        back_populates="config",
        cascade="all, delete-orphan"
    )


class PreprocessingConfig(Base):
    __tablename__ = "preprocessing_config"

    id = Column(Integer, primary_key=True, autoincrement=True)

    config_id = Column(Integer, ForeignKey("config.id"), nullable=False)

    scope = Column(
        Enum(ScopeEnum),
        nullable=False
    )

    enabled = Column(Boolean, nullable=False, default=True)

    clip_limit = Column(Float, nullable=False, default=2.0)
    grid_w = Column(Integer, nullable=False, default=8)
    grid_h = Column(Integer, nullable=False, default=8)

    gamma_correction = Column(Boolean, nullable=True)
    dark_threshold = Column(Integer, nullable=True)

    config = relationship("Config", back_populates="preprocessing")