"""
Main parking pipeline orchestrator.

The heavy helpers are split across focused modules so this file stays centered
on lifecycle and control flow.
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import logging
import numpy as np

from src.camera_manager import CameraManager
from src.config import AppConfig
from src.core.engine.camera_pipeline import CameraPipeline
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin
from src.core.engine.engine_tracking import ParkingEngineTrackingMixin
from src.core.engine.engine_visualization import ParkingEngineVisualizationMixin
from src.detection.tracker import TrackedDetector
from src.events.event_bus import EventBus
from src.models.slot import load_slots
from src.services.named_slot_service import is_named_slot

logger = logging.getLogger(__name__)




class ParkingEngine(
    ParkingEngineRuntimeMixin,
    ParkingEngineTrackingMixin,
    ParkingEngineVisualizationMixin,
):
    """
    Main orchestrator for the parking management system.

    Multi-camera mode: round-robin across all cameras.
    Single-camera mode: processes one video source.
    """

    def __init__(self, config: AppConfig, vehicle_registry=None, db_manager=None):
        self.config = config
        self.vehicle_registry = vehicle_registry
        self.db_manager = db_manager

        self.detector = TrackedDetector(
            detector_config=config.detector,
            tracker_config=config.tracker,
            preprocessing_config=config.preprocessing.detector,
        )
        self.event_bus = EventBus(log_file=config.output.log_file)

        self.pipelines: Dict[str, CameraPipeline] = {}
        self.special_zones: Dict[str, Dict] = {}
        self._park_entry_track_to_candidate: Dict[int, str] = {}
        self._tracks_inside_zones: Dict[tuple, set] = {}
        self._confirmation_bursts: Dict[tuple, Dict] = {}
        self._latest_zone_vehicle_crops: Dict[tuple, Dict] = {}
        self._recent_violators = []
        self._violation_match_threshold = 0.4
        self._violation_history_limit = 30

        self.is_running = False
        self.start_time = 0.0
        self.last_processed_at: Optional[datetime] = None
        self.model_loaded = True

        self._frame_count = 0
        self._start_time = 0.0
        self._reid_check_timer: Dict[tuple, float] = {}
        self._tracking_managers: Dict[str, object] = {}
        self._display_label_cache: Dict[tuple, Dict[str, object]] = {}
        self._display_label_ttl_seconds = 3.0

        # Legacy single-camera cooldown state.
        self._last_violation_alert_time = 0.0
        self._violation_cooldown_seconds = 5.0

    def get_engine_status(self) -> Dict:
        """Return real-time metrics for the /api/health endpoint."""
        return {
            "engine_running": self.is_running,
            "model_loaded": self.model_loaded,
            "camera_streams_count": len(self.pipelines),
            "camera_streams_ok": self.cam_manager.active_count if hasattr(self, "cam_manager") else 0,
            "total_cameras": self.cam_manager.total_count if hasattr(self, "cam_manager") else 0,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
            "uptime_seconds": int(time.time() - self.start_time) if self.is_running else 0,
            "frames_processed": self._frame_count,
            "db_ok": self.db_manager is not None,
        }

    def run_multi_camera(self) -> None:
        """Multi-camera round-robin processing loop."""
        if not self.config.cameras:
            print("[ERROR] No cameras defined in config.")
            return

        camera_configs = self._build_camera_configs()
        self.cam_manager = CameraManager(camera_configs)
        opened = self.cam_manager.open_all()
        if opened == 0:
            print("[ERROR] No cameras could be opened. Exiting.")
            return

        total_slots = self._initialize_camera_pipelines(camera_configs)
        print(f"[INFO] Total parking slots across all cameras: {total_slots}")
        print(f"[INFO] Processing mode: {self.config.processing.mode}")
        print(
            f"[INFO] Target FPS per camera: "
            f"{self.config.processing.target_fps_per_camera}\n"
        )

        show = self.config.output.show_video
        show_camera = self.config.output.show_camera

        self.is_running = True
        self.start_time = time.time()
        self._start_time = self.start_time

        summary_interval = max(1, len(camera_configs) * 10)
        grid_frames: Dict[str, np.ndarray] = {}
        grid_cell_width = 480
        grid_cell_height = 270
        floor_cameras = self._build_floor_camera_groups(camera_configs)
        floor_cols = 3

        try:
            while True:
                cam_id, frame = self.cam_manager.next_frame()
                if cam_id is None:
                    print("[WARN] All cameras unavailable. Retrying in 5s...")
                    time.sleep(5)
                    continue

                self._cleanup_stale_data()
                self.last_processed_at = datetime.now()

                pipeline = self.pipelines.get(cam_id)
                if pipeline is None:
                    if show:
                        self._store_passthrough_frame(frame, cam_id, grid_frames)
                    continue

                detection_frame = pipeline.apply_roi_mask(frame)
                detections = self.detector.detect_and_track(detection_frame)
                self._process_special_zones(cam_id, frame, detections)

                assignment = pipeline.assigner.assign(detections)
                all_events = self._update_slot_state(cam_id, frame, pipeline, assignment)
                if all_events:
                    final_events = self._filter_violation_events(
                        frame,
                        assignment,
                        cam_id,
                        all_events,
                    )
                    self._persist_final_events(final_events)

                self._frame_count += 1
                if self._frame_count % summary_interval == 0:
                    self._emit_full_summary()

                if show and self._show_multi_camera_output(
                    cam_id,
                    frame,
                    pipeline,
                    assignment,
                    detections,
                    grid_frames,
                    floor_cameras,
                    show_camera,
                    floor_cols,
                    grid_cell_width,
                    grid_cell_height,
                ):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted - shutting down.")
        finally:
            self.cam_manager.close_all()
            if show:
                cv2.destroyAllWindows()
            self.event_bus.close()
            print("[INFO] Engine stopped.")

    def run_single_camera(self, video_source: str, slots_file: str = "") -> None:
        """Single-camera mode for legacy/testing flows."""
        print(f"[INFO] Single-camera mode: {video_source}")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_source}")
            return

        slots_path = slots_file or self.config.slots_file
        slots, roi_polygon = load_slots(slots_path) if os.path.exists(slots_path) else ([], None)

        pipeline = CameraPipeline(
            camera_id="SINGLE",
            floor="",
            slots=slots,
            config=self.config,
            roi_polygon=roi_polygon,
        )
        self.pipelines["SINGLE"] = pipeline

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = self.config.video_target_fps or 2
        frame_skip = max(1, int(source_fps / target_fps))

        print(f"[INFO] Source FPS: {source_fps:.1f}, Target FPS: {target_fps}")
        print(f"[INFO] Processing every {frame_skip}th frame")

        show = self.config.output.show_video
        frame_idx = 0
        summary_interval = target_fps * 10

        self.is_running = True
        self.start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream.")
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                self.last_processed_at = datetime.now()
                detection_frame = pipeline.apply_roi_mask(frame)
                detections = self.detector.detect_and_track(detection_frame)
                assignment = pipeline.assigner.assign(detections)

                all_events = []
                for slot in pipeline.slots:
                    vehicle_in_slot = slot.id in assignment.slot_vehicle_map
                    track_id = None
                    if vehicle_in_slot:
                        track_id, _ = assignment.slot_vehicle_map[slot.id]
                    events = pipeline.state_machines[slot.id].update(
                        vehicle_present=vehicle_in_slot,
                        track_id=track_id,
                    )
                    all_events.extend(events)

                if all_events:
                    final_events = []
                    for event in all_events:
                        slot_state_machine = pipeline.state_machines.get(event.slot_id)
                        if (
                            slot_state_machine
                            and (slot_state_machine.is_violation_zone or is_named_slot(event.slot_id))
                        ):
                            if event.event_type == "vehicle_parked":
                                now_ts = time.time()
                                if (
                                    now_ts - self._last_violation_alert_time
                                    >= self._violation_cooldown_seconds
                                ):
                                    alert_type = self._get_slot_alert_type(
                                        event.slot_id,
                                        getattr(event, "plate_number", ""),
                                    )
                                    if alert_type is None:
                                        final_events.append(event)
                                    else:
                                        event.event_type = alert_type
                                        event.is_alert = True
                                        event.severity = "critical"
                                        self._last_violation_alert_time = now_ts
                                        final_events.append(event)
                                        print(
                                            f"[ALERT] {alert_type.replace('_', ' ').title()} "
                                            f"in {event.slot_id}!"
                                        )
                                else:
                                    final_events.append(event)
                            else:
                                final_events.append(event)
                        else:
                            final_events.append(event)

                    if final_events:
                        self.event_bus.emit_batch(final_events)

                self._frame_count += 1
                if summary_interval > 0 and self._frame_count % summary_interval == 0:
                    statuses = [
                        state_machine.get_status()
                        for state_machine in pipeline.state_machines.values()
                    ]
                    self.event_bus.emit_status_summary(statuses)

                if show:
                    self._draw_frame(frame, pipeline, assignment, "SINGLE", detections)
                    cv2.imshow("Parking Management System", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted - shutting down.")
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            self.event_bus.close()
            print("[INFO] Engine stopped.")
