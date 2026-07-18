"""
Main parking pipeline orchestrator.

The heavy helpers are split across focused modules so this file stays centered
on lifecycle and control flow.
"""

import os
import time
from dataclasses import replace
from datetime import datetime
from typing import Dict, Optional

import cv2
import logging
import numpy as np

from src.camera_manager import CameraManager
from src.config import AppConfig, norm_camera_id
from src.core.engine.camera_pipeline import CameraPipeline
from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin
from src.core.engine.engine_single_process import ParkingEngineSingleProcessMixin
from src.core.engine.engine_tracking import ParkingEngineTrackingMixin
from src.core.engine.engine_visualization import ParkingEngineVisualizationMixin
from src.detection.tracker import TrackedDetector
from src.events.event_bus import EventBus
from src.models.slot import load_slots
from src import perf_trace
from src.services.parking_service import load_camera_slots
from src.zoning import AreaRegistry, BoundaryCrossingDetector

logger = logging.getLogger(__name__)




class ParkingEngine(
    ParkingEngineRuntimeMixin,
    ParkingEngineSingleProcessMixin,
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

        # Detection / tracking are decoupled: ONE shared detection model (one
        # set of weights / one OpenVINO compiled model → RAM stays flat at 27
        # cameras), while ByteTrack *state* is kept per camera inside the
        # TrackedDetector (keyed by camera_id). This gives both low RAM and
        # stable per-camera track IDs — round-robin frames from different
        # cameras no longer corrupt each other's tracker state.
        self._shared_detector: Optional[TrackedDetector] = self._build_tracked_detector()
        # Extra models for cameras in detector.camera_overrides, keyed by the
        # override's settings so cameras sharing one override share one load.
        self._override_detectors: Dict[tuple, TrackedDetector] = {}
        print("[INFO] Shared detection model + per-camera tracker state (decoupled).")
        self.event_bus = EventBus(log_file=config.output.log_file)

        self.pipelines: Dict[str, CameraPipeline] = {}
        self.special_zones: Dict[str, Dict] = {}
        # camera_id -> {boundary_id: BoundaryZone}. Populated when pipelines load
        # DB boundaries; initialized here so zoning hooks (_drive_area_state) are
        # a safe no-op before/without any boundary polygons.
        self.boundaries: Dict[str, Dict] = {}
        self._park_entry_track_to_candidate: Dict[int, str] = {}
        self._tracks_inside_zones: Dict[tuple, set] = {}
        self._confirmation_bursts: Dict[tuple, Dict] = {}
        self._latest_zone_vehicle_crops: Dict[tuple, Dict] = {}
        self._recent_violators = []
        self._violation_match_threshold = 0.4
        self._violation_history_limit = 30
        self._reserved_for_map: dict[str, str | None] = {}
        self._special_slots: set[str] = set()

        self.is_running = False
        self.start_time = 0.0
        self.last_processed_at: Optional[datetime] = None
        self.model_loaded = True

        self._frame_count = 0
        self._start_time = 0.0
        # Window markers for the periodic effective-FPS readout: frame count
        # and timestamp captured at the previous summary.
        self._last_summary_frame = 0
        self._last_summary_ts = 0.0
        self._reid_check_timer: Dict[tuple, float] = {}
        self._tracking_managers: Dict[str, object] = {}
        self._display_label_cache: Dict[tuple, Dict[str, object]] = {}
        self._display_label_ttl_seconds = 3.0

        # Legacy single-camera cooldown state.
        self._last_violation_alert_time = 0.0
        self._violation_cooldown_seconds = 5.0

        # --- Zoning (no-ops on un-zoned deployments) ---------------------
        # AreaRegistry is a cheap read-only camera↔area index. The per-car area
        # lifecycle (AreaStateMachine) lives on the VehicleRegistry, alongside
        # the sessions it mutates; the engine only owns the boundary-crossing
        # *geometry* detector (it needs frame bboxes + polygons). Both are built
        # only when areas are defined AND a registry exists (identity is an
        # API-mode feature); otherwise the per-frame zoning hooks short-circuit
        # and behaviour is byte-for-byte the legacy un-zoned path.
        self.area_registry = AreaRegistry(config)
        self.boundary_crossing_detector = None
        if self.area_registry.enabled and self.vehicle_registry is not None:
            self.boundary_crossing_detector = BoundaryCrossingDetector()
            print(
                f"[INFO] Zoning enabled: {len(self.area_registry.all_area_ids())} "
                f"area(s) — per-area ownership active."
            )

    def _build_tracked_detector(self) -> TrackedDetector:
        """Construct a fresh TrackedDetector using the current config."""
        return TrackedDetector(
            detector_config=self.config.detector,
            tracker_config=self.config.tracker,
            preprocessing_config=self.config.preprocessing.detector,
        )

    def _detector_for(self, camera_id: str) -> TrackedDetector:
        """Return the detector serving this camera.

        Normally the shared one: per-camera ByteTrack state is isolated inside
        TrackedDetector (keyed by camera_id passed to detect_and_track), so a
        single shared model serves every camera without mixing track IDs.

        A camera listed in detector.camera_overrides gets its own model instead
        (see config.DetectorConfig.camera_overrides). Overridden detectors are
        cached by their *settings*, not by camera_id, so N cameras sharing one
        override still cost a single model load.
        """
        override = self.config.detector.camera_overrides.get(norm_camera_id(camera_id))
        if not override:
            if self._shared_detector is None:
                self._shared_detector = self._build_tracked_detector()
            return self._shared_detector

        key = tuple(sorted((k, str(v)) for k, v in override.items()))
        detector = self._override_detectors.get(key)
        if detector is None:
            print(f"[INFO] {camera_id}: detector override {dict(override)}")
            fields = dict(override)
            pp_override = fields.pop("preprocessing", None)
            preprocessing = self.config.preprocessing.detector
            if pp_override:
                preprocessing = replace(preprocessing, **pp_override)
            detector = TrackedDetector(
                detector_config=replace(
                    self.config.detector, camera_overrides={}, **fields
                ),
                tracker_config=self.config.tracker,
                preprocessing_config=preprocessing,
            )
            self._override_detectors[key] = detector
        return detector

    @property
    def detector(self) -> TrackedDetector:
        """Back-compat accessor — prefer _detector_for(camera_id).

        Some legacy call paths (tests, single-camera mode) reach for
        ``engine.detector`` expecting one global instance. We satisfy them
        by returning the shared detector when available, otherwise lazily
        materialising one so the attribute is always usable.
        """
        if self._shared_detector is None:
            self._shared_detector = self._build_tracked_detector()
        return self._shared_detector

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

    def _bootstrap_camera_runtime(self):
        """Shared setup for both processing loops: open cameras, build pipelines,
        prime the registry, start the OCR worker, mark the engine running.

        Returns the list of ``CameraConfig`` on success, or ``None`` if there are
        no cameras or none could be opened (the caller then returns early). Used
        by ``run_multi_camera`` and ``run_single_process`` so the two loops share
        identical startup behaviour.
        """
        if not self.config.cameras:
            print("[ERROR] No cameras defined in config.")
            return None

        camera_configs = self._build_camera_configs()
        self.cam_manager = CameraManager(
            camera_configs,
            max_grab_fps=self.config.processing.max_grab_fps,
        )
        opened = self.cam_manager.open_all()
        if opened == 0:
            print("[ERROR] No cameras could be opened. Exiting.")
            return None

        total_slots = self._initialize_camera_pipelines(camera_configs)
        print(f"[INFO] Total parking slots across all cameras: {total_slots}")

        # Tell the registry which cameras host parking slots so ReID ownership
        # prefers them over slotless transit/aisle cameras (a car's identity is
        # attributed to the camera that can actually bind its plate to a slot).
        if self.vehicle_registry is not None:
            slotted = {
                cam_id for cam_id, pipeline in self.pipelines.items() if pipeline.slots
            }
            self.vehicle_registry.set_cameras_with_slots(slotted)
            print(
                f"[INFO] Slot-hosting cameras (ReID ownership priority): "
                f"{sorted(slotted)}"
            )
            # Reload each still-inside car's persisted per-plate gallery so ReID
            # can re-identify it after a restart (enriches the vectorless
            # sessions _restore_plate_locks just created). No-op when the gallery
            # feature is disabled.
            self._restore_vehicle_galleries()
        print(f"[INFO] Processing mode: {self.config.processing.mode}")
        print(
            f"[INFO] Target FPS per camera: "
            f"{self.config.processing.target_fps_per_camera}\n"
        )

        # Background worker for slot-identify OCR reads (matching.slot_ocr_async).
        # A PaddleOCR read is 2-8s and, inline, freezes slot flips for the whole
        # group — off-thread, the loop submits the read and binds its result a
        # frame or two later. Only started when the registry can read plates.
        self._start_slot_ocr_worker()

        self.is_running = True
        self.start_time = time.time()
        self._start_time = self.start_time
        self._last_summary_frame = 0
        self._last_summary_ts = self.start_time
        return camera_configs

    def run_multi_camera(self) -> None:
        """Multi-camera round-robin processing loop."""
        camera_configs = self._bootstrap_camera_runtime()
        if camera_configs is None:
            return

        show = self.config.output.show_video
        show_camera = self.config.output.show_camera

        summary_interval = max(1, len(camera_configs) * 10)

        # Per-camera FPS pacing. ``target_fps_per_camera`` bounds how often any
        # single camera is processed: each camera is handled at most once per
        # ``min_interval`` seconds. A value <= 0 disables pacing (process as
        # fast as the CPU allows, the historical behaviour).
        target_fps = self.config.processing.target_fps_per_camera
        min_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0
        last_processed: Dict[str, float] = {}
        camera_ids = [c.id for c in camera_configs]
        idle_cycles = 0

        grid_frames: Dict[str, np.ndarray] = {}
        grid_cell_width = 480
        grid_cell_height = 270
        floor_cameras = self._build_floor_camera_groups(camera_configs)
        floor_cols = 3

        try:
            while True:
                # Bind any async OCR reads that finished since the last frame
                # (main thread; each is re-validated against its slot first).
                self._drain_slot_ocr_results()
                with perf_trace.stage("fetch"):
                    cam_id, frame = self.cam_manager.next_frame()
                if cam_id is None:
                    print("[WARN] All cameras unavailable. Retrying in 5s...")
                    time.sleep(5)
                    continue

                # Throttle to the configured per-camera FPS. Cameras still due
                # are processed immediately; if a full round passes with none
                # due we nap briefly so the loop doesn't busy-spin at 100% CPU.
                if min_interval > 0.0:
                    now = time.time()
                    if now - last_processed.get(cam_id, 0.0) < min_interval:
                        idle_cycles += 1
                        if idle_cycles >= len(camera_ids):
                            time.sleep(min(min_interval, 0.01))
                            idle_cycles = 0
                        continue
                    last_processed[cam_id] = now
                    idle_cycles = 0

                self._cleanup_stale_data()
                # Periodic sweep (gated to once per ~30s) that purges
                # in-memory tracking state for plates whose parking_sessions
                # row has been closed by PMS-AI without VA observing the
                # ANPR exit event. Catches missed CAM-EXIT triggers.
                self._exit_janitor_tick()
                # The inverse sweep (gated to once per ~10s): pull in cars that
                # ENTERED after this worker booted. Only the --api group gets the
                # ANPR webhook, so without this a car parking on any other
                # group's camera never matches and binds to plate=None.
                self._session_sync_tick()
                self.last_processed_at = datetime.now()

                pipeline = self.pipelines.get(cam_id)
                if pipeline is None:
                    if show:
                        self._store_passthrough_frame(frame, cam_id, grid_frames)
                    continue

                # clahe + infer stages are timed inside detect_and_track.
                # Detect on the FULL frame — masking to the ROI first blacks out
                # ~a quarter of the frame, which wrecks detection after the model
                # letterboxes to its small input (a fully-visible dark car went
                # undetected at int8/320, so its slot never flipped). The ROI is
                # applied AFTER detection as a bottom-center membership filter,
                # preserving the "exclude out-of-area cars" behaviour at no cost.
                detections = self._detector_for(cam_id).detect_and_track(frame, cam_id)
                assignment = self._process_detections_and_events(
                    cam_id, frame, pipeline, detections
                )

                self._frame_count += 1
                perf_trace.frame_done()
                perf_trace.record_camera_frame(cam_id)
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
            self._stop_slot_ocr_worker()
            self.cam_manager.close_all()
            if show:
                cv2.destroyAllWindows()
            self.event_bus.close()
            print("[INFO] Engine stopped.")

    def run_single_camera(
        self,
        video_source: str,
        slots_file: str = "",
        camera_id: str = "SINGLE",
        floor: str = "",
    ) -> None:
        """Single-camera mode for legacy/testing flows."""
        print(f"[INFO] Single-camera mode: {video_source}")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_source}")
            return

        slots = []
        special_zones = []
        roi_polygon = None
        if self.db_manager and camera_id != "SINGLE":
            matching_camera = next((cam for cam in self.config.cameras if cam.id == camera_id), None)
            if matching_camera is not None:
                self._bootstrap_camera_slots_if_needed(matching_camera)
            source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_res = (source_w, source_h) if source_w > 0 and source_h > 0 else None
            ref_res = (
                self.config.processing.slot_ref_width,
                self.config.processing.slot_ref_height,
            )
            session = self.db_manager.SessionLocal()
            try:
                slots, special_zones, roi_polygon, boundaries = load_camera_slots(
                    session,
                    camera_id=camera_id,
                    ref_resolution=ref_res,
                    actual_resolution=actual_res,
                )
            finally:
                session.close()
            self.special_zones[camera_id] = {zone.id: zone for zone in special_zones}
            if not hasattr(self, "boundaries"):
                self.boundaries = {}
            self.boundaries[camera_id] = {b.id: b for b in boundaries}
        else:
            slots_path = slots_file or self.config.slots_file
            slots, roi_polygon = load_slots(slots_path) if os.path.exists(slots_path) else ([], None)

        pipeline = CameraPipeline(
            camera_id=camera_id,
            floor=floor,
            slots=slots,
            config=self.config,
            roi_polygon=roi_polygon,
        )
        self.pipelines[camera_id] = pipeline

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
                # Detect on the full frame, then filter to the ROI (see the
                # multi-camera loop above — masking before detection wrecks the
                # detector at the model's small input size).
                detections = self._detector_for(camera_id).detect_and_track(frame, camera_id)
                detections = pipeline.filter_detections_to_roi(detections)
                self._process_special_zones(camera_id, frame, detections)
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
                            and (slot_state_machine.is_violation_zone
                             or event.slot_id in self._reserved_for_map
                             or event.slot_id in self._special_slots)
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
                    self._draw_frame(frame, pipeline, assignment, camera_id, detections)
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
