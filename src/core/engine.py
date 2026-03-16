"""
engine.py — Main pipeline orchestrator (multi-camera support).

Supports two modes:
  1. Multi-camera: round-robin processing of 12 cameras.
  2. Single-camera: legacy mode for testing with one stream.

Pipeline per frame:
  Frame Grabber → Detector/Tracker → Slot Assigner → State Machines → Event Bus
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import AppConfig
from src.models.slot import ParkingSlot, load_slots
from src.models.state_machine import SlotStateMachine, SlotState
from src.detection.tracker import TrackedDetector
from src.core.slot_assigner import SlotAssigner
from src.events.event_bus import EventBus
from src.camera_manager import CameraManager, CameraConfig


# Colors for visualization (BGR format)
COLORS = {
    SlotState.VACANT: (0, 255, 0),       # Green
    SlotState.ENTERING: (0, 255, 255),   # Yellow
    SlotState.OCCUPIED: (0, 0, 255),     # Red
    SlotState.LEAVING: (0, 165, 255),    # Orange
}


class CameraPipeline:
    """
    Per-camera processing state.

    Holds the slot polygons, state machines, and assigner
    specific to one camera's view.
    """

    def __init__(
        self,
        camera_id: str,
        floor: str,
        slots: List[ParkingSlot],
        config: AppConfig,
    ):
        self.camera_id = camera_id
        self.floor = floor
        self.slots = slots

        # Per-slot state machines
        self.state_machines: Dict[str, SlotStateMachine] = {}
        for slot in slots:
            self.state_machines[slot.id] = SlotStateMachine(
                slot_id=slot.id,
                confirm_enter_frames=config.state_machine.confirm_enter_frames,
                confirm_leave_frames=config.state_machine.confirm_leave_frames,
            )

        # Slot assigner for this camera's polygons
        self.assigner = SlotAssigner(slots=slots, config=config.assigner)

    @property
    def slot_count(self) -> int:
        return len(self.slots)


class ParkingEngine:
    """
    Main orchestrator for the parking management system.

    Multi-camera mode: round-robin across all cameras.
    Single-camera mode: processes one video source.
    """

    def __init__(self, config: AppConfig, vehicle_registry=None):
        self.config = config
        self.vehicle_registry = vehicle_registry

        # --- Shared detector (one YOLO model for all cameras) ---
        self.detector = TrackedDetector(
            detector_config=config.detector,
            tracker_config=config.tracker,
        )

        # --- Event bus ---
        self.event_bus = EventBus(log_file=config.output.log_file)

        # --- Per-camera pipelines ---
        self.pipelines: Dict[str, CameraPipeline] = {}

        # --- Frame counter for perf logging ---
        self._frame_count = 0
        self._start_time = 0.0

    def run_multi_camera(self) -> None:
        """
        Multi-camera round-robin processing loop.

        Cycles through all cameras, processing one frame at a time.
        """
        if not self.config.cameras:
            print("[ERROR] No cameras defined in config.")
            return

        # Build camera configs
        camera_configs: List[CameraConfig] = []
        for cam in self.config.cameras:
            cc = CameraConfig(
                id=cam.id,
                name=cam.name,
                floor=cam.floor,
                ip=cam.ip,
                user=cam.user,
                password=cam.password,
                slots_file=cam.slots_file,
            )
            cc.build_rtsp_url(channel=self.config.processing.stream_channel)
            camera_configs.append(cc)

        # Initialize camera manager
        cam_manager = CameraManager(camera_configs)
        opened = cam_manager.open_all()

        if opened == 0:
            print("[ERROR] No cameras could be opened. Exiting.")
            return

        # Load per-camera slot polygons and create pipelines
        total_slots = 0
        for cc in camera_configs:
            slots = []
            if cc.slots_file and os.path.exists(cc.slots_file):
                slots = load_slots(cc.slots_file)
            elif cc.slots_file:
                print(f"[WARN] Slots file '{cc.slots_file}' not found for {cc.id}. "
                      f"Run: python draw_slots.py --camera {cc.id}")

            pipeline = CameraPipeline(
                camera_id=cc.id,
                floor=cc.floor,
                slots=slots,
                config=self.config,
            )
            self.pipelines[cc.id] = pipeline
            total_slots += pipeline.slot_count

        print(f"[INFO] Total parking slots across all cameras: {total_slots}")
        print(f"[INFO] Processing mode: {self.config.processing.mode}")
        print(f"[INFO] Target FPS per camera: {self.config.processing.target_fps_per_camera}\n")

        # Visualization setup
        show = self.config.output.show_video
        show_camera = self.config.output.show_camera

        self._start_time = time.time()
        summary_interval = max(1, len(camera_configs) * 10)  # Every ~10 full cycles

        # Per-floor grid view: store latest annotated frame per camera
        grid_frames: Dict[str, np.ndarray] = {}
        grid_cell_width = 480
        grid_cell_height = 270

        # Group camera IDs by floor for per-floor windows
        floor_cameras: Dict[str, list] = {}
        for cfg in camera_configs:
            floor = cfg.floor
            if floor not in floor_cameras:
                floor_cameras[floor] = []
            floor_cameras[floor].append(cfg.id)
        # Use 3 columns per floor (6 cameras = 3×2 grid per floor)
        floor_cols = 3

        try:
            while True:
                # Get next frame (round-robin)
                cam_id, frame = cam_manager.next_frame()

                if cam_id is None:
                    print("[WARN] All cameras unavailable. Retrying in 5s...")
                    time.sleep(5)
                    continue

                t_start = time.time()

                # Process this camera's frame
                pipeline = self.pipelines.get(cam_id)
                if pipeline is None or pipeline.slot_count == 0:
                    # Still store raw frame for grid (cameras without slots)
                    if show:
                        label_frame = frame.copy()
                        cv2.putText(label_frame, cam_id, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        grid_frames[cam_id] = label_frame
                    continue

                # --- 1. Detect + Track ---
                detections = self.detector.detect_and_track(frame)

                # --- 1.1. Feed gate camera data for ANPR snapshot capture ---
                if self.vehicle_registry and cam_id == "CAM_01" and detections:
                    self.vehicle_registry.update_gate_snapshot(frame, detections)

                # --- 1.5. Assign ANPR plates to unplated detections ---
                # Strategy 1: Simple queue (oldest pending plate → first unplated car)
                # Strategy 2: Image matching (if ANPR images available)
                if self.vehicle_registry and detections:
                    h, w = frame.shape[:2]
                    for det in detections:
                        if det.track_id == -1:
                            continue
                        # Try simple queue first
                        plate = self.vehicle_registry.try_assign_plate(
                            track_id=det.track_id,
                            camera_id=cam_id,
                        )
                        # If queue didn't match, try image matching
                        if not plate:
                            x1, y1, x2, y2 = [int(v) for v in det.bbox]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            car_crop = frame[y1:y2, x1:x2]
                            if car_crop.size > 0:
                                self.vehicle_registry.try_match_by_image(
                                    car_crop=car_crop,
                                    track_id=det.track_id,
                                    camera_id=cam_id,
                                )

                # --- 2. Assign to slots ---
                assignment = pipeline.assigner.assign(detections)

                # --- 3. Update state machines ---
                all_events = []
                for slot in pipeline.slots:
                    vehicle_in_slot = slot.id in assignment.slot_vehicle_map
                    track_id = None
                    detection = None
                    if vehicle_in_slot:
                        track_id, detection = assignment.slot_vehicle_map[slot.id]

                    events = pipeline.state_machines[slot.id].update(
                        vehicle_present=vehicle_in_slot,
                        track_id=track_id,
                    )
                    # Add camera/floor context to events
                    for evt in events:
                        evt.camera_id = cam_id
                        evt.floor = pipeline.floor

                        # --- Auto-link ANPR plate when slot becomes OCCUPIED ---
                        if evt.event_type == "vehicle_parked" and self.vehicle_registry:
                            from datetime import datetime as dt
                            plate = self.vehicle_registry.try_link_to_slot(
                                slot_id=slot.id,
                                camera_id=cam_id,
                                floor=pipeline.floor,
                                track_id=track_id,
                                timestamp=dt.now(),
                            )
                            if plate:
                                evt.plate = plate
                                # Crop car image from frame for visual reference
                                if detection is not None:
                                    self._save_car_crop(frame, detection, plate, cam_id)

                    all_events.extend(events)

                # --- 4. Emit events ---
                if all_events:
                    self.event_bus.emit_batch(all_events)

                # --- 5. Periodic status summary ---
                self._frame_count += 1
                if self._frame_count % summary_interval == 0:
                    self._emit_full_summary()

                # --- 6. Visualization ---
                if show:
                    self._draw_frame(frame, pipeline, assignment, cam_id, detections)
                    grid_frames[cam_id] = frame

                    if show_camera:
                        # Single camera mode
                        if show_camera == cam_id:
                            cv2.imshow(f"PMS — {cam_id}", frame)
                    else:
                        # Per-floor grid windows
                        for floor_name, floor_cam_ids in floor_cameras.items():
                            grid = self._build_grid(
                                grid_frames, floor_cam_ids,
                                floor_cols, grid_cell_width, grid_cell_height,
                            )
                            cv2.imshow(f"Damanat PMS — {floor_name}", grid)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[INFO] 'q' pressed — exiting.")
                        break

                # --- 7. Pacing ---
                t_elapsed = time.time() - t_start
                if self._frame_count % (len(camera_configs) * 5) == 0:
                    total_elapsed = time.time() - self._start_time
                    avg_fps = self._frame_count / total_elapsed if total_elapsed > 0 else 0
                    print(f"[PERF] Total frames: {self._frame_count} | "
                          f"Camera: {cam_id} | "
                          f"Detections: {len(detections)} | "
                          f"Processing: {t_elapsed*1000:.0f}ms | "
                          f"Avg FPS: {avg_fps:.1f} | "
                          f"Active cams: {cam_manager.active_count}/{cam_manager.total_count}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted — shutting down.")

        finally:
            cam_manager.close_all()
            if show:
                cv2.destroyAllWindows()
            self.event_bus.close()
            print("[INFO] Engine stopped.")

    def run_single_camera(self, video_source: str, slots_file: str = "") -> None:
        """
        Single-camera mode (legacy/testing).

        Args:
            video_source: Video file path or RTSP URL.
            slots_file: Path to slot polygon JSON file.
        """
        print(f"[INFO] Single-camera mode: {video_source}")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_source}")
            return

        # Load slots
        sf = slots_file or self.config.slots_file
        slots = load_slots(sf) if os.path.exists(sf) else []

        pipeline = CameraPipeline(
            camera_id="SINGLE",
            floor="",
            slots=slots,
            config=self.config,
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

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream.")
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                t_start = time.time()

                detections = self.detector.detect_and_track(frame)
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
                    self.event_bus.emit_batch(all_events)

                self._frame_count += 1
                if summary_interval > 0 and self._frame_count % summary_interval == 0:
                    statuses = [sm.get_status() for sm in pipeline.state_machines.values()]
                    self.event_bus.emit_status_summary(statuses)

                if show:
                    self._draw_frame(frame, pipeline, assignment, "SINGLE", detections)
                    cv2.imshow("Parking Management System", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                t_elapsed = time.time() - t_start
                if self._frame_count % (target_fps * 5) == 0:
                    fps = 1.0 / t_elapsed if t_elapsed > 0 else 999
                    print(f"[PERF] Frame {frame_idx} | "
                          f"Detections: {len(detections)} | "
                          f"Processing: {t_elapsed*1000:.0f}ms | "
                          f"Effective FPS: {fps:.1f}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted — shutting down.")

        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
            self.event_bus.close()
            print("[INFO] Engine stopped.")

    def _emit_full_summary(self):
        """Emit status summary across all cameras."""
        all_statuses = []
        for cam_id, pipeline in self.pipelines.items():
            for sm in pipeline.state_machines.values():
                status = sm.get_status()
                status["camera_id"] = cam_id
                status["floor"] = pipeline.floor
                all_statuses.append(status)
        self.event_bus.emit_status_summary(all_statuses)

    def _draw_frame(self, frame, pipeline, assignment, cam_id, all_detections=None):
        """Draw slot polygons and detections on a frame."""
        # Draw camera label
        cv2.putText(
            frame, f"{cam_id} | {pipeline.floor}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        for slot in pipeline.slots:
            sm = pipeline.state_machines[slot.id]
            color = COLORS.get(sm.state, (128, 128, 128))

            coords = list(slot.polygon.exterior.coords)
            pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            cx, cy = int(slot.centroid_x), int(slot.centroid_y)
            label = f"{slot.id}: {sm.state.value}"
            cv2.putText(
                frame, label, (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        # Draw assigned detections (cyan bbox)
        assigned_track_ids = set()
        for det_info in assignment.slot_vehicle_map.values():
            track_id, detection = det_info
            assigned_track_ids.add(track_id)
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Show plate if known, otherwise show track ID
            label = f"ID:{track_id}"
            if self.vehicle_registry:
                plate = self.vehicle_registry.get_plate_for_any_camera(track_id)
                if plate:
                    label = f"[{plate}]"
                    # Green plate label above bbox
                    cv2.putText(
                        frame, label, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    label = f"ID:{track_id}"  # Still show track ID below

            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2,
            )
            bc_x, bc_y = detection.bottom_center
            cv2.circle(frame, (int(bc_x), int(bc_y)), 5, (0, 0, 255), -1)

        # Draw UNASSIGNED detections (magenta bbox) — helps debug slot alignment
        if all_detections:
            for det in all_detections:
                if det.track_id not in assigned_track_ids:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(
                        frame, f"ID:{det.track_id} NOT ASSIGNED",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1,
                    )
                    # Show the reference point used for slot assignment
                    bc_x, bc_y = det.bottom_center
                    cv2.circle(frame, (int(bc_x), int(bc_y)), 7, (0, 0, 255), -1)
                    cv2.putText(
                        frame, "ref",
                        (int(bc_x) + 10, int(bc_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1,
                    )

    def _save_car_crop(self, frame, detection, plate: str, cam_id: str):
        """Crop and save the detected car image for visual reference."""
        try:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            h, w = frame.shape[:2]
            # Add 10% padding
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                import os
                from datetime import datetime
                os.makedirs("vehicle_images", exist_ok=True)
                filename = f"vehicle_images/{plate}_{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, crop)
                print(f"[CROP] Saved car image: {filename}")
        except Exception as e:
            print(f"[WARN] Failed to save car crop: {e}")

    @staticmethod
    def _build_grid(
        frames: Dict[str, np.ndarray],
        camera_ids: List[str],
        cols: int = 4,
        cell_w: int = 480,
        cell_h: int = 270,
    ) -> np.ndarray:
        """
        Assemble camera frames into a single grid image.

        Args:
            frames: Dict of camera_id → annotated frame.
            camera_ids: Ordered list of all camera IDs.
            cols: Number of columns in the grid.
            cell_w: Width of each cell in pixels.
            cell_h: Height of each cell in pixels.

        Returns:
            Single numpy image with all cameras tiled.
        """
        import math
        rows = math.ceil(len(camera_ids) / cols)
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, cam_id in enumerate(camera_ids):
            row = idx // cols
            col = idx % cols
            y_off = row * cell_h
            x_off = col * cell_w

            if cam_id in frames:
                cell = cv2.resize(frames[cam_id], (cell_w, cell_h))
            else:
                # Black placeholder with camera label
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(
                    cell, f"{cam_id} — waiting...",
                    (10, cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1,
                )

            grid[y_off:y_off + cell_h, x_off:x_off + cell_w] = cell

        return grid

