"""
engine.py — Main pipeline orchestrator (multi-camera support).

Supports two modes:
  1. Multi-camera: round-robin processing of 14 cameras.
  2. Single-camera: legacy mode for testing with one stream.

Pipeline per frame:
  Frame Grabber → Detector/Tracker → Slot Assigner → State Machines → Event Bus
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon

from src.config import AppConfig
from src.models.slot import ParkingSlot, load_slots
from src.models.state_machine import SlotStateMachine, SlotState
from src.detection.tracker import TrackedDetector
from src.core.slot_assigner import SlotAssigner
from src.events.event_bus import EventBus
from src.camera_manager import CameraManager, CameraConfig
from src.model.parkingslot import ParkingSlot as DB_ParkingSlot
from src.services.parking_service import sync_slots_from_config
from src.services.slot_status_service import log_vehicle_event

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
        violation_slots: set = None,
        roi_polygon: Optional[Polygon] = None,
    ):
        self.camera_id = camera_id
        self.floor = floor
        self.slots = slots
        self.roi_polygon = roi_polygon
        self._mask = None # Cached binary mask

        # Per-slot state machines — pass is_violation_zone from DB
        self.state_machines: Dict[str, SlotStateMachine] = {}
        for slot in slots:
            is_restricted = (violation_slots is not None) and (slot.id in violation_slots)
            self.state_machines[slot.id] = SlotStateMachine(
                slot_id=slot.id,
                confirm_enter_frames=config.state_machine.confirm_enter_frames,
                confirm_leave_frames=config.state_machine.confirm_leave_frames,
                is_violation_zone=is_restricted,
            )

        # Slot assigner for this camera's polygons
        self.assigner = SlotAssigner(slots=slots, config=config.assigner)

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the ROI mask to the frame, blacking out areas outside the ROI.
        """
        if self.roi_polygon is None:
            return frame

        h, w = frame.shape[:2]
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(self._mask, [pts], 255)

        return cv2.bitwise_and(frame, frame, mask=self._mask)

    @property
    def slot_count(self) -> int:
        return len(self.slots)


class ParkingEngine:
    """
    Main orchestrator for the parking management system.

    Multi-camera mode: round-robin across all cameras.
    Single-camera mode: processes one video source.
    """

    def __init__(self, config: AppConfig, vehicle_registry=None, db_manager=None):
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
        self.cam_manager = CameraManager(camera_configs)
        opened = self.cam_manager.open_all()

        if opened == 0:
            print("[ERROR] No cameras could be opened. Exiting.")
            return

        # Load per-camera slot polygons and create pipelines
        total_slots = 0
        for cc in camera_configs:
            all_slots = []
            roi_polygon = None
            if cc.slots_file and os.path.exists(cc.slots_file):
                all_slots, roi_polygon = load_slots(cc.slots_file)
            
            # [NEW] Separate tracking zones from parking slots
            parking_slots, special_zones = self._split_special_zones(all_slots)
            self.special_zones[cc.id] = {z.id: z for z in special_zones}

            if self.db_manager and parking_slots:
                session = self.db_manager.SessionLocal()
                try:
                    sync_slots_from_config(session, parking_slots, cc.floor)
                    print(f"[DB] Synced {len(parking_slots)} slots for {cc.id}")
                except Exception as e:
                    session.rollback()
                    print(f"[ERROR] Failed to save slots to database: {e}")
                finally:
                    session.close()

            # Load violation zone flags from DB for this camera's slots
            violation_slots = set()
            if self.db_manager and parking_slots:
                session = self.db_manager.SessionLocal()
                try:
                    slot_ids = [s.id for s in parking_slots]
                    from src.repositories import ParkingSlotRepository
                    db_slots = [ParkingSlotRepository.get_by_id(session, sid) for sid in slot_ids]
                    violation_slots = {
                        db_s.slot_id for db_s in db_slots
                        if db_s and db_s.is_violation_zone
                    }
                except Exception as e:
                    print(f"[ERROR] Failed to load violation zone flags from DB: {e}")
                finally:
                    session.close()

            pipeline = CameraPipeline(
                camera_id=cc.id,
                floor=cc.floor,
                slots=parking_slots,  # Only standard and violation slots
                config=self.config,
                violation_slots=violation_slots,
                roi_polygon=roi_polygon,
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
                cam_id, frame = self.cam_manager.next_frame()

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
                # Apply ROI mask to focus detection on relevant areas (e.g. ignore street)
                detection_frame = pipeline.apply_roi_mask(frame)
                detections = self.detector.detect_and_track(detection_frame)

                # --- 1.2. Process Special Tracking Zones ---
                if self.vehicle_registry and detections:
                    cam_special_zones = self.special_zones.get(cam_id, {})
                    
                    # CAM_01 -> Park_Entry
                    if cam_id == "CAM_01" and "Park_Entry" in cam_special_zones:
                        self._process_park_entry_zone(cam_id, frame, detections, cam_special_zones["Park_Entry"])
                    
                    # CAM_03 (B1) or confirmation cams (e.g. CAM_09-14 for B2)
                    if "Entrence" in "".join(cam_special_zones.keys()): # Matches B1_Entrence, B2_Entrence
                        # Find the confirmation zone for this camera
                        conf_zone = next((z for k, z in cam_special_zones.items() if "Entrence" in k), None)
                        if conf_zone:
                            self._process_confirmation_zone(cam_id, frame, detections, conf_zone)

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
                            # 1. Try resolving plate from already confirmed tracks
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
                    final_events = []
                    for evt in all_events:
                        slot_sm = pipeline.state_machines.get(evt.slot_id)
                        if slot_sm and slot_sm.is_violation_zone:
                            if evt.event_type == "vehicle_parked":
                                # Extract crop for visual matching
                                track_id, detection = assignment.slot_vehicle_map.get(evt.slot_id, (None, None))
                                if detection:
                                    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                                    h, w = frame.shape[:2]
                                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                                    
                                    # Clean up old violators
                                    now_ts = time.time()
                                    self._recent_violators = [v for v in self._recent_violators 
                                                             if now_ts - v['timestamp'] < self._violation_history_limit]
                                    
                                    # Check visual similarity
                                    is_duplicate = False
                                    if crop.size > 0:
                                        for v in self._recent_violators:
                                            # Using the existing matcher in vehicle_registry
                                            score = self.vehicle_registry.matcher.compare(crop, v['crop'])
                                            if score > self._violation_match_threshold:
                                                is_duplicate = True
                                                break
                                    
                                    if not is_duplicate:
                                        # Distinguish violation zones vs intrusion (restricted) zones
                                        alert_type = "vehicle_violation" if "violation" in evt.slot_id.lower() else "vehicle_intrusion"
                                        evt.event_type = alert_type
                                        evt.is_alert = True
                                        evt.severity = "critical"
                                        if crop.size > 0:
                                            self._recent_violators.append({
                                                'crop': crop.copy(),
                                                'timestamp': now_ts,
                                                'camera_id': cam_id
                                            })
                                        final_events.append(evt)
                                        print(f"[ALERT] {alert_type.replace('_', ' ').title()}! {cam_id} | Slot:{evt.slot_id}")
                                    else:
                                        # Duplicate of a recent violator
                                        final_events.append(evt)
                                else:
                                    final_events.append(evt)
                            else:
                                final_events.append(evt)
                        else:
                            final_events.append(evt)

                    if final_events:
                        self.event_bus.emit_batch(final_events)
                        if self.db_manager:
                            session = self.db_manager.SessionLocal()
                            try:
                                for evt in final_events:
                                    if evt.event_type in ("vehicle_parked", "slot_vacant", "vehicle_violation"):
                                        is_parked = (evt.event_type in ("vehicle_parked", "vehicle_violation"))
                                        plate = getattr(evt, "plate", None) 
                                        log_vehicle_event(session, evt.slot_id, plate, is_parked)
                            except Exception as e:
                                session.rollback()
                                print(f"[ERROR] Failed to update slot DB status: {e}")
                            finally:
                                session.close()

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
                    #print(f"[PERF] Total frames: {self._frame_count} | "
                    #      f"Camera: {cam_id} | "
                    #      f"Detections: {len(detections)} | "
                    #      f"Processing: {t_elapsed*1000:.0f}ms | "
                    #      f"Avg FPS: {avg_fps:.1f} | "
                    #      f"Active cams: {cam_manager.active_count}/{cam_manager.total_count}")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted — shutting down.")

        finally:
            self.cam_manager.close_all()
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
        slots, roi_polygon = load_slots(sf) if os.path.exists(sf) else ([], None)

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
                    for evt in all_events:
                        slot_sm = pipeline.state_machines.get(evt.slot_id)
                        if slot_sm and slot_sm.is_violation_zone:
                            # In single camera mode floor might be empty, so check zone only
                            if evt.event_type == "vehicle_parked":
                                now_ts = time.time()
                                if now_ts - self._last_violation_alert_time >= self._violation_cooldown_seconds:
                                    # Distinguish violation zones vs intrusion (restricted) zones
                                    alert_type = "vehicle_violation" if "violation" in evt.slot_id.lower() else "vehicle_intrusion"
                                    evt.event_type = alert_type
                                    evt.is_alert = True
                                    evt.severity = "critical"
                                    self._last_violation_alert_time = now_ts
                                    final_events.append(evt)
                                    print(f"[ALERT] {alert_type.replace('_', ' ').title()} in {evt.slot_id}!")
                                else:
                                    final_events.append(evt)
                            else:
                                final_events.append(evt)
                        else:
                            final_events.append(evt)
                    
                    if final_events:
                        self.event_bus.emit_batch(final_events)

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
                    #print(f"[PERF] Frame {frame_idx} | "
                    #      f"Detections: {len(detections)} | "
                    #      f"Processing: {t_elapsed*1000:.0f}ms | "
                    #      f"Effective FPS: {fps:.1f}")

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
        # Draw ROI border if it exists (but don't mask the whole frame visually for humans)
        if pipeline.roi_polygon is not None:
            pts = np.array(pipeline.roi_polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            # Draw thin yellow line for the detection boundary
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                frame, "DETECTION ZONE", (pts[0][0][0], pts[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

        # Draw camera label
        cv2.putText(
            frame, f"{cam_id} | {pipeline.floor}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        for slot in pipeline.slots:
            sm = pipeline.state_machines[slot.id]
            
            # Violation visualization
            is_violation = sm.is_violation_zone
            if is_violation and sm.is_occupied:
                # Flickering red for violation
                if int(time.time() * 2) % 2 == 0:
                    color = (0, 0, 255) # Bright red
                else:
                    color = (0, 0, 150) # Dark red
                thickness = 4
            else:
                color = COLORS.get(sm.state, (128, 128, 128))
                thickness = 2

            coords = list(slot.polygon.exterior.coords)
            pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

            cx, cy = int(slot.centroid_x), int(slot.centroid_y)
            if is_violation and sm.is_occupied:
                label = "!!! VIOLATION !!!"
            else:
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
                plate = self.vehicle_registry.get_plate_for_track(cam_id, track_id)
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

        # Draw unassigned detections (red bbox) for debugging
        if hasattr(assignment, 'unassigned'):
            for detection in assignment.unassigned:
                x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"ID:{detection.track_id} (Unassigned)"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                )
                bc_x, bc_y = detection.bottom_center
                cv2.circle(frame, (int(bc_x), int(bc_y)), 5, (0, 0, 255), -1)

    # ------------------------------------------------------------------
    # Zone Processing Helpers
    # ------------------------------------------------------------------

    def _split_special_zones(self, slots: List[ParkingSlot]) -> Tuple[List[ParkingSlot], List[ParkingSlot]]:
        """Split polygons into standard parking slots and special tracking zones."""
        parking = []
        special = []
        for s in slots:
            if "Park_Entry" in s.id or "Entrence" in s.id:
                special.append(s)
            else:
                parking.append(s)
        return parking, special

    def _detection_in_zone(self, detection, zone_slot: ParkingSlot) -> bool:
        """Check if a detection's bottom-center is inside a zone polygon."""
        bc_x, bc_y = detection.bottom_center
        return zone_slot.polygon.contains(Point(bc_x, bc_y))

    def _crop_detection(self, frame: np.ndarray, detection) -> Optional[np.ndarray]:
        """Safely crop the vehicle detection from the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _score_snapshot_quality(self, detection) -> float:
        """Score snapshot quality based on detection area."""
        x1, y1, x2, y2 = detection.bbox
        return (x2 - x1) * (y2 - y1)

    def _process_park_entry_zone(self, cam_id: str, frame: np.ndarray, detections: List, zone: ParkingSlot):
        """Handle CAM_01 logic for Park_Entry."""
        currently_inside = set()
        
        for det in detections:
            if det.track_id == -1:
                continue
            
            if self._detection_in_zone(det, zone):
                currently_inside.add(det.track_id)
                
                # 1. Open candidate if first time seen
                candidate_id = self._park_entry_track_to_candidate.get(det.track_id)
                if not candidate_id:
                    candle = self.vehicle_registry.open_park_entry_candidate(cam_id, det.track_id)
                    candidate_id = candle.candidate_id
                    self._park_entry_track_to_candidate[det.track_id] = candidate_id
                
                # 2. Update snapshot with quality check
                crop = self._crop_detection(frame, det)
                if crop is not None:
                    quality = self._score_snapshot_quality(det)
                    # Feature vector is lazy-loaded in matcher if needed, passing None for now
                    self.vehicle_registry.update_park_entry_candidate_snapshot(
                        candidate_id, crop, quality
                    )
                
                # 3. Aggressively try binding to ANPR
                self.vehicle_registry.bind_next_pending_anpr_to_candidate(candidate_id)

        # Cleanup maps for tracks that left the zone
        last_track_ids = self._tracks_inside_zones.get((cam_id, zone.id), set())
        left_zone = last_track_ids - currently_inside
        for tid in left_zone:
            # Clean up the candidate mapping to prevent ID reuse poisoning
            if tid in self._park_entry_track_to_candidate:
                del self._park_entry_track_to_candidate[tid]
            
        self._tracks_inside_zones[(cam_id, zone.id)] = currently_inside

    def _process_confirmation_zone(self, cam_id: str, frame: np.ndarray, detections: List, zone: ParkingSlot):
        """Handle CAM_03 and other confirm zones for identity matching."""
        for det in detections:
            if det.track_id == -1:
                continue
            
            if self._detection_in_zone(det, zone):
                # 1. Check if already session-backed
                if self.vehicle_registry.get_plate_for_track(cam_id, det.track_id):
                    continue
                
                # 2. Try visual match vs provisional candidates
                crop = self._crop_detection(frame, det)
                if crop is not None:
                    self.vehicle_registry.confirm_at_b1_entrance(cam_id, det.track_id, crop)

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

