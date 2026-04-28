import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.camera_manager import CameraConfig
from src.core.engine.camera_pipeline import CameraPipeline
from src.detection.tracking_manager import TrackingManager
from src.model.parkingslot import ParkingSlot as DB_ParkingSlot
from src.models.slot import load_slots
from src.models.state_machine import SlotState
from src.services.parking_service import sync_slots_from_config
from src.services.named_slot_service import get_named_slot_title, is_named_slot
from src.services.slot_status_service import log_vehicle_event, update_current_slot_plate


class ParkingEngineRuntimeMixin:
    def _build_slot_snapshot_url(self, slot_id: str) -> str:
        return f"/api/slots/{slot_id}/snapshot/live"

    def _crop_vehicle_bbox_snapshot(
        self,
        frame,
        detection=None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        padding_ratio: float = 0.12,
    ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        source_bbox = bbox
        if source_bbox is None and detection is not None:
            source_bbox = tuple(float(v) for v in detection.bbox)
        if source_bbox is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in source_bbox]
        h, w = frame.shape[:2]
        pad_x = max(12, int((x2 - x1) * padding_ratio))
        pad_y = max(12, int((y2 - y1) * padding_ratio))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        return crop if crop.size > 0 else None

    def _crop_slot_snapshot(self, frame, slot) -> Optional[np.ndarray]:
        if frame is None or slot is None or frame.size == 0:
            return None

        polygon_points = np.array(
            [[int(x), int(y)] for x, y in slot.polygon.exterior.coords[:-1]],
            dtype=np.int32,
        )
        if polygon_points.size == 0:
            return None

        h, w = frame.shape[:2]
        x, y, width, height = cv2.boundingRect(polygon_points)
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        if width <= 0 or height <= 0:
            return None

        crop = frame[y:y + height, x:x + width].copy()
        shifted_points = polygon_points - np.array([x, y])
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_points], 255)
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
        return masked_crop if masked_crop.size > 0 else None

    def _save_slot_snapshot(
        self,
        frame,
        slot,
        detection=None,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> Optional[str]:
        crop = self._crop_vehicle_bbox_snapshot(frame, detection=detection, bbox=bbox)
        if crop is None or crop.size == 0:
            crop = self._crop_slot_snapshot(frame, slot)
        if crop is None or crop.size == 0:
            return None

        try:
            os.makedirs("vehicle_images", exist_ok=True)
            filename = f"slot_{slot.id}_latest.jpg"
            full_path = os.path.join("vehicle_images", filename)
            cv2.imwrite(full_path, crop)
            return filename
        except Exception as exc:
            print(f"[WARN] Failed to save slot snapshot for {slot.id}: {exc}")
            return None

    def _safe_snapshot_token(self, value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("._")
        return cleaned or fallback

    def _save_alert_snapshot(self, crop, alert_type: str, slot_id: str, camera_id: str) -> Optional[str]:
        if crop is None or crop.size == 0:
            return None

        try:
            directory = os.path.join("vehicle_images", "alerts")
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = (
                f"{self._safe_snapshot_token(alert_type, 'alert')}_"
                f"{self._safe_snapshot_token(slot_id, 'slot')}_"
                f"{self._safe_snapshot_token(camera_id, 'camera')}_"
                f"{timestamp}.jpg"
            )
            relative_path = os.path.join("vehicle_images", "alerts", filename)
            full_path = os.path.join(directory, filename)
            if not cv2.imwrite(full_path, crop):
                raise RuntimeError("cv2.imwrite returned False")
            return relative_path
        except Exception as exc:
            print(
                f"[WARN] Failed to save alert snapshot for {alert_type} "
                f"({camera_id} / {slot_id}): {exc}"
            )
            return None

    def _persist_slot_snapshot_path(self, slot_id: str, snapshot_filename: str) -> None:
        if not self.db_manager or not snapshot_filename:
            return

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            db_slot = ParkingSlotRepository.get_by_id(session, slot_id)
            if db_slot is None:
                return
            db_slot.last_snapshot_path = snapshot_filename
            session.commit()
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to persist slot snapshot path for {slot_id}: {exc}")
        finally:
            session.close()

    def _build_camera_configs(self) -> List[CameraConfig]:
        camera_configs: List[CameraConfig] = []
        for camera in self.config.cameras:
            camera_config = CameraConfig(
                id=camera.id,
                name=camera.name,
                floor=camera.floor,
                ip=camera.ip,
                user=camera.user,
                password=camera.password,
                slots_file=camera.slots_file,
            )
            camera_config.build_rtsp_url(channel=self.config.processing.stream_channel)
            camera_configs.append(camera_config)
        return camera_configs

    def _initialize_camera_pipelines(self, camera_configs: List[CameraConfig]) -> int:
        total_slots = 0
        all_active_slot_ids = set()

        for camera_config in camera_configs:
            pipeline, parking_slots = self._build_camera_pipeline(
                camera_config,
                all_active_slot_ids,
            )
            self.pipelines[camera_config.id] = pipeline
            total_slots += pipeline.slot_count
            for slot in parking_slots:
                all_active_slot_ids.add(slot.id)

        self._purge_stale_slots(all_active_slot_ids)
        return total_slots

    def _build_camera_pipeline(self, camera_config: CameraConfig, all_active_slot_ids: set):
        all_slots = []
        roi_polygon = None

        # Reference resolution (what the slot JSONs were drawn at)
        ref_res = (
            self.config.processing.slot_ref_width,
            self.config.processing.slot_ref_height,
        )

        # Actual stream resolution — read from the camera stream if available
        actual_res = None
        if hasattr(self, "cam_manager"):
            w, h = self.cam_manager.get_resolution(camera_config.id)
            if w > 0 and h > 0:
                actual_res = (w, h)

        if camera_config.slots_file:
            if os.path.exists(camera_config.slots_file):
                all_slots, roi_polygon = load_slots(
                    camera_config.slots_file,
                    default_zone_id=camera_config.name,
                    default_zone_name=camera_config.name,
                    ref_resolution=ref_res,
                    actual_resolution=actual_res,
                )
            else:
                print(
                    f"[WARN] Slots file not found for {camera_config.id}: "
                    f"'{camera_config.slots_file}'"
                )

        parking_slots, special_zones = self._split_special_zones(all_slots)
        self.special_zones[camera_config.id] = {zone.id: zone for zone in special_zones}
        if special_zones:
            print(
                f"[INFO] {camera_config.id} has {len(special_zones)} special zone(s): "
                f"{[zone.id for zone in special_zones]}"
            )

        self._sync_slots_for_camera(camera_config, parking_slots, special_zones)
        violation_slots, initial_statuses = self._load_camera_db_state(
            parking_slots,
            all_active_slot_ids,
        )

        pipeline = CameraPipeline(
            camera_id=camera_config.id,
            floor=camera_config.floor,
            slots=parking_slots,
            config=self.config,
            violation_slots=violation_slots,
            initial_statuses=initial_statuses,
            roi_polygon=roi_polygon,
        )
        return pipeline, parking_slots

    def _sync_slots_for_camera(self, camera_config, parking_slots, special_zones) -> None:
        if not self.db_manager or not parking_slots:
            return

        session = self.db_manager.SessionLocal()
        try:
            sync_slots_from_config(
                session,
                parking_slots,
                camera_config.floor,
                default_zone_id=camera_config.name,
                default_zone_name=camera_config.name,
            )
            sync_msg = (
                f"[DB] Synced {len(parking_slots)} parking slots for {camera_config.id}"
            )
            if special_zones:
                sync_msg += f" ({len(special_zones)} special zones excluded)"
            print(sync_msg)
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to save slots to database: {exc}")
        finally:
            session.close()

    def _load_camera_db_state(self, parking_slots, all_active_slot_ids: set):
        violation_slots = set()
        initial_statuses = {}

        if not self.db_manager or not parking_slots:
            return violation_slots, initial_statuses

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            slot_ids = [slot.id for slot in parking_slots]
            db_slots = [ParkingSlotRepository.get_by_id(session, slot_id) for slot_id in slot_ids]
            for db_slot in db_slots:
                if not db_slot:
                    continue
                if db_slot.is_violation_zone:
                    violation_slots.add(db_slot.slot_id)
                initial_statuses[db_slot.slot_id] = db_slot.is_available
                all_active_slot_ids.add(db_slot.slot_id)
        except Exception as exc:
            print(f"[ERROR] Failed to load initial slot states from DB: {exc}")
        finally:
            session.close()

        return violation_slots, initial_statuses

    def _purge_stale_slots(self, all_active_slot_ids: set) -> None:
        if not self.db_manager:
            return

        session = self.db_manager.SessionLocal()
        try:
            db_all_slots = session.query(DB_ParkingSlot).all()
            stale_slots = [
                slot for slot in db_all_slots if slot.slot_id not in all_active_slot_ids
            ]
            if stale_slots:
                print(
                    f"[DB] Purging {len(stale_slots)} stale slots from database: "
                    f"{[slot.slot_id for slot in stale_slots]}"
                )
                for slot in stale_slots:
                    session.delete(slot)
                session.commit()
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to purge stale slots: {exc}")
        finally:
            session.close()

    def _build_floor_camera_groups(self, camera_configs: List[CameraConfig]) -> Dict[str, List[str]]:
        floor_cameras: Dict[str, List[str]] = {}
        for camera_config in camera_configs:
            floor_cameras.setdefault(camera_config.floor, []).append(camera_config.id)
        return floor_cameras

    def _store_passthrough_frame(self, frame, cam_id: str, grid_frames: Dict[str, np.ndarray]):
        label_frame = frame.copy()
        cv2.putText(
            label_frame,
            cam_id,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        grid_frames[cam_id] = label_frame

    def _process_special_zones(self, cam_id: str, frame, detections) -> None:
        if not self.vehicle_registry or not detections:
            return

        camera_special_zones = self.special_zones.get(cam_id, {})

        if cam_id == "CAM_01" and "Park_Entry" in camera_special_zones:
            self._process_park_entry_zone(
                cam_id,
                frame,
                detections,
                camera_special_zones["Park_Entry"],
            )

        if "Entrence" in "".join(camera_special_zones.keys()):
            confirmation_zone = next(
                (zone for zone_id, zone in camera_special_zones.items() if "Entrence" in zone_id),
                None,
            )
            if confirmation_zone:
                self._process_confirmation_zone(
                    cam_id,
                    frame,
                    detections,
                    confirmation_zone,
                )

        if cam_id not in ["CAM_01", "CAM_02"] and detections:
            if cam_id not in self._tracking_managers:
                self._tracking_managers[cam_id] = TrackingManager(cam_id)
            tracking_manager = self._tracking_managers[cam_id]
            tracking_manager.process_detections(frame, detections)
            self._process_global_tracking(cam_id, frame, detections, tracking_manager)

    def _update_slot_state(self, cam_id: str, frame, pipeline, assignment):
        all_events = []

        for slot in pipeline.slots:
            state_machine = pipeline.state_machines[slot.id]
            vehicle_in_slot = slot.id in assignment.slot_vehicle_map
            track_id = None
            detection = None
            if vehicle_in_slot:
                track_id, detection = assignment.slot_vehicle_map[slot.id]
                if detection is not None:
                    state_machine.latest_detection_bbox = tuple(
                        float(v) for v in detection.bbox
                    )
            elif state_machine.state == SlotState.VACANT:
                state_machine.latest_detection_bbox = None

            events = state_machine.update(
                vehicle_present=vehicle_in_slot,
                track_id=track_id,
            )

            for event in events:
                event.camera_id = cam_id
                event.floor = pipeline.floor
                event.slot_name = slot.label
                event.zone_id = slot.zone_id
                event.zone_name = slot.zone_name

                if event.event_type == "vehicle_parked":
                    snapshot_filename = self._save_slot_snapshot(
                        frame,
                        slot,
                        detection=detection,
                        bbox=state_machine.latest_detection_bbox,
                    )
                    if snapshot_filename:
                        self._persist_slot_snapshot_path(slot.id, snapshot_filename)

                if event.event_type == "vehicle_parked" and self.vehicle_registry:
                    # Attempt to get plate first to save crop with correct filename
                    plate = self.vehicle_registry.get_plate_for_track(cam_id, track_id)
                    snapshot_path = None
                    if plate and detection is not None:
                        snapshot_path = self._save_car_crop(frame, detection, plate, cam_id)

                    linked_plate = self.vehicle_registry.try_link_to_slot(
                        slot_id=slot.id,
                        slot_name=slot.label,
                        zone_id=slot.zone_id,
                        zone_name=slot.zone_name,
                        camera_id=cam_id,
                        floor=pipeline.floor,
                        track_id=track_id,
                        timestamp=datetime.now(),
                        snapshot_path=snapshot_path,
                    )
                    if linked_plate:
                        event.plate_number = linked_plate
                        location = self.vehicle_registry.get_plate_location(linked_plate)
                        if location:
                            event.snapshot_url = location.get("snapshot_url", "")
                        state_machine.bind_identity(
                            linked_plate,
                            self._build_slot_snapshot_url(slot.id),
                        )
                    else:
                        state_machine.bind_identity(
                            None,
                            self._build_slot_snapshot_url(slot.id),
                        )
                elif event.event_type == "slot_vacant" and self.vehicle_registry:
                    plate = self.vehicle_registry.unlink_slot(slot.id)
                    if plate:
                        event.plate_number = plate

            if (
                self.vehicle_registry
                and vehicle_in_slot
                and pipeline.state_machines[slot.id].state
                in (SlotState.OCCUPIED, SlotState.LEAVING)
            ):
                previous_plate = pipeline.state_machines[slot.id].plate_number
                plate = self.vehicle_registry.try_link_to_slot(
                    slot_id=slot.id,
                    slot_name=slot.label,
                    zone_id=slot.zone_id,
                    zone_name=slot.zone_name,
                    camera_id=cam_id,
                    floor=pipeline.floor,
                    track_id=track_id,
                    timestamp=datetime.now(),
                )
                if plate:
                    state_machine.bind_identity(
                        plate,
                        self._build_slot_snapshot_url(slot.id),
                    )
                    if self.db_manager and plate != previous_plate:
                        self._persist_late_slot_plate(slot.id, plate, cam_id)

            all_events.extend(events)

        return all_events

    def _persist_late_slot_plate(self, slot_id: str, plate: str, camera_id: str) -> None:
        session = self.db_manager.SessionLocal()
        try:
            update_current_slot_plate(
                session,
                slot_id=slot_id,
                plate=plate,
                camera_id=camera_id,
            )
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to persist late slot plate for {slot_id}: {exc}")
        finally:
            session.close()

    def _filter_violation_events(self, frame, assignment, cam_id: str, events):
        final_events = []
        for event in events:
            slot_state_machine = self.pipelines[cam_id].state_machines.get(event.slot_id)
            is_named_reserved_slot = is_named_slot(event.slot_id)
            if (
                not slot_state_machine
                or (not slot_state_machine.is_violation_zone and not is_named_reserved_slot)
            ):
                final_events.append(event)
                continue

            if event.event_type != "vehicle_parked":
                final_events.append(event)
                continue

            _, detection = assignment.slot_vehicle_map.get(event.slot_id, (None, None))
            if not detection:
                final_events.append(event)
                continue

            crop = self._crop_vehicle_bbox_snapshot(frame, detection=detection)
            if crop is None:
                crop = np.empty((0, 0, 3), dtype=np.uint8)

            now_ts = time.time()
            self._recent_violators = [
                violator
                for violator in self._recent_violators
                if now_ts - violator["timestamp"] < self._violation_history_limit
            ]

            is_duplicate = False
            if crop.size > 0 and self.vehicle_registry:
                for violator in self._recent_violators:
                    score = self.vehicle_registry.matcher.compare(crop, violator["crop"])
                    if score > self._violation_match_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                alert_type = self._get_slot_alert_type(event.slot_id, getattr(event, "plate_number", ""))
                if alert_type is None:
                    final_events.append(event)
                    continue
                event.event_type = alert_type
                event.is_alert = True
                event.severity = "critical"
                event.snapshot_path = self._save_alert_snapshot(
                    crop,
                    alert_type=alert_type,
                    slot_id=event.slot_id,
                    camera_id=cam_id,
                )
                if crop.size > 0:
                    self._recent_violators.append(
                        {"crop": crop.copy(), "timestamp": now_ts, "camera_id": cam_id}
                    )
                final_events.append(event)
                print(f"[ALERT] {alert_type.replace('_', ' ').title()}! {cam_id} | Slot:{event.slot_id}")
            else:
                final_events.append(event)

        return final_events

    def _get_slot_alert_type(self, slot_id: str, plate_number: str) -> str | None:
        named_slot_title = get_named_slot_title(slot_id)
        if named_slot_title:
            if self._is_named_slot_vehicle_allowed(plate_number, named_slot_title):
                return None
            return "named_slot_violation"

        return (
            "vehicle_violation"
            if "violation" in slot_id.lower()
            else "vehicle_intrusion"
        )

    def _is_named_slot_vehicle_allowed(self, plate_number: str, expected_title: str) -> bool:
        if not self.db_manager or not plate_number or not expected_title:
            return False

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import VehicleRepository

            vehicle = VehicleRepository.get_by_plate(session, plate_number)
            return bool(vehicle and vehicle.title == expected_title)
        except Exception as exc:
            print(
                f"[ERROR] Failed to validate named-slot ownership for plate "
                f"{plate_number}: {exc}"
            )
            return False
        finally:
            session.close()

    def _persist_final_events(self, events) -> None:
        if not events:
            return

        if not self.db_manager:
            self.event_bus.emit_batch(events)
            return

        session = self.db_manager.SessionLocal()
        try:
            for event in events:
                if event.event_type in (
                    "vehicle_parked",
                    "slot_vacant",
                    "vehicle_violation",
                    "vehicle_intrusion",
                    "named_slot_violation",
                ):
                    is_parked = event.event_type in (
                        "vehicle_parked",
                        "vehicle_violation",
                        "vehicle_intrusion",
                        "named_slot_violation",
                    )
                    plate = getattr(event, "plate_number", None)
                    # Capture the alert_id from log_vehicle_event
                    _, db_alert_id = log_vehicle_event(
                        session,
                        event.slot_id,
                        plate,
                        is_parked,
                        camera_id=event.camera_id,
                        severity=event.severity,
                        snapshot_path=getattr(event, "snapshot_path", None),
                    )
                    # Enrich the event with the database-generated ID
                    if db_alert_id:
                        event.alert_id = db_alert_id

            # Emit AFTER updating with database IDs
            self.event_bus.emit_batch(events)

        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to update slot DB status: {exc}")
        finally:
            session.close()

    def _show_multi_camera_output(
        self,
        cam_id: str,
        frame,
        pipeline,
        assignment,
        detections,
        grid_frames,
        floor_cameras,
        show_camera,
        floor_cols: int,
        grid_cell_width: int,
        grid_cell_height: int,
    ) -> bool:
        self._draw_frame(frame, pipeline, assignment, cam_id, detections)
        grid_frames[cam_id] = frame

        if show_camera:
            if show_camera == cam_id:
                cv2.imshow(f"PMS - {cam_id}", frame)
        else:
            for floor_name, floor_cam_ids in floor_cameras.items():
                grid = self._build_grid(
                    grid_frames,
                    floor_cam_ids,
                    floor_cols,
                    grid_cell_width,
                    grid_cell_height,
                )
                cv2.imshow(f"Damanat PMS - {floor_name}", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] 'q' pressed - exiting.")
            return True
        return False
