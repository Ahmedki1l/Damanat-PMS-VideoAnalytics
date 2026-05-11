import logging
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
from src.models.state_machine import SlotState
from src.services.parking_service import (
    bootstrap_camera_slots_from_json,
    load_camera_slots,
)
from src.services.slot_status_service import log_vehicle_event, update_current_slot_plate


logger = logging.getLogger(__name__)


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
            base_dir = self.config.output.snapshot_base_dir
            os.makedirs(base_dir, exist_ok=True)
            filename = f"slot_{slot.id}_latest.jpg"
            full_path = os.path.join(base_dir, filename)
            cv2.imwrite(full_path, crop)
            # Return the externally-reachable URL so the alerts table /
            # parking_slots.last_snapshot_path / Gateway responses all
            # carry full URLs instead of bare relative filenames that
            # frontends can't render directly.
            return self._build_snapshot_url(filename)
        except Exception as exc:
            print(f"[WARN] Failed to save slot snapshot for {slot.id}: {exc}")
            return None

    def _safe_snapshot_token(self, value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("._")
        return cleaned or fallback

    def _build_snapshot_url(self, relative_path: str) -> str:
        """Turn a snapshot_base_dir-relative path (e.g. 'alerts/foo.jpg' or
        'slot_B11_CFO_latest.jpg') into the externally-reachable URL served
        by api.py's `/pms-video-analytics/snapshots/{filepath:path}` route.

        Reads `output.public_base_url`, `output.snapshot_url_prefix`, and
        `output.gateway_path_prefix` from config (same precedence as
        VehicleRegistryQueryMixin._get_snapshot_url) so the URL shape stays
        consistent across every consumer of VA snapshots — alerts, vehicle
        registry queries, and the API's own slot views.

        When `public_base_url` is empty, returns a site-relative URL
        (legacy behaviour, lets dev environments without an external host
        keep working).
        """
        if not relative_path:
            return ""
        rel = relative_path.replace(os.sep, "/").lstrip("/")
        out = self.config.output
        base = (getattr(out, "public_base_url", "") or "").rstrip("/")
        gateway = (getattr(out, "gateway_path_prefix", "") or "").strip("/")
        prefix = (getattr(out, "snapshot_url_prefix", "snapshots") or "snapshots").strip("/")
        path_parts = "/".join(part for part in [gateway, prefix, rel] if part)
        return f"{base}/{path_parts}" if base else f"/{path_parts}"

    def _save_alert_snapshot(self, crop, alert_type: str, slot_id: str, camera_id: str,
                             fallback_frame=None) -> Optional[str]:
        """Save the alert evidence image to disk, return its public URL.

        Prefers the vehicle-bbox `crop` (tighter framing for the operator).
        If the crop is empty/None — which happens when the vehicle's bbox is
        missing for the frame the alert fired on (Bug #v0-1 in Version 0:
        intrusion alerts landing without an evidence snapshot) — falls back
        to `fallback_frame` (the full camera frame). The full frame is wider
        but always meaningful, so the alert never lands snapshot-less.

        The returned value is the externally-reachable URL (e.g.
        ``http://localhost:8000/pms-video-analytics/snapshots/alerts/<file>.jpg``)
        built from `output.public_base_url` + `output.snapshot_url_prefix`,
        so callers can persist it directly to `alerts.snapshot_path` and
        downstream consumers (Gateway, frontend) render it without further
        rewriting. When `public_base_url` is empty, returns a site-relative
        URL — legacy behaviour for dev environments.

        Returns None only if BOTH crop and fallback_frame are empty/None, or
        if cv2.imwrite fails on disk.
        """
        # Decide which image to save: crop preferred, full frame as fallback.
        image = None
        if crop is not None and crop.size > 0:
            image = crop
        elif fallback_frame is not None and fallback_frame.size > 0:
            image = fallback_frame
            print(
                f"[INFO] Alert snapshot fallback: full frame for "
                f"{alert_type} ({camera_id} / {slot_id}); vehicle bbox crop was empty."
            )
        else:
            print(
                f"[WARN] Alert snapshot UNAVAILABLE: no crop and no fallback frame for "
                f"{alert_type} ({camera_id} / {slot_id})."
            )
            return None

        try:
            base_dir = self.config.output.snapshot_base_dir
            directory = os.path.join(base_dir, "alerts")
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = (
                f"{self._safe_snapshot_token(alert_type, 'alert')}_"
                f"{self._safe_snapshot_token(slot_id, 'slot')}_"
                f"{self._safe_snapshot_token(camera_id, 'camera')}_"
                f"{timestamp}.jpg"
            )
            relative_path = os.path.join("alerts", filename)
            full_path = os.path.join(directory, filename)
            if not cv2.imwrite(full_path, image):
                raise RuntimeError("cv2.imwrite returned False")
            # Return the externally-reachable URL so consumers can render
            # the snapshot directly without rewriting the path. The same
            # file is still available on disk under snapshot_base_dir/alerts.
            return self._build_snapshot_url(relative_path)
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

    # In-process rate gate: at most one vehicles-row write per plate per
    # _PRESENCE_MIN_INTERVAL_S seconds. Without this, the per-frame loop
    # would issue an UPDATE every camera tick (~14/s across all cameras)
    # for every actively-tracked plate, which is pointless DB churn.
    _PRESENCE_MIN_INTERVAL_S = 5.0

    # Exit-janitor cadence — how often the engine sweeps the registry to
    # purge plates whose parking_sessions row has been closed by PMS-AI
    # without VA seeing the corresponding ANPR exit event.
    _EXIT_JANITOR_INTERVAL_S = 30.0

    def _exit_janitor_tick(self) -> None:
        """Once per `_EXIT_JANITOR_INTERVAL_S`, find plates VA still has in
        memory whose latest parking_sessions row is closed (per PMS-AI), and
        call vehicle_registry._handle_exit(plate, now) to purge the in-memory
        tracking state. Catches missed CAM-EXIT ANPR events and stops VA from
        re-id-matching cars that have already left the garage.

        Called from the main loop (next to _cleanup_stale_data). The gate
        ensures it doesn't run on every frame.
        """
        if not self.db_manager or not self.vehicle_registry:
            return
        now_ts = time.time()
        last = getattr(self, "_exit_janitor_last_run_at", 0.0)
        if now_ts - last < self._EXIT_JANITOR_INTERVAL_S:
            return
        self._exit_janitor_last_run_at = now_ts

        # Snapshot the plates the registry currently holds. Done under the
        # registry's lock-protected accessor (or via a stable copy) so we
        # don't iterate a dict that another thread is mutating.
        try:
            tracked_plates = self.vehicle_registry.get_tracked_plates()
        except AttributeError:
            # Older registry without the helper — fall back to _parked +
            # session map plates.
            tracked_plates = set()
            for sess in getattr(self.vehicle_registry, "_parked", {}).values():
                if sess.plate:
                    tracked_plates.add(sess.plate)
            for sess in getattr(self.vehicle_registry, "_sessions", {}).values():
                if sess.plate:
                    tracked_plates.add(sess.plate)
        if not tracked_plates:
            return

        try:
            from sqlalchemy import bindparam, text as _text

            session = self.db_manager.SessionLocal()
            try:
                # One round-trip: get the latest status per plate. Plates
                # with no rows aren't in the result — those are fine, they
                # haven't entered yet. `expanding=True` lets SQLAlchemy
                # turn the IN binding into a parameterized list at execute time.
                stmt = _text(
                    "SELECT plate_number, status FROM ("
                    "  SELECT plate_number, status, "
                    "         ROW_NUMBER() OVER (PARTITION BY plate_number ORDER BY entry_time DESC) AS rn "
                    "  FROM dbo.parking_sessions "
                    "  WHERE plate_number IN :plates"
                    ") t WHERE rn = 1"
                ).bindparams(bindparam("plates", expanding=True))
                rows = session.execute(stmt, {"plates": list(tracked_plates)}).fetchall()
            finally:
                session.close()
        except Exception as exc:
            logger.warning("[exit_janitor] DB probe failed: %r", exc)
            return

        closed = [r[0] for r in rows if r[1] == "closed"]
        if not closed:
            return

        purged_at = datetime.now()
        for plate in closed:
            try:
                self.vehicle_registry._handle_exit(plate, purged_at)
                logger.info(
                    "[exit_janitor] purged in-memory state for plate=%s "
                    "(parking_sessions.status=closed)",
                    plate,
                )
            except Exception as exc:
                logger.warning("[exit_janitor] _handle_exit(%s) failed: %r", plate, exc)

    def update_vehicle_presence(
        self,
        plate: str,
        *,
        floor: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> None:
        """Write `vehicles.floor` / `floor_id` AND mirror the same value onto
        the OPEN `parking_sessions` row so the Gateway's entry-exit endpoint
        (which JOINs parking_sessions for floor / floor_id / parked_at) shows
        the live floor even before a slot is bound.

        Called from every track-confirmation path (Park_Entry capture,
        B1_Entrence confirmation, slot bind, plus the per-frame
        TrackingManager observation). The slot-bind path is also written by
        PMS-AI's parking_session_service.bind_slot — both sources are
        idempotent so they can race safely.

        Rate-gated to once per plate per ~5s so per-frame callers don't
        hammer the DB. Safe to call with floor=None — only updates fields
        that are actually known.

        `parked_at` semantics: set ONLY when currently NULL on the open
        session (i.e. the session has no slot-bind timestamp yet). This
        marks "first floor observation" for sessions that arrive at the
        slot detection cameras before any slot bind, while preserving the
        slot-bind timestamp once bind_slot has set it.
        """
        if not plate or not self.db_manager:
            return

        # Lazy-init the per-plate gate map.
        gate = getattr(self, "_presence_last_write_at", None)
        if gate is None:
            gate = {}
            self._presence_last_write_at = gate
        now_ts = time.time()
        last = gate.get(plate, 0.0)
        if now_ts - last < self._PRESENCE_MIN_INTERVAL_S:
            return
        gate[plate] = now_ts

        session = self.db_manager.SessionLocal()
        try:
            from sqlalchemy import text as _text

            # Check if vehicle exists (raw SQL — VA has no Vehicle ORM model;
            # the vehicles table is owned by the Gateway's schema).
            row = session.execute(
                _text("SELECT id, floor FROM dbo.vehicles WHERE plate_number = :p"),
                {"p": plate},
            ).first()
            if row is None:
                # No registry row yet — VA's Park_Entry pipeline will create
                # it once ANPR matches. Don't create a partial row here.
                return

            vehicle_id, current_floor = row
            if not floor or current_floor == floor:
                return

            # Resolve floor_id from the floors lookup table once; reused for
            # both the vehicles UPDATE and the parking_sessions UPDATE.
            fid = session.execute(
                _text("SELECT id FROM dbo.floors WHERE name = :n"),
                {"n": floor},
            ).scalar()

            # 1. vehicles row (canonical "where is the car right now").
            session.execute(
                _text("UPDATE dbo.vehicles SET floor = :f, floor_id = :fid WHERE id = :vid"),
                {"f": floor, "fid": fid, "vid": vehicle_id},
            )

            # 2. open parking_sessions row (drives the Gateway's entry-exit
            #    response shape via JOIN). Only one open session per plate
            #    by invariant (UC1 dedup + close_session). Update the latest
            #    open row; no-op if the plate isn't currently inside.
            #    parked_at is COALESCE so a slot-bind timestamp from
            #    parking_session_service.bind_slot wins; we only fill it in
            #    the gap where VA observed the car on a floor before the
            #    slot detection camera reported a bind.
            # Bind a Python facility-local naive datetime instead of using
            # MSSQL's SYSUTCDATETIME() — the DB convention is naive facility-
            # local (operator wall clock), and SYSUTCDATETIME() returns UTC
            # which would land 3h behind the wall clock.
            from src.utils.datetime_helper import facility_now_naive
            now_naive = facility_now_naive()
            session.execute(
                _text(
                    "UPDATE dbo.parking_sessions "
                    "SET floor = :f, floor_id = :fid, "
                    "    parked_at = COALESCE(parked_at, :now), "
                    "    updated_at = :now "
                    "WHERE plate_number = :p AND status = 'open'"
                ),
                {"f": floor, "fid": fid, "p": plate, "now": now_naive},
            )

            session.commit()
            logger.debug(
                "[presence] plate=%s floor=%s camera=%s (vehicles + parking_sessions)",
                plate, floor, camera_id,
            )
        except Exception as exc:
            session.rollback()
            logger.warning("[presence] write failed for plate=%s: %r", plate, exc)
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

        return total_slots

    def _build_camera_pipeline(self, camera_config: CameraConfig, all_active_slot_ids: set):
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

        parking_slots = []
        special_zones = []
        roi_polygon = None
        self._bootstrap_camera_slots_if_needed(camera_config)
        if self.db_manager:
            session = self.db_manager.SessionLocal()
            try:
                parking_slots, special_zones, roi_polygon = load_camera_slots(
                    session,
                    camera_id=camera_config.id,
                    ref_resolution=ref_res,
                    actual_resolution=actual_res,
                )
            except Exception as exc:
                print(f"[ERROR] Failed to load slots from database for {camera_config.id}: {exc}")
            finally:
                session.close()

        self.special_zones[camera_config.id] = {zone.id: zone for zone in special_zones}
        if special_zones:
            print(
                f"[INFO] {camera_config.id} has {len(special_zones)} special zone(s): "
                f"{[zone.id for zone in special_zones]}"
            )

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

    def _bootstrap_camera_slots_if_needed(self, camera_config) -> None:
        if not self.db_manager:
            return

        session = self.db_manager.SessionLocal()
        try:
            from src.repositories import ParkingSlotRepository

            existing_rows = ParkingSlotRepository.filter_camera_slots(session, camera_config.id)
            if existing_rows:
                return

            migrated = bootstrap_camera_slots_from_json(
                session,
                camera_id=camera_config.id,
                floor=camera_config.floor,
                slots_file=camera_config.slots_file,
                default_zone_id=camera_config.name,
                default_zone_name=camera_config.name,
            )
            if migrated:
                print(
                    f"[DB] Bootstrapped slot definitions for {camera_config.id} "
                    f"from legacy JSON '{camera_config.slots_file}'"
                )
        except Exception as exc:
            session.rollback()
            print(f"[ERROR] Failed to bootstrap slots for {camera_config.id}: {exc}")
        finally:
            session.close()

    def _load_camera_db_state(self, parking_slots, all_active_slot_ids: set):
        violation_slots = set()
        initial_statuses = {}
        reserved_for_map = getattr(self, "_reserved_for_map", {})
        special_slots = getattr(self, "_special_slots", set())

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
                if db_slot.reservation_type == "EMPLOYEE":
                    violation_slots.add(db_slot.slot_id)
                    reserved_for_map[db_slot.slot_id] = db_slot.reserved_for
                elif db_slot.reservation_type == "SPECIAL":
                    violation_slots.add(db_slot.slot_id)
                    special_slots.add(db_slot.slot_id)
                initial_statuses[db_slot.slot_id] = db_slot.is_available
                all_active_slot_ids.add(db_slot.slot_id)
        except Exception as exc:
            print(f"[ERROR] Failed to load initial slot states from DB: {exc}")
        finally:
            session.close()

        self._reserved_for_map = reserved_for_map
        self._special_slots = special_slots
        return violation_slots, initial_statuses

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

        if cam_id == "CAM-01" and "Park_Entry" in camera_special_zones:
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

        if cam_id not in ["CAM-01", "CAM-02"] and detections:
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
                elif previous_plate:
                    # Registry says no plate (moved or unlinked), but machine has one.
                    # Clear it to avoid ghost labels in the UI.
                    state_machine.bind_identity(
                        None,
                        self._build_slot_snapshot_url(slot.id),
                    )
                    if self.db_manager:
                        self._persist_late_slot_plate(slot.id, None, cam_id)

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
            is_named_reserved_slot = (
                event.slot_id in self._reserved_for_map
                or event.slot_id in self._special_slots
            )
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
            # Don't early-return when detection is missing — that bypasses
            # _save_alert_snapshot and the alert ends up either with the rolling
            # `slot_<id>_latest.jpg` (stale) or no snapshot at all. Carry an
            # empty crop instead; _save_alert_snapshot's fallback_frame=frame
            # path will save the full camera frame as evidence (Bug fix:
            # production audit 2026-05-05 showed 49/151 intrusion alerts
            # using the rolling fallback path purely because of this branch).
            if detection:
                crop = self._crop_vehicle_bbox_snapshot(frame, detection=detection)
                if crop is None:
                    crop = np.empty((0, 0, 3), dtype=np.uint8)
            else:
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
                # Pass the full frame as fallback so the alert always carries an
                # evidence image even when the per-vehicle crop is empty
                # (Version 0 / Issue #v0-1 fix). Operators previously got
                # intrusion alerts with no snapshot when the bbox was missing.
                event.snapshot_path = self._save_alert_snapshot(
                    crop,
                    alert_type=alert_type,
                    slot_id=event.slot_id,
                    camera_id=cam_id,
                    fallback_frame=frame,
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
        if slot_id in self._special_slots:
            return "special_needs_violation"
        named_slot_title = self._reserved_for_map.get(slot_id)
        if named_slot_title:
            if self._is_named_slot_vehicle_allowed(plate_number, named_slot_title):
                return None
            return "vehicle_intrusion"
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
                    "special_needs_violation",
                ):
                    is_parked = event.event_type in (
                        "vehicle_parked",
                        "vehicle_violation",
                        "vehicle_intrusion",
                        "special_needs_violation",
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
