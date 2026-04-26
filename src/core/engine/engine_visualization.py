import math
import time
from typing import Dict, List

import cv2
import numpy as np

from src.models.state_machine import SlotState

COLORS = {
    SlotState.VACANT: (0, 255, 0),
    SlotState.ENTERING: (0, 255, 255),
    SlotState.OCCUPIED: (0, 0, 255),
    SlotState.LEAVING: (0, 165, 255),
}


class ParkingEngineVisualizationMixin:
    def _resolve_display_label(self, cam_id: str, track_id: int):
        """Keep a confirmed plate label visually stable for a short time."""
        fallback_label = track_id
        if not self.vehicle_registry:
            return fallback_label, False

        display_id = self.vehicle_registry.get_display_id_for_track(cam_id, track_id)
        cache_key = (cam_id, track_id)
        now_ts = time.time()
        cache = self._display_label_cache.get(cache_key)

        is_confirmed_label = isinstance(display_id, str) and not display_id.startswith("ID-")
        if is_confirmed_label:
            self._display_label_cache[cache_key] = {
                "label": display_id,
                "expires_at": now_ts + self._display_label_ttl_seconds,
            }
            return display_id, True

        if cache and cache.get("expires_at", 0.0) >= now_ts:
            return cache.get("label", display_id), True

        if cache_key in self._display_label_cache:
            del self._display_label_cache[cache_key]
        return display_id, False

    def _emit_full_summary(self):
        """Emit status summary across all cameras."""
        all_statuses = []
        for cam_id, pipeline in self.pipelines.items():
            for state_machine in pipeline.state_machines.values():
                status = state_machine.get_status()
                status["camera_id"] = cam_id
                status["floor"] = pipeline.floor
                all_statuses.append(status)
        self.event_bus.emit_status_summary(all_statuses)

    def _draw_frame(self, frame, pipeline, assignment, cam_id, all_detections=None):
        """Draw slot polygons and detections on a frame."""
        if pipeline.roi_polygon is not None:
            pts = np.array(
                pipeline.roi_polygon.exterior.coords,
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )
            cv2.putText(
                frame,
                "DETECTION ZONE",
                (pts[0][0][0], pts[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        cv2.putText(
            frame,
            f"{cam_id} | {pipeline.floor}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        for slot in pipeline.slots:
            state_machine = pipeline.state_machines[slot.id]
            is_violation = state_machine.is_violation_zone
            if is_violation and state_machine.is_occupied:
                color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 0, 150)
                thickness = 4
            else:
                color = COLORS.get(state_machine.state, (128, 128, 128))
                thickness = 2

            coords = list(slot.polygon.exterior.coords)
            pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

            cx, cy = int(slot.centroid_x), int(slot.centroid_y)
            label = "!!! VIOLATION !!!" if is_violation and state_machine.is_occupied else (
                f"{slot.id}: {state_machine.state.value}"
            )
            cv2.putText(
                frame,
                label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        for track_id, detection in assignment.slot_vehicle_map.values():
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            display_id = track_id
            is_sticky_confirmed = False
            if self.vehicle_registry:
                display_id, is_sticky_confirmed = self._resolve_display_label(cam_id, track_id)

            cv2.putText(
                frame,
                f"{display_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if is_sticky_confirmed else (255, 255, 0),
                2,
            )
            bc_x, bc_y = detection.bottom_center
            cv2.circle(frame, (int(bc_x), int(bc_y)), 5, (0, 0, 255), -1)

        if hasattr(assignment, "unassigned"):
            for detection in assignment.unassigned:
                track_id = detection.track_id
                x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                display_id = track_id
                is_sticky_confirmed = False
                if self.vehicle_registry:
                    display_id, is_sticky_confirmed = self._resolve_display_label(
                        cam_id,
                        track_id,
                    )

                cv2.putText(
                    frame,
                    f"{display_id} (?)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255) if is_sticky_confirmed else (0, 0, 255),
                    2,
                )
                bc_x, bc_y = detection.bottom_center
                cv2.circle(frame, (int(bc_x), int(bc_y)), 5, (0, 0, 255), -1)

    @staticmethod
    def _build_grid(
        frames: Dict[str, np.ndarray],
        camera_ids: List[str],
        cols: int = 4,
        cell_w: int = 480,
        cell_h: int = 270,
    ) -> np.ndarray:
        """Assemble camera frames into a single grid image."""
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
                cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(
                    cell,
                    f"{cam_id} - waiting...",
                    (10, cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 100),
                    1,
                )

            grid[y_off:y_off + cell_h, x_off:x_off + cell_w] = cell

        return grid
