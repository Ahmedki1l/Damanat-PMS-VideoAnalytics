import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point

from src.models.slot import ParkingSlot

logger = logging.getLogger(__name__)


class ParkingEngineTrackingMixin:
    def _split_special_zones(
        self,
        slots: List[ParkingSlot],
    ) -> Tuple[List[ParkingSlot], List[ParkingSlot]]:
        """Split polygons into standard parking slots and special tracking zones."""
        parking = []
        special = []
        for slot in slots:
            if "Park_Entry" in slot.id or "Entrence" in slot.id:
                special.append(slot)
            else:
                parking.append(slot)
        return parking, special

    def _detection_in_zone(self, detection, zone_slot: ParkingSlot) -> bool:
        """Check if a detection's bottom-center is inside a zone polygon."""
        bc_x, bc_y = detection.bottom_center
        return zone_slot.polygon.contains(Point(bc_x, bc_y))

    def _zone_depth_score(self, detection, zone_slot: ParkingSlot) -> float:
        """Measure how deep the detection is inside a zone."""
        bc_x, bc_y = detection.bottom_center
        point = Point(bc_x, bc_y)
        if not zone_slot.polygon.contains(point):
            return 0.0
        return float(point.distance(zone_slot.polygon.exterior))

    def _make_confirmation_candidate(
        self,
        detection,
        crop: np.ndarray,
        depth: float,
        quality: float,
    ) -> Dict:
        """Build a scored candidate frame from the confirmation-zone burst."""
        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        return {
            "crop": crop,
            "quality": quality,
            "depth": depth,
            "center_x": (x1 + x2) / 2.0,
            "aspect_ratio": width / height,
        }

    def _select_confirmation_reference_frames(
        self,
        candidates: List[Dict],
        max_refs: int = 2,
    ) -> List[Dict]:
        """Pick up to two distinct reference frames for one confirmed car."""
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda item: (item["depth"], item["quality"]),
            reverse=True,
        )
        selected = [ranked[0]]
        if max_refs <= 1 or len(ranked) == 1:
            return selected

        primary = ranked[0]
        secondary = None
        for candidate in ranked[1:]:
            center_delta = abs(candidate["center_x"] - primary["center_x"])
            aspect_delta = abs(candidate["aspect_ratio"] - primary["aspect_ratio"])
            depth_delta = abs(candidate["depth"] - primary["depth"])
            if center_delta >= 25.0 or aspect_delta >= 0.12 or depth_delta >= 4.0:
                secondary = candidate
                break

        if secondary is None and len(ranked) > 1:
            secondary = ranked[1]
        if secondary is not None:
            selected.append(secondary)
        return selected

    def _crop_detection(
        self,
        frame: np.ndarray,
        detection,
        padding_ratio: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Safely crop the vehicle detection from the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        if padding_ratio > 0.0:
            pad_x = int((x2 - x1) * padding_ratio)
            pad_y = int((y2 - y1) * padding_ratio)
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _score_snapshot_quality(self, detection, crop: Optional[np.ndarray]) -> float:
        """Score snapshot quality using both size and clarity."""
        x1, y1, x2, y2 = detection.bbox
        area_score = max(0.0, float((x2 - x1) * (y2 - y1)))
        if crop is None or crop.size == 0:
            return area_score

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_factor = min(sharpness_score, 250.0) / 250.0
        return area_score * (1.0 + 0.5 * sharpness_factor)

    def _process_park_entry_zone(
        self,
        cam_id: str,
        frame: np.ndarray,
        detections: List,
        zone: ParkingSlot,
    ):
        """Handle CAM_01 logic for Park_Entry."""
        currently_inside = set()

        for detection in detections:
            if detection.track_id == -1:
                continue

            if self._detection_in_zone(detection, zone):
                currently_inside.add(detection.track_id)

                candidate_id = self._park_entry_track_to_candidate.get(detection.track_id)
                if not candidate_id:
                    candidate = self.vehicle_registry.open_park_entry_candidate(
                        cam_id,
                        detection.track_id,
                    )
                    candidate_id = candidate.candidate_id
                    self._park_entry_track_to_candidate[detection.track_id] = candidate_id

                crop = self._crop_detection(frame, detection)
                if crop is not None:
                    quality = self._score_snapshot_quality(detection, crop)
                    self.vehicle_registry.update_park_entry_candidate_snapshot(
                        candidate_id,
                        crop,
                        quality,
                    )

                self.vehicle_registry.bind_next_pending_anpr_to_candidate(candidate_id)

        last_track_ids = self._tracks_inside_zones.get((cam_id, zone.id), set())
        left_zone = last_track_ids - currently_inside
        for track_id in left_zone:
            if track_id in self._park_entry_track_to_candidate:
                del self._park_entry_track_to_candidate[track_id]

        self._tracks_inside_zones[(cam_id, zone.id)] = currently_inside

    def _process_confirmation_zone(
        self,
        cam_id: str,
        frame: np.ndarray,
        detections: List,
        zone: ParkingSlot,
    ):
        """
        Handle CAM_03 and other confirm zones for identity matching.
        Uses the zone entry transition as a virtual crossing event.
        """
        currently_inside = set()
        last_track_ids = self._tracks_inside_zones.get((cam_id, zone.id), set())

        for detection in detections:
            if detection.track_id == -1:
                continue

            if self._detection_in_zone(detection, zone):
                currently_inside.add(detection.track_id)

                if self.vehicle_registry.get_plate_for_track(cam_id, detection.track_id):
                    continue

                burst_key = (cam_id, detection.track_id)
                crop = self._crop_detection(frame, detection, padding_ratio=0.12)
                if crop is None:
                    continue

                quality = self._score_snapshot_quality(detection, crop)
                depth = self._zone_depth_score(detection, zone)

                if burst_key not in self._confirmation_bursts:
                    self._confirmation_bursts[burst_key] = {
                        "best_crop": crop,
                        "best_quality": quality,
                        "best_depth": depth,
                        "frames_collected": 1,
                        "candidates": [
                            self._make_confirmation_candidate(
                                detection,
                                crop,
                                depth,
                                quality,
                            )
                        ],
                    }
                    continue

                burst = self._confirmation_bursts[burst_key]
                burst["frames_collected"] += 1
                burst["candidates"].append(
                    self._make_confirmation_candidate(detection, crop, depth, quality)
                )
                best_depth = burst.get("best_depth", 0.0)
                best_quality = burst.get("best_quality", 0.0)
                if depth > best_depth + 3.0 or (
                    abs(depth - best_depth) <= 3.0 and quality > best_quality
                ):
                    burst["best_crop"] = crop
                    burst["best_quality"] = quality
                    burst["best_depth"] = depth

        left_zone = last_track_ids - currently_inside
        for track_id in left_zone:
            burst_key = (cam_id, track_id)
            if burst_key not in self._confirmation_bursts:
                continue

            burst = self._confirmation_bursts.pop(burst_key)
            selected_refs = self._select_confirmation_reference_frames(
                burst.get("candidates", []),
                max_refs=2,
            )
            reference_images = [item["crop"] for item in selected_refs]
            primary_image = reference_images[0] if reference_images else burst["best_crop"]

            plate = self.vehicle_registry.confirm_at_b1_entrance(
                cam_id,
                track_id,
                primary_image,
                reference_images=reference_images,
            )
            depth_text = f"{burst.get('best_depth', 0.0):.1f}"
            refs_count = max(1, len(reference_images))
            if plate:
                print(
                    f"[SNAPSHOT] Final session snapshot updated from {cam_id}/{zone.id} "
                    f"for {plate} | Deep frame selected ({depth_text}px inside zone) | "
                    f"References: {refs_count} | "
                    f"Sync across {burst['frames_collected']} frames | "
                    f"Quality: {burst['best_quality']:.0f}"
                )
            else:
                print(
                    f"[LINE] Car {track_id} finished crossing {zone.id} on {cam_id} | "
                    f"Deepest frame: {depth_text}px inside zone | "
                    f"References: {refs_count} | "
                    f"Sync across {burst['frames_collected']} frames | "
                    f"Quality: {burst['best_quality']:.0f}"
                )

        self._tracks_inside_zones[(cam_id, zone.id)] = currently_inside

    def _process_global_tracking(
        self,
        cam_id: str,
        frame: np.ndarray,
        detections: List,
        tracking_manager=None,
    ):
        """
        Identify confirmed vehicles as they move between CAM_03 and CAM_14.
        """
        if not self.vehicle_registry:
            return

        now_ts = time.time()
        for detection in detections:
            if detection.track_id == -1:
                continue

            track_key = (cam_id, detection.track_id)
            session_id = self.vehicle_registry.get_session_id_for_track(
                cam_id,
                detection.track_id,
            )

            if session_id:
                if tracking_manager:
                    smoothed = tracking_manager.get_track_feature(detection.track_id)
                    if smoothed is not None:
                        self.vehicle_registry.update_session_feature(session_id, smoothed)
                        self.vehicle_registry.reattach_track_to_confirmed_session(
                            cam_id,
                            detection.track_id,
                            smoothed,
                        )
                continue

            if now_ts - self._reid_check_timer.get(track_key, 0) < 1.0:
                continue
            self._reid_check_timer[track_key] = now_ts

            query_vector = None
            if tracking_manager:
                query_vector = tracking_manager.get_track_feature(detection.track_id)

            if query_vector is None:
                crop = self._crop_detection(frame, detection)
                if crop is not None:
                    query_vector = self.vehicle_registry.reid_matcher.extract_feature(crop)

            if query_vector is None:
                continue

            matched_session = self.vehicle_registry.match_global_session(
                query_vector,
                camera_id=cam_id,
                track_id=detection.track_id,
            )
            if matched_session:
                self.vehicle_registry.attach_session_to_track(
                    cam_id,
                    detection.track_id,
                    matched_session,
                )
                logger.info(
                    "[GLOBAL] Track (%s, %d) identified via ReID -> session %s",
                    cam_id,
                    detection.track_id,
                    matched_session,
                )
            elif tracking_manager:
                track_state = tracking_manager.tracks.get(detection.track_id)
                if track_state and track_state.update_count >= 10:
                    new_session_id = self.vehicle_registry.create_appearance_session(
                        cam_id,
                        detection.track_id,
                        query_vector,
                    )
                    logger.info(
                        "[GLOBAL] Track (%s, %d) is new -> created session %s",
                        cam_id,
                        detection.track_id,
                        new_session_id,
                    )

    def _save_car_crop(self, frame, detection, plate: str, cam_id: str):
        """Crop and save the detected car image for visual reference."""
        try:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                os.makedirs("vehicle_images", exist_ok=True)
                filename = (
                    f"vehicle_images/{plate}_{cam_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                cv2.imwrite(filename, crop)
                print(f"[CROP] Saved car image: {filename}")
        except Exception as exc:
            print(f"[WARN] Failed to save car crop: {exc}")

    def _cleanup_stale_data(self):
        """Prune stale state to prevent memory leaks."""
        now = time.time()

        stale_reid = [
            key for key, timestamp in self._reid_check_timer.items() if now - timestamp > 30.0
        ]
        for key in stale_reid:
            del self._reid_check_timer[key]

        self._recent_violators = [
            violator
            for violator in self._recent_violators
            if now - violator["timestamp"] < self._violation_history_limit
        ]
