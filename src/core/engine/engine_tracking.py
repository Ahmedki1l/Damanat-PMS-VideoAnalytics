import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, box

from src.models.slot import ParkingSlot

logger = logging.getLogger(__name__)


class ParkingEngineTrackingMixin:
    def _find_special_zone(
        self,
        cam_id: str,
        zone_name_hints: Tuple[str, ...],
    ) -> Optional[ParkingSlot]:
        """Resolve the first configured special zone matching any hint."""
        camera_special_zones = self.special_zones.get(cam_id, {})
        for zone_id, zone in camera_special_zones.items():
            if any(hint in zone_id for hint in zone_name_hints):
                return zone
        return None

    def _crop_special_zone(
        self,
        frame: np.ndarray,
        cam_id: str,
        zone_name_hints: Tuple[str, ...],
    ) -> Optional[np.ndarray]:
        """Crop a configured special-zone bounding rectangle from a live frame."""
        zone = self._find_special_zone(cam_id, zone_name_hints)
        if zone is None or frame is None or frame.size == 0:
            return None

        minx, miny, maxx, maxy = zone.polygon.bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(minx)), max(0, int(miny))
        x2, y2 = min(w, int(maxx)), min(h, int(maxy))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def get_special_zone_snapshot(
        self,
        cam_id: str,
        zone_name_hints: Tuple[str, ...],
        max_age_seconds: float = 5.0,
    ) -> Optional[np.ndarray]:
        """
        Capture a fresh special-zone crop first, then fall back to cached vehicle crops.
        """
        if hasattr(self, "cam_manager") and self.cam_manager:
            success, frame = self.cam_manager.read_camera(cam_id)
            if success and frame is not None:
                crop = self._crop_special_zone(frame, cam_id, zone_name_hints)
                if crop is not None:
                    return crop

        for zone_name_hint in zone_name_hints:
            recent_crop = self.get_recent_zone_vehicle_crop(
                cam_id,
                zone_name_hint=zone_name_hint,
                max_age_seconds=max_age_seconds,
            )
            if recent_crop is not None and recent_crop.size > 0:
                return recent_crop
        return None

    def _try_confirm_b1_track(
        self,
        cam_id: str,
        track_id: int,
        burst: Dict,
    ) -> Optional[str]:
        """Attempt an early B1 entrance confirmation for a live in-zone track."""
        ordered_images, primary_index = self._build_confirmation_timeline_gallery(
            burst,
            include_exit_view=False,
        )
        if not ordered_images:
            return None

        primary_image = ordered_images[primary_index]
        return self.vehicle_registry.confirm_at_b1_entrance(
            cam_id,
            track_id,
            primary_image,
            ordered_images=ordered_images,
            primary_snapshot_index=primary_index,
        )

    def get_recent_zone_vehicle_crop(
        self,
        cam_id: str,
        zone_name_hint: str = "Entrence",
        max_age_seconds: float = 5.0,
    ) -> Optional[np.ndarray]:
        """
        Return a recent vehicle crop captured while a car was inside a special zone.

        Prefers the first crop seen when the car entered the zone so API-triggered
        snapshots line up with the crossing moment instead of a late empty frame.
        """
        now_ts = time.time()
        best_entry = None
        best_fallback = None

        for (cached_cam_id, zone_id), payload in self._latest_zone_vehicle_crops.items():
            if cached_cam_id != cam_id or zone_name_hint not in zone_id:
                continue
            age = now_ts - payload.get("timestamp", 0.0)
            if age > max_age_seconds:
                continue
            if payload.get("entry_crop") is not None:
                if best_entry is None or payload.get("timestamp", 0.0) > best_entry.get("timestamp", 0.0):
                    best_entry = payload
            elif payload.get("crop") is not None:
                if best_fallback is None or payload.get("timestamp", 0.0) > best_fallback.get("timestamp", 0.0):
                    best_fallback = payload

        if best_entry is not None:
            return best_entry["entry_crop"].copy()
        if best_fallback is not None:
            return best_fallback["crop"].copy()
        return None

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

    def _detection_overlaps_zone(
        self,
        detection,
        zone_slot: ParkingSlot,
        min_overlap_ratio: float = 0.12,
    ) -> bool:
        """Accept fast-moving cars when enough of the bbox overlaps the zone."""
        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        if x2 <= x1 or y2 <= y1:
            return False

        detection_box = box(x1, y1, x2, y2)
        if detection_box.is_empty:
            return False

        intersection = detection_box.intersection(zone_slot.polygon)
        if intersection.is_empty:
            return False

        overlap_ratio = float(intersection.area) / max(1.0, float(detection_box.area))
        return overlap_ratio >= min_overlap_ratio

    def _zone_overlap_ratio(
        self,
        detection,
        zone_slot: ParkingSlot,
    ) -> float:
        """Measure how much of the YOLO bbox overlaps the confirmation zone."""
        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        if x2 <= x1 or y2 <= y1:
            return 0.0

        detection_box = box(x1, y1, x2, y2)
        if detection_box.is_empty:
            return 0.0

        intersection = detection_box.intersection(zone_slot.polygon)
        if intersection.is_empty:
            return 0.0

        return float(intersection.area) / max(1.0, float(detection_box.area))

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
        visibility: float,
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
            "visibility": visibility,
            "entry_score": visibility * 100000.0 + quality,
        }

    @staticmethod
    def _make_stage_view(
        crop: np.ndarray,
        quality: float,
        depth: float,
        visibility: float,
    ) -> Dict:
        return {
            "crop": crop.copy(),
            "quality": quality,
            "depth": depth,
            "visibility": visibility,
        }

    def _build_confirmation_timeline_gallery(
        self,
        burst: Dict,
        include_exit_view: bool = True,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Build an ordered CAM_03 gallery: entry, deep, then exit.
        """
        ordered_views = []

        entry_view = burst.get("entry_view") or burst.get("earliest_view")
        deep_view = burst.get("deep_view") or burst.get("best_view")
        exit_view = None
        if include_exit_view:
            exit_view = burst.get("exit_view") or burst.get("latest_view") or deep_view

        entry_crop = (
            entry_view.get("crop")
            if entry_view and entry_view.get("crop") is not None
            else None
        )
        if entry_crop is not None and deep_view and deep_view.get("crop") is not None:
            if not self._is_consistent_confirmation_crop(entry_crop, deep_view["crop"]):
                deep_view = None
        if entry_crop is not None and include_exit_view and exit_view and exit_view.get("crop") is not None:
            if not self._is_consistent_confirmation_crop(entry_crop, exit_view["crop"]):
                exit_view = None

        if entry_view and entry_view.get("crop") is not None:
            ordered_views.append(("entry", entry_view["crop"]))
        if deep_view and deep_view.get("crop") is not None:
            ordered_views.append(("deep", deep_view["crop"]))
        if include_exit_view and exit_view and exit_view.get("crop") is not None:
            ordered_views.append(("exit", exit_view["crop"]))

        images: List[np.ndarray] = []
        labels: List[str] = []
        for label, crop in ordered_views:
            if crop is None or crop.size == 0:
                continue
            if any(np.array_equal(crop, existing) for existing in images):
                continue
            labels.append(label)
            images.append(crop.copy())

        if not images:
            return [], 0

        primary_index = labels.index("deep") if "deep" in labels else 0
        return images, primary_index

    def _is_consistent_confirmation_crop(
        self,
        reference_crop: Optional[np.ndarray],
        candidate_crop: Optional[np.ndarray],
        min_color_similarity: float = 0.45,
    ) -> bool:
        """
        Reject timeline crops that clearly look like a different vehicle.
        """
        if reference_crop is None or candidate_crop is None:
            return False
        if reference_crop.size == 0 or candidate_crop.size == 0:
            return False
        if not self.vehicle_registry:
            return True

        try:
            color_similarity = self.vehicle_registry.matcher._compare_dominant_colors(
                reference_crop,
                candidate_crop,
            )
        except Exception:
            return True
        return color_similarity >= min_color_similarity

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

    def _bbox_view_quality(self, frame: np.ndarray, detection) -> float:
        """How fully this camera sees the car: 1.0 when the bbox sits entirely
        inside the frame (with a 1% edge margin), decreasing toward 0.0 as the
        box is clipped by a frame edge (part of the car out of view).

        Estimated as the fraction of the detection's *reported* bbox area that
        falls inside the frame — a box hard against an edge has been clamped by
        the detector, so its reported extent beyond the border is the missing
        part of the car. Used only as an ownership tie-breaker (display/
        attribution), never a detection decision.
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in detection.bbox)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                return 0.0
            mx, my = 0.01 * w, 0.01 * h
            ix = max(0.0, min(x2, w - mx) - max(x1, mx))
            iy = max(0.0, min(y2, h - my) - max(y1, my))
            return max(0.0, min(1.0, (ix * iy) / (bw * bh)))
        except Exception:
            return 0.0

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
            pad_x = max(16, int((x2 - x1) * padding_ratio))
            pad_y = max(12, int((y2 - y1) * padding_ratio))
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _crop_detection_to_zone(
        self,
        frame: np.ndarray,
        detection,
        zone: ParkingSlot,
        padding_ratio: float = 0.0,
        mask_outside_zone: bool = True,
    ) -> Optional[np.ndarray]:
        """Crop only the overlapping detection area inside the configured zone."""
        if frame is None or frame.size == 0 or zone is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        if padding_ratio > 0.0:
            pad_x = max(0, int((x2 - x1) * padding_ratio))
            pad_y = max(0, int((y2 - y1) * padding_ratio))
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

        minx, miny, maxx, maxy = zone.polygon.bounds
        zx1, zy1 = max(0, int(minx)), max(0, int(miny))
        zx2, zy2 = min(w, int(maxx)), min(h, int(maxy))

        x1 = max(0, max(x1, zx1))
        y1 = max(0, max(y1, zy1))
        x2 = min(w, min(x2, zx2))
        y2 = min(h, min(y2, zy2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0 or not mask_outside_zone:
            return crop if crop.size > 0 else None

        polygon_points = np.array(
            [[int(px) - x1, int(py) - y1] for px, py in zone.polygon.exterior.coords[:-1]],
            dtype=np.int32,
        )
        if polygon_points.size == 0:
            return crop

        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
        return masked_crop if masked_crop.size > 0 else None

    def _visibility_score(self, frame: np.ndarray, detection) -> float:
        """Higher score when the full vehicle box is comfortably inside the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        left_margin = max(0.0, x1) / box_w
        top_margin = max(0.0, y1) / box_h
        right_margin = max(0.0, w - x2) / box_w
        bottom_margin = max(0.0, h - y2) / box_h
        return min(left_margin, top_margin, right_margin, bottom_margin)

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

    def _bbox_is_snapshot_ready(
        self,
        frame: np.ndarray,
        detection,
        min_visibility: float = 0.035,
        min_area: float = 14000.0,
    ) -> bool:
        """
        Wait until the YOLO bbox is reasonably formed before taking the snapshot.
        """
        if frame is None or detection is None:
            return False

        visibility = self._visibility_score(frame, detection)
        x1, y1, x2, y2 = [float(v) for v in detection.bbox]
        area = max(0.0, (x2 - x1) * (y2 - y1))
        return visibility >= min_visibility and area >= min_area

    def _select_primary_zone_detection(
        self,
        frame: np.ndarray,
        detections: List,
        zone: ParkingSlot,
    ):
        """
        Pick the single best entrance car for this frame and ignore nearby cars.
        """
        candidates = []
        for detection in detections:
            if detection.track_id == -1:
                continue

            overlap_ratio = self._zone_overlap_ratio(detection, zone)
            in_zone = (
                self._detection_in_zone(detection, zone)
                or overlap_ratio >= 0.12
            )
            if not in_zone:
                continue
            if not self._bbox_is_snapshot_ready(frame, detection):
                continue

            depth = self._zone_depth_score(detection, zone)
            x1, y1, x2, y2 = [float(v) for v in detection.bbox]
            area = max(0.0, (x2 - x1) * (y2 - y1))
            _, bc_y = detection.bottom_center

            score = (
                overlap_ratio * 100000.0
                + depth * 1000.0
                + area * 0.25
                + float(bc_y)
            )
            candidates.append((score, detection))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

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
        primary_detection = self._select_primary_zone_detection(frame, detections, zone)

        for detection in detections:
            if detection.track_id == -1:
                continue

            if primary_detection is not None and detection.track_id != primary_detection.track_id:
                continue

            in_confirmation_zone = (
                self._detection_in_zone(detection, zone)
                or self._detection_overlaps_zone(detection, zone, min_overlap_ratio=0.12)
            )
            if in_confirmation_zone:
                currently_inside.add(detection.track_id)
                entered_now = detection.track_id not in last_track_ids

                if self.vehicle_registry.get_plate_for_track(cam_id, detection.track_id):
                    continue

                burst_key = (cam_id, detection.track_id)
                crop = self._crop_detection(
                    frame,
                    detection,
                    padding_ratio=0.12,
                )
                if crop is None:
                    continue

                quality = self._score_snapshot_quality(detection, crop)
                depth = self._zone_depth_score(detection, zone)
                visibility = self._visibility_score(frame, detection)

                if burst_key not in self._confirmation_bursts:
                    stage_view = self._make_stage_view(crop, quality, depth, visibility)
                    self._confirmation_bursts[burst_key] = {
                        "entry_crop": crop.copy(),
                        "entry_quality": quality,
                        "entry_visibility": visibility,
                        "entry_depth": depth,
                        "best_crop": crop,
                        "best_quality": quality,
                        "best_depth": depth,
                        "latest_crop": crop.copy(),
                        "latest_quality": quality,
                        "latest_depth": depth,
                        "earliest_view": stage_view,
                        "entry_view": stage_view,
                        "deep_view": stage_view,
                        "best_view": stage_view,
                        "latest_view": stage_view,
                        "exit_view": stage_view,
                        "confirmed": False,
                        "confirmed_plate": None,
                        "frames_collected": 1,
                        "candidates": [
                            self._make_confirmation_candidate(
                                detection,
                                crop,
                                depth,
                                quality,
                                visibility,
                            )
                        ],
                    }
                    self._latest_zone_vehicle_crops[(cam_id, zone.id)] = {
                        "crop": crop.copy(),
                        "entry_crop": crop.copy(),
                        "track_id": detection.track_id,
                        "timestamp": time.time(),
                        "depth": depth,
                        "quality": quality,
                    }
                    continue

                burst = self._confirmation_bursts[burst_key]
                burst["frames_collected"] += 1
                current_view = self._make_stage_view(crop, quality, depth, visibility)
                entry_reference = (
                    burst.get("entry_view", {}).get("crop")
                    if burst.get("entry_view")
                    else None
                )
                is_consistent_with_entry = self._is_consistent_confirmation_crop(
                    entry_reference,
                    crop,
                )
                burst["candidates"].append(
                    self._make_confirmation_candidate(
                        detection,
                        crop,
                        depth,
                        quality,
                        visibility,
                    )
                )
                burst["latest_crop"] = crop.copy()
                burst["latest_quality"] = quality
                burst["latest_depth"] = depth
                burst["latest_view"] = current_view
                if is_consistent_with_entry:
                    burst["exit_view"] = current_view
                best_depth = burst.get("best_depth", 0.0)
                best_quality = burst.get("best_quality", 0.0)
                if is_consistent_with_entry and (
                    depth > best_depth + 3.0 or (
                    abs(depth - best_depth) <= 3.0 and quality > best_quality
                    )
                ):
                    burst["best_crop"] = crop
                    burst["best_quality"] = quality
                    burst["best_depth"] = depth
                    burst["best_view"] = current_view
                    burst["deep_view"] = current_view

                cache_key = (cam_id, zone.id)
                cached = self._latest_zone_vehicle_crops.get(cache_key)
                entry_crop = burst.get("entry_crop")
                if (
                    cached is None
                    or detection.track_id == cached.get("track_id")
                    or (
                        is_consistent_with_entry
                        and depth >= cached.get("depth", 0.0)
                    )
                ):
                    self._latest_zone_vehicle_crops[cache_key] = {
                        "crop": burst.get("best_crop", crop).copy(),
                        "entry_crop": entry_crop.copy() if entry_crop is not None else None,
                        "track_id": detection.track_id,
                        "timestamp": time.time(),
                        "depth": burst.get("best_depth", depth),
                        "quality": burst.get("best_quality", quality),
                    }

                should_try_early_confirm = (
                    (
                        entered_now
                        and not burst.get("confirmed", False)
                    )
                    or (
                        not burst.get("confirmed", False)
                        and burst["frames_collected"] >= 2
                        and (
                            burst.get("best_depth", 0.0) >= 4.0
                            or burst.get("entry_visibility", 0.0) >= 0.08
                        )
                    )
                )
                if should_try_early_confirm:
                    plate = self._try_confirm_b1_track(
                        cam_id,
                        detection.track_id,
                        burst,
                    )
                    if plate:
                        burst["confirmed"] = True
                        burst["confirmed_plate"] = plate
                        print(
                            f"[B1] Early confirmation on {cam_id}/{zone.id}: "
                            f"track {detection.track_id} -> {plate}"
                        )

        left_zone = last_track_ids - currently_inside
        for track_id in left_zone:
            burst_key = (cam_id, track_id)
            if burst_key not in self._confirmation_bursts:
                continue

            burst = self._confirmation_bursts.pop(burst_key)
            ordered_images, primary_index = self._build_confirmation_timeline_gallery(
                burst,
                include_exit_view=True,
            )
            if not ordered_images:
                continue

            primary_image = ordered_images[primary_index]

            plate = burst.get("confirmed_plate")
            if not plate:
                plate = self.vehicle_registry.confirm_at_b1_entrance(
                    cam_id,
                    track_id,
                    primary_image,
                    ordered_images=ordered_images,
                    primary_snapshot_index=primary_index,
                )
            else:
                self.vehicle_registry.update_confirmed_session_gallery(
                    cam_id,
                    track_id,
                    ordered_images,
                    primary_snapshot_index=primary_index,
                )

            depth_text = f"{burst.get('best_depth', 0.0):.1f}"
            refs_count = len(ordered_images)
            if plate:
                print(
                    f"[SNAPSHOT] Final session snapshot updated from {cam_id}/{zone.id} "
                    f"for {plate} | Timeline views: {refs_count} | "
                    f"Deep frame selected ({depth_text}px inside zone) | "
                    f"Sync across {burst['frames_collected']} frames | "
                    f"Quality: {burst['best_quality']:.0f}"
                )
            else:
                print(
                    f"[LINE] Car {track_id} finished crossing {zone.id} on {cam_id} | "
                    f"Deepest frame: {depth_text}px inside zone | "
                    f"Timeline views: {refs_count} | "
                    f"Sync across {burst['frames_collected']} frames | "
                    f"Quality: {burst['best_quality']:.0f}"
                )

            self._latest_zone_vehicle_crops[(cam_id, zone.id)] = {
                "crop": burst["best_crop"].copy(),
                "entry_crop": (
                    burst["entry_crop"].copy()
                    if burst.get("entry_crop") is not None
                    else None
                ),
                "track_id": track_id,
                "timestamp": time.time(),
                "depth": burst.get("best_depth", 0.0),
                "quality": burst.get("best_quality", 0.0),
            }

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

            # Record how fully this camera sees the car — feeds the ReID
            # ownership tie-breaker (a full view outranks a clipped one).
            self.vehicle_registry.record_track_view(
                cam_id,
                detection.track_id,
                self._bbox_view_quality(frame, detection),
            )

            track_key = (cam_id, detection.track_id)
            session_id = self.vehicle_registry.get_session_id_for_track(
                cam_id,
                detection.track_id,
            )

            if session_id:
                # If this track already has a confirmed plate, the car is
                # already identified — skip all ReID computation.
                # Only refresh timestamps to keep the binding alive.
                plate = self.vehicle_registry.get_plate_for_track(cam_id, detection.track_id)
                if plate:
                    self.vehicle_registry.refresh_track_binding(cam_id, detection.track_id, session_id)
                    # Persist "where is this car right now" — vehicles.floor /
                    # last_seen_at — so the Gateway can answer presence queries
                    # for cars driving across the floor without parking. The
                    # writer is rate-gated (see _PRESENCE_MIN_INTERVAL_S),
                    # so calling it on every frame is cheap.
                    pipeline = self.pipelines.get(cam_id)
                    if pipeline is not None:
                        self.update_vehicle_presence(
                            plate, floor=pipeline.floor, camera_id=cam_id,
                        )
                    continue

                if tracking_manager:
                    smoothed = tracking_manager.get_track_feature(detection.track_id)
                    if smoothed is not None:
                        self.vehicle_registry.update_session_feature(
                            session_id,
                            smoothed,
                            camera_id=cam_id,
                            track_id=detection.track_id,
                        )
                        self.vehicle_registry.reattach_track_to_confirmed_session(
                            cam_id,
                            detection.track_id,
                            smoothed,
                        )
                else:
                    # No tracking manager — still refresh the binding so this
                    # camera stays in the session's observing_tracks.
                    self.vehicle_registry.refresh_track_binding(
                        cam_id,
                        detection.track_id,
                        session_id,
                    )

                continue

            if now_ts - self._reid_check_timer.get(track_key, 0) < 1.0:
                continue
            self._reid_check_timer[track_key] = now_ts

            query_vector = None
            crop = None
            if tracking_manager:
                query_vector = tracking_manager.get_track_feature(detection.track_id)

            if query_vector is None:
                crop = self._crop_detection(frame, detection)
                if crop is not None:
                    query_vector = self.vehicle_registry.reid_matcher.extract_feature(crop)

            if query_vector is None:
                continue

            # Crop for rank-5 OCR disambiguation inside match_global_session
            # (the smoothed-feature path above skips cropping). Gated to one
            # ReID check/sec per track, so this extra crop is cheap.
            if crop is None:
                crop = self._crop_detection(frame, detection)

            # Bound the candidate pool to this camera's area (+ the cross-area
            # handoff pool of cars that recently departed an adjacent area) so a
            # parked car can't be ReID-matched to a session sitting in an
            # unrelated area/floor — the cross-area false-lock that mislabels a
            # B2 car with a B1 entrant's plate. Resolves to "" when zoning is off
            # or the camera is un-zoned, in which case match_global_session falls
            # back to the legacy all-sessions pool (no behaviour change there).
            area_id = ""
            if self.area_registry is not None and self.area_registry.enabled:
                area_id = self.area_registry.area_for_camera(cam_id)

            matched_session = self.vehicle_registry.match_global_session(
                query_vector,
                camera_id=cam_id,
                track_id=detection.track_id,
                area_id=area_id or None,
                query_crop=crop,
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

    def _save_car_crop(self, frame, detection, plate: str, cam_id: str) -> Optional[str]:
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
                base_dir = self.config.output.snapshot_base_dir
                os.makedirs(base_dir, exist_ok=True)
                filename = (
                    f"{plate}_{cam_id}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                full_path = os.path.join(base_dir, filename)
                cv2.imwrite(full_path, crop)
                print(f"[CROP] Saved car image: {full_path}")
                return filename
        except Exception as exc:
            print(f"[WARN] Failed to save car crop: {exc}")
        return None

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
