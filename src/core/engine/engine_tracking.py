import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, box

from src.models.slot import ParkingSlot
from src.models.state_machine import SlotState

logger = logging.getLogger(__name__)

# Apparent-size ramp for _bbox_view_quality: a car whose on-screen HEIGHT is at
# or below _VQ_MIN_H px yields a useless upscaled crop for ReID and scores 0 on
# size; at or above _VQ_GOOD_H px it is well-resolved and scores 1.0. This makes
# a far camera whose tiny car is fully framed rank BELOW a near camera that sees
# the car large — for both identity ownership and gallery reference capture.
#
# _VQ_GOOD_H was 90px, which made this ramp nearly inert: ANY car >=90px tall
# scored a PERFECT 1.0 on size and sailed through gallery_min_view_quality (0.9).
# Measured against the real on-disk gallery, that is far too generous:
#     genuine close references (CAM-03 / CAM-23 entry shots):  300-537 px tall
#     far aisle crops that poisoned DJS-7842 (CAM-07/CAM-04):  119-139 px tall
# Both scored 1.0. 250px sits cleanly between the two populations, so a distant
# car down an aisle now scores well under the 0.9 gate and can no longer become
# a gallery reference or win the ownership contest against the camera that
# actually sees the car up close.
# These were calibrated against 720p crops. They MUST NOT be absolute pixels: the
# cameras were switched from 1280x720 to 1920x1080 on 2026-07-11, which makes every car
# 1.5x taller on screen at the same physical distance — so a fixed 250px gate silently
# became a 167px-at-720p gate, 50% more permissive, letting back in exactly the mid-size
# junk it was added to exclude. Expressed as a FRACTION of frame height, the gate means
# the same thing at any stream resolution.
_VQ_REF_H = 720.0          # the resolution the numbers below were measured at
_VQ_MIN_H_720 = 45.0       # below this: a useless upscaled crop
_VQ_GOOD_H_720 = 250.0     # at/above this: well-resolved (real refs were 300-537px)


def _vq_size_thresholds(frame_h: float) -> Tuple[float, float]:
    """Scale the size ramp to the actual frame height, so the gate is resolution-free."""
    s = max(1e-6, float(frame_h)) / _VQ_REF_H
    return _VQ_MIN_H_720 * s, _VQ_GOOD_H_720 * s

# A bbox within this many px of the frame border was CLAMPED by the detector: the car
# runs off the edge and the box holds only the visible fragment. Such a crop is a piece
# of a car, and must never become a ReID reference. 2px (not 0) because the detector
# rounds, so a truly touching box can report 1.0 rather than 0.0.
_VQ_BORDER_PX = 2.0
# Widest a whole car may plausibly appear. Facility crops have a MEDIAN w/h of 1.40;
# CAM-04's junk was 707x222 = 3.2 — a sliver of a car cut off at the top of frame, not
# a vehicle. Catches badly-merged boxes too, which can be sliver-shaped without ever
# touching a border.
_VQ_MAX_ASPECT = 2.2


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

    def _detection_in_occupied_slot(self, cam_id: str, detection) -> bool:
        """True when this detection sits inside a slot that is ALREADY occupied
        (or leaving) on this camera.

        A car in an occupied slot is a parked car — it cannot be the newly
        entered car we are building a reference for — so its crop must not be
        pulled into a gallery. Uses the slot state from the previous frame's
        occupancy pass (this runs before ``_update_slot_state`` for the current
        frame), which is the correct "already occupied" signal. No-op (False)
        when the camera has no pipeline / slots."""
        pipeline = self.pipelines.get(cam_id) if hasattr(self, "pipelines") else None
        if pipeline is None:
            return False
        for slot in getattr(pipeline, "slots", []) or []:
            sm = pipeline.state_machines.get(slot.id)
            if sm is None or sm.state not in (SlotState.OCCUPIED, SlotState.LEAVING):
                continue
            if self._detection_in_zone(detection, slot):
                return True
        return False

    def _detection_in_own_slot(self, cam_id: str, detection) -> bool:
        """True when this detection sits inside a slot THIS camera hosts.

        Slot authority for gallery teaching. A camera may host one or two slots
        while its frame covers a whole aisle full of OTHER cameras' slots — CAM-07
        hosts a single slot that is 4% of its frame, yet its ROI is 74% of it. Owning
        a car (the only camera check on the teach path) was therefore enough licence
        to write references for a car parked in someone else's slot, and that is how
        four crops of a black Ford ended up in grey-Hyundai DJS-7842's gallery.

        Note this can only ever ask "is it in MY slot", never "is it in SOMEONE
        ELSE'S": slot polygons are expressed in each camera's own image coordinates,
        and the cameras are split across worker processes, so another camera's
        polygon is neither comparable nor in scope here.

        No-op (False) when the camera has no pipeline / hosts no slots — a slotless
        transit camera (CAM-03, CAM-05, CAM-23) has no slot authority and so teaches
        nothing through this path. Its references come from the seed paths instead.
        """
        pipeline = self.pipelines.get(cam_id) if hasattr(self, "pipelines") else None
        if pipeline is None:
            return False
        for slot in getattr(pipeline, "slots", []) or []:
            if self._detection_in_zone(detection, slot):
                return True
        return False

    def _slot_authority_required(self) -> bool:
        cfg = getattr(self, "config", None)
        matching = getattr(cfg, "matching", None) if cfg is not None else None
        return bool(getattr(matching, "gallery_require_slot_authority", True))

    def _track_in_entrance_zone(self, cam_id: str, track_id: int) -> bool:
        """True while ``track_id`` is currently inside a B1_Entrance
        (``Entrence``) confirmation zone on this camera.

        Reference snapshots must not be captured for a car until it has LEFT the
        B1_Entrance (CAM-03) zone — while inside, the view is mid-transit /
        partial, not the clean profile we want. ``_process_confirmation_zone``
        refreshes ``_tracks_inside_zones`` earlier in the same frame, so this
        reads the current-frame membership. Cameras without an entrance zone
        simply have no such key → False (accumulation proceeds normally)."""
        for (c, zone_id), ids in self._tracks_inside_zones.items():
            if c == cam_id and "Entrence" in zone_id and track_id in ids:
                return True
        return False

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

        # Extra rear-side views. The single `exit` frame above is only the last
        # consistent crop and is often far/small, so the car's BACK is under-
        # represented by a single frame. Sample additional frames from the
        # crossing tail (car driving away = rear-facing) so the gallery carries
        # real rear coverage. Consistency-gated against the entry crop so a
        # mis-tracked frame (a different car) never enters; evenly spread in time
        # so the frames are distinct rear viewpoints, not adjacent duplicates.
        # Bounded by config; the downstream cosine-dedup drops any redundant one.
        extra_rear = 0
        if include_exit_view and self.vehicle_registry is not None:
            cfg = getattr(self.vehicle_registry, "matching_config", None)
            extra_rear = int(getattr(cfg, "confirmation_extra_rear_views", 0) or 0)
        if extra_rear > 0:
            tail = (burst.get("candidates") or [])[len(burst.get("candidates") or []) // 2:]
            rear_pool = [
                c for c in tail
                if c.get("crop") is not None and c["crop"].size > 0
                and (
                    entry_crop is None
                    or self._is_consistent_confirmation_crop(entry_crop, c["crop"])
                )
            ]
            if rear_pool:
                step = max(1, len(rear_pool) // extra_rear)
                for c in rear_pool[::step][:extra_rear]:
                    ordered_views.append(("rear", c["crop"]))

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

    def _neighbour_clearance(self, detection, detections) -> float:
        """1.0 when this car's box is unobstructed; →0 as other boxes cover it.

        A car parked shoulder-to-shoulder in a garage has a box that overlaps its
        neighbour's, so a crop of it contains part of the wrong car and makes a
        contaminated ReID reference. Returns the fraction of THIS box NOT covered
        by the single most-overlapping other detection — an asymmetric
        intersection-over-self, not IoU, so a large neighbour that swallows a
        small distant box correctly scores that small box as heavily occluded.
        Fails open (1.0) on any error: clearance only ever WEIGHTS view quality,
        it is never a detection decision.
        """
        try:
            if not detections or len(detections) < 2:
                return 1.0
            ax1, ay1, ax2, ay2 = (float(v) for v in detection.bbox)
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            if a_area <= 0.0:
                return 1.0
            self_tid = getattr(detection, "track_id", None)
            worst = 0.0
            for other in detections:
                if other is detection:
                    continue
                other_tid = getattr(other, "track_id", None)
                if other_tid == -1 or (
                    self_tid is not None and other_tid == self_tid
                ):
                    continue
                bx1, by1, bx2, by2 = (float(v) for v in other.bbox)
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                worst = max(worst, (ix * iy) / a_area)
            return max(0.0, 1.0 - worst)
        except Exception:
            return 1.0

    def _clearance_enforced(self) -> bool:
        """True when D9 neighbour-clearance should MULTIPLY view quality (gating
        occluded crops out of the gallery). Default False ⇒ log-only. Reads the
        registry's matching config; absent (bare mixin in a unit test) ⇒ False."""
        cfg = getattr(getattr(self, "vehicle_registry", None), "matching_config", None)
        return bool(getattr(cfg, "gallery_neighbour_clearance_enforce", False))

    def _bbox_view_quality(self, frame: np.ndarray, detection, detections=None) -> float:
        """How well this camera sees the car — the product of two (or three)
        factors:

        * TRUNCATION: 0.0 when the bbox touches the frame border. The detector
          reports only the VISIBLE fragment of a car that runs off the edge, so the
          box is not "slightly clipped" — it is a partial car, and the missing part
          is invisible to the box geometry. The old `edge` term could not see this:
          it only ever docked the width of the 1% margin, so a car with a third of
          its body out of frame (bbox 222x707, y1=0) scored edge = (222-7.2)/222 =
          0.968 and sailed through the 0.9 gallery gate. That is exactly how CAM-04
          wrote a 707px-wide sliver into DJS-7842's gallery.
        * ASPECT: 0.0 when the crop is far too wide to be a whole car. Facility crops
          have a median w/h of 1.40; a 707x222 box (3.2) is a sliver, not a vehicle.
          Truncation and aspect catch overlapping-but-different failures — a car cut
          off at the BOTTOM of frame stays plausibly-shaped, and a badly-merged box
          can be sliver-shaped without touching any border.
        * edge fraction: 1.0 when the bbox sits entirely inside the frame (1%
          edge margin), toward 0.0 as an edge clips it (part of the car out of
          view). Retained as a soft term for boxes that are near, but not on, a border.
        * apparent-size factor: ramps 0→1 with the car's on-screen HEIGHT
          (_VQ_MIN_H → _VQ_GOOD_H). A distant, low-resolution car makes a poor
          ReID crop regardless of framing, so it must NOT rank as a "full view".
        * neighbour clearance (D9): 1.0 unobstructed → 0.0 covered by another
          car's box. Folded in ONLY when ``detections`` is supplied AND
          gallery_neighbour_clearance_enforce is set; otherwise it is computed
          and logged (log-only) so its distribution can be studied before it
          gates. Passing ``detections=None`` (e.g. unit tests) skips it entirely.

        Used as an ownership tie-breaker (display/attribution) AND as the
        gallery-reference quality gate — so a far camera whose tiny car is fully
        framed no longer outranks a near camera that sees the car large, and its
        low-resolution crops are kept out of the gallery. Never a detection
        decision.
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in detection.bbox)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                return 0.0

            # Truncated: the box runs off the frame, so this is a fragment of a car.
            if (
                x1 <= _VQ_BORDER_PX
                or y1 <= _VQ_BORDER_PX
                or x2 >= w - _VQ_BORDER_PX
                or y2 >= h - _VQ_BORDER_PX
            ):
                logger.debug(
                    "[quality] track=%s TRUNCATED at frame border "
                    "(bbox=%.0f,%.0f,%.0f,%.0f frame=%dx%d) -> quality 0",
                    getattr(detection, "track_id", "?"), x1, y1, x2, y2, w, h,
                )
                return 0.0

            # Sliver: too wide to be a whole car (median facility aspect is 1.40).
            if (bw / bh) > _VQ_MAX_ASPECT:
                logger.debug(
                    "[quality] track=%s ASPECT %.2f > %.2f (%.0fx%.0f) — a sliver, "
                    "not a car -> quality 0",
                    getattr(detection, "track_id", "?"), bw / bh, _VQ_MAX_ASPECT, bw, bh,
                )
                return 0.0

            mx, my = 0.01 * w, 0.01 * h
            ix = max(0.0, min(x2, w - mx) - max(x1, mx))
            iy = max(0.0, min(y2, h - my) - max(y1, my))
            edge = max(0.0, min(1.0, (ix * iy) / (bw * bh)))
            vq_min_h, vq_good_h = _vq_size_thresholds(h)
            size = max(0.0, min(1.0, (bh - vq_min_h) / (vq_good_h - vq_min_h)))
            base = edge * size
            if detections is None:
                return base
            clearance = self._neighbour_clearance(detection, detections)
            enforce = self._clearance_enforced()
            if clearance < 1.0:
                logger.info(
                    "[quality] track=%s clearance=%.3f base=%.3f -> %.3f (enforce=%s)",
                    getattr(detection, "track_id", "?"),
                    clearance, base, base * clearance, enforce,
                )
            return base * clearance if enforce else base
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
        ranked = self._rank_zone_detections(frame, detections, zone)
        return ranked[0] if ranked else None

    def _rank_zone_detections(
        self,
        frame: np.ndarray,
        detections: List,
        zone: ParkingSlot,
    ) -> List:
        """
        Rank the snapshot-ready in-zone cars best-first (the primary is [0]).

        The Park_Entry bind walks this order and stops at the first car the
        registry accepts, so a car that CANNOT bind (a D3 lingerer, or one
        already bound) no longer blocks the car behind it.
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

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [detection for _, detection in candidates]

    def _process_park_entry_zone(
        self,
        cam_id: str,
        frame: np.ndarray,
        detections: List,
        zone: ParkingSlot,
    ):
        """Handle CAM_01 logic for Park_Entry.

        D1: the pending plate goes to the PRIMARY car in the zone, not to whoever
        the tracker happened to list first.

        D3: the bind walks the cars in primary order and stops at the first one the
        registry accepts. A car that cannot legitimately own the plate — a lingerer
        that was already sitting in the zone when the plate was read, or one that is
        already bound — is skipped rather than allowed to steal it. Walking on to the
        next car also matters: the bind requires status == "open" and only the
        best-ranked car used to attempt it, so a stationary lingerer (which scores
        HIGH on overlap/depth/area, and so wins the primary slot) would otherwise
        block every car behind it from ever binding.
        """
        in_zone_detections = [
            d for d in detections
            if d.track_id != -1 and self._detection_in_zone(d, zone)
        ]
        currently_inside = {d.track_id for d in in_zone_detections}

        if cam_id == "CAM-23":
            for detection in detections:
                if detection.track_id == -1:
                    continue
                logger.debug(
                    "[PARK_ENTRY] Track %d zone-contains: %s (bottom_center=%s)",
                    detection.track_id,
                    detection.track_id in currently_inside,
                    detection.bottom_center,
                )

        # Pass 1 — every in-zone car gets a candidate and a fresh snapshot.
        candidate_ids = {}
        for detection in in_zone_detections:
            candidate_id = self._park_entry_track_to_candidate.get(detection.track_id)
            if not candidate_id:
                candidate = self.vehicle_registry.open_park_entry_candidate(
                    cam_id,
                    detection.track_id,
                )
                candidate_id = candidate.candidate_id
                self._park_entry_track_to_candidate[detection.track_id] = candidate_id
            candidate_ids[detection.track_id] = candidate_id

            crop = self._crop_detection(frame, detection)
            if crop is not None:
                quality = self._score_snapshot_quality(detection, crop)
                self.vehicle_registry.update_park_entry_candidate_snapshot(
                    candidate_id,
                    crop,
                    quality,
                )
            elif cam_id == "CAM-23":
                logger.debug(
                    "[PARK_ENTRY] Track %d crop is None (bbox=%s)",
                    detection.track_id, detection.bbox,
                )

        # Pass 2 — bind at most ONE pending plate, walking the cars in primary order.
        # Solo fallback (b3ef313): a lone car must still bind when _rank_zone_detections
        # abstains, because its snapshot-ready test is stricter than _detection_in_zone.
        # With several cars present only the ranked (snapshot-ready) ones are eligible,
        # so the order is deterministic and D1's anti-tailgate preference holds.
        if len(in_zone_detections) == 1:
            bind_order = [in_zone_detections[0].track_id]
        else:
            bind_order = [
                d.track_id
                for d in self._rank_zone_detections(frame, detections, zone)
                if d.track_id in candidate_ids
            ]

        bound_plates = {}
        for track_id in bind_order:
            bound_plate = self.vehicle_registry.bind_next_pending_anpr_to_candidate(
                candidate_ids[track_id]
            )
            if cam_id == "CAM-23":
                logger.info(
                    "[PARK_ENTRY] Track %d bind_next_pending_anpr_to_candidate -> plate=%s",
                    track_id, bound_plate,
                )
            if bound_plate:
                bound_plates[track_id] = bound_plate
                break

        # Pass 3 — a successful bind fires once (the candidate leaves "open"), so this
        # is the natural moment to create the car's durable per-plate gallery folder
        # from its first good Park_Entry view — guaranteeing every entering car gets a
        # folder here, at CAM-23, not the gate. Retried every frame (Fix 4): even when
        # THIS frame did not bind (bound earlier, or the pending event was consumed),
        # the candidate still carries its plate — resolve it and re-attempt the
        # idempotent seed so a good LATER crop still lands the CAM-23 top view instead
        # of being lost when the first in-zone frame was poor.
        for track_id, candidate_id in candidate_ids.items():
            plate_for_seed = (
                bound_plates.get(track_id)
                or self.vehicle_registry.plate_for_park_entry_candidate(candidate_id)
            )
            if not plate_for_seed:
                continue
            seeded_ok = self.vehicle_registry.seed_gallery_from_park_entry(
                candidate_id, plate_for_seed
            )
            if cam_id == "CAM-23":
                logger.info(
                    "[PARK_ENTRY] seed_gallery_from_park_entry candidate=%s plate=%s -> %s",
                    candidate_id, plate_for_seed, seeded_ok,
                )

        last_track_ids = self._tracks_inside_zones.get((cam_id, zone.id), set())
        left_zone = last_track_ids - currently_inside
        for track_id in left_zone:
            if track_id in self._park_entry_track_to_candidate:
                # Drop only the track->candidate map entry. Do NOT expire the
                # candidate here: a one-frame ByteTrack dropout/ID-switch briefly
                # removes a still-present car from currently_inside, and actively
                # expiring on that flicker permanently kills a candidate that was
                # about to bind. The liveness sweep (last_seen_at) retires genuinely
                # departed candidates.
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
                crop = self._crop_detection_to_zone(
                    frame,
                    detection,
                    zone,
                    padding_ratio=0.12,
                    mask_outside_zone=True,
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

    # Budget per TRACK. PaddleOCR is ~200ms on the frame loop, so this must stay tight:
    # a car drives up the aisle for several seconds and we only need ONE frame in which
    # its plate faces the camera.
    _OCR_TRACK_MAX_ATTEMPTS = 8
    _OCR_TRACK_MIN_INTERVAL_S = 1.0

    def _try_ocr_identify_tracks(self, cam_id, frame, detections, now_ts) -> None:
        """Read the plate off a still-driving car and bind its identity to the track.

        Only for tracks that are ANONYMOUS (no plate yet) on a camera that hosts slots —
        i.e. a car heading for a parking space on this camera. Bounded per track, and the
        budget dies with the track.
        """
        registry = self.vehicle_registry
        if registry is None or not getattr(self, "pipelines", None):
            return
        pipeline = self.pipelines.get(cam_id)
        if pipeline is None or not getattr(pipeline, "slots", None):
            return  # a camera with no slots has no car to identify for a slot

        if not hasattr(self, "_ocr_track_attempts"):
            self._ocr_track_attempts, self._ocr_track_last_at = {}, {}
            self._ocr_track_first_seen = {}

        for detection in detections:
            tid = getattr(detection, "track_id", -1)
            if tid == -1:
                continue
            key = (cam_id, tid)
            # When did we FIRST see this car? The transit hop needs it: a car already
            # being tracked before another car's plate was read cannot be that car.
            self._ocr_track_first_seen.setdefault(key, now_ts)
            if registry.get_plate_for_track(cam_id, tid):
                continue  # already identified — nothing to read
            if self._ocr_track_attempts.get(key, 0) >= self._OCR_TRACK_MAX_ATTEMPTS:
                continue
            if now_ts - self._ocr_track_last_at.get(key, 0.0) < self._OCR_TRACK_MIN_INTERVAL_S:
                continue

            crop = self._crop_detection(frame, detection)
            if crop is None or crop.size == 0:
                continue

            self._ocr_track_attempts[key] = self._ocr_track_attempts.get(key, 0) + 1
            self._ocr_track_last_at[key] = now_ts

            plate = registry.try_ocr_identify_track(cam_id, tid, crop)
            if plate:
                logger.info(
                    "[ocr-id] track (%s, %s) IDENTIFIED as %s while driving — "
                    "the plate is known BEFORE it parks",
                    cam_id, tid, plate,
                )
                continue

            # THE TRANSIT HOP. This camera cannot read a plate — CAM-21 is mounted
            # side-on to its aisle and has produced ZERO successful reads, parked or
            # moving, because a car is in profile the entire time it is in frame. The
            # information simply is not in the image.
            #
            # But every car entering this facility passes CAM-20, which reads plates
            # reliably. So the car in front of us was READ seconds ago, metres away, and
            # has not parked yet. Adopt that identity — then capture the SIDE-VIEW
            # reference, which is the thing ReID has never had. Every gallery reference
            # today is a front-on gate photo, which is why a side view scores 0.583 for
            # the right car and 0.634 for a wrong one. One side reference turns that into
            # a same-view match (~0.9), and from the car's SECOND visit ReID does this on
            # its own and this hop is never needed again.
            self._try_adopt_transit_identity(cam_id, tid, key, crop, now_ts, detection)

    def _try_adopt_transit_identity(self, cam_id, tid, key, crop, now_ts, detection) -> None:
        registry = self.vehicle_registry
        cfg = getattr(getattr(registry, "_matching_config", None), "__dict__", {})
        if not cfg.get("slot_acquire_by_ocr_transit", True):
            return

        first_seen = self._ocr_track_first_seen.get(key)
        if first_seen is None:
            return

        # A car sitting in an OCCUPIED slot is PARKED. It cannot be the car that just
        # drove past CAM-20, no matter how new its track looks.
        #
        # This is the hole the "appeared after the read" guard does not close: ByteTrack
        # loses and re-acquires IDs, so a car that has been parked for hours can suddenly
        # present a brand-new track whose first_seen is AFTER someone else's plate was
        # read. Unattended, in a full garage, that would stamp wrong plates across many
        # slots — the same way acquisition-by-elimination put DJS-7842 on a parked Nissan
        # Sunny. Ask where the car IS, not just how old its track is.
        if detection is not None and self._detection_in_occupied_slot(cam_id, detection):
            return

        transit = registry.ocr_transit_candidates()
        if len(transit) != 1:
            if len(transit) > 1:
                logger.info(
                    "[ocr-id] cam=%s transit hop declined: %d cars in transit (%s) — "
                    "ambiguous, refusing to guess",
                    cam_id, len(transit), ",".join(s.plate for s, _ in transit),
                )
            return

        session, identified_at = transit[0]
        # The car we are looking at must have appeared AFTER the other car was read.
        # A car already being tracked here beforehand was here first, so it cannot be
        # the one that just drove past CAM-20. This is the guard that stopped the
        # Nissan Sunny from being handed someone else's plate.
        if first_seen < identified_at.timestamp():
            logger.info(
                "[ocr-id] cam=%s transit hop REFUSED for %s: this car was already "
                "being tracked before that plate was read — it was here first",
                cam_id, session.plate,
            )
            return

        plate = registry.adopt_transit_identity(cam_id, tid, session.session_id)
        if not plate:
            return
        logger.info(
            "[ocr-id] cam=%s track %s ADOPTED %s in transit (read at %s, the only car "
            "in transit) — this camera cannot read a plate, so the read one is carried",
            cam_id, tid, plate, identified_at.strftime("%H:%M:%S"),
        )
        # The whole point: learn what this car looks like FROM THIS CAMERA.
        registry.save_parked_reference(plate, crop, cam_id)

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

        # OCR the car WHILE IT IS STILL DRIVING, before it reaches a slot.
        #
        # Slot-level OCR cannot solve a side-on slot: CAM-21 frames B1_CRO in pure
        # profile, so a car parked there shows no plate at all — not when settled, not
        # even while turning in. The plate is only ever visible on the APPROACH, when the
        # car is driving up the aisle with its front or rear toward the camera. Those are
        # the frames we were throwing away by only looking inside slot polygons.
        #
        # Identify the track here, and the plate is already known by the time it parks —
        # the existing slot-linking then carries it into current_plate with no guessing.
        self._try_ocr_identify_tracks(cam_id, frame, detections, now_ts)

        for detection in detections:
            if detection.track_id == -1:
                continue

            # Record how fully this camera sees the car — feeds the ReID
            # ownership tie-breaker (a full view outranks a clipped one).
            self.vehicle_registry.record_track_view(
                cam_id,
                detection.track_id,
                self._bbox_view_quality(frame, detection, detections),
            )

            track_key = (cam_id, detection.track_id)
            session_id = self.vehicle_registry.get_session_id_for_track(
                cam_id,
                detection.track_id,
            )

            if session_id:
                # If the session already has a confirmed plate, the car is
                # identified — skip all ReID computation. Gate on the
                # SESSION's plate, not the ownership-filtered
                # get_plate_for_track(): on a non-owner camera that getter
                # returns None even for a plated session, which used to drop
                # us into the anonymous-track branch below and overwrite the
                # confirmed session's primary feature vector with this
                # camera's (possibly clipped) view every frame.
                if self.vehicle_registry.get_session_plate(session_id):
                    self.vehicle_registry.refresh_track_binding(cam_id, detection.track_id, session_id)
                    # Presence writes and gallery accumulation remain
                    # owner-only side effects, attributed via the
                    # ownership-filtered getter.
                    plate = self.vehicle_registry.get_plate_for_track(cam_id, detection.track_id)
                    if plate:
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
                        # Grow this car's persistent gallery with a fresh full-view
                        # crop (throttled/gated/deduped inside the registry) so its
                        # ReID profile keeps improving and survives restart. Two
                        # capture guards, so a reference is only ever taken from
                        # the genuine moving entrant:
                        #   * skip a car sitting in an already-occupied slot — it
                        #     is parked, not the newly entered car (its crop must
                        #     not enter a gallery);
                        #   * skip while the car is still inside the B1_Entrance
                        #     (CAM-03) zone — wait until it has LEFT, so the ref is
                        #     a clean profile, not a mid-transit partial view.
                        if self._detection_in_occupied_slot(cam_id, detection):
                            continue
                        if self._track_in_entrance_zone(cam_id, detection.track_id):
                            continue
                        # Slot authority: only teach a car that is inside a slot THIS
                        # camera hosts. Owning a car used to be the only camera check
                        # here, so any aisle camera that won the ownership contest
                        # could write references for a car parked in another camera's
                        # slot. Combined with the guard above (skip already-occupied
                        # slots), what survives is precisely the car currently
                        # settling into MY not-yet-confirmed slot — which is the
                        # parked-pose reference the gallery was missing, and the one
                        # a slot camera needs to re-identify that car later.
                        if (
                            self._slot_authority_required()
                            and not self._detection_in_own_slot(cam_id, detection)
                        ):
                            logger.debug(
                                "[gallery] %s refused reference for plate=%s: car is "
                                "not in a slot this camera hosts",
                                cam_id, plate,
                            )
                            continue
                        ref_crop = self._crop_detection(frame, detection)
                        if ref_crop is not None:
                            self.vehicle_registry.record_reference_for_track(
                                cam_id,
                                detection.track_id,
                                ref_crop,
                                self._bbox_view_quality(frame, detection, detections),
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
