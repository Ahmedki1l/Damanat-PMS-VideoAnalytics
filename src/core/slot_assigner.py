"""
slot_assigner.py — Assigns detected vehicles to parking slots.

Assignment strategy:
  1. PRIMARY: Compute the bottom-center point of each vehicle's bounding box.
     Test this point against each slot polygon using Shapely's point-in-polygon.
  2. FALLBACK: If the bottom-center is not inside any slot, compute the overlap
     ratio between the vehicle's bounding box and each slot polygon. If the
     overlap exceeds a threshold, assign the vehicle to that slot.
  3. TIE-BREAKING: If multiple vehicles map to the same slot, the one whose
     bottom-center is closest to the slot's centroid wins.

Why bottom-center?
  - In a fixed overhead or angled camera, the bottom-center of a bounding box
    approximates the vehicle's ground contact point.
  - This is more stable than using the bbox center, which shifts with vehicle
    height and perspective angle.
"""

import math
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon, box as shapely_box

from src.config import AssignerConfig
from src.detection.detector import Detection
from src.models.slot import ParkingSlot


class SlotAssignment:
    """
    Result of slot assignment for one frame.

    Attributes:
        slot_vehicle_map: Maps slot_id -> (track_id, detection).
        unassigned: Detections that don't fall in any slot.
    """

    def __init__(self):
        self.slot_vehicle_map: Dict[str, Tuple[int, Detection]] = {}
        self.unassigned: List[Detection] = []


class SlotAssigner:
    """Assigns vehicle detections to parking slot polygons."""

    def __init__(self, slots: List[ParkingSlot], config: AssignerConfig):
        """
        Args:
            slots: List of ParkingSlot instances with Shapely polygons.
            config: AssignerConfig with overlap_threshold.
        """
        self.slots = slots
        self.overlap_threshold = config.overlap_threshold

    def assign(self, detections: List[Detection]) -> SlotAssignment:
        """
        Assign each detection to the best matching slot.

        Args:
            detections: List of Detection objects from this frame.

        Returns:
            SlotAssignment with slot→vehicle mapping.
        """
        result = SlotAssignment()

        # Collect all candidate (slot, detection, distance) triples
        candidates: List[Tuple[str, Detection, float]] = []

        # Track a simple counter for detections without stable IDs
        temp_id_counter = -100

        for det in detections:
            # Assign a temporary ID if tracker hasn't assigned one
            # (common in round-robin multi-cam where tracker state resets)
            if det.track_id == -1:
                det.track_id = temp_id_counter
                temp_id_counter -= 1

            assigned = False
            bc_x, bc_y = det.bottom_center
            bc_point = Point(bc_x, bc_y)

            # --- PRIMARY: bottom-center point-in-polygon ---
            for slot in self.slots:
                if slot.polygon.contains(bc_point):
                    dist = self._distance_to_centroid(bc_x, bc_y, slot)
                    candidates.append((slot.id, det, dist))
                    assigned = True
                    break  # One vehicle can only be in one slot

            if assigned:
                continue

            # --- FALLBACK: bbox-polygon overlap ---
            det_box = shapely_box(
                det.bbox[0], det.bbox[1],
                det.bbox[2], det.bbox[3],
            )

            best_overlap = 0.0
            best_slot_id = None
            best_dist = float("inf")

            for slot in self.slots:
                overlap = self._compute_overlap(det_box, slot.polygon)
                # Debug: show why assignment fails
                contains = slot.polygon.contains(bc_point)
                # print(f"[DEBUG-ASSIGN] Track:{det.track_id} → Slot:{slot.id} | "
                #       f"center=({bc_x:.0f},{bc_y:.0f}) in_poly={contains} | "
                #       f"overlap={overlap:.2%} (thresh={self.overlap_threshold:.0%})")

                if overlap > self.overlap_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_slot_id = slot.id
                    best_dist = self._distance_to_centroid(bc_x, bc_y, slot)

            if best_slot_id is not None:
                candidates.append((best_slot_id, det, best_dist))
            else:
                result.unassigned.append(det)

        # --- TIE-BREAKING: if multiple vehicles map to the same slot ---
        # Group by slot, keep the one closest to centroid
        slot_candidates: Dict[str, List[Tuple[Detection, float]]] = {}
        for slot_id, det, dist in candidates:
            if slot_id not in slot_candidates:
                slot_candidates[slot_id] = []
            slot_candidates[slot_id].append((det, dist))

        for slot_id, entries in slot_candidates.items():
            # Sort by distance to centroid — closest wins
            entries.sort(key=lambda x: x[1])
            winner_det, _ = entries[0]
            result.slot_vehicle_map[slot_id] = (winner_det.track_id, winner_det)

            # Others are unassigned
            for det, _ in entries[1:]:
                result.unassigned.append(det)

        return result

    @staticmethod
    def _distance_to_centroid(x: float, y: float, slot: ParkingSlot) -> float:
        """Euclidean distance from point (x, y) to slot centroid."""
        dx = x - slot.centroid_x
        dy = y - slot.centroid_y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _compute_overlap(det_box: Polygon, slot_polygon: Polygon) -> float:
        """
        Overlap ratio = intersection_area / detection_box_area — the fraction
        of the *vehicle's* bounding box that lies inside the slot polygon.

        A ratio > threshold means the vehicle is likely in the slot even if
        its ground-contact point is slightly outside. Normalising by the
        detection box (not the slot) keeps the gate perspective-robust: a small
        car fully inside a large near slot still scores ~1.0, and a truck merely
        clipping a small far slot scores low. Previously this divided by
        ``slot_polygon.area``, which scaled with slot pixel size under
        perspective (a truck could "steal" a far slot; a small car in a near
        slot was rejected). NOTE: ``overlap_threshold`` was tuned under the old
        slot-area semantics — re-validate it on real footage.
        """
        if not det_box.intersects(slot_polygon):
            return 0.0

        try:
            intersection_area = det_box.intersection(slot_polygon).area
            det_area = det_box.area
            if det_area == 0:
                return 0.0
            return intersection_area / det_area
        except Exception:
            return 0.0
