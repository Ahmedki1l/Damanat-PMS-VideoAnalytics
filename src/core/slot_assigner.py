"""
slot_assigner.py — Assigns detected vehicles to parking slots.

Two assignment modes, selected by ``AssignerConfig.assignment_mode``.

POINT MODE (``"point"``, the default and the historical behaviour):
  1. PRIMARY: Compute the bottom-center point of each vehicle's bounding box.
     Test this point against each slot polygon using Shapely's point-in-polygon.
  2. FALLBACK: If the bottom-center is not inside any slot, compute the overlap
     ratio between the vehicle's bounding box and each slot polygon. If the
     overlap exceeds ``overlap_threshold``, assign the vehicle to that slot.
  3. TIE-BREAKING: If multiple vehicles map to the same slot, the one whose
     bottom-center is closest to the slot's centroid wins.

COVERAGE MODE (``"coverage"``):
  1. There is no point probe. A vehicle is in the slot that its box covers most,
     provided that coverage reaches ``coverage_threshold``.
  2. TIE-BREAKING: If multiple vehicles map to the same slot, the one covering
     it most wins; centroid distance only breaks exact coverage ties.

Why bottom-center? (point mode)
  - In a fixed overhead or angled camera, the bottom-center of a bounding box
    approximates the vehicle's ground contact point.
  - This is more stable than using the bbox center, which shifts with vehicle
    height and perspective angle.

Why a mode flag rather than replacing one with the other?
  The production slot polygons were operator-authored against the point probe's
  particular character (it sits at ``(y1+y2)/1.5``, ABOVE the bbox bottom, and
  the polygons compensate). Coverage mode changes membership at every polygon
  edge, so it needs per-camera revalidation and probably a threshold re-tune
  before it can be trusted. ``assigner.camera_overrides`` lets that happen one
  camera at a time; CameraPipeline resolves it when it builds this object.
  See tests/test_slot_probe_geometry.py for why unconditional changes to slot
  membership have a bad history here.

NOTE: ROI filtering (``CameraPipeline.filter_detections_to_roi``) still drops
detections by the POINT probe, before the assigner ever runs. In coverage mode
a vehicle whose probe falls outside the ROI is therefore never considered, no
matter how much slot area it covers. That is a separate, deliberate decision —
see the same test module.
"""

import math
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, Polygon, box as shapely_box

from src.config import ASSIGNMENT_MODES, AssignerConfig
from src.detection.detector import Detection
from src.models.slot import ParkingSlot


class SlotAssignment:
    """
    Result of slot assignment for one frame.

    Attributes:
        slot_vehicle_map: Maps slot_id -> (track_id, detection).
        unassigned: Detections that don't fall in any slot.
        evidence: Maps slot_id -> why that detection won the slot. Diagnostic
            only; nothing in the pipeline branches on it.

    ``evidence`` exists because "slot X is occupied" was previously unfalsifiable
    from the logs: the assigner knew the track id, the box, the probe point and
    the overlap fraction, and discarded all of it. When a slot stuck OCCUPIED
    there was no way to tell a real parked car from a neighbour's box bleeding
    over a polygon edge without guessing. Each entry carries:

        track_id  : the (possibly synthetic) id that held the slot
        confidence: the detector's confidence for that box [0, 1] — a persistent
                    low-confidence holder points at a ghost/false detection
        class_id  : the detected class id (2 = car) — a non-vehicle class holding
                    a slot points at a misclassification, not a parked car
        bbox      : the detection box, (x1, y1, x2, y2)
        probe     : the ground-contact point tested against the polygon
        method    : "point"    -> probe was inside the polygon (point mode,
                                  primary rule)
                    "overlap"  -> probe was OUTSIDE; won on bbox-overlap
                                  fallback (point mode)
                    "coverage" -> won on box coverage alone (coverage mode)
        overlap   : fraction of the vehicle's box inside this slot's polygon
        rivals    : [(slot_id, overlap), ...] for other slots this box also
                    overlaps — a high rival overlap means the box straddles a
                    boundary and the winner was decided by centroid distance
    """

    def __init__(self):
        self.slot_vehicle_map: Dict[str, Tuple[int, Detection]] = {}
        self.unassigned: List[Detection] = []
        self.evidence: Dict[str, dict] = {}


class SlotAssigner:
    """Assigns vehicle detections to parking slot polygons."""

    #: Assignment modes this class knows how to run. Shared with the config
    #: layer so the YAML validator and the runtime cannot drift apart.
    MODES = ASSIGNMENT_MODES

    def __init__(self, slots: List[ParkingSlot], config: AssignerConfig):
        """
        Args:
            slots: List of ParkingSlot instances with Shapely polygons.
            config: AssignerConfig with overlap_threshold, assignment_mode and
                coverage_threshold.
        """
        self.slots = slots
        self.overlap_threshold = config.overlap_threshold
        self.coverage_threshold = getattr(config, "coverage_threshold", 0.5)

        mode = str(getattr(config, "assignment_mode", "point") or "point").lower()
        if mode not in self.MODES:
            # Degrade to the calibrated default rather than raising — a typo in
            # deployed config must not take the engine down.
            print(
                f"[WARN] Unknown assigner.assignment_mode={mode!r}; "
                f"falling back to 'point'. Valid: {', '.join(self.MODES)}"
            )
            mode = "point"
        self.assignment_mode = mode

    def assign(self, detections: List[Detection]) -> SlotAssignment:
        """
        Assign each detection to the best matching slot.

        Args:
            detections: List of Detection objects from this frame.

        Returns:
            SlotAssignment with slot→vehicle mapping.
        """
        result = SlotAssignment()

        # Collect all candidate (slot, detection, rank_key) triples. rank_key
        # sorts ASCENDING and decides who keeps a contested slot:
        #   point mode    -> (0.0, centroid_distance)  — nearest wins
        #   coverage mode -> (-coverage, centroid_distance) — most covered wins,
        #                    distance only breaks exact coverage ties
        candidates: List[Tuple[str, Detection, Tuple[float, float]]] = []
        # Diagnostic side-tables, keyed by detection identity. Never read by the
        # assignment logic — only folded into result.evidence at the end.
        how: Dict[int, Tuple[str, float]] = {}
        rivals: Dict[int, List[Tuple[str, float]]] = {}

        # Track a simple counter for detections without stable IDs
        temp_id_counter = -100

        for det in detections:
            # Assign a temporary ID if tracker hasn't assigned one
            # (common in round-robin multi-cam where tracker state resets)
            if det.track_id == -1:
                det.track_id = temp_id_counter
                temp_id_counter -= 1

            bc_x, bc_y = det.bottom_center

            # --- COVERAGE MODE: box coverage is the whole rule ---
            if self.assignment_mode == "coverage":
                if not self._assign_by_coverage(
                    det, bc_x, bc_y, candidates, how, rivals
                ):
                    result.unassigned.append(det)
                continue

            assigned = False
            bc_point = Point(bc_x, bc_y)

            # --- PRIMARY: bottom-center point-in-polygon ---
            for slot in self.slots:
                if slot.polygon.contains(bc_point):
                    dist = self._distance_to_centroid(bc_x, bc_y, slot)
                    candidates.append((slot.id, det, (0.0, dist)))
                    how[id(det)] = ("point", self._overlap_for(det, slot))
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
                if overlap > 0.0:
                    rivals.setdefault(id(det), []).append((slot.id, overlap))

                if overlap > self.overlap_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_slot_id = slot.id
                    best_dist = self._distance_to_centroid(bc_x, bc_y, slot)

            if best_slot_id is not None:
                candidates.append((best_slot_id, det, (0.0, best_dist)))
                how[id(det)] = ("overlap", best_overlap)
            else:
                result.unassigned.append(det)

        # --- TIE-BREAKING: if multiple vehicles map to the same slot ---
        # Group by slot, keep the best-ranked claimant (see rank_key above).
        slot_candidates: Dict[str, List[Tuple[Detection, Tuple[float, float]]]] = {}
        for slot_id, det, rank_key in candidates:
            if slot_id not in slot_candidates:
                slot_candidates[slot_id] = []
            slot_candidates[slot_id].append((det, rank_key))

        for slot_id, entries in slot_candidates.items():
            # Ascending rank_key — point mode: closest to centroid wins;
            # coverage mode: highest coverage wins.
            entries.sort(key=lambda x: x[1])
            winner_det, _ = entries[0]
            result.slot_vehicle_map[slot_id] = (winner_det.track_id, winner_det)

            method, own_overlap = how.get(id(winner_det), ("unknown", 0.0))
            result.evidence[slot_id] = {
                "track_id": winner_det.track_id,
                "confidence": round(float(winner_det.confidence), 3),
                "class_id": int(winner_det.class_id),
                "bbox": tuple(round(float(v), 1) for v in winner_det.bbox),
                "probe": tuple(round(float(v), 1) for v in winner_det.bottom_center),
                "method": method,
                "overlap": round(float(own_overlap), 3),
                "rivals": sorted(
                    (
                        (sid, round(ov, 3))
                        for sid, ov in rivals.get(id(winner_det), [])
                        if sid != slot_id
                    ),
                    key=lambda t: -t[1],
                )[:3],
            }

            # Others are unassigned
            for det, _ in entries[1:]:
                result.unassigned.append(det)

        return result

    def _assign_by_coverage(
        self,
        det: Detection,
        bc_x: float,
        bc_y: float,
        candidates: List[Tuple[str, Detection, Tuple[float, float]]],
        how: Dict[int, Tuple[str, float]],
        rivals: Dict[int, List[Tuple[str, float]]],
    ) -> bool:
        """Claim the slot this detection's box covers most, if it covers enough.

        Unlike the point-mode fallback this scans every slot and takes the
        maximum rather than the first slot over the bar, so a box straddling a
        boundary lands in the slot it is actually mostly in. The threshold is
        inclusive (``>=``): ``coverage_threshold=0.5`` means "at least half the
        vehicle's box is inside".

        Returns True if a slot was claimed. Appends every non-zero coverage to
        ``rivals`` regardless, so the log shows what the box was straddling.
        """
        try:
            det_box = shapely_box(det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3])
        except Exception:
            return False

        best_slot_id: Optional[str] = None
        best_coverage = 0.0
        best_dist = float("inf")

        for slot in self.slots:
            coverage = self._compute_overlap(det_box, slot.polygon)
            if coverage <= 0.0:
                continue

            rivals.setdefault(id(det), []).append((slot.id, coverage))

            if coverage >= self.coverage_threshold and coverage > best_coverage:
                best_slot_id = slot.id
                best_coverage = coverage
                best_dist = self._distance_to_centroid(bc_x, bc_y, slot)

        if best_slot_id is None:
            return False

        candidates.append((best_slot_id, det, (-best_coverage, best_dist)))
        how[id(det)] = ("coverage", best_coverage)
        return True

    def _overlap_for(self, det: Detection, slot: ParkingSlot) -> float:
        """Fraction of ``det``'s box inside ``slot``. Diagnostic only.

        Computed for the point-in-polygon winner so the log can distinguish a car
        genuinely sitting in the slot (large fraction) from one whose probe merely
        clipped the polygon edge (small fraction) — the two look identical in the
        assignment result but mean very different things when a slot sticks.
        """
        try:
            det_box = shapely_box(det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3])
            return self._compute_overlap(det_box, slot.polygon)
        except Exception:
            return 0.0

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
