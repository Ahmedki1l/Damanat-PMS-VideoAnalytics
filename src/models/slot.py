"""
slot.py — Parking slot data model and JSON loader.

Each parking slot is defined by:
  - A unique ID (e.g., "A1")
  - A polygon representing the slot boundary in pixel coordinates
  - An optional human-readable label

Polygons are loaded from a JSON file and converted to Shapely Polygon
objects for efficient geometric operations.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import Polygon


@dataclass
class ParkingSlot:
    """
    Represents a single parking slot in the camera view.

    Attributes:
        id: Unique identifier (e.g., "A1").
        polygon: Shapely Polygon of the slot boundary.
        label: Human-readable label.
        centroid_x: Pre-computed centroid X for distance calculations.
        centroid_y: Pre-computed centroid Y for distance calculations.
    """
    id: str
    polygon: Polygon
    label: str = ""
    zone_id: str = ""
    zone_name: str = ""
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    def __post_init__(self):
        """Pre-compute centroid for fast distance lookups."""
        centroid = self.polygon.centroid
        self.centroid_x = centroid.x
        self.centroid_y = centroid.y


def load_slots(
    json_path: str,
    default_zone_id: str = "",
    default_zone_name: str = "",
    ref_resolution: Optional[Tuple[int, int]] = None,
    actual_resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[List["ParkingSlot"], Optional[Polygon]]:
    """
    Load parking slot definitions from a JSON file.

    Expected JSON format:
    [
      {
        "id": "A1",
        "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "label": "Slot A1"  // optional
      },
      ...
      {
        "id": "roi",
        "polygon": [[x1, y1], ...] // optional global ROI
      }
    ]

    Args:
        json_path: Path to the parking slots JSON file.
        ref_resolution: (width, height) that polygon coords were drawn at.
        actual_resolution: (width, height) of the live stream frame.
            When both are provided and differ, all polygon vertices are
            scaled proportionally:  x' = x * (actual_w / ref_w)

    Returns:
        Tuple containing:
          - List of ParkingSlot instances.
          - Optional Shapely Polygon representing the ROI.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # --- Compute scale factors ---
    sx, sy = 1.0, 1.0
    if ref_resolution and actual_resolution:
        ref_w, ref_h = ref_resolution
        act_w, act_h = actual_resolution
        if ref_w > 0 and ref_h > 0 and (ref_w != act_w or ref_h != act_h):
            sx = act_w / ref_w
            sy = act_h / ref_h
            print(
                f"[INFO] Scaling polygons in '{json_path}': "
                f"{ref_w}×{ref_h} → {act_w}×{act_h} "
                f"(sx={sx:.4f}, sy={sy:.4f})"
            )

    slots = []
    roi_polygon = None

    for entry in raw_data:
        slot_id = entry["id"]

        # Skip virtual_line entries — they are processed separately by LineCrossingDetector
        if entry.get("type") == "virtual_line":
            print(f"[INFO] Skipping virtual_line entry '{slot_id}' (not a parking slot)")
            continue

        points = entry.get("polygon", [])

        if len(points) < 3:
            print(f"[WARN] Entry '{slot_id}' has {len(points)} points — skipping.")
            continue

        # Apply resolution scaling (no-op when sx=sy=1.0)
        scaled_points = [[p[0] * sx, p[1] * sy] for p in points]
        polygon = Polygon(scaled_points)

        # Check if this is the global ROI definition
        if slot_id.lower() == "roi":
            roi_polygon = polygon
            print(f"[INFO] Found ROI polygon in '{json_path}'")
            continue

        label = entry.get("label", slot_id)
        zone_id = entry.get("zone_id", default_zone_id or "")
        zone_name = entry.get("zone_name", default_zone_name or zone_id or label)
        slot = ParkingSlot(
            id=slot_id,
            polygon=polygon,
            label=label,
            zone_id=zone_id,
            zone_name=zone_name,
        )
        slots.append(slot)

    print(f"[INFO] Loaded {len(slots)} slots from '{json_path}'")
    return slots, roi_polygon