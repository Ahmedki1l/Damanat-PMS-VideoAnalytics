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
) -> Tuple[List[ParkingSlot], Optional[Polygon]]:
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

    Returns:
        Tuple containing:
          - List of ParkingSlot instances.
          - Optional Shapely Polygon representing the ROI.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

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

        polygon = Polygon(points)

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