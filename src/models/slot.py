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
from typing import List, Optional

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
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    def __post_init__(self):
        """Pre-compute centroid for fast distance lookups."""
        centroid = self.polygon.centroid
        self.centroid_x = centroid.x
        self.centroid_y = centroid.y


def load_slots(json_path: str) -> List[ParkingSlot]:
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
    ]

    Each polygon must have at least 3 points. The polygon is automatically
    closed by Shapely (no need to repeat the first point).

    Args:
        json_path: Path to the parking slots JSON file.

    Returns:
        List of ParkingSlot instances.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        ValueError: If a polygon has fewer than 3 points.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_slots = json.load(f)

    slots = []
    for entry in raw_slots:
        slot_id = entry["id"]
        points = entry["polygon"]

        if len(points) < 3:
            raise ValueError(
                f"Slot '{slot_id}' has {len(points)} points — need at least 3."
            )

        polygon = Polygon(points)
        label = entry.get("label", slot_id)

        slot = ParkingSlot(
            id=slot_id,
            polygon=polygon,
            label=label,
        )
        slots.append(slot)

    print(f"[INFO] Loaded {len(slots)} parking slots from '{json_path}'")
    return slots
