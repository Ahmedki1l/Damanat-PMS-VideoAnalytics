"""
area_registry.py — runtime view of the parking-area topology (zoning).

``AreaRegistry`` is a thin, read-only index built from :class:`AppConfig`
(``areas`` + per-camera ``area``). It answers the questions the bounded
matcher / handoff matcher / state machine ask:

  - which area does a camera belong to?            ``area_for_camera``
  - which cameras cover an area?                    ``cameras``
  - what is an area's physical capacity?            ``capacity``
  - which areas are adjacent, and how far (secs)?   ``adjacent`` / ``transit_seconds``

It holds no live state — that lives on the registry's ``_area_sessions`` index
and on each :class:`VehicleSession`. When ``AppConfig`` has no areas every
lookup returns an empty/zero default, which keeps the whole system in its
legacy un-zoned mode.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config import AppConfig, AreaEntry


class AreaRegistry:
    """Read-only resolver for camera→area + area topology."""

    def __init__(self, config: AppConfig):
        self._config = config
        # area_id → AreaEntry
        self._areas: Dict[str, AreaEntry] = {
            a.area_id: a for a in config.areas if a.area_id
        }
        # camera_id → area_id (only zoned cameras)
        self._camera_area: Dict[str, str] = {
            c.id: c.area for c in config.cameras if c.area
        }
        # area_id → [camera_id, ...]
        self._area_cameras: Dict[str, List[str]] = {}
        for cam_id, area_id in self._camera_area.items():
            self._area_cameras.setdefault(area_id, []).append(cam_id)

    # --- presence -----------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when at least one area is defined (zoned deployment)."""
        return bool(self._areas)

    def exists(self, area_id: str) -> bool:
        return area_id in self._areas

    def all_area_ids(self) -> List[str]:
        return list(self._areas.keys())

    # --- camera ↔ area ------------------------------------------------
    def area_for_camera(self, camera_id: str) -> str:
        """Return the area_id a camera covers, or "" if un-zoned/unknown."""
        return self._camera_area.get(camera_id, "")

    def cameras(self, area_id: str) -> List[str]:
        """Return the camera ids assigned to an area."""
        return list(self._area_cameras.get(area_id, ()))

    # --- area attributes ----------------------------------------------
    def get(self, area_id: str) -> Optional[AreaEntry]:
        return self._areas.get(area_id)

    def floor(self, area_id: str) -> str:
        area = self._areas.get(area_id)
        return area.floor if area else ""

    def capacity(self, area_id: str) -> int:
        area = self._areas.get(area_id)
        return area.capacity if area else 0

    # --- topology -----------------------------------------------------
    def adjacent(self, area_id: str) -> Dict[str, float]:
        """Return {neighbor_area_id: transit_seconds} for an area."""
        area = self._areas.get(area_id)
        return dict(area.adjacency) if area else {}

    def is_adjacent(self, area_a: str, area_b: str) -> bool:
        return area_b in self.adjacent(area_a)

    def transit_seconds(self, area_from: str, area_to: str) -> Optional[float]:
        """Expected transit time between two adjacent areas, or None if they
        are not directly connected."""
        return self.adjacent(area_from).get(area_to)
