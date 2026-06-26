"""
boundary_detector.py — area-to-area boundary crossing detection (zoning).

A *boundary* is a polygon painted (in one camera's view) across the lane where
two areas meet — typically a ramp throat. When a tracked car's bottom-centre
point enters that polygon, the car is crossing from ``area_from`` to
``area_to``. This drives the area state machine (DEPARTING → IN_TRANSIT).

Foundation status: the point-in-polygon geometry below is real and mirrors the
engine's existing special-zone logic (``shapely`` ``contains`` on the bottom
-centre point). It stays inert until boundary polygons are authored
(``draw_slots.py`` boundary mode, Step 10) — with no boundaries it returns no
crossings. Debounce/anti-jitter tuning is the only deferred piece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from shapely.geometry import Point


@dataclass
class BoundaryZone:
    """A boundary polygon between two areas (loaded from the slot pipeline)."""
    id: str
    polygon: object          # shapely Polygon
    area_from: str
    area_to: str


@dataclass
class BoundaryCrossing:
    """Emitted when a track transitions outside→inside a boundary polygon."""
    camera_id: str
    track_id: int
    boundary_id: str
    area_from: str
    area_to: str


def _bottom_center(bbox: Sequence[float]) -> Point:
    """Approximate the vehicle's ground-contact point (matches slot logic)."""
    x1, y1, x2, y2 = bbox[:4]
    return Point((x1 + x2) / 2.0, y2)


class BoundaryCrossingDetector:
    """Frame-to-frame inside/outside tracker for boundary polygons.

    Stateless w.r.t. areas; it only remembers, per (camera, track), which
    boundaries the track was inside last frame, so it can emit a crossing on the
    outside→inside edge (not every frame the car sits in the zone).
    """

    def __init__(self):
        # (camera_id, track_id) -> set(boundary_id) the track was inside last tick
        self._inside: Dict[Tuple[str, int], set] = {}

    def detect(
        self,
        camera_id: str,
        tracks: Sequence[Tuple[int, Sequence[float]]],
        boundaries: Sequence[BoundaryZone],
    ) -> List[BoundaryCrossing]:
        """Return crossing events for this frame. ``tracks`` is a sequence of
        ``(track_id, bbox)``; ``boundaries`` the camera's boundary zones."""
        crossings: List[BoundaryCrossing] = []
        if not boundaries:
            return crossings

        for track_id, bbox in tracks:
            key = (camera_id, track_id)
            pt = _bottom_center(bbox)
            now_inside = set()
            for b in boundaries:
                try:
                    if b.polygon.contains(pt):
                        now_inside.add(b.id)
                except Exception:
                    continue
            was_inside = self._inside.get(key, set())
            # outside→inside edge = a crossing
            for b in boundaries:
                if b.id in now_inside and b.id not in was_inside:
                    crossings.append(
                        BoundaryCrossing(
                            camera_id=camera_id,
                            track_id=track_id,
                            boundary_id=b.id,
                            area_from=b.area_from,
                            area_to=b.area_to,
                        )
                    )
            self._inside[key] = now_inside
        return crossings

    def forget_track(self, camera_id: str, track_id: int) -> None:
        """Drop per-track state when a track ends (avoids unbounded growth)."""
        self._inside.pop((camera_id, track_id), None)
