"""
multiview_fusion.py — one identity per car across an area's cameras (zoning).

Within an area, several overlapping cameras may see the same car simultaneously.
The system already solves this with single-camera ownership: ``observing_tracks``
/ ``observing_scores`` on the session and the registry's owner-camera resolution
pick the single best camera to "own" the identity. ``IntraAreaFusion`` simply
**restricts that ownership contest to the cameras of one area**, so a car is
owned by the best view *within its area* rather than across the whole building.

Foundation status: this is a thin wrapper that delegates to the existing
ownership logic; the area-restricted owner resolution is a deferred tuning task.
For now ``resolve_owner`` returns the session's current owner unchanged (safe
no-op), so wiring it into the engine does not alter today's behaviour.
"""

from __future__ import annotations

from typing import Optional

from src.zoning.area_registry import AreaRegistry


class IntraAreaFusion:
    def __init__(self, area_registry: AreaRegistry):
        self._areas = area_registry

    def area_of(self, session) -> str:
        """Best-known area for a session, from its owner camera's area."""
        owner = session.owner_camera or session.last_seen_camera
        return self._areas.area_for_camera(owner) if owner else ""

    def resolve_owner(self, session) -> Optional[str]:
        """Return the camera that should own this car's identity, considering
        only cameras in the car's area.

        Scaffolding: delegates to the existing owner today (no behaviour
        change). The area-restricted contest using ``observing_scores`` filtered
        to ``AreaRegistry.cameras(area)`` is the deferred AI-Dev task.
        """
        # TODO(zoning): pick max observing_scores among cameras in this area.
        return session.owner_camera
