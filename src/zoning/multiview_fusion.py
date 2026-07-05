"""
multiview_fusion.py — one identity per car across an area's cameras (zoning).

Within an area, several overlapping cameras may see the same car simultaneously.
The system already solves this with single-camera ownership: ``observing_tracks``
/ ``observing_scores`` on the session and the registry's owner-camera resolution
pick the single best camera to "own" the identity. ``IntraAreaFusion`` simply
**restricts that ownership contest to the cameras of one area**, so a car is
owned by the best view *within its area* rather than across the whole building.

``resolve_owner`` runs the area-restricted ownership contest: among the cameras
``observing`` the session, it keeps only those assigned to the car's settled
``current_area`` and returns the highest-scoring one (deterministic tie-break).
When zoning is off or the car is un-zoned / in-transit it returns the existing
owner unchanged. It is implemented and unit-tested in isolation; the engine does
not consult it yet, so it is inert until a later wiring step.
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
        """Pick the camera that should own this car's identity, restricted to the
        cameras of the area the car is currently settled in.

        Hysteresis: when zoning is off, or the car is un-zoned / in-transit
        (``current_area == ""``), the existing owner is returned unchanged —
        moving a car between areas is :class:`AreaStateMachine`'s job, never this
        method's. Within the settled area, the highest ``observing_scores`` camera
        wins, with a deterministic tie-break that prefers the incumbent owner (no
        needless churn) and otherwise the lexicographically smallest ``camera_id``.

        All ownership policy lives here: future signals (detection confidence,
        bbox size, track age, view quality, composite score) slot into this method
        without touching :class:`AreaRegistry`.
        """
        if not self._areas.enabled or not session.current_area:
            return session.owner_camera

        allowed = set(self._areas.cameras(session.current_area))
        scores = {
            cam: s for cam, s in session.observing_scores.items() if cam in allowed
        }
        if not scores:
            return session.owner_camera

        best = max(scores.values())
        tied = sorted(cam for cam, s in scores.items() if s == best)
        if session.owner_camera in tied:
            return session.owner_camera
        return tied[0]
