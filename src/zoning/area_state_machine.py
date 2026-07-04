"""
area_state_machine.py — per-car area lifecycle (zoning).

Drives each :class:`VehicleSession` through:

    IN_AREA → DEPARTING → IN_TRANSIT → ARRIVING → IN_AREA

A car is IN_AREA while observed inside one area. A boundary crossing (or losing
the car from every camera of its area) moves it to DEPARTING/IN_TRANSIT, during
which the cross-area handoff matcher treats it as an eligible candidate for the
adjacent area it is travelling to. Re-acquisition in the destination area moves
it to ARRIVING then settles to IN_AREA, and updates the registry's per-area
index via ``set_session_area``.

Foundation status: the transitions + index updates are wired here, but the
debounce thresholds (how many frames / seconds confirm a DEPARTING or an
ARRIVING) are placeholders — they need real footage to tune. Until the boundary
detector and intra-area fusion feed it, this object is constructed but not yet
called from the engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from src.zoning.area_registry import AreaRegistry

# TODO(zoning): tune on real footage. Debounce so a single flicker frame does
# not bounce a car between areas.
ARRIVING_CONFIRM_OBSERVATIONS = 2


class AreaStateMachine:
    def __init__(
        self,
        registry,
        area_registry: AreaRegistry,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self._registry = registry
        self._areas = area_registry
        self._clock = clock or datetime.now

    def _is_ramp(self, area_id: str) -> bool:
        """True for a defined inter-floor ramp: it exists in the topology but
        carries no floor (``floor == ""``). An unknown/blank area is not a
        ramp."""
        return bool(area_id) and self._areas.exists(area_id) and not self._areas.floor(area_id)

    def on_boundary_cross(self, session, area_from: str, area_to: str) -> None:
        """A boundary connecting two areas was crossed. The boundary's stored
        direction is ``area_from -> area_to``, but the same band can be crossed
        either way (e.g. a ramp used both down and up). Resolve the ACTUAL
        direction from where the car currently is: if it is sitting in
        ``area_to``, this is the reverse crossing (``area_to -> area_from``)."""
        if session.current_area == area_to:
            src, dst = area_to, area_from            # reverse crossing
        else:
            src, dst = area_from, area_to            # forward (or unknown area)
        # Crossing INTO an inter-floor ramp: record the ramp itself as the
        # departure area. A ramp with no interior camera (the DOWN ramp: only
        # entrance/exit bands on the adjacent aisle cameras) never settles a car
        # into its own area, so ``departed_from_area`` would otherwise stay the
        # source aisle. But the aisle the car later emerges into is adjacent to
        # the RAMP, not to that source aisle — so the far-end handoff would never
        # match. Keying the departure to the ramp lets the emerging aisle
        # recognise the car as an in-transit arrival off that ramp (and makes the
        # entrance band's source-aisle label irrelevant). Normal aisle→aisle
        # crossings keep the source area unchanged.
        if self._is_ramp(dst):
            session.departed_from_area = dst
        else:
            session.departed_from_area = src or session.current_area
        session.area_state = "IN_TRANSIT"
        session.area_entered_at = self._clock()
        # The car has left the source area — remove it from that area's in-area
        # bucket (and clear current_area) so a bounded match in the old area no
        # longer treats it as an in-area candidate. The handoff matcher still
        # reaches it via ``departed_from_area`` while it is in transit.
        self._registry.set_session_area(session, "")

    def on_fov_exit(self, session) -> None:
        """The car left every camera of its current area without a boundary
        crossing (blind gap). Mark DEPARTING; the handoff window still applies.
        TODO: debounce N seconds before committing this transition."""
        if session.area_state == "IN_AREA" and session.current_area:
            session.departed_from_area = session.current_area
            session.area_state = "DEPARTING"
            session.area_entered_at = self._clock()
            # Left every camera of the area — drop it from the in-area bucket so
            # a bounded match in that area no longer counts it as present.
            self._registry.set_session_area(session, "")

    def on_area_observed(self, session, area_id: str) -> None:
        """The car is being observed in ``area_id`` (the owner camera's area).

        Settles arrival/transit bookkeeping and keeps the registry's per-area
        index in sync. Idempotent when the car is already settled in the area."""
        if not area_id or not self._areas.exists(area_id):
            return
        if session.current_area == area_id and session.area_state == "IN_AREA":
            return  # already settled here

        # Arriving from elsewhere (or first placement) → settle into the area.
        session.area_state = "IN_AREA"
        session.departed_from_area = ""
        self._registry.set_session_area(session, area_id)
