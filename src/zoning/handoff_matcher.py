"""
handoff_matcher.py — cross-area handoff candidate builder (zoning).

When a car appears in an area, the bounded matcher should compare it not only
against cars currently *in* that area, but also against cars that **recently
departed an adjacent area** and could plausibly have travelled here within the
expected transit time. ``CrossAreaHandoffMatcher`` builds exactly that candidate
set; the actual ReID scoring/threshold decision still runs in
``match_global_session`` (this class only narrows *which* sessions are eligible).

Foundation status: the candidate-set construction below is real and used by the
bounded matcher. The only deferred ("scaffolding") part is operating-point
tuning — slack factor and any per-boundary threshold drop — which needs real
footage to calibrate. ``DEFAULT_TRANSIT_SLACK`` is a conservative placeholder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Optional, Set

from src.zoning.area_registry import AreaRegistry

# A car may legitimately take longer than the nominal transit time (traffic,
# waiting). Multiply the configured transit seconds by this slack before
# expiring a departed car from the handoff pool. TODO: tune on real data.
DEFAULT_TRANSIT_SLACK = 2.0

# States a session must be in to be considered "in flight" between areas.
_IN_FLIGHT_STATES = ("DEPARTING", "IN_TRANSIT")


class CrossAreaHandoffMatcher:
    def __init__(
        self,
        area_registry: AreaRegistry,
        clock: Optional[Callable[[], datetime]] = None,
        transit_slack: float = DEFAULT_TRANSIT_SLACK,
    ):
        self._areas = area_registry
        self._clock = clock or datetime.now
        self._slack = transit_slack

    def candidate_session_ids(
        self,
        area_id: str,
        sessions: Iterable,
    ) -> Set[str]:
        """Return session ids of cars that recently departed an area adjacent to
        ``area_id`` and remain within their (slack-extended) transit window.

        ``sessions`` is any iterable of :class:`VehicleSession`. Empty result
        when zoning is off, the area is unknown, or nothing is in flight."""
        out: Set[str] = set()
        if not area_id or not self._areas.enabled:
            return out

        adjacency = self._areas.adjacent(area_id)  # {neighbor_area_id: transit_s}
        if not adjacency:
            return out

        now = self._clock()
        for session in sessions:
            if getattr(session, "area_state", "IN_AREA") not in _IN_FLIGHT_STATES:
                continue
            source = session.departed_from_area or session.current_area
            transit = adjacency.get(source)
            if transit is None:
                continue  # not departing from an adjacent area
            entered = session.area_entered_at
            if entered is None:
                # No timestamp yet — be permissive and include it.
                out.add(session.session_id)
                continue
            elapsed = (now - entered).total_seconds()
            if elapsed <= transit * self._slack:
                out.add(session.session_id)
        return out
