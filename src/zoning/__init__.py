"""
Zoning package — parking-area architecture (B1/B2).

Foundation (working now):
  - :class:`AreaRegistry` — resolves camera→area, area capacity, adjacency,
    transit times from :class:`src.config.AppConfig`.

Scaffolding (wired but inert until real cameras + tuning):
  - :class:`BoundaryCrossingDetector`
  - :class:`AreaStateMachine`
  - :class:`IntraAreaFusion`
  - :class:`CrossAreaHandoffMatcher`

Everything degrades to the legacy single-area behaviour when no areas are
configured (un-zoned deployments).
"""

from src.zoning.area_registry import AreaRegistry
from src.zoning.area_state_machine import AreaStateMachine
from src.zoning.boundary_detector import (
    BoundaryCrossing,
    BoundaryCrossingDetector,
    BoundaryZone,
)
from src.zoning.handoff_matcher import CrossAreaHandoffMatcher
from src.zoning.multiview_fusion import IntraAreaFusion

__all__ = [
    "AreaRegistry",
    "AreaStateMachine",
    "BoundaryCrossing",
    "BoundaryCrossingDetector",
    "BoundaryZone",
    "CrossAreaHandoffMatcher",
    "IntraAreaFusion",
]
