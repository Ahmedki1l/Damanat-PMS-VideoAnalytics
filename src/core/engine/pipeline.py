import cv2
import numpy as np
from shapely.geometry import Polygon
from typing import Dict, List, Optional

from src.config import AppConfig
from src.models.slot import ParkingSlot
from src.models.state_machine import SlotStateMachine
from src.core.slot_assigner import SlotAssigner




class CameraPipeline:
    """
    Per-camera processing state.

    Holds the slot polygons, state machines, and assigner
    specific to one camera's view.
    """

    def __init__(
        self,
        camera_id: str,
        floor: str,
        slots: List[ParkingSlot],
        config: AppConfig,
        violation_slots: set = None,
        roi_polygon: Optional[Polygon] = None,
    ):
        self.camera_id = camera_id
        self.floor = floor
        self.slots = slots
        self.roi_polygon = roi_polygon
        self._mask = None # Cached binary mask

        # Per-slot state machines — pass is_violation_zone from DB
        self.state_machines: Dict[str, SlotStateMachine] = {}
        for slot in slots:
            is_restricted = (violation_slots is not None) and (slot.id in violation_slots)
            self.state_machines[slot.id] = SlotStateMachine(
                slot_id=slot.id,
                confirm_enter_frames=config.state_machine.confirm_enter_frames,
                confirm_leave_frames=config.state_machine.confirm_leave_frames,
                is_violation_zone=is_restricted,
            )

        # Slot assigner for this camera's polygons
        self.assigner = SlotAssigner(slots=slots, config=config.assigner)

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the ROI mask to the frame, blacking out areas outside the ROI.
        """
        if self.roi_polygon is None:
            return frame

        h, w = frame.shape[:2]
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(self._mask, [pts], 255)

        return cv2.bitwise_and(frame, frame, mask=self._mask)

    @property
    def slot_count(self) -> int:
        return len(self.slots)
