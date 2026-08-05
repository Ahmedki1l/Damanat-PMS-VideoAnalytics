from typing import Dict, List, Optional

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from src.config import AppConfig
from src.core.slot_assigner import SlotAssigner
from src.models.slot import ParkingSlot
from src.models.state_machine import SlotState, SlotStateMachine


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
        initial_statuses: Dict[str, bool] = None,
        roi_polygon: Optional[Polygon] = None,
    ):
        self.camera_id = camera_id
        self.floor = floor
        self.slots = slots
        self.roi_polygon = roi_polygon
        self._mask = None
        initial_statuses = initial_statuses or {}

        # Per-camera debounce, if this camera has a state_machine.camera_overrides
        # entry; otherwise the global (DB-owned) counters, unchanged.
        sm_config = config.state_machine.resolve_for_camera(camera_id)
        if sm_config is not config.state_machine:
            print(
                f"[INFO] {camera_id}: state machine override "
                f"confirm_enter_frames={sm_config.confirm_enter_frames}, "
                f"confirm_leave_frames={sm_config.confirm_leave_frames}"
            )

        self.state_machines: Dict[str, SlotStateMachine] = {}
        for slot in slots:
            is_restricted = (violation_slots is not None) and (slot.id in violation_slots)
            db_avail = initial_statuses.get(slot.id, True)
            start_state = SlotState.VACANT if db_avail else SlotState.OCCUPIED

            self.state_machines[slot.id] = SlotStateMachine(
                slot_id=slot.id,
                confirm_enter_frames=sm_config.confirm_enter_frames,
                confirm_leave_frames=sm_config.confirm_leave_frames,
                is_violation_zone=is_restricted,
                initial_state=start_state,
                observation_policy=config.state_machine.observation_policy,
            )

        self.assigner = SlotAssigner(slots=slots, config=config.assigner)

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Apply the ROI mask to the frame, blacking out areas outside the ROI."""
        if self.roi_polygon is None:
            return frame

        h, w = frame.shape[:2]
        if self._mask is None or self._mask.shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(self._mask, [pts], 255)

        return cv2.bitwise_and(frame, frame, mask=self._mask)

    def filter_detections_to_roi(self, detections):
        """Drop detections whose ground-contact point falls outside the ROI.

        This is the post-detection equivalent of ``apply_roi_mask``: it keeps
        out-of-area vehicles (e.g. a neighbouring camera's car visible at the
        frame edge) from reaching the assigner/zoning, WITHOUT blacking out the
        frame before detection. Masking the frame first paints large black
        regions that wreck the detector after it letterboxes 1280x720 down to
        the model's small input (a fully-visible dark car was being missed at
        int8/320). Detecting on the full frame preserves detection quality; the
        ROI is then applied here as a cheap membership test on the same
        bottom-center point ``SlotAssigner`` uses, so the exclusion semantics are
        unchanged. No-op when the camera has no ROI polygon.
        """
        if self.roi_polygon is None:
            return detections
        return [
            d for d in detections
            if self.roi_polygon.contains(Point(*d.bottom_center))
        ]

    @property
    def slot_count(self) -> int:
        return len(self.slots)
