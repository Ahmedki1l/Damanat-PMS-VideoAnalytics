"""
vehicle_registry.py — Vehicle plate-to-slot tracking.

Stores ANPR events (plate + image) and links them to parking slots
via time-window correlation.

Thread-safe for concurrent access from the API and engine.
"""

import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class VehicleRecord:
    """Record of a vehicle seen by ANPR."""
    plate: str
    direction: str                     # "entry" or "exit"
    timestamp: datetime
    image_path: Optional[str] = None   # Path to saved car crop image
    linked_slot: Optional[str] = None  # Slot ID if linked
    linked_camera: Optional[str] = None
    linked_floor: Optional[str] = None
    linked_at: Optional[datetime] = None


class VehicleRegistry:
    """
    Central registry linking ANPR plates to parking slots.

    Flow:
      1. ANPR server sends plate + image → stored as pending entry
      2. Engine detects vehicle entering a slot → tries to link to a pending plate
      3. When linked: we know plate "ABC-1234" is in slot A2 on floor B1
      4. On exit ANPR event → mark the vehicle as departed
    """

    def __init__(self, image_dir: str = "vehicle_images"):
        self._lock = threading.Lock()
        self._pending_entries: List[VehicleRecord] = []   # Awaiting slot link
        self._parked: Dict[str, VehicleRecord] = {}       # slot_id → record
        self._history: List[VehicleRecord] = []           # Completed visits
        self._image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

    def register_anpr_event(
        self,
        plate: str,
        direction: str,
        image: Optional[np.ndarray] = None,
        image_bytes: Optional[bytes] = None,
    ) -> VehicleRecord:
        """
        Register a new ANPR event.

        Args:
            plate: License plate string.
            direction: "entry" or "exit".
            image: OpenCV image (numpy array) of the vehicle.
            image_bytes: Raw image bytes (alternative to numpy array).

        Returns:
            The created VehicleRecord.
        """
        now = datetime.now()
        record = VehicleRecord(plate=plate, direction=direction, timestamp=now)

        # Save image if provided
        if image is not None or image_bytes is not None:
            filename = f"{plate}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self._image_dir, filename)
            if image_bytes is not None:
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
            elif image is not None:
                cv2.imwrite(filepath, image)
            record.image_path = filepath

        with self._lock:
            if direction == "entry":
                self._pending_entries.append(record)
                print(f"[ANPR] Entry: plate={plate} | image={'yes' if record.image_path else 'no'}")
            elif direction == "exit":
                self._handle_exit(plate, now)
                print(f"[ANPR] Exit: plate={plate}")

        return record

    def try_link_to_slot(
        self,
        slot_id: str,
        camera_id: str,
        floor: str,
        timestamp: datetime,
        time_window_seconds: int = 120,
    ) -> Optional[str]:
        """
        Try to link a newly occupied slot to a pending ANPR entry.

        Uses time-window matching: the most recent unlinked plate
        within the time window is linked to this slot.

        Args:
            slot_id: The slot that just became occupied.
            camera_id: Camera that detected the vehicle.
            floor: Floor of the slot.
            timestamp: When the slot became occupied.
            time_window_seconds: How far back to search for matching plates.

        Returns:
            The linked plate string, or None if no match found.
        """
        with self._lock:
            # Search pending entries in reverse (most recent first)
            for record in reversed(self._pending_entries):
                age = (timestamp - record.timestamp).total_seconds()
                if age <= time_window_seconds and record.linked_slot is None:
                    # Link this plate to this slot
                    record.linked_slot = slot_id
                    record.linked_camera = camera_id
                    record.linked_floor = floor
                    record.linked_at = timestamp
                    self._parked[slot_id] = record
                    print(f"[LINK] Plate {record.plate} → {slot_id} ({floor}, {camera_id})")
                    return record.plate
            return None

    def _handle_exit(self, plate: str, timestamp: datetime):
        """Handle an exit ANPR event — find and archive the parked record."""
        # Find the slot this plate is linked to
        slot_to_remove = None
        for slot_id, record in self._parked.items():
            if record.plate == plate:
                slot_to_remove = slot_id
                break

        if slot_to_remove:
            record = self._parked.pop(slot_to_remove)
            self._history.append(record)
            print(f"[EXIT] Plate {plate} left slot {slot_to_remove}")

        # Also remove from pending entries
        self._pending_entries = [
            r for r in self._pending_entries if r.plate != plate
        ]

    def get_slot_plate(self, slot_id: str) -> Optional[str]:
        """Get the plate linked to a specific slot."""
        with self._lock:
            record = self._parked.get(slot_id)
            return record.plate if record else None

    def get_plate_location(self, plate: str) -> Optional[Dict]:
        """Find where a specific plate is parked."""
        with self._lock:
            for slot_id, record in self._parked.items():
                if record.plate == plate:
                    return {
                        "plate": record.plate,
                        "slot_id": slot_id,
                        "camera_id": record.linked_camera,
                        "floor": record.linked_floor,
                        "parked_at": record.linked_at.isoformat() if record.linked_at else None,
                        "entry_time": record.timestamp.isoformat(),
                    }
            return None

    def get_all_parked(self) -> List[Dict]:
        """Get all currently parked vehicles with slot info."""
        with self._lock:
            return [
                {
                    "plate": r.plate,
                    "slot_id": sid,
                    "camera_id": r.linked_camera,
                    "floor": r.linked_floor,
                    "parked_at": r.linked_at.isoformat() if r.linked_at else None,
                    "entry_time": r.timestamp.isoformat(),
                }
                for sid, r in self._parked.items()
            ]

    def get_pending_entries(self) -> List[Dict]:
        """Get plates that entered but haven't been linked to a slot yet."""
        with self._lock:
            return [
                {
                    "plate": r.plate,
                    "entry_time": r.timestamp.isoformat(),
                    "image_path": r.image_path,
                    "linked": r.linked_slot is not None,
                }
                for r in self._pending_entries
                if r.linked_slot is None
            ]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        with self._lock:
            return {
                "parked_count": len(self._parked),
                "pending_entries": sum(1 for r in self._pending_entries if r.linked_slot is None),
                "total_visits": len(self._history),
            }
