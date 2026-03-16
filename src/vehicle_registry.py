"""
vehicle_registry.py — Vehicle plate-to-slot tracking with track ID matching.

Stores ANPR events (plate + image) and links them to parking slots.

Linking strategy (in priority order):
  1. Track ID match: if the track_id already has a plate → known car moving slots
  2. Time-window match: most recent unlinked ANPR entry within 2 minutes

This prevents a known parked car from consuming a pending ANPR entry
when it simply moves to a different slot.

Thread-safe for concurrent access from the API and engine.
"""

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VehicleRecord:
    """Record of a vehicle seen by ANPR."""
    plate: str
    direction: str                     # "entry" or "exit"
    timestamp: datetime
    image_path: Optional[str] = None   # Path to saved ANPR image
    anpr_image: Optional[np.ndarray] = None  # ANPR image for visual matching
    linked_slot: Optional[str] = None  # Slot ID if linked
    linked_camera: Optional[str] = None
    linked_floor: Optional[str] = None
    linked_at: Optional[datetime] = None
    track_id: Optional[int] = None     # ByteTrack ID when linked


class VehicleRegistry:
    """
    Central registry linking ANPR plates to parking slots.

    Maintains three key mappings:
      - _pending_entries: ANPR plates awaiting slot assignment
      - _parked: slot_id → VehicleRecord (currently parked)
      - _track_plate_map: (camera_id, track_id) → plate
        Ensures known cars are recognized when they move slots.
    """

    def __init__(self, image_dir: str = "vehicle_images"):
        self._lock = threading.Lock()
        self._pending_entries: List[VehicleRecord] = []    # Awaiting slot link
        self._parked: Dict[str, VehicleRecord] = {}        # slot_id → record
        self._history: List[VehicleRecord] = []            # Completed visits

        # Track ID → plate mapping (per camera)
        # Key: (camera_id, track_id), Value: plate
        self._track_plate_map: Dict[Tuple[str, int], str] = {}

        # Cross-camera re-identification:
        # plate → latest car crop (used to identify the car on ANY camera)
        self._plate_crops: Dict[str, np.ndarray] = {}

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
        Register a new ANPR event (entry or exit).

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

        # Save image and keep in memory for visual matching
        if image is not None or image_bytes is not None:
            filename = f"{plate}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self._image_dir, filename)
            if image_bytes is not None:
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                # Decode bytes to numpy for visual matching
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                record.anpr_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif image is not None:
                cv2.imwrite(filepath, image)
                record.anpr_image = image.copy()
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
        track_id: Optional[int],
        timestamp: datetime,
        time_window_seconds: int = 120,
    ) -> Optional[str]:
        """
        Try to link a newly occupied slot to a plate.

        Priority:
          1. Check if this track_id already has a known plate
             → known car moving to a different slot (just update location)
          2. Search pending ANPR entries (time-window match)
             → new car just entered via ANPR

        Args:
            slot_id: The slot that just became occupied.
            camera_id: Camera that detected the vehicle.
            floor: Floor of the slot.
            track_id: ByteTrack ID of the vehicle.
            timestamp: When the slot became occupied.
            time_window_seconds: How far back to search for ANPR entries.

        Returns:
            The linked plate string, or None if no match found.
        """
        with self._lock:
            # --- Strategy 1: Track ID already has a plate ---
            if track_id is not None:
                key = (camera_id, track_id)
                existing_plate = self._track_plate_map.get(key)
                if existing_plate:
                    # Known car — update its slot location
                    self._move_car_to_slot(
                        existing_plate, slot_id, camera_id, floor, track_id, timestamp
                    )
                    print(f"[MOVE] Known car {existing_plate} (track {track_id}) "
                          f"→ {slot_id} ({floor})")
                    return existing_plate

            # --- Strategy 2: Time-window match with pending ANPR entries ---
            for record in reversed(self._pending_entries):
                age = (timestamp - record.timestamp).total_seconds()
                if age <= time_window_seconds and record.linked_slot is None:
                    # Link this plate to this slot
                    record.linked_slot = slot_id
                    record.linked_camera = camera_id
                    record.linked_floor = floor
                    record.linked_at = timestamp
                    record.track_id = track_id
                    self._parked[slot_id] = record

                    # Register track_id → plate mapping
                    if track_id is not None:
                        self._track_plate_map[(camera_id, track_id)] = record.plate

                    print(f"[LINK] Plate {record.plate} → {slot_id} ({floor}, "
                          f"track={track_id})")
                    return record.plate

            return None

    def _move_car_to_slot(
        self,
        plate: str,
        new_slot_id: str,
        camera_id: str,
        floor: str,
        track_id: int,
        timestamp: datetime,
    ):
        """Move a known car from its current slot to a new slot."""
        # Find and remove from old slot
        old_slot = None
        record = None
        for sid, rec in self._parked.items():
            if rec.plate == plate:
                old_slot = sid
                record = rec
                break

        if old_slot:
            del self._parked[old_slot]
            print(f"[MOVE] {plate}: {old_slot} → {new_slot_id}")

        if record is None:
            # This shouldn't happen, but handle gracefully
            record = VehicleRecord(
                plate=plate, direction="entry", timestamp=timestamp
            )

        # Update slot info
        record.linked_slot = new_slot_id
        record.linked_camera = camera_id
        record.linked_floor = floor
        record.linked_at = timestamp
        record.track_id = track_id
        self._parked[new_slot_id] = record

    def update_track_plate(
        self, camera_id: str, track_id: int, plate: str
    ):
        """
        Manually associate a track_id with a plate.

        Useful when cross-camera matching identifies a car.
        """
        with self._lock:
            self._track_plate_map[(camera_id, track_id)] = plate

    def get_plate_for_track(
        self, camera_id: str, track_id: int
    ) -> Optional[str]:
        """Get the plate associated with a track_id (if any)."""
        with self._lock:
            return self._track_plate_map.get((camera_id, track_id))

    def try_match_by_image(
        self,
        car_crop: np.ndarray,
        track_id: int,
        camera_id: str,
        similarity_threshold: float = 0.4,
    ) -> Optional[str]:
        """
        Match a car crop against pending ANPR images AND known plate crops
        from other cameras (cross-camera re-identification).

        Priority:
          1. Already known on this camera → return immediately
          2. Compare vs pending ANPR images (entrance camera match)
          3. Compare vs known plate crops (cross-camera re-ID)

        When matched, the crop is stored for future cross-camera matching.

        Args:
            car_crop: Cropped car image from the camera frame (BGR).
            track_id: Track ID of the detected car.
            camera_id: Camera that detected the car.
            similarity_threshold: Minimum histogram correlation to match (0-1).

        Returns:
            Plate string if matched, None otherwise.
        """
        if car_crop is None or car_crop.size == 0:
            return None

        with self._lock:
            # Already known on THIS camera?
            existing = self._track_plate_map.get((camera_id, track_id))
            if existing:
                # Update crop for better matching on next camera
                self._plate_crops[existing] = car_crop.copy()
                return existing

            best_score = 0.0
            best_plate = None

            # --- Strategy 1: Compare vs pending ANPR images ---
            for record in self._pending_entries:
                if record.linked_slot is not None:
                    continue
                if record.anpr_image is None:
                    continue

                score = self._compare_images(car_crop, record.anpr_image)
                if score > best_score:
                    best_score = score
                    best_plate = record.plate

            # --- Strategy 2: Compare vs known plate crops (cross-camera) ---
            for plate, known_crop in self._plate_crops.items():
                score = self._compare_images(car_crop, known_crop)
                if score > best_score:
                    best_score = score
                    best_plate = plate

            if best_plate and best_score >= similarity_threshold:
                # Match found! Link plate → track_id on this camera
                self._track_plate_map[(camera_id, track_id)] = best_plate
                # Store/update crop for future cross-camera matching
                self._plate_crops[best_plate] = car_crop.copy()
                print(f"[MATCH] Plate {best_plate} → Track:{track_id} "
                      f"(cam={camera_id}, similarity={best_score:.2%})")
                return best_plate

        return None

    def get_plate_for_any_camera(self, track_id: int) -> Optional[str]:
        """
        Get the plate for a track_id regardless of which camera identified it.

        Searches all camera entries in the track_plate_map.
        """
        with self._lock:
            for (cam, tid), plate in self._track_plate_map.items():
                if tid == track_id:
                    return plate
            return None

    @staticmethod
    def _compare_images(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compare two car images using HSV histogram correlation.

        Returns similarity score between 0.0 (different) and 1.0 (identical).
        """
        try:
            # Resize both to same size for fair comparison
            size = (128, 128)
            a = cv2.resize(img1, size)
            b = cv2.resize(img2, size)

            # Convert to HSV (more robust to lighting changes)
            a_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
            b_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

            # Compute histograms (H: 50 bins, S: 60 bins)
            h_bins, s_bins = 50, 60
            channels = [0, 1]
            ranges = [0, 180, 0, 256]
            hist_size = [h_bins, s_bins]

            hist_a = cv2.calcHist([a_hsv], channels, None, hist_size, ranges)
            cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)

            hist_b = cv2.calcHist([b_hsv], channels, None, hist_size, ranges)
            cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)

            # Correlation: 1.0 = perfect match, -1.0 = inverse
            score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

            # Clamp to [0, 1]
            return max(0.0, score)
        except Exception:
            return 0.0

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
            # Clean up track_id mapping
            if record.track_id and record.linked_camera:
                key = (record.linked_camera, record.track_id)
                self._track_plate_map.pop(key, None)
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
                        "track_id": record.track_id,
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
                    "track_id": r.track_id,
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
                "tracked_ids": len(self._track_plate_map),
            }
