import os
from typing import Dict, List, Optional


class VehicleRegistryQueryMixin:
    def get_slot_plate(self, slot_id: str) -> Optional[str]:
        with self._lock:
            session = self._parked.get(slot_id)
            return session.plate if session else None

    def _get_snapshot_url(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        # Preserve relative path for subdirectories (e.g., alerts/)
        rel = path.replace(os.sep, "/")
        base = (getattr(self, "_public_base_url", "") or "").rstrip("/")
        return f"{base}/images/{rel}" if base else f"/images/{rel}"

    def _get_snapshot_urls(self, paths: List[str]) -> List[str]:
        return [
            url
            for url in (self._get_snapshot_url(path) for path in paths)
            if url is not None
        ]

    def get_plate_location(self, plate: str) -> Optional[Dict]:
        with self._lock:
            for slot_id, session in self._parked.items():
                if session.plate == plate:
                    return {
                        "plate_number": session.plate,
                        "slot_id": slot_id,
                        "slot_name": session.linked_slot_name,
                        "zone_id": session.linked_zone_id,
                        "zone_name": session.linked_zone_name,
                        "camera_id": session.linked_camera,
                        "floor": session.linked_floor,
                        "track_id": session.last_seen_track_id,
                        "parked_at": session.linked_at.isoformat()
                        if session.linked_at
                        else None,
                        "confirmed_at": session.first_seen_at.isoformat(),
                        "snapshot_url": self._get_snapshot_url(session.snapshot_path),
                        "gate_snapshot_urls": self._get_snapshot_urls(
                            session.gate_snapshot_paths,
                        ),
                        "gallery_snapshot_urls": self._get_snapshot_urls(
                            session.reference_snapshot_paths,
                        ),
                    }
            return None

    def get_all_parked(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "plate_number": session.plate,
                    "slot_id": slot_id,
                    "slot_name": session.linked_slot_name,
                    "zone_id": session.linked_zone_id,
                    "zone_name": session.linked_zone_name,
                    "camera_id": session.linked_camera,
                    "floor": session.linked_floor,
                    "track_id": session.last_seen_track_id,
                    "parked_at": session.linked_at.isoformat()
                    if session.linked_at
                    else None,
                    "confirmed_at": session.first_seen_at.isoformat(),
                    "snapshot_url": self._get_snapshot_url(session.snapshot_path),
                    "gate_snapshot_urls": self._get_snapshot_urls(
                        session.gate_snapshot_paths,
                    ),
                    "gallery_snapshot_urls": self._get_snapshot_urls(
                        session.reference_snapshot_paths,
                    ),
                }
                for slot_id, session in self._parked.items()
            ]

    def get_pending_entries(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "event_id": event.event_id,
                    "plate_number": event.plate,
                    "timestamp": event.timestamp.isoformat(),
                    "status": event.status,
                    "candidate_id": event.candidate_id,
                    "session_id": event.session_id,
                }
                for event in self._pending_events.values()
                if event.direction == "entry"
                and event.status in ("pending", "provisional")
            ]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "parked_count": len(self._parked),
                "pending_entries": sum(
                    1
                    for event in self._pending_events.values()
                    if event.direction == "entry"
                    and event.status in ("pending", "provisional")
                ),
                "total_visits": len(self._history),
                "active_sessions": len(self._sessions),
                "tracked_ids": len(self._track_session_map),
            }
