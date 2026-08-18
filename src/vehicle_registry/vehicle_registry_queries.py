import os
from typing import Dict, List, Optional


class VehicleRegistryQueryMixin:
    def get_slot_plate(self, slot_id: str) -> Optional[str]:
        with self._lock:
            session = self._parked.get(slot_id)
            return session.plate if session else None

    def rename_plate(self, old_plate: str, new_plate: str) -> int:
        """Rewrite a corrected plate onto every live parked session holding it.

        The DB column and the gallery folder are not enough on their own: for a
        camera this worker owns, the in-memory registry is authoritative and the
        slot API reads `get_slot_plate` in preference to `current_plate`. Leave
        this stale and VA writes the misread straight back over the correction on
        the next slot update.

        Returns how many sessions were rewritten. Zero is normal — the car has
        usually left by the time an exit correction arrives.
        """
        if not old_plate or not new_plate or old_plate == new_plate:
            return 0
        updated = 0
        with self._lock:
            for session in self._parked.values():
                if session is not None and session.plate == old_plate:
                    session.plate = new_plate
                    updated += 1
        return updated

    def _get_snapshot_url(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        # Preserve relative path for subdirectories (e.g., alerts/)
        rel = path.replace(os.sep, "/")
        base = (getattr(self, "_public_base_url", "") or "").rstrip("/")
        gateway = getattr(self, "_gateway_path_prefix", "").strip("/")
        prefix = getattr(self, "_snapshot_url_prefix", "snapshots").strip("/")
        # Build path: /{gateway}/{prefix}/{file} — gateway may be empty
        path_parts = "/".join(part for part in [gateway, prefix, rel] if part)
        return f"{base}/{path_parts}" if base else f"/{path_parts}"

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
